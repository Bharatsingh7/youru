import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from youru import tools
from youru.agent import Agent, is_hindi
from youru.assistant import strip_wake
from youru.audio import SAMPLE_RATE
from youru.config import DATA
from youru.confirm import VoiceConfirmer, parse_yes_no
from youru.pipeline import Pipeline, SpeakerGate
from youru.stt import pick_language
from youru.tts import clean, sentences
from youru.vad import SileroVAD

from test_pipeline import LISTEN, FixedGate, ScriptedVAD, ScriptedWake, run


# ---------------------------------------------------------------- tools

def test_schemas_are_complete():
    for s in tools.schemas():
        f = s["function"]
        assert f["description"], f["name"]
        for name, prop in f["parameters"]["properties"].items():
            assert "type" in prop, (f["name"], name)
    vol = tools.load_all()["volume"].schema["function"]["parameters"]
    assert vol["properties"]["action"]["enum"][:2] == ["get", "set"] and vol["required"] == ["action"]


def test_risky_tools_need_confirmation():
    asked = []
    out = tools.execute("run_command", {"command": "notify-send hi"}, lambda q: asked.append(q) or False)
    assert out.startswith("cancelled") and asked == ["run the command: notify-send hi"]
    out = tools.execute("run_command", '{"command": "echo hi"}', lambda q: pytest.fail("read-only must not ask"))
    assert "exit code 0" in out and "hi" in out


def test_safe_tools_never_ask():
    out = tools.execute("get_time", {}, lambda q: pytest.fail("should not ask"))
    assert any(day in out for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))


def test_bad_calls_return_errors():
    assert tools.execute("nope", {}, lambda q: True).startswith("error: unknown tool")
    assert tools.execute("get_time", "{not json", lambda q: True).startswith("error: arguments")
    assert tools.execute("get_time", {"x": 1}, lambda q: True).startswith("error: bad arguments")
    assert "outside your home" in tools.execute("read_text_file", {"path": "/etc/hostname"}, lambda q: True)


def test_voice_data_is_protected():
    out = tools.execute("write_file", {"path": str(DATA / "should_not_exist.txt"), "content": "x"}, lambda q: True)
    assert "protected" in out and not (DATA / "should_not_exist.txt").exists()


# ---------------------------------------------------------------- language, confirmation, wake stripping

@pytest.mark.parametrize("text,expected", [
    ("yes", True), ("Haan, kar do", True), ("हाँ", True), ("okay go ahead", True),
    ("no", False), ("nahi", False), ("mat karo", False), ("नहीं", False), ("yes no", False),
    ("what?", None), ("", None)])
def test_yes_no(text, expected):
    assert parse_yes_no(text) is expected


def test_voice_confirmer_retries_then_defaults_to_no():
    heard = iter(["umm", "what"])
    c = VoiceConfirmer(None, listen=lambda: np.zeros(10), transcribe=lambda a: next(heard))
    assert c("run ls") is False and c.log == ["umm", "what"]
    c = VoiceConfirmer(None, listen=lambda: np.zeros(10), transcribe=lambda a: "haan")
    assert c("run ls") is True


def test_language_allow_list_blocks_urdu():
    probs = [("ur", 0.55), ("hi", 0.30), ("en", 0.10), ("pa", 0.05)]
    assert pick_language(probs, ["en", "hi"]) == ("hi", 0.30)
    assert pick_language([("fr", 0.9)], ["en", "hi"])[0] == "en"


@pytest.mark.parametrize("text,expected", [
    ("Hey Jarvis, what's the time?", "what's the time?"), ("hey travis open firefox", "open firefox"),
    ("Jarvis. Volume kam karo", "Volume kam karo"), ("हे जार्विस, मौसम बताओ", "मौसम बताओ"),
    ("what is jarvis", "what is jarvis"), ("Hey Jarvis.", "")])
def test_strip_wake(text, expected):
    assert strip_wake(text) == expected


def test_tts_text_handling():
    assert list(sentences(["Hello the", "re. How are", " you? Fine"])) == ["Hello there.", "How are you?", "Fine"]
    assert list(sentences(["मौसम साफ़ है। तापमान ३० है"])) == ["मौसम साफ़ है।", "तापमान ३० है"]
    assert clean("**Bold** see https://x.com/a now") == "Bold see link now"
    assert is_hindi("आज") and not is_hindi("aaj ka weather")


# ---------------------------------------------------------------- agent loop with a fake LLM

def _chunk(content=None, tool_calls=None):
    return NS(choices=[NS(delta=NS(content=content, tool_calls=tool_calls))])


def _call(index, id=None, name=None, args=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class FakeLLM:
    def __init__(self, rounds):
        self.rounds = iter(rounds)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.requests.append(kw)
        return iter(next(self.rounds))


def test_agent_runs_tools_then_answers():
    llm = FakeLLM([
        [_chunk(tool_calls=[_call(0, "c1", "get_time", "")]), _chunk(tool_calls=[_call(0, args="{}")])],
        [_chunk("It is "), _chunk("eight PM.")],
    ])
    seen = []
    agent = Agent("http://x", confirm=lambda q: True, client=llm, on_tool=lambda n, a, r: seen.append(n))
    assert agent.ask("what time is it") == "It is eight PM."
    assert seen == ["get_time"]
    second = llm.requests[1]["messages"]
    assert second[-2]["tool_calls"][0]["function"]["name"] == "get_time"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert agent.history[-1] == {"role": "assistant", "content": "It is eight PM."}


def test_agent_hides_think_and_uses_hindi_prompt():
    llm = FakeLLM([[_chunk("<think>hmm</think>"), _chunk("नमस्ते!")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm)
    assert agent.ask("नमस्ते") == "नमस्ते!"
    assert "Devanagari" in llm.requests[0]["messages"][0]["content"]


def test_agent_memory_resets_when_idle():
    llm = FakeLLM([[_chunk("one")], [_chunk("two")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm, memory_reset_s=0)
    agent.ask("a")
    agent.ask("b")
    assert [m["role"] for m in llm.requests[1]["messages"]] == ["system", "user"]


# ---------------------------------------------------------------- pipeline: hotkey + bypassed voice lock

def test_trigger_listens_without_wake_word():
    pipe = Pipeline(ScriptedVAD([(1.0, 2.0)]), ScriptedWake(None), FixedGate(), LISTEN)
    run(pipe, 0.5)
    pipe.trigger()
    decisions = run(pipe, 3.0)
    assert len(decisions) == 1 and decisions[0].wake_score == 1.0
    assert decisions[0].t == pytest.approx(2.0 + 0.8, abs=0.1)


def test_bypassed_voice_lock_accepts_any_speaker(vad_model, clips):
    vad = SileroVAD(vad_model, 0.5, 0.35, 0.25, 0.8)
    gate = SpeakerGate(None, None, vad, NS(enabled=False))
    pad = np.zeros(SAMPLE_RATE, np.float32)
    assert gate.check(np.concatenate([pad, clips["b1"], pad]))[0] == "accept"
    assert gate.check(np.zeros(2 * SAMPLE_RATE, np.float32))[0] == "too_short"


def test_hinglish_speech_gets_hindi_reply_hint():
    llm = FakeLLM([[_chunk("ठीक है")], [_chunk("ok")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm)
    agent.ask("aap kaise ho", language="hi")
    assert "Devanagari" in llm.requests[0]["messages"][-1]["content"]
    agent.ask("how are you", language="en")
    assert "Devanagari" not in llm.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("text,expected", [
    ("Aaj ka mausam kaisa hai?", True), ("Gaana chalao", True), ("Firefox kholo", True),
    ("Computer ki battery kitni hai?", True), ("Open the file manager", False),
    ("What is the capital of Australia?", False), ("play some music", False), ("is it raining in Delhi", False)])
def test_looks_hinglish(text, expected):
    from youru.agent import looks_hinglish

    assert looks_hinglish(text) is expected


@pytest.mark.parametrize("text,tool", [
    ("Computer ki battery kitni hai?", "system_info"), ("Volume kitna hai abhi?", "volume"),
    ("Mujhe aaj ki taaza khabren batao", "news_search"), ("aaj ka mausam", "weather"),
    ("kitne baje hain", "get_time"), ("run the command uname -a", "run_command"), ("gaana chalao", "play_youtube"),
    ("Firefox kholo", "open_app")])
def test_tool_hints(text, tool):
    from youru.agent import tool_hints

    assert tool in tool_hints(text)


def test_no_hint_for_plain_questions():
    from youru.agent import tool_hints

    assert tool_hints("Explain in one line what a black hole is") == []


def test_cancelled_action_stops_the_model():
    llm = FakeLLM([[_chunk("Sure. "), _chunk(tool_calls=[_call(0, "c1", "run_command", '{"command": "rm notes.txt"}')])],
                   [_chunk("Here is the output: invented")]])
    agent = Agent("http://x", confirm=lambda q: False, client=llm)
    reply = agent.ask("run ls")
    assert "cancelled" in reply and "invented" not in reply and len(llm.requests) == 1


def test_forced_tool_call_when_model_only_talks():
    llm = FakeLLM([
        [_chunk("I am setting the brightness to 50 percent.")],                       # talks, no tool
        [_chunk(tool_calls=[_call(0, "c1", "get_time", "{}")])],                      # forced retry
        [_chunk("It is 9 PM.")],
    ])
    seen = []
    agent = Agent("http://x", confirm=lambda q: True, client=llm, on_tool=lambda n, a, r: seen.append(n))
    reply = agent.ask("kitne baje hain")
    assert seen == ["get_time"] and reply == "It is 9 PM." and "setting" not in reply
    assert llm.requests[1]["tool_choice"] == "required"
    assert llm.requests[1]["tools"] == llm.requests[0]["tools"]   # same list keeps the prompt cache valid


def test_hinted_request_without_tool_still_answers():
    llm = FakeLLM([[_chunk("Open source means the code is public.")], [_chunk("still talking")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm)
    assert agent.ask("what does open source mean") == "Open source means the code is public."


@pytest.mark.parametrize("text,wanted,unwanted", [
    ("VS Code kholo", "open_app", "find_code"), ("take a screenshot", "screenshot", "brightness"),
    ("meri beach wali photo dhundo", "find_anything", None), ("find the code that sets the wallpaper", "find_code", None),
    ("external monitor ki brightness 50 karo", "brightness", None)])
def test_tool_hint_routing(text, wanted, unwanted):
    from youru.agent import tool_hints

    hints = tool_hints(text)
    assert wanted in hints and (unwanted is None or unwanted not in hints)



@pytest.mark.parametrize("command,level", [
    ("ls ~/Downloads", "low"), ("df -h | grep nvme", "low"), ("git status", "low"), ("uptime && free -h", "low"),
    ("echo hi > ~/.bashrc", "medium"), ("rm notes.txt", "medium"), ("git push", "medium"),
    ("find ~ -name x -delete", "medium"), ("notify-send hi", "medium"), ("sudo pacman -Syu", "high"),
    ("rm -rf /", "critical"), ("dd if=/dev/zero of=/dev/nvme0n1", "critical")])
def test_shell_risk(command, level):
    from youru.tools import risk

    assert risk.shell({"command": command})[0] == level


@pytest.mark.parametrize("code,level", [
    ("print(2**10)", "low"), ("import math; math.sqrt(2)", "low"), ("open('x','w').write('a')", "medium"),
    ("import subprocess; subprocess.run(['ls'])", "high"), ("import shutil; shutil.rmtree('/home/cyberwolf')", "critical")])
def test_python_risk(code, level):
    from youru.tools import risk

    assert risk.python({"code": code})[0] == level


def test_critical_is_blocked_even_if_user_says_yes():
    out = tools.execute("run_command", {"command": "rm -rf /"}, lambda q: True)
    assert out.startswith("blocked")


def test_python_session_keeps_variables_and_survives_errors():
    from youru.tools.pyrun import PythonSession

    py = PythonSession(timeout_s=10)
    assert py.run("x = 20") == "(ran with no output)"
    assert py.run("x + 1") == "21"
    assert "ZeroDivisionError" in py.run("1/0")
    assert py.run("x") == "20"


def test_python_timeout_restarts_session():
    from youru.tools.pyrun import PythonSession

    py = PythonSession(timeout_s=1)
    py.run("y = 1")
    assert "stopped" in py.run("while True: pass")
    assert "NameError" in py.run("y")


def test_edit_file_and_grep(tmp_path, monkeypatch):
    from youru.tools import files

    monkeypatch.setattr(files, "HOME", tmp_path.resolve())
    f = tmp_path / "contacts.txt"
    f.write_text("9414060832\n")
    out = tools.execute("edit_file", {"path": str(f), "find_text": "9414060832", "replace_text": "Bharat: 9414060832"},
                        lambda q: True)
    assert "Replaced 1" in out and f.read_text() == "Bharat: 9414060832\n"
    assert "contacts.txt:1" in tools.execute("search_in_files", {"text": "Bharat", "folder": str(tmp_path)},
                                             lambda q: True)



@pytest.mark.parametrize("text,tool", [
    ("Arijit Singh ka gaana chalao", "play_youtube"), ("play GTA 6 trailer video", "play_youtube"),
    ("agla gaana", "media"), ("pause", "media"), ("music ruko", "media"),
    ("If I have 3 apples and eat one, then buy twice as many as I have left, how many do I have?", "run_python"),
    ("what is 234 * 17", "run_python")])
def test_media_and_maths_hints(text, tool):
    from youru.agent import tool_hints

    assert tool in tool_hints(text)


def test_play_youtube_video_and_music(monkeypatch):
    from youru.tools import media

    results = [{"id": f"id{i}", "title": f"Song {i}", "duration": 200, "channel": "T"} for i in range(1, 6)]
    monkeypatch.setattr(media, "MPV_SOCKET", Path("/nonexistent-test-dir/youru-mpv.sock"))   # never the real one
    monkeypatch.setattr(media, "youtube_search", lambda q, n=5: results)
    monkeypatch.setattr(media, "music_playing", lambda: False)
    monkeypatch.setattr(media, "mpv_command", lambda *a: None)
    monkeypatch.setattr(media, "_player_pids", lambda: [])
    spawned = []
    monkeypatch.setattr(media.subprocess, "Popen", lambda args, **kw: spawned.append(args))
    out = media.play_youtube("some video", kind="video")
    assert spawned[-1] == ["xdg-open", "https://www.youtube.com/watch?v=id1"] and "Song 1" in out
    out = media.play_youtube("", kind="music", pick=2)   # "play the second one" of the last results
    args = spawned[-1]
    assert args[0] == "mpv" and "--no-video" in args and args[-4:] == [
        f"https://www.youtube.com/watch?v=id{i}" for i in (2, 3, 4, 5)]
    assert "Song 2" in out and "3 more queued" in out


def test_media_tool_prefers_youru_music_player(monkeypatch):
    from youru.tools import desktop, media

    monkeypatch.setattr(media, "music_playing", lambda: True)
    sent = []
    monkeypatch.setattr(media, "mpv_command", lambda *a: sent.append(a) or {"error": "success", "data": "x"})
    assert desktop.media("next") == "next song" and sent[-1] == ("playlist-next", "force")


def test_wake_gate_keeps_detection(vad_model, clips):
    """The wake model is skipped in silence but sees the audio just before speech."""
    from youru.audio import FRAME
    from youru.wakeword import WakeWord
    from youru import models

    w = WakeWord(models.ensure("hey_jarvis"), gate=True)
    calls = []
    real = w.model.predict
    w.model.predict = lambda x: calls.append(1) or real(x)
    silence = np.zeros(FRAME, np.float32)
    for i in range(50):                       # 4 s of silence, no speech
        w.process(silence, i * 0.08, False)
    idle_calls = len(calls)
    w.process(silence, 4.08, True)            # speech starts: backlog (1.5 s) is fed first
    assert idle_calls <= 13 and len(calls) - idle_calls >= 18


def test_stt_unload_and_reload(monkeypatch):
    from youru import stt as stt_mod

    loads = []
    monkeypatch.setattr(stt_mod.STT, "load", lambda self: loads.append(1) or setattr(self, "model", object()))
    s = stt_mod.STT("x", "cpu")
    assert s.loaded and loads == [1]
    s.unload()
    assert not s.loaded


# ---------------------------------------------------------------- thinking, solve mode, learning

@pytest.fixture
def mem(tmp_path, monkeypatch):
    from youru import memory

    monkeypatch.setattr(memory, "HOME", tmp_path / "youru")
    return memory


@pytest.mark.parametrize("text,expected", [
    ("Why is the sky blue?", True), ("Which is better for gaming, SSD or HDD?", True), ("Neend kyun zaroori hai?", True),
    ("Kitne baje hain?", False), ("Why is my volume so low? set it to 50", False), ("Open Firefox", False)])
def test_needs_thinking(text, expected):
    from youru.agent import needs_thinking

    assert needs_thinking(text, tool_hints_of(text)) is expected


def tool_hints_of(text):
    from youru.agent import tool_hints

    return tool_hints(text)


def test_thinking_filler_comes_first_and_is_not_remembered():
    llm = FakeLLM([[_chunk("Because of Rayleigh scattering.")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm, thinking=True)
    chunks = list(agent.ask_stream("Why is the sky blue?"))
    assert chunks[0].startswith("Hmm, let me think") and "Rayleigh" in "".join(chunks)
    assert llm.requests[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert agent.turns[-1][-1]["content"] == "Because of Rayleigh scattering."


def test_refusal_is_not_spoken_and_solve_mode_learns_a_recipe(mem):
    llm = FakeLLM([
        [_chunk("I cannot change that setting. "), _chunk("Sorry.")],                     # refusal: held back
        [_chunk(tool_calls=[_call(0, "c1", "how_to", '{"task": "say hello"}')])],
        [_chunk(tool_calls=[_call(0, "c2", "run_command", '{"command": "echo hello"}')])],
        [_chunk("Done, it printed hello.")],
    ])
    agent = Agent("http://x", confirm=lambda q: True, client=llm, solve_on_refusal=True, learn=True)
    chunks = list(agent.ask_stream("say hello in the terminal"))
    text = "".join(chunks)
    assert "cannot" not in text and text.startswith("Let me find a way") and "Done" in text
    assert mem.learned_recipes()[0]["command"] == "echo hello"


def test_normal_first_sentence_is_released_unchanged():
    llm = FakeLLM([[_chunk("Paris is the "), _chunk("capital. It is in France.")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm, solve_on_refusal=True)
    assert agent.ask("capital of France") == "Paris is the capital. It is in France."


def test_missing_step_is_nudged():
    llm = FakeLLM([
        [_chunk(tool_calls=[_call(0, "c1", "get_time", "{}")])],
        [_chunk("Here is the time and I copied it.")],                    # stops early: held, not spoken
        [_chunk(tool_calls=[_call(0, "c2", "clipboard_get", "{}")])],     # forced: the missing step
        [_chunk("Done.")],
    ])
    seen = []
    agent = Agent("http://x", confirm=lambda q: True, client=llm, on_tool=lambda n, a, r: seen.append(n))
    reply = agent.ask("kitne baje hain aur clipboard copy dikhao")
    assert seen == ["get_time", "clipboard_get"] and reply == "Done."


def test_memory_facts_and_relevance(mem):
    mem.remember_fact("The user's name is Bharat.")
    mem.remember_fact("The user lives in Beawar, Rajasthan.")
    mem.remember_fact("The user lives in Jaipur, Rajasthan.")       # an update replaces the old fact
    assert len(mem.facts()) == 2
    assert mem.relevant_facts("open firefox") == []                  # unrelated: nothing attached
    assert "The user's name is Bharat." in mem.relevant_facts("mera naam kya hai")
    assert any("Jaipur" in f for f in mem.relevant_facts("weather in my city"))
    assert "forgot 1" in mem.forget_fact("Jaipur Rajasthan")


def test_corrections(mem):
    assert mem.correction_target("Nahi, browser kholo") == "browser kholo"
    assert mem.correction_target("no") is None and mem.correction_target("browser kholo") is None
    mem.save_correction("Officer kholo.", "browser kholo")
    assert mem.apply_correction("officer kholo") == "browser kholo"
    assert mem.apply_correction("terminal kholo") is None


def test_vocab_comes_from_facts_not_corrections(mem):
    mem.save_correction("Open drive browser", "PRAVE browser")
    assert mem.vocab() == []
    mem.remember_fact("The user studies at AMITY University in Jaipur")
    assert mem.vocab() == ["AMITY", "University", "Jaipur"]


def test_multi_step_success_becomes_a_skill(mem):
    llm = FakeLLM([
        [_chunk(tool_calls=[_call(0, "c1", "get_time", "{}"), _call(1, "c2", "system_info", "{}")])],
        [_chunk("Done.")]])
    agent = Agent("http://x", confirm=lambda q: True, client=llm, learn=True)
    agent.ask("tell me time and system status")
    assert [s["tool"] for s in mem.learned_skills()[0]["steps"]] == ["get_time", "system_info"]


def test_how_to_finds_curated_and_learned(mem):
    from youru.tools.recipes import how_to

    assert "xdg-settings" in how_to("make firefox the default browser")
    assert "asusctl profile" in how_to("make fans quieter")
    mem.save_recipe("show my calendar", "cal")
    assert "(learned)" in how_to("show calendar")


def test_review_filters(mem):
    from youru.learn import command_ok, failures

    assert command_ok("asusctl profile get") and not command_ok("echo '[ACTION]'")
    assert not command_ok("distro_update_check") and not command_ok("rm -rf /")
    rows = [{"transcript": "Make my browser default", "reply": "I cannot change that.", "tools": []},
            {"transcript": "Open drive browser", "reply": "ok", "tools": [{"name": "open_app", "result": "opened"}]},
            {"transcript": "Nahi, Brave browser kholo", "reply": "ok", "tools": []}]
    problems = [p for _, p, _ in failures(rows)]
    assert problems[:2] == ["it said it couldn't do it", "the user corrected it"]


@pytest.mark.parametrize("command,level", [
    ("powerprofilesctl get", "low"), ("asusctl profile get", "low"), ("bluetoothctl show", "low"),
    ("nmcli radio wifi", "low"), ("nmcli radio wifi off", "medium"), ("xdg-settings get default-web-browser", "low"),
    ("xdg-settings set default-web-browser firefox.desktop", "medium"), ("distro_update_check", "low"),
    ("fakecmd; rm -rf ~/x", "high")])
def test_read_only_forms_and_missing_commands(command, level):
    from youru.tools import risk

    assert risk.shell({"command": command})[0] == level


# ---------------------------------------------------------------- reminders, screen, organizer

def test_timer_and_reminder_args():
    from datetime import datetime

    from youru.tools import reminders as r

    a = r.timer_args("youru-rem-x", "tea timer", "Tea is ready", seconds=300)
    assert a[:3] == ["systemd-run", "--user", "--unit=youru-rem-x"] and "--on-active=300s" in a
    assert a[-3:] == ["youru", "alert", "Tea is ready"]
    when = datetime(2099, 1, 1, 7, 0)
    b = r.timer_args("youru-rem-y", "call mom", "Call mom", at=when)
    assert "--on-calendar=2099-01-01 07:00:00" in b and "--timer-property=Persistent=true" in b


@pytest.mark.parametrize("text,expected", [
    ("2026-10-23 06:00", (2026, 10, 23, 6, 0)), ("19:30", (2026, 10, 22, 19, 30)),
    ("7 pm", (2026, 10, 22, 19, 0)), ("9 am", (2026, 10, 23, 9, 0)), ("12 am", (2026, 10, 23, 0, 0))])
def test_parse_when(text, expected):
    from datetime import datetime

    from youru.tools.reminders import parse_when

    got = parse_when(text, now=datetime(2026, 10, 22, 11, 0))
    assert (got.year, got.month, got.day, got.hour, got.minute) == expected


def test_reminder_in_the_past_is_refused(monkeypatch):
    from youru.tools import reminders as r

    monkeypatch.setattr(r, "_run", lambda args: pytest.fail("must not create a timer"))
    assert r.set_reminder("2001-01-01 10:00", "x").startswith("error")


def test_read_screen_uses_window_geometry(monkeypatch):
    from youru.tools import screen

    calls = []
    monkeypatch.setattr(screen, "active_window", lambda: {"class": "kitty", "title": "zsh", "at": [10, 20],
                                                           "size": [800, 600]})

    def fake_run(args, **kw):
        calls.append(args)
        return type("R", (), {"returncode": 0, "stdout": ""})()
    monkeypatch.setattr(screen.subprocess, "run", fake_run)
    monkeypatch.setattr(screen, "ocr", lambda png: "error: permission denied")
    out = screen.read_screen("window")
    assert calls[0][:3] == ["grim", "-g", "10,20 800x600"] and "permission denied" in out and "kitty" in out


def test_deadlines_wrapper(monkeypatch):
    from youru.tools import organizer

    store = [{"id": 3, "name": "Physics exam", "completed": False, "due_str": "Nov 05", "remaining": "14 days",
              "remaining_seconds": 14 * 86400}]
    sent = []

    def fake_tracker(*args):
        sent.append(args)
        return json.dumps(store) if args[0] == "list" else '{"ok": true}'
    monkeypatch.setattr(organizer, "_tracker", fake_tracker)
    assert "Physics exam" in organizer.list_deadlines(days=30) and "nothing due" in organizer.list_deadlines(days=7)
    assert organizer.complete_deadline("physics") == "marked done: Physics exam" and sent[-1] == ("complete", "3")
    organizer.add_deadline("Maths submission", "2026-11-10", "17:00")
    assert sent[-1] == ("add", "--name", "Maths submission", "--date", "2026-11-10", "--time", "17:00")
    with pytest.raises(ValueError):
        organizer.add_deadline("x", "5 November")


def test_notes_and_todo(tmp_path, monkeypatch):
    from youru.tools import organizer

    monkeypatch.setattr(organizer, "NOTES_DIR", tmp_path)
    organizer.add_note("buy milk")
    assert "buy milk" in organizer.read_notes()
    organizer.add_todo("finish robotics report")
    organizer.add_todo("call mom")
    assert organizer.done_todo("robotics") == "done: finish robotics report"
    assert organizer.list_todo() == "- [ ] call mom"


@pytest.mark.parametrize("text,tool", [
    ("5 minute ka timer lagao", "set_timer"), ("remind me at 7 pm to call mom", "set_reminder"),
    ("kal subah 6 baje uthana", "set_reminder"), ("what's on my screen", "read_screen"),
    ("is error ko padho", "read_screen"), ("add exam deadline on 5 November", "add_deadline"),
    ("is hafte kya due hai", "list_deadlines"), ("note likh lo: buy milk", "add_note"),
    ("mere todo dikhao", "list_todo")])
def test_new_tool_hints(text, tool):
    from youru.agent import tool_hints

    assert tool in tool_hints(text)


def test_detail_mode_speaks_only_the_start():
    from types import SimpleNamespace

    from youru.assistant import spoken_part

    agent = SimpleNamespace(detail=True)
    chunks = ["One. ", "Two. ", "Three. ", "Four. ", "Five."]
    assert "".join(spoken_part(iter(chunks), agent)) == "One. Two. Three.  The rest is on your screen."
    agent.detail = False
    assert "".join(spoken_part(iter(chunks), agent)) == "".join(chunks)


@pytest.mark.skipif(not (Path(__file__).parents[1] / "models/tts/kokoro/kokoro-v1.0.onnx").exists(),
                    reason="Kokoro model not downloaded")
def test_kokoro_voice_and_cute_pitch():
    from youru.tts import KokoroVoice

    v = KokoroVoice("af_sky:0.6,jf_nezumi:0.4", pitch=1.07)
    audio, sr = v.synth("Hi!")
    assert len(audio) > 2400 and sr == int(24000 * 1.07)


def test_malformed_tool_call_is_retried_then_given_up():
    class Broken(FakeLLM):
        def create(self, **kw):
            self.requests.append(kw)
            raise RuntimeError("Error code: 500 - Failed to parse tool call arguments as JSON")
    agent = Agent("http://x", confirm=lambda q: True, client=Broken([]))
    assert "say that again" in agent.ask("Mera naam Bharat hai, yaad rakhna")


def test_malformed_tool_call_recovers_on_retry():
    rounds = iter([RuntimeError("Failed to parse tool call arguments as JSON"),
                   [_chunk(tool_calls=[_call(0, "c1", "get_time", "{}")])], [_chunk("It is 9.")]])

    class Flaky(FakeLLM):
        def create(self, **kw):
            self.requests.append(kw)
            r = next(rounds)
            if isinstance(r, Exception):
                raise r
            return iter(r)
    llm = Flaky([])
    agent = Agent("http://x", confirm=lambda q: True, client=llm)
    assert agent.ask("what time") == "It is 9." and llm.requests[1]["temperature"] == 0.0


@pytest.mark.parametrize("text,has,hasnt", [
    ("Use the GLaDOS voice.", "set_voice", "volume"), ("Apni normal awaaz mein wapas aao.", "set_voice", "volume"),
    ("Awaaz 30 kar do.", "volume", "set_voice")])
def test_voice_vs_volume_hints(text, has, hasnt):
    from youru.agent import tool_hints

    hints = tool_hints(text)
    assert has in hints and hasnt not in hints


def test_set_voice_switches_and_remembers(tmp_path, monkeypatch):
    from youru import tts
    from youru.tools import voice

    saved = {}
    monkeypatch.setattr(voice, "update_config", lambda u: saved.update(u))
    sp = tts.Speaker(__import__("youru.config", fromlist=["load_config"]).load_config().tts)
    assert voice.set_voice("GLaDOS") == "switched the English voice to glados" and sp.english == "glados"
    assert saved == {"tts": {"english_voice": "glados"}}
    assert voice.set_voice("normal").endswith("default") and sp.english == "default"
    assert "no voice" in voice.set_voice("darth")
