"""Phase 2: the assistant loop. Wake word / hotkey -> STT -> local LLM + tools -> spoken reply."""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator

import numpy as np
from rich.console import Console

from .config import resolve

console = Console()

WAKE_PREFIX = re.compile(
    r"^\s*((hey|hi|hay|he|ok|okay|a)[\s,]+)?(jarvis|jarvais|javis|travis|service|youru|yuru|yooru)\b[\s,.!?]*"
    r"|^\s*(हे\s*)?(जार्विस|जारविस|यूरू|युरु)[\s,।.!?]*", re.I)


def strip_wake(text: str) -> str:
    return WAKE_PREFIX.sub("", text, count=1).strip()


def make_stt(cfg):
    from .stt import STT

    s = cfg.stt
    model = str(resolve(s.model)) if str(s.model).startswith(("models/", "/", "~")) else s.model
    return STT(model, s.device, s.compute_type, s.initial_prompt, s.get("languages"))


def ensure_llm(cfg):
    from .llm import LlamaServer

    srv = LlamaServer(cfg.llm)
    if not srv.healthy():
        with console.status(f"starting llama-server ({srv.model_path.name})..."):
            srv.start()
    return srv


def make_agent(cfg, srv, confirm):
    from .agent import Agent

    def on_tool(name, args, result):
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        console.print(f"  [dim]⚙ {name}({shown}) → {result[:120].replace(chr(10), ' ')}[/]")

    a = cfg.assistant
    agent = Agent(srv.url, confirm, cfg.llm.temperature, cfg.llm.max_tokens, a.max_tool_rounds, a.memory_turns,
                  a.memory_reset_s, on_tool=on_tool, top_p=cfg.llm.get("top_p", 0.95), top_k=cfg.llm.get("top_k", 64),
                  normalize_hinglish=a.get("normalize_hinglish", True), thinking=a.get("thinking", True),
                  think_budget=cfg.llm.get("think_budget", 128), solve_on_refusal=a.get("solve_on_refusal", True),
                  learn=a.get("learn", True))
    with console.status("warming up the prompt cache..."):
        srv.warm_up([agent.system_message(), {"role": "user", "content": "hi"}], agent.tool_schemas)

    def restart_server() -> bool:
        """If llama-server died (e.g. the GPU ran out of memory), start it again once."""
        if srv.healthy():
            return False
        try:
            srv.stop()
            srv.start()
            srv.warm_up([agent.system_message(), {"role": "user", "content": "hi"}], agent.tool_schemas)
            return True
        except SystemExit as e:
            console.print(f"[red]{e}[/]")
            return False

    agent.restart_server = restart_server
    return agent


def turn_tools(agent) -> list[dict]:
    """The tool calls of the agent's last turn, for the session log."""
    if not agent.turns:
        return []
    calls, results = [], {}
    for m in agent.turns[-1]:
        if m["role"] == "assistant":
            calls += m.get("tool_calls") or []
        elif m["role"] == "tool":
            results[m["tool_call_id"]] = m["content"][:200]
    return [{"name": c["function"]["name"], "args": c["function"]["arguments"], "result": results.get(c["id"], "")}
            for c in calls]


def notify(title: str, body: str) -> None:
    subprocess.Popen(["notify-send", "-a", "Youru", title, body[:300]],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tee(chunks: Iterable[str], out: list[str]) -> Iterator[str]:
    """Print chunks as they stream and remember them."""
    for c in chunks:
        out.append(c)
        console.print(c, end="", soft_wrap=True, highlight=False, markup=False)
        yield c


def spoken_part(chunks: Iterable[str], agent, max_sentences: int = 3) -> Iterator[str]:
    """In detail mode the whole answer is shown, but only the first sentences are spoken."""
    from .tts import SENTENCE_END

    said = 0
    for c in chunks:
        if not agent.detail:
            yield c
            continue
        if said < max_sentences:
            said += len(SENTENCE_END.findall(c))
            yield c
    if agent.detail and said >= max_sentences:
        yield " The rest is on your screen."


def answer(agent, speaker, text: str, language: str | None = None, retry: bool = True) -> str:
    console.print("[bold green]youru>[/] ", end="")
    parts: list[str] = []
    try:
        stream = tee(agent.ask_stream(text, language), parts)
        if speaker is not None:
            speaker.say_stream(spoken_part(stream, agent))
        else:
            for _ in stream:
                pass
    except Exception as e:  # noqa: BLE001
        console.print(f"\n[red]LLM error: {e}[/]")
        restart = getattr(agent, "restart_server", None)
        if retry and restart is not None and restart():
            console.print("[yellow]llama-server restarted; asking again[/]")
            return answer(agent, speaker, text, language, retry=False)
        msg = "Sorry, my brain is not responding right now."
        if speaker is not None:
            speaker.say(msg)
        return msg
    console.print()
    return "".join(parts).strip()


# ---------------------------------------------------------------- typed chat

def run_chat(cfg, speak: bool = False) -> None:
    from .confirm import TerminalConfirmer
    from .tts import Speaker

    srv = ensure_llm(cfg)
    agent = make_agent(cfg, srv, TerminalConfirmer(console))
    speaker = Speaker(cfg.tts) if speak else None
    console.print(f"[dim]model {srv.running_model() or srv.model_path.name} · /reset clears memory · "
                  "Ctrl+D or /exit quits[/]")
    while True:
        try:
            text = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/reset":
            agent.history = []
            continue
        if text:
            answer(agent, speaker, text)


# ---------------------------------------------------------------- voice assistant

def beep(freq: float = 880, dur: float = 0.12) -> None:
    import sounddevice as sd

    t = np.arange(int(24000 * dur)) / 24000
    tone = 0.15 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    sd.play(tone.astype(np.float32), 24000)


def run_assist(cfg, speak: bool = True) -> None:
    from .audio import MicSource
    from .confirm import VoiceConfirmer
    from .hotkey import SOCKET, TriggerServer
    from .pipeline import Decision, build
    from .record import capture_utterance
    from .tts import Speaker
    from .wakeword import WakeEvent

    # Whisper first, so llama-server fits itself into whatever VRAM is left.
    with console.status(f"loading whisper {cfg.stt.model}..."):
        stt = make_stt(cfg)
    srv = ensure_llm(cfg)
    speaker = None
    if speak and cfg.tts.enabled:
        with console.status("loading voices..."):
            speaker = Speaker(cfg.tts)
            speaker.warm_up()
    pipe = build(cfg)
    mic = MicSource(cfg.audio.device, cfg.audio.channel)
    confirm_vad = pipe.gate.vad   # the gate's detector is free while we wait for a yes/no; no third copy
    confirmer = VoiceConfirmer(
        speaker,
        listen=lambda: capture_utterance(mic, confirm_vad, start_timeout_s=cfg.assistant.confirm_timeout_s, max_s=4),
        transcribe=lambda audio: stt.transcribe(audio).text,
        notify=notify if cfg.assistant.notify else None)
    agent = make_agent(cfg, srv, confirmer)
    trigger = TriggerServer()

    def on_command(d: Decision) -> None:
        tr = stt.transcribe(d.audio, cfg.stt.language)
        d.transcript, d.language = tr.text, tr.language
        text = strip_wake(tr.text)
        console.print(f"[bold cyan]you>[/] {tr.text} [dim]({tr.language}, {tr.seconds:.2f}s)[/]")
        if not text:
            if speaker:
                speaker.say("Yes?")
            return
        t0 = time.perf_counter()
        reply = answer(agent, speaker, text, tr.language)
        console.print(f"[dim]  {time.perf_counter() - t0:.1f}s[/]")
        d.reply, d.meaning, d.tools = reply, agent.last_meaning or None, turn_tools(agent)
        if cfg.assistant.notify and reply:
            notify("Youru", reply)

    pipe.on_owner_utterance(on_command)

    sleep_after = float(cfg.assistant.get("sleep_after_min", 15)) * 60
    activity = [time.monotonic()]

    def on_event(ev) -> None:
        if isinstance(ev, WakeEvent):
            activity[0] = time.monotonic()
            console.print(f"[magenta]● listening[/] [dim](score {ev.score:.2f})[/]")
            beep()
            if not stt.loaded:   # asleep: reload Whisper while the command is still being spoken
                console.print("[dim]  waking up...[/]")
                threading.Thread(target=stt.load, daemon=True).start()
        elif isinstance(ev, Decision) and ev.outcome != "accept":
            console.print(f"[dim]  ({ev.outcome}, {ev.speech_s:.1f}s speech)[/]")

    pipe.on_event(on_event)
    from . import memory

    if waiting := memory.pending():
        console.print(f"[yellow]I learned {len(waiting)} new command(s) from yesterday's mistakes; "
                      "approve them with: youru learn --review[/]")
        if speaker:
            speaker.say(f"I learned {len(waiting)} new things yesterday. Approve them with youru learn review.")
    console.rule("[bold]Youru is ready")
    console.print(f"Say [bold]\"{pipe.wake.name.split('_v')[0].replace('_', ' ')}\"[/] or press your hotkey "
                  f"([dim]youru trigger → {SOCKET}[/]). Ctrl+C quits.")
    try:
        with mic:
            for frame in mic.frames():
                if trigger.take():
                    pipe.trigger()
                if pipe.feed(frame) is not None:
                    mic.drain()      # drop audio heard while we were thinking/speaking
                    pipe.resync()
                    activity[0] = time.monotonic()
                elif sleep_after and stt.loaded and time.monotonic() - activity[0] > sleep_after:
                    stt.unload()     # idle: give the GPU memory back; the wake word still works
                    console.print(f"[dim]💤 idle for {sleep_after / 60:.0f} min: Whisper unloaded "
                                  "(wakes on the wake word)[/]")
    except KeyboardInterrupt:
        pass
    finally:
        trigger.close()


# ---------------------------------------------------------------- llm management

def run_llm(cfg, action: str, models: list[str] | None = None) -> None:
    from .llm import LOG_FILE, LlamaServer

    srv = LlamaServer(cfg.llm)
    if action == "start":
        ensure_llm(cfg)
        console.print(f"[green]llama-server running[/] at {srv.url} ({srv.running_model()}); log: {LOG_FILE}")
    elif action == "stop":
        console.print("stopped" if srv.stop() else "not running (no pid file)")
    elif action == "status":
        console.print(f"{'[green]running[/]' if srv.healthy() else '[red]not running[/]'} {srv.url} "
                      f"{srv.running_model() or ''}")
    elif action == "bench":
        _bench(cfg, models)


def _vram_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def _bench(cfg, models: list[str] | None) -> None:
    import copy

    import psutil
    from rich.table import Table

    from .llm import PID_FILE, LlamaServer

    candidates = models or ["models/llm/Qwen3.5-4B-Q4_K_M.gguf", "models/llm/Qwen3.5-2B-Q4_K_M.gguf"]
    with console.status("loading whisper (to share the GPU like the real assistant)..."):
        stt = make_stt(cfg)  # noqa: F841  (kept alive on purpose)
    base = _vram_mb()
    t = Table(title=f"LLM benchmark (Whisper already using the GPU: {base} MB VRAM)")
    for col in ("model", "load s", "first token s", "prompt tok/s", "gen tok/s", "VRAM MB", "server RAM MB"):
        t.add_column(col)
    for m in candidates:
        c = copy.deepcopy(dict(cfg.llm))
        c["model"] = m
        srv = LlamaServer(type(cfg.llm)(c))
        srv.stop()
        time.sleep(1)
        t0 = time.perf_counter()
        srv.start()
        load = time.perf_counter() - t0
        r = srv.bench()
        pid = int(PID_FILE.read_text())
        rss = psutil.Process(pid).memory_info().rss / 2**20
        t.add_row(m.split("/")[-1], f"{load:.1f}", f"{r['ttft_s']:.2f}", f"{r['prompt_tok_s']:.0f}",
                  f"{r['gen_tok_s']:.1f}", str(_vram_mb()), f"{rss:.0f}")
        srv.stop()
        time.sleep(1)
    console.print(t)
    console.print("Pick one by setting llm.model in config.yaml.")


# ---------------------------------------------------------------- session history

def show_log(last: int = 20) -> None:
    import json

    from .config import DATA

    rows = []
    for f in sorted((DATA / "sessions").glob("*/sessions.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for r in rows[-last:]:
        if not r.get("transcript"):
            continue
        console.print(f"[dim]{r['time'][:19]}[/] [cyan]you>[/] {r['transcript']}")
        if r.get("meaning"):
            console.print(f"  [dim]meaning:[/] {r['meaning']}")
        for t in r.get("tools") or []:
            console.print(f"  [dim]⚙ {t['name']}({t['args']}) → {t['result'][:100]}[/]")
        if r.get("tools") == [] and r.get("reply") is not None:
            console.print("  [yellow](no tool used)[/]")
        if r.get("reply"):
            console.print(f"  [green]youru>[/] {r['reply']}")
