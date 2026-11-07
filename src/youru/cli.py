"""`youru` command line."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from . import models
from .audio import SAMPLE_RATE, MicSource, FileSource, clip_ratio, dbfs, load_wav, to_mono
from .config import DATA, ROOT, load_config, resolve, update_config

console = Console()


def make_vad(cfg, min_silence_s: float | None = None):
    from .vad import SileroVAD

    return SileroVAD(resolve(cfg.vad.model), cfg.vad.threshold, cfg.vad.neg_threshold, cfg.vad.min_speech_s,
                     cfg.vad.min_silence_s if min_silence_s is None else min_silence_s)


def mic(cfg) -> MicSource:
    return MicSource(cfg.audio.device, cfg.audio.channel)


def bar(v: float, width: int = 20, thr: float | None = None) -> str:
    n = int(max(0.0, min(1.0, v)) * width)
    color = "green" if thr is not None and v >= thr else "cyan"
    return f"[{color}]{'█' * n}[/][dim]{'░' * (width - n)}[/] {v:.2f}"


# ---------------------------------------------------------------- devices / mic-check

def cmd_devices(cfg, args) -> None:
    import sounddevice as sd

    default = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "[green]*[/]" if i == default else " "
            console.print(f"{mark} {i:2d}  {d['name']}  [dim]({d['max_input_channels']} ch, "
                          f"{int(d['default_samplerate'])} Hz)[/]")
    console.print(f"\nUsing: [bold]{mic(cfg).name}[/]  (set audio.device in config.yaml to change)")


def _source_volume() -> str:
    try:
        return subprocess.run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"], capture_output=True,
                              text=True, timeout=3).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def cmd_mic_check(cfg, args) -> None:
    import sounddevice as sd
    import soxr

    from .denoise import save_noise_profile

    info = sd.query_devices(cfg.audio.device, "input")
    rate, ch = int(info["default_samplerate"]), min(2, int(info["max_input_channels"]))
    console.print(f"Mic: [bold]{info['name']}[/], {ch} ch @ {rate} Hz, PipeWire volume: {_source_volume()}")

    def rec(seconds):
        x = sd.rec(int(seconds * rate), samplerate=rate, channels=ch, dtype="float32", device=cfg.audio.device)
        sd.wait()
        return x

    console.input("\n[bold]Step 1/2:[/] stay [bold]silent[/] for 5 s (leave the fan as it usually is). Enter to start ")
    silence = rec(5)
    console.input("[bold]Step 2/2:[/] read this aloud at normal volume for 5 s:\n"
                  "  [cyan]\"Youru, aaj ka weather batao aur phir music chalao, please.\"[/]  Enter to start ")
    speech = rec(5)

    vad = make_vad(cfg)
    mono16 = lambda x: soxr.resample(to_mono(x, cfg.audio.channel), rate, SAMPLE_RATE).astype(np.float32)
    noise16, speech16 = mono16(silence), mono16(speech)
    voiced = vad.speech_only(speech16, pad_s=0.0)
    floor, level = dbfs(noise16), dbfs(voiced) if len(voiced) else -100.0
    snr = level - floor
    clipped = clip_ratio(speech)
    save_noise_profile(noise16)

    t = Table(title="Mic check", show_header=False)
    t.add_row("Background noise floor", f"{floor:.1f} dBFS")
    t.add_row("Speech level", f"{level:.1f} dBFS  ({len(voiced) / SAMPLE_RATE:.1f}s detected as speech)")
    t.add_row("SNR", f"{snr:.1f} dB  " + ("[green]good[/]" if snr >= 20 else "[yellow]ok[/]" if snr >= 12
                                          else "[red]low[/]"))
    t.add_row("Clipping", f"{clipped:.2%}")
    if ch == 2:
        a, b = speech[:, 0], speech[:, 1]
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        t.add_row("Left / right level", f"{dbfs(a):.1f} / {dbfs(b):.1f} dBFS, correlation {corr:.2f}")
        if dbfs(a) - dbfs(b) > 10:
            t.add_row("Channel advice", "right channel is weak -> set audio.channel: 0")
        elif dbfs(b) - dbfs(a) > 10:
            t.add_row("Channel advice", "left channel is weak -> set audio.channel: 1")
        else:
            t.add_row("Channel advice", "both channels fine -> keep audio.channel: mix")
    console.print(t)

    if clipped > 0.001 or level > -8:
        console.print("[red]Too loud.[/] Lower mic gain: [bold]wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 60%[/]")
    elif level < -38:
        console.print("[yellow]Quiet.[/] Raise mic gain: [bold]wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 100%[/] "
                      "(then re-run mic-check)")
    else:
        console.print("[green]Levels look good.[/]")
    if snr < 12:
        console.print("[yellow]Background noise is high relative to your voice: sit closer, or set "
                      "speaker.denoise after checking `youru eval`.[/]")
    console.print(f"[dim]Fan-noise profile saved to {DATA / 'noise_profile.wav'}[/]")


# ---------------------------------------------------------------- recording

def cmd_enroll(cfg, args) -> None:
    from .pipeline import PROFILE_PATH
    from .record import ENROLL_DIR, ENROLL_PROMPTS, console as rc, prompted_session

    if not args.rebuild:
        existing = len(list(ENROLL_DIR.glob("*.wav")))
        if existing:
            rc.print(f"{existing} enrollment clips already exist; new ones are added. "
                     "(Use --rebuild to only rebuild the profile from them.)")
        if not (DATA / "noise_profile.wav").exists():
            rc.print("[yellow]Tip: run `youru mic-check` first so quality checks know your noise floor.[/]")
        rc.print("\nRead each sentence naturally, the way you'd talk to the assistant. "
                 "Recording stops automatically when you stop speaking.\n")
        with mic(cfg) as m:
            prompted_session(m, make_vad(cfg, 0.8), ENROLL_PROMPTS, ENROLL_DIR, "enroll", 2.0)
    _build_profile(cfg, PROFILE_PATH)


def _build_profile(cfg, profile_path: Path) -> None:
    from .record import ENROLL_DIR
    from .speaker import Embedder, OwnerProfile

    clips = sorted(ENROLL_DIR.glob("*.wav"))
    if len(clips) < 5:
        raise SystemExit(f"Only {len(clips)} clips; need at least 5 (20 recommended).")
    vad = make_vad(cfg)
    emb = Embedder(models.ensure(cfg.speaker.model))
    E = np.array([emb.embed(vad.speech_only(load_wav(p))) for p in clips])
    OwnerProfile(emb.model_name, E).save(profile_path)
    console.print(f"[green]Owner profile built from {len(E)} clips[/] -> {profile_path}")
    console.print("Next: `youru record impostor --name <someone>` (optional) then `youru eval --speaker`.")


def cmd_record(cfg, args) -> None:
    from . import record as r

    with mic(cfg) as m:
        if args.kind == "wake":
            n = r.wake_session(m, make_vad(cfg, 0.35))
            console.print(f"[green]{n} wake word takes saved[/] in {r.WAKE_DIR}")
        elif args.kind == "confusables":
            prompts = [("normal voice - do NOT say Youru", s) for s in r.CONFUSABLES]
            r.prompted_session(m, make_vad(cfg, 0.6), prompts, r.CONFUSABLE_DIR, "conf", 0.6)
        elif args.kind == "background":
            console.print(f"Recording {args.minutes} min of normal life: work, talk, play videos - just don't "
                          "say the wake word. Ctrl+C stops early (finished minutes are kept).")
            try:
                r.background_session(m, args.minutes)
            except KeyboardInterrupt:
                pass
        elif args.kind == "impostor":
            if not args.name:
                raise SystemExit("--name is required for impostor recordings")
            console.print(f"[bold]{args.name}[/] should read these (anyone except you).")
            r.prompted_session(m, make_vad(cfg, 0.8), r.ENROLL_PROMPTS[: args.count],
                               r.IMPOSTOR_DIR / re.sub(r"\W+", "_", args.name), "imp", 2.0)


# ---------------------------------------------------------------- wake word training

def cmd_train_pack(cfg, args) -> None:
    from .record import CONFUSABLE_DIR, WAKE_DIR

    out = DATA / "youru_train_pack.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for split in ("train", "test"):
            for p in sorted((WAKE_DIR / split).glob("*.wav")):
                z.write(p, f"real_positive_{split}/{p.name}")
                n += 1
        for p in sorted(CONFUSABLE_DIR.glob("*.wav")):
            z.write(p, f"real_negative/{p.name}")
        z.write(ROOT / "training" / "youru.yaml", "youru.yaml")
    if n < 50:
        console.print(f"[yellow]Only {n} wake word takes; 150 recommended (`youru record wake`).[/]")
    console.print(f"[green]Pack ready:[/] {out}  ({n} positive takes)\nFollow training/README.md to train in Colab.")


def cmd_train_verifier(cfg, args) -> None:
    import openwakeword

    from .record import BACKGROUND_DIR, CONFUSABLE_DIR, ENROLL_DIR, WAKE_DIR
    from .wakeword import oww_kwargs

    model = resolve(cfg.wakeword.model)
    if "hey_jarvis" in model.name:
        raise SystemExit("The verifier must be trained on top of your custom youru model. "
                         "Train it first (training/README.md) and set wakeword.model.")
    pos = [str(p) for p in sorted((WAKE_DIR / "train").glob("*.wav"))]
    neg = [str(p) for p in sorted(CONFUSABLE_DIR.glob("*.wav"))]
    # Your enrollment sentences that don't contain the wake word are good negatives too.
    neg += [str(p) for p in sorted(ENROLL_DIR.glob("*.wav"))
            if p.with_suffix(".txt").exists() and "youru" not in p.with_suffix(".txt").read_text().lower()]
    neg += [str(p) for p in sorted(BACKGROUND_DIR.glob("*.wav"))[:10]]
    if len(pos) < 20 or len(neg) < 10:
        raise SystemExit(f"Need >=20 wake takes and >=10 negatives (have {len(pos)} / {len(neg)}).")
    out = model.with_name(model.stem + "_verifier.pkl")
    openwakeword.train_custom_verifier(positive_reference_clips=pos, negative_reference_clips=neg,
                                       output_path=str(out), model_name=str(model), **oww_kwargs())
    update_config({"wakeword": {"verifier": str(out.relative_to(ROOT))}})
    console.print(f"[green]Verifier saved[/] -> {out} (config updated). Run `youru eval --wake` to tune it.")


# ---------------------------------------------------------------- eval

def cmd_eval(cfg, args) -> None:
    from . import evaluate as ev

    do_speaker = args.speaker or not args.wake
    do_wake = args.wake or not args.speaker
    speaker = wake = None
    updates: dict = {}
    if do_speaker:
        names = models.SPEAKER_MODELS if args.all_models else [cfg.speaker.model]
        with console.status("scoring speaker models..."):
            speaker = ev.speaker_eval(make_vad(cfg), names, cfg.speaker.min_speech_s)
        best = ev.best_speaker(speaker)
        updates["speaker"] = {"model": best.model, "accept": best.accept, "reject": best.reject,
                              "denoise": best.denoise}
    if do_wake:
        from .wakeword import WakeWord

        w = cfg.wakeword
        with console.status("scoring wake word..."):
            wake_model = WakeWord(resolve(w.model), w.threshold, resolve(w.verifier) if w.verifier else None,
                                  w.verifier_threshold, w.require_speech, w.refractory_s)
            wake = ev.wake_eval(wake_model)
        updates["wakeword"] = {"threshold": wake.suggested}
    report = ev.write_report(speaker, wake)
    console.print(report.read_text())
    console.print(f"[dim]Report saved to {report}[/]")
    console.print(f"Suggested config: {updates}")
    if args.apply or console.input("Apply suggested thresholds to config.yaml? [y/N] ").strip().lower() == "y":
        if "speaker" in updates and updates["speaker"]["model"] != cfg.speaker.model:
            console.print(f"[yellow]Speaker model changed to {updates['speaker']['model']}; "
                          "rebuilding your profile.[/]")
        update_config(updates)
        if "speaker" in updates and updates["speaker"]["model"] != cfg.speaker.model:
            from .pipeline import PROFILE_PATH

            _build_profile(load_config(), PROFILE_PATH)
        console.print("[green]config.yaml updated[/]")


# ---------------------------------------------------------------- listen

def cmd_listen(cfg, args) -> None:
    from .pipeline import Decision, State, build
    from .vad import SpeechEnd, SpeechStart
    from .wakeword import WakeEvent

    pipe = build(cfg)
    if pipe.gate.profile is None:
        console.print("[yellow]No owner profile yet - every command will be 'no_profile'. "
                      "Run `youru enroll`.[/]")
    stt = None
    if args.transcribe:
        from .assistant import make_stt

        with console.status(f"loading whisper {cfg.stt.model}..."):
            stt = make_stt(cfg)
        console.print(f"[dim]whisper on {stt.device}[/]")

        def transcribe(d: Decision) -> None:
            tr = stt.transcribe(d.audio, cfg.stt.language)
            d.transcript, d.language = tr.text, tr.language
        pipe.on_owner_utterance(transcribe)

    history: deque[str] = deque(maxlen=10)
    colors = {"accept": "green", "uncertain": "yellow", "reject": "red", "too_short": "yellow",
              "no_profile": "yellow"}

    def on_event(ev) -> None:
        ts = f"[dim]{pipe.t:7.1f}s[/]"
        if isinstance(ev, SpeechStart):
            history.append(f"{ts} speech started")
        elif isinstance(ev, SpeechEnd):
            history.append(f"{ts} speech ended ({ev.duration:.1f}s)")
        elif isinstance(ev, WakeEvent):
            history.append(f"{ts} [bold magenta]WAKE[/] (score {ev.score:.2f}) - listening for command")
        elif isinstance(ev, Decision):
            sc = "-" if ev.speaker_score is None else f"{ev.speaker_score:.3f}"
            msg = {"accept": "OWNER VERIFIED", "uncertain": "not sure it's you - please repeat",
                   "reject": "REJECTED - not the owner", "too_short": "too short to verify",
                   "no_profile": "no owner profile"}[ev.outcome]
            line = (f"{ts} [bold {colors[ev.outcome]}]{msg}[/] speaker {sc}, {ev.speech_s:.1f}s speech, "
                    f"{ev.latency_ms:.0f} ms")
            if ev.transcript is not None:
                line += f"\n         [bold]» {ev.transcript}[/] [dim]({ev.language})[/]"
            history.append(line)

    pipe.on_event(on_event)

    def view() -> Table:
        t = Table.grid(padding=(0, 2))
        state = pipe.state
        t.add_row("[bold]state[/]", "[magenta]LISTENING[/]" if state is State.LISTENING else "[dim]idle[/]")
        t.add_row("[bold]speech[/]", bar(pipe.vad.prob, thr=cfg.vad.threshold) +
                  ("  [green]● speaking[/]" if pipe.vad.speaking else ""))
        t.add_row("[bold]wake[/]", bar(pipe.wake.score, thr=cfg.wakeword.threshold) + f"  [dim]{pipe.wake.name}[/]")
        t.add_row("", "")
        for h in history:
            t.add_row("", h)
        return t

    source = FileSource(args.file) if args.file else mic(cfg)
    console.print(f"Listening on [bold]{source.name}[/] - wake word model [bold]{pipe.wake.name}[/]. Ctrl+C to stop.")
    try:
        with source, Live(view(), console=console, refresh_per_second=12, transient=False) as live:
            for frame in source.frames():
                pipe.feed(frame)
                live.update(view())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- STT accent test

def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def cmd_transcribe_test(cfg, args) -> None:
    from .assistant import make_stt
    from .record import CONFUSABLE_DIR, ENROLL_DIR

    clips = [p for p in sorted(ENROLL_DIR.glob("*.wav")) + sorted(CONFUSABLE_DIR.glob("*.wav"))
             if p.with_suffix(".txt").exists()]
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit("No recorded sentences yet. Run `youru enroll` first.")
    with console.status(f"loading whisper {cfg.stt.model}..."):
        stt = make_stt(cfg)
    console.print(f"whisper {cfg.stt.model} on [bold]{stt.device}[/], {len(clips)} clips\n")
    modes = {"auto": None, "en": "en", "hi": "hi"}
    totals = {m: [] for m in modes}
    for p in clips:
        ref = p.with_suffix(".txt").read_text().strip()
        console.rule(f"[dim]{p.name}[/]")
        console.print(f"[bold]said:[/] {ref}")
        for m, lang in modes.items():
            tr = stt.transcribe(load_wav(p), lang)
            e = wer(ref, tr.text)
            totals[m].append(e)
            console.print(f"  [cyan]{m:4s}[/] [dim]({tr.language}, {tr.seconds:.2f}s, WER {e:.0%})[/] {tr.text}")
    console.rule("summary")
    console.print("Average word error rate vs the sentence you read (lower is better; Hindi in Devanagari "
                  "counts as errors because the reference is romanized):")
    for m, v in totals.items():
        console.print(f"  {m:4s}  {np.mean(v):.0%}")


# ---------------------------------------------------------------- phase 2: assistant

def cmd_chat(cfg, args) -> None:
    from .assistant import run_chat

    run_chat(cfg, speak=args.speak)


def cmd_assist(cfg, args) -> None:
    from .assistant import run_assist

    run_assist(cfg, speak=not args.quiet)


def cmd_say(cfg, args) -> None:
    from .tts import Speaker

    Speaker(cfg.tts).say(args.text)


def cmd_voices(cfg, args) -> None:
    """Write voice samples to listen to, so you can pick Youru's voice."""
    import time

    import soundfile as sf
    import soxr

    from .tts import KokoroVoice

    out = Path.home() / "Music" / "youru-voice-samples"
    out.mkdir(parents=True, exist_ok=True)
    en = "Hi! I'm Youru, your assistant. It's sunny in Delhi today, thirty one degrees. Shall I play some music?"
    hi = "नमस्ते! मैं यूरू हूँ, आपकी असिस्टेंट। आज दिल्ली में धूप है। क्या मैं गाना चलाऊँ?"
    samples = [("af_sky", "en-us", en), ("af_nicole", "en-us", en), ("af_bella", "en-us", en),
               ("af_heart", "en-us", en), ("jf_nezumi", "en-us", en), ("af_sky:0.6,jf_nezumi:0.4", "en-us", en),
               ("hf_alpha", "hi", hi), ("hf_beta", "hi", hi)]
    table = Table(title=f"Voice samples in {out}")
    for col in ("file", "voice", "made in", "length"):
        table.add_column(col)
    for voice, lang, text in samples:
        for cute in (False, True):
            v = KokoroVoice(voice, lang, speed=1.08 if cute else 1.0, pitch=1.07 if cute else 1.0)
            t = time.perf_counter()
            audio, rate = v.synth(text)
            took = time.perf_counter() - t
            audio = soxr.resample(audio, rate, 24000) if rate != 24000 else audio   # bake the pitch in
            name = voice.replace(":", "").replace(",", "+").replace(".", "") + ("_cute" if cute else "") + ".wav"
            sf.write(out / name, audio, 24000)
            table.add_row(name, voice + (" (cute)" if cute else ""), f"{took:.2f}s", f"{len(audio) / 24000:.1f}s")
    console.print(table)
    console.print("Listen to them, then tell me which one you like.")


def cmd_alert(cfg, args) -> None:
    """What a timer/reminder runs when it fires: sound, notification, then the message spoken."""
    sound = "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga"
    subprocess.Popen(["notify-send", "-u", "critical", "-a", "Youru", "⏰ Youru", args.message])
    if Path(sound).exists():
        subprocess.run(["pw-play", sound], timeout=15, check=False)
    from .tts import Speaker

    try:
        Speaker(cfg.tts).say(args.message)
    except Exception as e:  # noqa: BLE001  (the notification already showed it)
        console.print(f"[red]could not speak: {e}[/]")


def cmd_llm(cfg, args) -> None:
    from .assistant import run_llm

    run_llm(cfg, args.action, args.model)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="youru", description="Youru phase 1: wake word, VAD, owner voice lock")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices", help="list microphones")
    sub.add_parser("download", help="download all models")
    sub.add_parser("mic-check", help="measure noise floor, levels, SNR; save fan-noise profile")
    e = sub.add_parser("enroll", help="record your voice and build the owner profile")
    e.add_argument("--rebuild", action="store_true", help="only rebuild the profile from existing clips")
    r = sub.add_parser("record", help="record wake word / confusables / background / impostor clips")
    r.add_argument("kind", choices=["wake", "confusables", "background", "impostor"])
    r.add_argument("--minutes", type=float, default=45)
    r.add_argument("--name", help="impostor name")
    r.add_argument("--count", type=int, default=8, help="sentences per impostor")
    sub.add_parser("train-pack", help="zip your wake word clips for Colab training")
    sub.add_parser("train-verifier", help="train the personal wake word verifier")
    ev = sub.add_parser("eval", help="measure accuracy and suggest thresholds")
    ev.add_argument("--speaker", action="store_true", help="only the voice lock")
    ev.add_argument("--wake", action="store_true", help="only the wake word")
    ev.add_argument("--all-models", action="store_true", help="compare all speaker models")
    ev.add_argument("--apply", action="store_true", help="write suggestions to config.yaml without asking")
    li = sub.add_parser("listen", help="run the live pipeline")
    li.add_argument("--transcribe", action="store_true", help="transcribe verified commands with whisper")
    li.add_argument("--file", help="replay a WAV file instead of the mic")
    tt = sub.add_parser("transcribe-test", help="check how well whisper understands your recorded sentences")
    tt.add_argument("--limit", type=int, default=0)
    ch = sub.add_parser("chat", help="type to the assistant (LLM + tools, no mic)")
    ch.add_argument("--speak", action="store_true", help="also speak the replies")
    asg = sub.add_parser("assist", help="voice assistant: wake word / hotkey -> LLM -> spoken reply")
    asg.add_argument("--quiet", action="store_true", help="text replies only, no speech")
    sub.add_parser("trigger", help="make a running `youru assist` start listening (bind this to a hotkey)")
    sub.add_parser("voices", help="write voice samples to ~/Music/youru-voice-samples to pick a voice")
    al = sub.add_parser("alert", help="ring + notify + speak a message (what timers and reminders run)")
    al.add_argument("message")
    sy = sub.add_parser("say", help="speak a sentence (TTS test)")
    sy.add_argument("text")
    lg = sub.add_parser("log", help="what you said, which tools ran, and what Youru replied")
    lg.add_argument("--last", type=int, default=20)
    ln = sub.add_parser("learn", help="review today's mistakes and learn from them (runs nightly)")
    ln.add_argument("--review", action="store_true", help="approve or reject learned commands")
    ln.add_argument("--day", help="YYYY-MM-DD (default: today)")
    ln.add_argument("--quiet", action="store_true")
    sub.add_parser("learned", help="everything Youru has learned: facts, corrections, commands, skills")
    lm = sub.add_parser("llm", help="manage the local LLM server")
    lm.add_argument("action", choices=["start", "stop", "status", "bench"])
    lm.add_argument("--model", action="append", help="GGUF path(s) to benchmark")
    args = p.parse_args(argv)

    if args.cmd == "learned":
        from .learn import show_learned

        show_learned()
        return
    if args.cmd == "learn":
        from .learn import approve_pending, review

        if args.review:
            approve_pending()
            return
        from datetime import date

        from .assistant import ensure_llm

        srv = ensure_llm(load_config())
        result = review(srv.url, date.fromisoformat(args.day) if args.day else None, args.quiet)
        console.print(f"Looked at {result['looked_at']} mistakes: learned {result['facts']} facts, "
                      f"{result['aliases']} corrections; {result['pending']} new commands wait for "
                      f"`youru learn --review`.")
        return
    if args.cmd == "log":
        from .assistant import show_log

        show_log(args.last)
        return
    if args.cmd == "trigger":
        from .hotkey import send_trigger

        if not send_trigger():
            raise SystemExit("youru assist is not running")
        return

    if args.cmd == "download":
        models.ensure_all(models.DEFAULT_SET + models.SPEAKER_MODELS)
        return
    models.ensure_all()
    cfg = load_config()
    {"devices": cmd_devices, "mic-check": cmd_mic_check, "enroll": cmd_enroll, "record": cmd_record,
     "train-pack": cmd_train_pack, "train-verifier": cmd_train_verifier, "eval": cmd_eval,
     "listen": cmd_listen, "transcribe-test": cmd_transcribe_test, "chat": cmd_chat, "assist": cmd_assist,
     "say": cmd_say, "llm": cmd_llm, "voices": cmd_voices,
     "alert": cmd_alert}[args.cmd](cfg, args)


if __name__ == "__main__":
    main(sys.argv[1:])
