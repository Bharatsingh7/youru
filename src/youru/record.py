"""Guided recording sessions: enrollment, wake word takes, hard negatives, background, impostors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

from . import denoise
from .audio import SAMPLE_RATE, RingBuffer, clip_ratio, dbfs, save_wav
from .config import DATA
from .vad import SpeechEnd, SpeechStart

console = Console()

ENROLL_DIR = DATA / "enroll"
WAKE_DIR = DATA / "wake"
CONFUSABLE_DIR = DATA / "confusables"
BACKGROUND_DIR = DATA / "background"
IMPOSTOR_DIR = DATA / "impostors"

# 20 sentences mixing English, Hindi and Hinglish, with a recording condition each.
ENROLL_PROMPTS = [
    ("normal voice, normal distance", "Youru, open Firefox and search for today's cricket score."),
    ("normal voice", "Mera naam aur meri awaaz hi is computer ka password hai."),
    ("normal voice", "Youru, music chalao aur volume thoda kam karo."),
    ("normal voice", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("normal voice", "Kal subah saat baje ka alarm laga do, please."),
    ("fan ON (if it's off, just speak normally)", "Youru, mujhe aaj ka weather batao."),
    ("fan ON", "Please check my emails and tell me if anything is urgent."),
    ("fan ON", "Ye file Documents folder mein save kar do."),
    ("fan ON", "I am going to the market to buy some vegetables and milk."),
    ("lean back ~1 m from the laptop", "Youru, screen ki brightness badha do."),
    ("lean back ~1 m", "What is the capital of Australia and how far is it from India?"),
    ("lean back ~1 m", "Thoda sa ruko, main abhi wapas aata hoon."),
    ("soft / tired voice", "Youru, lights band karo, mujhe neend aa rahi hai."),
    ("soft voice", "Can you remind me to call my mother in the evening?"),
    ("fast speech", "Youru jaldi se terminal kholo aur system update chalao."),
    ("fast speech", "Yesterday I watched a movie, it was really good, you should watch it too."),
    ("loud / excited voice", "Youru, India ne match jeet liya, celebration song bajao!"),
    ("normal voice, turn your head slightly away", "Next song play karo, ye wala acha nahi hai."),
    ("normal voice", "Open Visual Studio Code and create a new Python project for me."),
    ("normal voice", "Aaj ka din bahut accha tha, kal phir se kaam shuru karna hai."),
]

# ~150 takes of the wake word in varied styles: (instruction, count)
WAKE_BATCHES = [
    ("Say \"Youru\" normally, one per take", 25),
    ("Say \"Youru\" fast", 15),
    ("Say \"Youru\" slowly / stretched", 15),
    ("Say \"Youru\" softly", 15),
    ("Say \"Youru\" loudly, like calling from across the room", 10),
    ("Lean back ~1 m and say \"Youru\"", 20),
    ("Fan ON (or noisy room), say \"Youru\"", 20),
    ("Say \"Youru\" with a command after it, e.g. \"Youru, music chalao\"", 20),
    ("Say \"Youru\" with a questioning tone (\"Youru?\")", 10),
]

# Everyday sentences with sounds close to "Youru" (hard negatives).
CONFUSABLES = [
    "Is this your phone or mine?", "You're going to love this song.", "Your room is very clean today.",
    "I went to Europe last summer.", "The euro is stronger than the dollar.", "You rule, bro!",
    "Your order has been shipped.", "You're right, I forgot about that.", "Where is your charger?",
    "Yaar, ye kya ho gaya?", "Your turn now, roll the dice.", "Europe ki trip bahut mehngi hai.",
    "You were late again today.", "Is your mom at home?", "You're welcome, anytime.",
    "Your guru told you that?", "I booked an Uber for you.", "You really did it!",
    "Yours sincerely, Rahul.", "Your exam is on Monday, right?", "You're the best, yaar.",
    "Your voice sounds different today.", "Euro cup final kab hai?", "Yeh tumhara hai ya mera?",
    "You know what I mean?", "Your laptop is overheating.", "Is it your birthday today?",
    "You're making too much noise.", "Your rules, your game.", "You owe me a pizza.",
    "Uri movie dekhi hai tumne?", "Yoga roz karna chahiye.", "Your room or my room?",
    "You remember that day?", "You're using my headphones again.", "Your Urdu is really good.",
    "Euro or rupee, which one should I carry?", "Your router needs a restart.",
    "You ruined the surprise!", "Use your brain, yaar.",
]


@dataclass
class Quality:
    ok: bool
    speech_s: float
    level_db: float
    snr_db: float
    clipped: float
    reasons: list[str]


def noise_floor_db() -> float | None:
    noise = denoise.load_noise_profile()
    return dbfs(noise) if noise is not None else None


def check_quality(audio: np.ndarray, vad, min_speech_s: float, floor_db: float | None) -> Quality:
    speech = vad.speech_only(audio, pad_s=0.0)
    speech_s = len(speech) / SAMPLE_RATE
    level = dbfs(speech) if len(speech) else -100.0
    floor = floor_db if floor_db is not None else -60.0
    snr = level - floor
    clipped = clip_ratio(audio)
    reasons = []
    if speech_s < min_speech_s:
        reasons.append(f"only {speech_s:.1f}s of speech (need {min_speech_s:.1f}s)")
    if snr < 10:
        reasons.append(f"too quiet vs background (SNR {snr:.0f} dB, need 10)")
    if clipped > 0.001:
        reasons.append(f"clipping ({clipped:.1%}) - speak softer or lower mic gain")
    return Quality(not reasons, speech_s, level, snr, clipped, reasons)


def capture_utterance(mic, vad, start_timeout_s: float = 8.0, max_s: float = 12.0,
                      pad_s: float = 0.3) -> np.ndarray | None:
    """Wait for speech, record until the VAD says it ended. None on timeout."""
    vad.reset()
    mic.drain()
    ring = RingBuffer(pad_s + 0.4)
    chunks: list[np.ndarray] = []
    started = False
    t = 0.0
    tail = 0
    for frame in mic.frames():
        t += len(frame) / SAMPLE_RATE
        events = vad.process(frame)
        if not started:
            ring.append(frame)
            if any(isinstance(e, SpeechStart) for e in events):
                started = True
                chunks.append(ring.last(pad_s + 0.4))
                t = 0.0
            elif t > start_timeout_s:
                return None
            continue
        chunks.append(frame)
        if any(isinstance(e, SpeechEnd) for e in events) or tail:
            # keep a little audio after the end, then stop
            tail += len(frame)
            if tail >= pad_s * SAMPLE_RATE:
                break
        if t > max_s:
            break
    audio = np.concatenate(chunks)
    # drop most of the trailing silence the VAD waited through
    trim = int(max(0.0, vad.min_silence / SAMPLE_RATE - pad_s) * SAMPLE_RATE)
    return audio[: len(audio) - trim] if trim and len(audio) > trim * 2 else audio


def next_index(folder: Path) -> int:
    nums = [int(p.stem.split("_")[-1]) for p in folder.glob("*.wav") if p.stem.split("_")[-1].isdigit()]
    return max(nums, default=-1) + 1


def show_quality(q: Quality) -> None:
    color = "green" if q.ok else "red"
    console.print(f"  [{color}]{'OK' if q.ok else 'REDO'}[/] speech {q.speech_s:.1f}s, level {q.level_db:.0f} dBFS, "
                  f"SNR {q.snr_db:.0f} dB" + ("" if q.ok else "  -> " + "; ".join(q.reasons)))


def prompted_session(mic, vad, prompts: list[tuple[str, str]], folder: Path, prefix: str,
                     min_speech_s: float) -> list[Path]:
    """Read-aloud session: Enter to start each sentence, auto-stops on silence."""
    folder.mkdir(parents=True, exist_ok=True)
    floor = noise_floor_db()
    saved = []
    i = 0
    while i < len(prompts):
        cond, text = prompts[i]
        console.rule(f"[bold]{i + 1}/{len(prompts)}[/]  [dim]{cond}[/]")
        console.print(f"[bold cyan]  {text}[/]")
        answer = console.input("  [dim]Enter = record, s = skip, q = quit[/] ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            i += 1
            continue
        console.print("  [yellow]listening... speak now[/]")
        audio = capture_utterance(mic, vad)
        if audio is None:
            console.print("  [red]didn't hear anything, try again[/]")
            continue
        q = check_quality(audio, vad, min_speech_s, floor)
        show_quality(q)
        if not q.ok:
            continue
        path = folder / f"{prefix}_{next_index(folder):03d}.wav"
        save_wav(path, audio)
        path.with_suffix(".txt").write_text(text + "\n")  # reference text for the STT accent test
        saved.append(path)
        i += 1
    return saved


def wake_session(mic, vad, batches=WAKE_BATCHES, folder: Path = WAKE_DIR) -> int:
    """Rapid-fire wake word takes; every 5th take goes to the held-out test set."""
    floor = noise_floor_db()
    total = 0
    for instruction, count in batches:
        console.rule(f"[bold]{instruction}[/]  ({count} takes)")
        answer = console.input("  [dim]Enter = start batch, s = skip, q = quit[/] ").strip().lower()
        if answer == "q":
            break
        if answer == "s":
            continue
        n = 0
        while n < count:
            console.print(f"  [yellow]take {n + 1}/{count} - say it now[/]", end=" ")
            audio = capture_utterance(mic, vad, start_timeout_s=6, max_s=4 if "command" not in instruction else 8,
                                      pad_s=0.2)
            if audio is None:
                console.print("[red]timeout[/]")
                continue
            q = check_quality(audio, vad, 0.25, floor)
            if not q.ok:
                console.print(f"[red]redo: {'; '.join(q.reasons)}[/]")
                continue
            idx = sum(1 for _ in folder.glob("*/youru_*.wav"))
            split = "test" if idx % 5 == 4 else "train"
            save_wav(folder / split / f"youru_{idx:04d}.wav", audio)
            console.print(f"[green]ok[/] [dim]{q.speech_s:.1f}s[/]")
            n += 1
            total += 1
    return total


def background_session(mic, minutes: float, folder: Path = BACKGROUND_DIR) -> int:
    """Passive recording in 60 s chunks."""
    folder.mkdir(parents=True, exist_ok=True)
    chunks = int(np.ceil(minutes))
    mic.drain()
    buf: list[np.ndarray] = []
    done = 0
    with console.status("recording background...") as status:
        for frame in mic.frames():
            buf.append(frame)
            if sum(len(b) for b in buf) >= 60 * SAMPLE_RATE:
                save_wav(folder / f"bg_{next_index(folder):04d}.wav", np.concatenate(buf))
                buf = []
                done += 1
                status.update(f"recording background... {done}/{chunks} min")
                if done >= chunks:
                    break
    return done
