"""Spoken replies with Piper voices (sherpa-onnx): English and Hindi, ~0.08x real time on CPU."""

from __future__ import annotations

import glob
import queue
import re
import threading
from collections.abc import Iterable

import numpy as np

from .agent import is_hindi
from .config import resolve

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")
URL = re.compile(r"https?://\S+")


def clean(text: str) -> str:
    """Make text speakable: no markdown symbols, URLs read as 'link'."""
    text = URL.sub("link", text)
    text = re.sub(r"[*_`#>|~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sentences(chunks: Iterable[str]) -> Iterable[str]:
    """Regroup a stream of text chunks into whole sentences (so speech can start early)."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        parts = SENTENCE_END.split(buf)
        for s in parts[:-1]:
            if s.strip():
                yield s.strip()
        buf = parts[-1]
        # Long clause still going: speak up to the last comma now instead of waiting for the full stop.
        if len(buf) > 70 and ", " in buf[30:]:
            head, buf = buf.rsplit(", ", 1)
            yield head.strip() + ","
    if buf.strip():
        yield buf.strip()


class Voice:
    def __init__(self, folder: str, threads: int = 4, speed: float = 1.0):
        import sherpa_onnx as so

        d = resolve(folder)
        model = [m for m in glob.glob(f"{d}/*.onnx")][0]
        cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(
            vits=so.OfflineTtsVitsModelConfig(model=model, tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
            num_threads=threads))
        self.tts = so.OfflineTts(cfg)
        self.speed = speed

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        a = self.tts.generate(text, sid=0, speed=self.speed)
        return np.asarray(a.samples, dtype=np.float32), a.sample_rate


class KokoroVoice:
    """Kokoro-82M (Apache 2.0): cute-leaning English voices and Hindi female voices in one model, so the
    same girl speaks both languages. `voice` may be one name or a blend like "af_sky:0.7,jf_nezumi:0.3".
    `pitch` > 1 plays the audio slightly faster than it was made: higher, younger-sounding, zero cost
    (`speed` is lowered by the same factor so the pace stays natural)."""

    MODEL = "models/tts/kokoro/kokoro-v1.0.onnx"
    VOICES = "models/tts/kokoro/voices-v1.0.bin"

    def __init__(self, voice: str = "af_sky", lang: str = "en-us", speed: float = 1.0, pitch: float = 1.0):
        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(str(resolve(self.MODEL)), str(resolve(self.VOICES)))
        self.style = self._style(voice)
        self.lang, self.pitch = lang, pitch
        self.speed = speed / pitch

    def _style(self, voice: str):
        if ":" not in voice and "," not in voice:
            return voice
        parts = [p.split(":") for p in voice.split(",")]
        weights = np.array([float(w) for _, w in parts], dtype=np.float32)
        styles = [self.kokoro.get_voice_style(n.strip()) for n, _ in parts]
        return sum(w * st for w, st in zip(weights / weights.sum(), styles))

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        audio, sr = self.kokoro.create(text, voice=self.style, speed=self.speed, lang=self.lang)
        return np.asarray(audio, dtype=np.float32), int(sr * self.pitch)


ACTIVE = None   # the Speaker of the running assistant, if any


class Speaker:
    """Speaks text; Devanagari goes to the Hindi voice, everything else to English."""

    def __init__(self, cfg):
        global ACTIVE
        self.cfg = cfg
        self._voices: dict[str, Voice] = {}
        self.english = cfg.get("english_voice", "default")
        ACTIVE = self   # so the set_voice tool can switch the running speaker

    def english_voices(self) -> list[str]:
        return ["default", *(self.cfg.get("voices") or {})]

    def switch(self, name: str) -> None:
        """Change the English voice ("glados", "default"); Hindi keeps its voice."""
        if name not in self.english_voices():
            raise ValueError(f"no voice called {name}; have: {', '.join(self.english_voices())}")
        self.english = name
        self._voices.pop("en", None)

    def voice(self, lang: str):
        if lang == "en" and self.english != "default" and "en" not in self._voices:
            self._voices["en"] = Voice(self.cfg.voices[self.english], speed=self.cfg.get("speed", 1.0))
        if lang not in self._voices:
            c = self.cfg
            if c.get("engine", "piper") == "kokoro":
                name = c.get("kokoro_hindi_voice", "hf_alpha") if lang == "hi" else c.get("kokoro_voice", "af_sky")
                self._voices[lang] = KokoroVoice(name, "hi" if lang == "hi" else "en-us",
                                                 c.get("speed", 1.0), c.get("pitch", 1.0))
            else:
                self._voices[lang] = Voice(c.hi_voice if lang == "hi" else c.en_voice, speed=c.speed)
        return self._voices[lang]

    def warm_up(self) -> None:
        self.voice("en")
        self.voice("hi")

    def say(self, text: str) -> None:
        self.say_stream([text])

    def say_stream(self, chunks: Iterable[str]) -> None:
        """Synthesize sentence N+1 while sentence N plays."""
        import sounddevice as sd

        q: queue.Queue = queue.Queue(maxsize=3)

        def producer():
            try:
                for s in sentences(chunks):
                    s = clean(s)
                    if s:
                        q.put((*self.voice("hi" if is_hindi(s) else "en").synth(s), s))
            finally:
                q.put(None)

        threading.Thread(target=producer, daemon=True).start()
        while (item := q.get()) is not None:
            samples, sr, text = item
            threading.Thread(target=_halo_say, args=(text, samples, sr), daemon=True).start()
            sd.play(samples, sr)
            sd.wait()


def _halo_say(text, samples, sr) -> None:
    """Tell the Halo companion (~/Projects/halo) what is being said so she lip-syncs.

    Fire-and-forget: a 0.25 s timeout and every error ignored, so Youru never waits
    on or breaks because of Halo. Supports both local and remote (spare laptop 'hades').
    """
    import json
    import os
    from pathlib import Path
    import urllib.request

    try:
        import numpy as np

        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        hop = max(1, int(sr / 60))
        n = len(x) // hop
        env = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0)
        peak = float(env.max()) if env.size else 0.0
        env = (env / peak if peak else env).round(3).tolist()
        body = json.dumps({"text": text, "env": env, "rate": 60}).encode()

        port = os.environ.get("HALO_PORT", "8787")
        host = os.environ.get("HALO_HOST", "hades")

        headers = {"Content-Type": "application/json"}
        token = os.environ.get("HALO_TOKEN")
        if not token:
            try:
                cfg = json.loads((Path.home() / "Projects/halo/config.json").read_text())
                token = cfg.get("token")
            except Exception:
                pass
        if token:
            headers["X-Halo-Key"] = token

        target_hosts = [host] if host in ("127.0.0.1", "localhost") else [host, "127.0.0.1"]
        for h in target_hosts:
            try:
                req = urllib.request.Request(f"http://{h}:{port}/say", body, headers=headers)
                urllib.request.urlopen(req, timeout=0.25).close()
                break
            except Exception:
                continue
    except Exception:
        pass

