"""faster-whisper speech-to-text (accent test), GPU with CPU fallback."""

from __future__ import annotations

import ctypes
import threading
import site
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _preload_cuda_libs() -> None:
    """CTranslate2 needs cuBLAS/cuDNN; load them from the nvidia-* pip wheels."""
    for sp in site.getsitepackages():
        base = Path(sp) / "nvidia"
        for lib in ["cublas/lib/libcublasLt.so.12", "cublas/lib/libcublas.so.12",
                    "cudnn/lib/libcudnn_ops.so.9", "cudnn/lib/libcudnn_cnn.so.9", "cudnn/lib/libcudnn.so.9"]:
            p = base / lib
            if p.exists():
                try:
                    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float
    seconds: float


def pick_language(probs: list[tuple[str, float]], allowed: list[str]) -> tuple[str, float]:
    """Best language among the allowed ones (Whisper may say 'ur' for Hindi; we never accept it)."""
    scores = {lang: p for lang, p in probs}
    best = max(allowed, key=lambda lang: scores.get(lang, 0.0))
    return best, scores.get(best, 0.0)


class STT:
    def __init__(self, model: str = "large-v3-turbo", device: str = "cuda", compute_type: str = "int8_float16",
                 initial_prompt: str | dict | None = None, languages: list[str] | None = None):
        self.initial_prompt = initial_prompt
        self.languages = list(languages or [])
        self._last_lang = self.languages[0] if self.languages else "en"  # picks the prompt
        self.model_id = model
        self.want_device = device
        self.compute_type = compute_type
        self._cpu_model = None
        self.model = None
        self.device = "unloaded"
        self._loading = threading.Lock()
        self.load()

    def load(self) -> None:
        """Load the model (again after unload()). Safe to call from a background thread."""
        from faster_whisper import WhisperModel

        with self._loading:
            if self.model is not None:
                return
            if self.want_device == "cuda":
                _preload_cuda_libs()
                try:
                    model = WhisperModel(self.model_id, device="cuda", compute_type=self.compute_type)
                    model.transcribe(np.zeros(16000, np.float32))  # fail fast if CUDA is broken
                    self.model, self.device = model, "cuda"
                    return
                except Exception as e:  # noqa: BLE001
                    print(f"[stt] CUDA unavailable ({e}); falling back to CPU")
            self.model, self.device = WhisperModel(self.model_id, device="cpu", compute_type="int8"), "cpu"

    def unload(self) -> None:
        """Free the GPU memory (and the host memory the CUDA side holds) while idle."""
        import gc

        with self._loading:
            self.model = None
            self._cpu_model = None
            self.device = "unloaded"
        gc.collect()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def transcribe(self, audio: np.ndarray, language: str | None = None) -> Transcript:
        """One Whisper pass in the usual case. Language detection happens inside that pass; only if it
        lands outside the allow-list (e.g. Urdu for Hindi speech) do we re-run with the best allowed one."""
        import time

        t0 = time.perf_counter()
        if self.model is None:
            self.load()   # woken from sleep (normally already loading since the wake word)
        segments, info = self._run(audio, language, language)
        if language is None and self.languages and info.language not in self.languages:
            best, _ = pick_language(info.all_language_probs or [], self.languages)
            segments, info = self._run(audio, best, best)
        text = " ".join(s.text.strip() for s in segments).strip()
        if text.lower().strip(" .") in ("nan", "none"):   # Hinglish-Apex quirk on near-silent audio
            text = ""
        if language is None:
            self._last_lang = info.language
        return Transcript(text, info.language, info.language_probability, time.perf_counter() - t0)

    def _run(self, audio: np.ndarray, prompt_lang: str | None, language: str | None):
        lang = prompt_lang or self._last_lang
        prompt = self.initial_prompt.get(lang) if isinstance(self.initial_prompt, dict) else self.initial_prompt
        from .memory import vocab

        if learned := vocab():   # names/words learned from your corrections (memory.add_vocab)
            prompt = f"{prompt or ''} {', '.join(learned[-30:])}.".strip()
        kwargs = dict(language=language, beam_size=5, vad_filter=False, initial_prompt=prompt,
                      condition_on_previous_text=False, without_timestamps=True)
        try:
            segments, info = self.model.transcribe(audio, **kwargs)
            return list(segments), info
        except RuntimeError as e:
            if self.device != "cuda" or "memory" not in str(e).lower():
                raise
            # The GPU is shared with the LLM, the browser and the desktop; if it is full right now,
            # do this one on the CPU instead of crashing. The next command tries the GPU again.
            print("[stt] GPU out of memory - transcribing this one on the CPU")
            return self._on_cpu(audio, kwargs)

    def _on_cpu(self, audio: np.ndarray, kwargs: dict):
        from faster_whisper import WhisperModel

        if self._cpu_model is None:
            self._cpu_model = WhisperModel(self.model_id, device="cpu", compute_type="int8", cpu_threads=12)  # ~8 s: last resort only
        segments, info = self._cpu_model.transcribe(audio, **kwargs)
        return list(segments), info
