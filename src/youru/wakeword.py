"""openWakeWord wrapper: ONNX backend, VAD gating, personal verifier, refractory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import models
from .audio import FRAME, SAMPLE_RATE, to_int16


@dataclass
class WakeEvent:
    t: float
    score: float


def oww_kwargs() -> dict:
    return dict(inference_framework="onnx",
                melspec_model_path=str(models.ensure("oww_melspectrogram")),
                embedding_model_path=str(models.ensure("oww_embedding")))


class WakeWord:
    def __init__(self, model_path: str | Path, threshold: float = 0.5, verifier: str | Path | None = None,
                 verifier_threshold: float = 0.3, require_speech: bool = True, refractory_s: float = 1.5,
                 gate: bool = True):
        from openwakeword.model import Model

        self.name = Path(model_path).stem
        verifiers = {self.name: str(verifier)} if verifier else {}
        self.model = Model(wakeword_models=[str(model_path)], custom_verifier_models=verifiers,
                           custom_verifier_threshold=verifier_threshold, **oww_kwargs())
        self.threshold = threshold
        self.require_speech = require_speech
        self.refractory = refractory_s
        self.score = 0.0
        self._last_trigger = -1e9
        self._recent_speech = 0.0  # time of the last frame the VAD called speech
        # Gating: openWakeWord costs most of the idle CPU, so it only runs while someone is speaking.
        # The last 1.5 s are kept and fed in when speech starts, so a wake word at the start is caught.
        self.gate = gate
        self._backlog: deque[np.ndarray] = deque(maxlen=int(1.5 * SAMPLE_RATE / FRAME))
        self._active = True

    def reset(self) -> None:
        self.model.reset()
        self.score = 0.0

    def process(self, frame: np.ndarray, t: float, speech: bool) -> WakeEvent | None:
        """`frame` is one 80 ms float32 frame; `speech` is the VAD state for it."""
        assert len(frame) == FRAME
        if speech:
            self._recent_speech = t
        if self.gate and t - self._recent_speech > 1.0:
            if self._active:            # went quiet: forget the stale context
                self.model.reset()
                self._active = False
            self._backlog.append(frame)
            self.score = 0.0
            return None
        if not self._active:            # speech started: catch up on the audio just before it
            for f in self._backlog:
                self.model.predict(to_int16(f))
            self._backlog.clear()
            self._active = True
        pred = self.model.predict(to_int16(frame))
        self.score = float(max(pred.values()))
        # The VAD needs ~250 ms to confirm speech, so allow a short grace window.
        speech_ok = not self.require_speech or t - self._recent_speech <= 0.5
        if self.score >= self.threshold and speech_ok and t - self._last_trigger >= self.refractory:
            self._last_trigger = t
            return WakeEvent(t, self.score)
        return None

    def max_score(self, audio: np.ndarray) -> float:
        """Offline: highest score over a clip (padded with silence), for eval."""
        self.reset()
        pad = np.zeros(SAMPLE_RATE, dtype=np.float32)
        x = np.concatenate([pad, audio, pad])
        x = np.concatenate([x, np.zeros((-len(x)) % FRAME, np.float32)])
        best = 0.0
        for i in range(0, len(x), FRAME):
            best = max(best, float(max(self.model.predict(to_int16(x[i:i + FRAME])).values())))
        self.reset()
        return best
