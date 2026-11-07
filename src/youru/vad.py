"""Silero VAD v5 (ONNX) with a hysteresis segmenter.

Run directly on onnxruntime so we get the per-window speech probability
(for the live meter) as well as start/end events.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .audio import SAMPLE_RATE

WINDOW = 512   # 32 ms at 16 kHz, required by Silero v5
CONTEXT = 64


@dataclass
class SpeechStart:
    t: float


@dataclass
class SpeechEnd:
    t: float
    duration: float


class SileroVAD:
    def __init__(self, model_path, threshold: float = 0.5, neg_threshold: float | None = None,
                 min_speech_s: float = 0.25, min_silence_s: float = 0.8):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self.neg_threshold = threshold - 0.15 if neg_threshold is None else neg_threshold
        self.min_speech = int(min_speech_s * SAMPLE_RATE)
        self.min_silence = int(min_silence_s * SAMPLE_RATE)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.samples = 0          # total samples processed
        self.speaking = False
        self.prob = 0.0
        self._run_speech = 0      # consecutive speech samples while not speaking
        self._run_silence = 0     # consecutive silence samples while speaking
        self._start = 0

    def window_prob(self, w: np.ndarray) -> float:
        x = np.concatenate([self.context, w])[None, :].astype(np.float32)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state,
                                               "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
        self.context = w[-CONTEXT:]
        return float(out[0][0])

    def process(self, frame: np.ndarray) -> list[SpeechStart | SpeechEnd]:
        """Feed audio of any length; returns events that completed within it."""
        events: list[SpeechStart | SpeechEnd] = []
        self.pending = np.concatenate([self.pending, frame])
        probs = []
        while len(self.pending) >= WINDOW:
            w, self.pending = self.pending[:WINDOW], self.pending[WINDOW:]
            p = self.window_prob(w)
            probs.append(p)
            self.prob = max(probs)  # highest window probability in this call
            self.samples += WINDOW
            if not self.speaking:
                self._run_speech = self._run_speech + WINDOW if p >= self.threshold else 0
                if self._run_speech >= self.min_speech:
                    self.speaking = True
                    self._run_silence = 0
                    self._start = self.samples - self._run_speech
                    events.append(SpeechStart(self._start / SAMPLE_RATE))
            else:
                self._run_silence = self._run_silence + WINDOW if p < self.neg_threshold else 0
                if self._run_silence >= self.min_silence:
                    self.speaking = False
                    self._run_speech = 0
                    end = self.samples - self._run_silence
                    events.append(SpeechEnd(end / SAMPLE_RATE, (end - self._start) / SAMPLE_RATE))
        return events

    def segments(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Offline: (start, end) sample indices of speech in a whole clip."""
        self.reset()
        segs, start = [], None
        for ev in self.process(audio):
            if isinstance(ev, SpeechStart):
                start = int(ev.t * SAMPLE_RATE)
            else:
                segs.append((start, int(ev.t * SAMPLE_RATE)))
                start = None
        if self.speaking and start is not None:
            segs.append((start, len(audio) - self._run_silence))
        self.reset()
        return segs

    def speech_only(self, audio: np.ndarray, pad_s: float = 0.1) -> np.ndarray:
        """Concatenate the speech parts of a clip (empty array if none)."""
        pad = int(pad_s * SAMPLE_RATE)
        parts = [audio[max(0, s - pad):min(len(audio), e + pad)] for s, e in self.segments(audio)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
