"""Stationary noise reduction (fan noise) using a profile recorded by mic-check.

Only applied to finished utterances before speaker check / STT, never to the
live wake word or VAD input.
"""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE, load_wav, save_wav
from .config import DATA

NOISE_PROFILE = DATA / "noise_profile.wav"


def save_noise_profile(noise: np.ndarray) -> None:
    save_wav(NOISE_PROFILE, noise)


def load_noise_profile() -> np.ndarray | None:
    return load_wav(NOISE_PROFILE) if NOISE_PROFILE.exists() else None


def reduce(audio: np.ndarray, noise: np.ndarray | None = None, strength: float = 0.8) -> np.ndarray:
    import noisereduce as nr

    if noise is None:
        noise = load_noise_profile()
    if noise is None or len(audio) < SAMPLE_RATE // 4:
        return audio
    return nr.reduce_noise(y=audio, sr=SAMPLE_RATE, y_noise=noise, stationary=True,
                           prop_decrease=strength).astype(np.float32)
