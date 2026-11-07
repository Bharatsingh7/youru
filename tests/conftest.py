"""Shared fixtures. Real speech clips (sherpa-onnx test data) are downloaded once into tests/fixtures/."""

import urllib.request
from pathlib import Path

import numpy as np
import pytest

from youru import models
from youru.audio import SAMPLE_RATE, load_wav

FIXTURES = Path(__file__).parent / "fixtures"
SR_DATA = "https://github.com/csukuangfj/sr-data/raw/main"
CLIPS = {
    "a1": "enroll/fangjun-sr-1.wav", "a2": "enroll/fangjun-sr-2.wav", "a3": "enroll/fangjun-sr-3.wav",
    "a_test": "test/fangjun-test-sr-1.wav",
    "b1": "enroll/leijun-sr-1.wav", "c1": "test/3d-speaker/speaker1_a_en_16k.wav",
}


@pytest.fixture(scope="session")
def clips() -> dict[str, np.ndarray]:
    FIXTURES.mkdir(exist_ok=True)
    out = {}
    for key, rel in CLIPS.items():
        dest = FIXTURES / rel.replace("/", "_")
        if not dest.exists():
            urllib.request.urlretrieve(f"{SR_DATA}/{rel}", dest)
        out[key] = load_wav(dest)
    return out


@pytest.fixture(scope="session")
def vad_model() -> Path:
    return models.ensure("silero_vad")


@pytest.fixture
def silence() -> np.ndarray:
    return np.zeros(3 * SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def fan_noise() -> np.ndarray:
    """Synthetic fan: low-passed noise plus a motor hum at 120 Hz, around -35 dBFS."""
    rng = np.random.default_rng(0)
    n = 5 * SAMPLE_RATE
    white = rng.standard_normal(n)
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")
    hum = 0.3 * np.sin(2 * np.pi * 120 * np.arange(n) / SAMPLE_RATE)
    x = brown / np.abs(brown).max() + hum
    return (0.02 * x / np.abs(x).max()).astype(np.float32)
