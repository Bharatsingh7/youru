"""Audio input (mic or file), ring buffer and level helpers.

Everything downstream sees 16 kHz mono float32 in 80 ms frames (1280 samples),
which is openWakeWord's native frame size.
"""

from __future__ import annotations

import queue
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000
FRAME = 1280  # 80 ms


def to_mono(x: np.ndarray, channel: str | int = "mix") -> np.ndarray:
    if x.ndim == 1:
        return x
    if channel == "mix":
        return x.mean(axis=1)
    return x[:, int(channel)]


def load_wav(path: str | Path) -> np.ndarray:
    """Read any audio file as 16 kHz mono float32."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    x = to_mono(x)
    if sr != SAMPLE_RATE:
        x = soxr.resample(x, sr, SAMPLE_RATE)
    return x.astype(np.float32)


def save_wav(path: str | Path, x: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1, 1), SAMPLE_RATE, subtype="PCM_16")


def to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0
    return 20 * np.log10(max(rms, 1e-10))


def clip_ratio(x: np.ndarray, level: float = 0.99) -> float:
    return float(np.mean(np.abs(x) >= level)) if len(x) else 0.0


class RingBuffer:
    """Keeps the most recent `seconds` of audio."""

    def __init__(self, seconds: float):
        self.buf = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
        self.filled = 0

    def append(self, x: np.ndarray) -> None:
        n = len(x)
        if n >= len(self.buf):
            self.buf[:] = x[-len(self.buf):]
        else:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = x
        self.filled = min(len(self.buf), self.filled + n)

    def last(self, seconds: float) -> np.ndarray:
        n = min(int(seconds * SAMPLE_RATE), self.filled)
        return self.buf[len(self.buf) - n:].copy()


class MicSource:
    """Live microphone frames via PortAudio/PipeWire.

    Opens the device at 16 kHz directly when PipeWire allows it, otherwise at the
    device's native rate with streaming resampling.
    """

    def __init__(self, device: str | int | None = None, channel: str | int = "mix"):
        import sounddevice as sd

        self.sd = sd
        self.device = device
        self.channel = channel
        info = sd.query_devices(device, "input")
        self.name = info["name"]
        self.channels = min(2, int(info["max_input_channels"])) or 1
        self.q: queue.Queue[np.ndarray] = queue.Queue()
        self.rate = SAMPLE_RATE
        try:
            sd.check_input_settings(device, channels=self.channels, samplerate=SAMPLE_RATE, dtype="float32")
        except Exception:
            self.rate = int(info["default_samplerate"])
        self.resampler = None if self.rate == SAMPLE_RATE else soxr.ResampleStream(
            self.rate, SAMPLE_RATE, 1, dtype="float32")
        self.stream = None
        self._pending = np.zeros(0, dtype=np.float32)
        self.raw_last: np.ndarray | None = None  # last raw multi-channel block (for mic-check)

    def _callback(self, indata, frames, time_info, status):
        self.q.put(indata.copy())

    def __enter__(self) -> "MicSource":
        block = int(self.rate * FRAME / SAMPLE_RATE)
        self.stream = self.sd.InputStream(device=self.device, channels=self.channels, samplerate=self.rate,
                                          dtype="float32", blocksize=block, callback=self._callback)
        self.stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            raw = self.q.get()
            self.raw_last = raw
            x = to_mono(raw, self.channel)
            if self.resampler is not None:
                x = self.resampler.resample_chunk(x)
            self._pending = np.concatenate([self._pending, x])
            while len(self._pending) >= FRAME:
                yield self._pending[:FRAME]
                self._pending = self._pending[FRAME:]

    def drain(self) -> None:
        """Drop queued audio (e.g. after a slow STT call or between prompts)."""
        while not self.q.empty():
            self.q.get_nowait()
        self._pending = np.zeros(0, dtype=np.float32)


class FileSource:
    """Replays a WAV file as frames, so the pipeline can be tested offline."""

    def __init__(self, path_or_audio: str | Path | np.ndarray):
        self.audio = path_or_audio if isinstance(path_or_audio, np.ndarray) else load_wav(path_or_audio)
        self.name = "file"

    def __enter__(self) -> "FileSource":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def frames(self) -> Iterator[np.ndarray]:
        x = self.audio
        pad = (-len(x)) % FRAME
        x = np.concatenate([x, np.zeros(pad, dtype=np.float32)])
        for i in range(0, len(x), FRAME):
            yield x[i:i + FRAME]

    def drain(self) -> None:
        pass


def record(source, seconds: float) -> np.ndarray:
    out = []
    n = 0
    for frame in source.frames():
        out.append(frame)
        n += len(frame)
        if n >= seconds * SAMPLE_RATE:
            break
    return np.concatenate(out)
