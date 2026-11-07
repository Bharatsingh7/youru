"""Model registry and downloader. All models live under models/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from .config import MODELS

OWW = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
SHERPA_SPK = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"

REGISTRY: dict[str, tuple[str, str]] = {
    # name: (relative path under models/, url)
    "silero_vad": ("vad/silero_vad.onnx",
                   "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"),
    "oww_melspectrogram": ("wakeword/melspectrogram.onnx", f"{OWW}/melspectrogram.onnx"),
    "oww_embedding": ("wakeword/embedding_model.onnx", f"{OWW}/embedding_model.onnx"),
    "hey_jarvis": ("wakeword/hey_jarvis_v0.1.onnx", f"{OWW}/hey_jarvis_v0.1.onnx"),
    "wespeaker_resnet34": ("speaker/wespeaker_en_voxceleb_resnet34.onnx",
                           f"{SHERPA_SPK}/wespeaker_en_voxceleb_resnet34.onnx"),
    "wespeaker_campp": ("speaker/wespeaker_en_voxceleb_CAM++.onnx", f"{SHERPA_SPK}/wespeaker_en_voxceleb_CAM++.onnx"),
    "titanet_large": ("speaker/nemo_en_titanet_large.onnx", f"{SHERPA_SPK}/nemo_en_titanet_large.onnx"),
}

SPEAKER_MODELS = ["titanet_large", "wespeaker_resnet34", "wespeaker_campp"]
DEFAULT_SET = ["silero_vad", "oww_melspectrogram", "oww_embedding", "hey_jarvis", "titanet_large"]


def path(name: str) -> Path:
    return MODELS / REGISTRY[name][0]


def ensure(name: str) -> Path:
    dest = path(name)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = REGISTRY[name][1]
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f, Progress(
            "[bold]{task.description}", BarColumn(), DownloadColumn(), TransferSpeedColumn()) as prog:
        task = prog.add_task(dest.name, total=int(r.headers.get("Content-Length", 0)) or None)
        while chunk := r.read(1 << 16):
            f.write(chunk)
            prog.advance(task, len(chunk))
    tmp.rename(dest)
    return dest


def ensure_all(names: list[str] | None = None) -> None:
    for n in names or DEFAULT_SET:
        ensure(n)
