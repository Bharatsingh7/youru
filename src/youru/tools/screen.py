"""Read the screen with OCR (grim + Tesseract): "what's on my screen", "read this error",
"translate this". Uses the CPU only, ~0.5-1 s."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Literal

from . import tool


def active_window() -> dict:
    try:
        out = subprocess.run(["hyprctl", "activewindow", "-j"], capture_output=True, text=True, timeout=2).stdout
        return json.loads(out or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}


def window_geometry(w: dict) -> str | None:
    if not w.get("at") or not w.get("size"):
        return None
    (x, y), (wd, ht) = w["at"], w["size"]
    return f"{x},{y} {wd}x{ht}"


def prepare(png: str) -> None:
    """Grayscale, and dark themes inverted to dark-on-light, so Tesseract can skip its own inverted second
    pass: 6.2 s -> 2.1 s on a 1825x980 VS Code window with nearly the same text (measured)."""
    from PIL import Image, ImageOps

    g = ImageOps.grayscale(Image.open(png))
    if sum(g.resize((64, 36)).getdata()) / (64 * 36) < 128:
        g = ImageOps.invert(g)
    g.save(png)


def ocr(png: str) -> str:
    prepare(png)
    r = subprocess.run(["tesseract", png, "stdout", "-l", "eng", "--psm", "3", "-c", "tessedit_do_invert=0"],
                       capture_output=True, text=True, timeout=30,
                       env={**os.environ, "OMP_THREAD_LIMIT": "1"})   # its threading is slower here
    lines = [ln.rstrip() for ln in r.stdout.splitlines()]
    text, blank = [], 0
    for ln in lines:   # squeeze runs of empty lines
        blank = blank + 1 if not ln.strip() else 0
        if blank <= 1:
            text.append(ln)
    return "\n".join(text).strip()


@tool()
def read_screen(area: Literal["window", "screen", "select"] = "window") -> str:
    """Read the text that is on the screen right now (OCR). Use for "what's on my screen", "read this error",
    "translate this", "summarize this page", "what does it say".

    Args:
        area: window (the focused window), screen (everything), or select (the user drags a box)
    """
    w = active_window()
    geom = None
    if area == "window":
        geom = window_geometry(w)
    elif area == "select":
        r = subprocess.run(["slurp"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return "cancelled: no area was selected"
        geom = r.stdout.strip()
    fd, png = tempfile.mkstemp(suffix=".png", dir=os.environ.get("XDG_RUNTIME_DIR"))
    os.close(fd)
    try:
        subprocess.run(["grim", *(["-g", geom] if geom else []), png], check=True, capture_output=True, timeout=10)
        text = ocr(png)
    finally:
        os.unlink(png)
    where = f"{w.get('class', '?')} - {w.get('title', '')}" if area == "window" else area
    if not text:
        return f"no readable text on the {area} ({where})"
    return f"Text on the {area} ({where}):\n{text[:3500]}"
