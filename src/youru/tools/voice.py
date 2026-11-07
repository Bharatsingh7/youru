"""Switch Youru's English speaking voice ("use GLaDOS voice", "normal voice")."""

from __future__ import annotations

from ..config import load_config, update_config
from . import tool


@tool()
def set_voice(name: str) -> str:
    """Change Youru's English speaking voice, e.g. "glados" or "default" (normal). Hindi keeps its voice.

    Args:
        name: the voice name: default or glados
    """
    from .. import tts

    name = name.lower().strip().replace(" ", "")
    name = {"normal": "default", "original": "default", "standard": "default"}.get(name, name)
    available = ["default", *(load_config().tts.get("voices") or {})]
    if name not in available:
        return f"no voice called '{name}'. Available: {', '.join(available)}"
    if tts.ACTIVE is not None:
        tts.ACTIVE.switch(name)
    update_config({"tts": {"english_voice": name}})   # remembered for next time
    return f"switched the English voice to {name}"
