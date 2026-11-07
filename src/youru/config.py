"""Config loading and project paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("YOURU_HOME", Path(__file__).resolve().parents[2]))
CONFIG_PATH = ROOT / "config.yaml"
DATA = ROOT / "data"
MODELS = ROOT / "models"


class Section(dict):
    """A dict whose keys can also be read as attributes (cfg.vad.threshold)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Section(value) if isinstance(value, dict) else value


def load_config(path: Path = CONFIG_PATH) -> Section:
    with open(path) as f:
        return Section(yaml.safe_load(f))


def update_config(updates: dict[str, dict[str, Any]], path: Path = CONFIG_PATH) -> None:
    """Rewrite individual `key: value` lines in place so comments survive."""
    lines = path.read_text().splitlines(keepends=True)
    section = None
    for i, line in enumerate(lines):
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            section = line.strip()[:-1]
            continue
        stripped = line.lstrip()
        key = stripped.split(":", 1)[0]
        if section in updates and key in updates[section] and ":" in stripped:
            indent = line[: len(line) - len(stripped)]
            comment = ""
            if "#" in stripped:
                comment = "  #" + stripped.split("#", 1)[1].rstrip("\n")
            value = yaml.safe_dump(updates[section][key], default_flow_style=True).strip()
            value = value.removesuffix("\n...").removesuffix("...").strip()
            lines[i] = f"{indent}{key}: {value}{comment}\n"
    path.write_text("".join(lines))


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
