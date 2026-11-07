"""Run a shell command. Its risk (risk.shell) decides: read-only runs, changes ask, dangerous is blocked."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import risk, tool


@tool(assess=risk.shell, describe=lambda a: f"run the command: {a.get('command')}")
def run_command(command: str) -> str:
    """Run a bash command on this Linux (CachyOS, Hyprland) computer and return its output.
    Read-only commands run directly; anything that changes things asks the user first.
    Use how_to first to find the right command. Avoid commands needing sudo or interaction.

    Args:
        command: the bash command line
    """
    try:
        r = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=30,
                           cwd=Path.home(), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "error: the command took longer than 30 s and was stopped"
    out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
    if r.returncode == 127:
        out += "\n(that command doesn't exist here: call how_to to find the right one)"
    return f"exit code {r.returncode}\n{out[:3000] or '(no output)'}"
