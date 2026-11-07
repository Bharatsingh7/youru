"""File tools, limited to the home folder. Deleting means moving to the Trash, never rm."""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import time
from pathlib import Path

from ..config import DATA
from . import risk, tool

HOME = Path.home().resolve()
PROTECTED = [DATA.resolve()]          # your voice recordings
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".cache", ".local", ".cargo", ".npm"}


def _path(p: str, must_exist: bool = True) -> Path:
    raw = Path(os.path.expanduser(p.strip() or "~"))
    path = (raw if raw.is_absolute() else HOME / raw).resolve()   # "notes.txt" means ~/notes.txt
    if path != HOME and HOME not in path.parents:
        raise PermissionError(f"{path} is outside your home folder")
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    return path


def _writable(path: Path) -> Path:
    for prot in PROTECTED:
        if path == prot or prot in path.parents:
            raise PermissionError(f"{path} is protected (Youru voice data)")
    return path


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@tool()
def list_dir(path: str = "~", show_hidden: bool = False) -> str:
    """List files and folders in a folder (newest first).

    Args:
        path: folder, e.g. "~/Downloads"
        show_hidden: include dotfiles
    """
    d = _path(path)
    entries = [e for e in d.iterdir() if show_hidden or not e.name.startswith(".")]
    entries.sort(key=lambda e: e.stat().st_mtime if e.exists() else 0, reverse=True)
    lines = [f"{e.name}/" if e.is_dir() else f"{e.name} ({_size(e.stat().st_size)})" for e in entries[:80]]
    more = f"\n... and {len(entries) - 80} more" if len(entries) > 80 else ""
    return f"{d} ({len(entries)} items):\n" + "\n".join(lines) + more if lines else f"{d} is empty"


@tool()
def find_files(pattern: str, folder: str = "~", max_results: int = 25) -> str:
    """Find files by name anywhere under a folder. Case-insensitive; * wildcards allowed.

    Args:
        pattern: file name or wildcard, e.g. "*.pdf" or "resume"
        folder: where to search (default: home)
        max_results: how many matches to return
    """
    root = _path(folder)
    pat = pattern.lower() if any(c in pattern for c in "*?[") else f"*{pattern.lower()}*"
    found, deadline = [], time.monotonic() + 8
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames + dirnames:
            if fnmatch.fnmatch(name.lower(), pat):
                found.append(os.path.join(dirpath, name))
                if len(found) >= max_results:
                    return "\n".join(found)
        if time.monotonic() > deadline:
            found.append("(search stopped after 8 s)")
            break
    return "\n".join(found) if found else f"nothing matching '{pattern}' under {root}"


@tool()
def read_text_file(path: str) -> str:
    """Read a text file (first 4000 characters).

    Args:
        path: file path, e.g. "~/notes.txt"
    """
    p = _path(path)
    if p.is_dir():
        return f"{p} is a folder; use list_dir"
    with open(p, errors="replace") as f:
        return f.read(4000)


@tool(assess=risk.file_write("fs.write_file", "path"),
      describe=lambda a: f"{'append to' if a.get('append') else 'overwrite'} the file {a.get('path')}")
def write_file(path: str, content: str, append: bool = False) -> str:
    """Create or overwrite a text file (or append to it).

    Args:
        path: file path
        content: the text to write
        append: add to the end instead of replacing
    """
    p = _writable(_path(path, must_exist=False))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w") as f:
        f.write(content)
    return f"wrote {len(content)} characters to {p}"


@tool(risk="confirm", describe=lambda a: f"move {a.get('source')} to {a.get('destination')}")
def move_file(source: str, destination: str) -> str:
    """Move or rename a file or folder.

    Args:
        source: current path
        destination: new path or target folder
    """
    src = _writable(_path(source))
    dst = _writable(_path(destination, must_exist=False))
    shutil.move(src, dst)
    return f"moved {src} to {dst}"


@tool(risk="confirm", describe=lambda a: f"move {a.get('path')} to the Trash")
def trash_file(path: str) -> str:
    """Delete a file or folder by moving it to the Trash (can be restored).

    Args:
        path: what to delete
    """
    p = _writable(_path(path))
    if p == HOME:
        raise PermissionError("refusing to trash the home folder")
    subprocess.run(["gio", "trash", str(p)], check=True, capture_output=True, timeout=20)
    return f"moved {p} to the Trash"
