"""DigUp (https://github.com/ARahim3/DigUp): find files by what's *in* them (photos, PDFs, documents,
recordings, videos, code), with EmbeddingGemma 2 running locally.

Searches go to `digup search-service`, kept running with the index open and the query encoder warm, so a
search takes tens of milliseconds instead of a process start. The model runs on the Intel iGPU (see the
`digup` wrapper in ~/dotfiles/bin), never on the NVIDIA GPU that Whisper and the LLM need.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Literal

from ..config import load_config
from . import tool

KINDS = ["screenshot", "image", "pdf", "doc", "audio", "video"]


class DigUpService:
    def __init__(self, folders: list[str], code_folders: list[str], idle_s: int = 600):
        self.folders = [os.path.expanduser(f) for f in folders]
        self.code_folders = [os.path.expanduser(f) for f in code_folders]
        self.idle_s = idle_s
        self.proc: subprocess.Popen | None = None
        self.ids = itertools.count(1)
        self.lock = threading.Lock()
        self._stamp = 0.0

    def _start(self) -> None:
        binary = shutil.which("digup")
        if binary is None:
            raise RuntimeError("DigUp is not installed (expected ~/.local/bin/digup)")
        args = [binary, "search-service", *self.folders, "--encoder-idle", str(self.idle_s)]
        for c in self.code_folders:
            args += ["--code-root", c]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def request(self, op: dict, timeout_lines: int = 1000) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            rid = next(self.ids)
            self.proc.stdin.write(json.dumps({**op, "id": rid}) + "\n")
            self.proc.stdin.flush()
            for _ in range(timeout_lines):
                line = self.proc.stdout.readline()
                if not line:
                    self.proc = None
                    raise RuntimeError("digup search-service stopped")
                try:
                    reply = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if reply.get("id") == rid:
                    if "error" in reply:
                        raise RuntimeError(str(reply["error"]))
                    return reply
            raise RuntimeError("no reply from digup")

    def _reload_if_changed(self) -> None:
        """The timer (digup-index.timer) re-indexes in the background; pick up its new vectors."""
        index = Path.home() / ".local/share/digup/index"
        stamp = max((f.stat().st_mtime for f in index.glob("*.sqlite*")), default=0.0)
        if self.proc is not None and self.proc.poll() is None and stamp > self._stamp:
            self.request({"op": "reload"})
        self._stamp = stamp

    def search(self, query: str, code: bool = False, kinds: list[str] | None = None, limit: int = 5) -> list[dict]:
        self._reload_if_changed()
        op = {"op": "search", "q": query, "code": code, "limit": limit, "meaning": True, "prefix_last": False}
        if kinds:
            op["kinds"] = kinds
        return self.request(op).get("hits", [])


_service: DigUpService | None = None


def service() -> DigUpService:
    global _service
    if _service is None:
        cfg = load_config().get("digup", {}) or {}
        _service = DigUpService(cfg.get("folders", ["~/Documents", "~/Downloads", "~/Pictures"]),
                                cfg.get("code_folders", ["~/Projects"]), cfg.get("encoder_idle_s", 600))
    return _service


def _where(hit: dict) -> str:
    loc, seg = hit.get("loc"), hit.get("segment", "")
    if loc is None:
        return ""
    if seg in ("page", "pageImage"):
        return f"page {int(loc)}"
    if seg in ("audio", "frame"):
        return f"at {int(loc) // 60}:{int(loc) % 60:02d}"
    if seg == "lines":
        return f"line {int(hit.get('focus') or loc)}"
    return ""


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "nothing found"
    out = []
    for i, h in enumerate(hits, 1):
        path = h.get("path", "?")
        where = _where(h)
        excerpt = " ".join(str(h.get("excerpt") or "").split())[:140]
        out.append(f"{i}. {h.get('kind', '')} {path}" + (f" [{where}]" if where else "")
                   + (f" - \"{excerpt}\"" if excerpt else ""))
    return "\n".join(out)


@tool()
def find_anything(query: str, kind: Literal["any", "image", "screenshot", "pdf", "doc", "audio", "video"] = "any",
                  max_results: int = 5) -> str:
    """Find files on this PC by what is IN them (DigUp): photos by what they show, PDFs and documents by
    their content, recordings and videos by what is said or shown (with the page or moment). Use for
    "find the photo of...", "where is the PDF about...", "that video where...". For file NAMES use find_files.

    Args:
        query: describe what you are looking for, e.g. "dog on the beach" or "rent agreement pet clause"
        kind: limit to one type of file, or any
        max_results: how many results
    """
    hits = service().search(query, kinds=None if kind == "any" else [kind], limit=max(1, min(10, max_results)))
    return format_hits(hits)


@tool()
def find_code(query: str, max_results: int = 5) -> str:
    """Search the code in ~/Projects by what it does or by a name (DigUp code search), e.g.
    "function that sets the wallpaper" or "parse_yes_no". Returns files and line numbers.

    Args:
        query: what the code does, or an identifier
        max_results: how many results
    """
    return format_hits(service().search(query, code=True, limit=max(1, min(10, max_results))))


@tool()
def open_file(path: str, page: int = 0, seconds: int = 0) -> str:
    """Open a file found by find_anything / find_code / find_files: PDFs at a page, audio and video at a moment,
    code in VS Code at a line (pass the line as page), anything else with its default app.

    Args:
        path: the file path
        page: PDF page, or code line number (0 = start)
        seconds: where to start audio/video (0 = start)
    """
    p = Path(os.path.expanduser(path)).resolve()
    if not p.exists():
        return f"{p} does not exist"
    home = Path.home().resolve()
    if home not in p.parents:
        return f"{p} is outside your home folder"
    suffix = p.suffix.lower()
    spawn = lambda *cmd: subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          start_new_session=True)
    if suffix in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".mp3", ".m4a", ".wav", ".flac", ".ogg") and seconds:
        spawn("mpv", f"--start={int(seconds)}", str(p))
        return f"playing {p.name} from {seconds // 60}:{seconds % 60:02d}"
    if suffix == ".pdf" and page:
        spawn("brave", f"{p.as_uri()}#page={int(page)}")   # the default PDF viewer here; it honours #page=
        return f"opened {p.name} at page {page}"
    if shutil.which("code") and page and suffix in (".py", ".js", ".ts", ".go", ".rs", ".lua", ".sh", ".md", ".kt",
                                                     ".java", ".swift", ".c", ".cpp", ".h", ".json", ".yaml", ".toml"):
        spawn("code", "--goto", f"{p}:{int(page)}")
        return f"opened {p.name} at line {page} in VS Code"
    spawn("xdg-open", str(p))
    return f"opened {p.name}"
