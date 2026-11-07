"""Tools borrowed from OmniPilot (https://github.com/xrchris7/OmniPilot, MIT, omnipilot-agent==0.1.0).

Only its tool implementations are used, wrapped with short names and descriptions a small model can pick,
Youru's path rules (home folder only, voice recordings protected) and Youru's risk levels. OmniPilot's agent
loop (built for large cloud models) and its GUI/screen tools (PyAutoGUI, X11-only, so not usable on
Hyprland) are not used. Its run_python is replaced by pyrun.py (separate process).
"""

from __future__ import annotations

import json
from typing import Literal

from omnipilot.config.settings import Settings
from omnipilot.tools.base import ToolContext
from omnipilot.tools.filesystem import CopyFile, EditFile, GrepFiles
from omnipilot.tools.system_info import ListDrives
from omnipilot.tools.web import DownloadFile, HttpRequest

from . import risk, tool
from .files import _path, _writable

_CTX = ToolContext(settings=Settings())


def _call(cls, **arguments) -> str:
    result = cls().run(_CTX, **arguments)
    return result.message_for_model()


@tool(assess=risk.file_write("fs.edit_file", "path"),
      describe=lambda a: f"change \"{str(a.get('find_text', ''))[:60]}\" to "
                         f"\"{str(a.get('replace_text', ''))[:60]}\" in {a.get('path')}")
def edit_file(path: str, find_text: str, replace_text: str, replace_all: bool = False) -> str:
    """Edit a text file: replace exact text with new text. To add something next to existing text,
    put the existing text in find_text and the combined text in replace_text. Read the file first if unsure.

    Args:
        path: the file, e.g. "~/contacts.txt"
        find_text: exact text currently in the file
        replace_text: what it should become
        replace_all: replace every occurrence (default: exactly one)
    """
    p = _writable(_path(path))
    return _call(EditFile, path=str(p), find_text=find_text, replace_text=replace_text, replace_all=replace_all)


@tool(assess=risk.file_write("fs.copy_file", "destination"),
      describe=lambda a: f"copy {a.get('source')} to {a.get('destination')}")
def copy_file(source: str, destination: str) -> str:
    """Copy a file or folder.

    Args:
        source: what to copy
        destination: new path or target folder
    """
    src = _path(source)
    dst = _writable(_path(destination, must_exist=False))
    return _call(CopyFile, source=str(src), destination=str(dst))


@tool()
def search_in_files(text: str, folder: str = "~", file_pattern: str = "*", max_results: int = 30) -> str:
    """Find which files contain some exact text or a regular expression (like grep), with line numbers.
    For finding files by meaning use find_anything; by name use find_files.

    Args:
        text: text or regex to look for
        folder: where to search (default home)
        file_pattern: limit to files like "*.txt" or "*.py"
        max_results: how many matching lines to return
    """
    return _call(GrepFiles, directory=str(_path(folder)), pattern=text, file_glob=file_pattern,
                 regex=True, max_results=max(1, min(100, max_results)))


@tool(assess=lambda a: (("low", "") if str(a.get("method", "GET")).upper() == "GET"
                        else ("medium", "sends data to a website")),
      describe=lambda a: f"send an HTTP {a.get('method', 'GET')} request to {a.get('url')}")
def http_request(url: str, method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET",
                 body: str = "") -> str:
    """Call a web API / URL and return the response (status and body). GET only reads.

    Args:
        url: full URL
        method: HTTP method
        body: request body (JSON text) for POST/PUT/PATCH
    """
    args = {"url": url, "method": method}
    if body:
        try:
            args["json"] = json.loads(body)
        except json.JSONDecodeError:
            args["json"] = body
    return _call(HttpRequest, **args)


@tool(assess=risk.file_write("web.download_file", "destination"),
      describe=lambda a: f"download {a.get('url')} to {a.get('destination')}")
def download_file(url: str, destination: str = "~/Downloads") -> str:
    """Download a file from the internet.

    Args:
        url: file URL
        destination: folder or file path (default ~/Downloads)
    """
    dst = _writable(_path(destination, must_exist=False))
    return _call(DownloadFile, url=url, destination=str(dst))


@tool()
def list_drives() -> str:
    """Disks and partitions with their free space."""
    return _call(ListDrives)
