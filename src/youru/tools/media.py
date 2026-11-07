"""Play music and videos from YouTube.

Videos open their watch page in the browser (YouTube starts playing). Music plays audio-only in a hidden
mpv (much lighter than a browser tab), with the top results queued so "next" moves to the next song.
That mpv is controlled over its JSON IPC socket; desktop.media falls back to playerctl for other players.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Literal

from . import tool

MPV_SOCKET = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "youru-mpv.sock"
YT_DLP = str(Path(sys.executable).with_name("yt-dlp"))   # the venv's yt-dlp, for mpv's ytdl hook

_last_results: list[dict] = []   # the latest search, so "play the second one" works


def youtube_search(query: str, n: int = 5) -> list[dict]:
    from yt_dlp import YoutubeDL

    with YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True, "noprogress": True}) as y:
        info = y.extract_info(f"ytsearch{n}:{query}", download=False)
    return [{"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
             "channel": e.get("channel") or ""} for e in info.get("entries") or [] if e.get("id")]


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def mpv_command(*command) -> dict | None:
    """Send one command to Youru's music player; None if it isn't running."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(str(MPV_SOCKET))
            s.sendall((json.dumps({"command": list(command)}) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
            for line in buf.splitlines():
                reply = json.loads(line)
                if "error" in reply:   # skip async event lines
                    return reply
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _player_pids() -> list[int]:
    """Youru's background mpv, found by its socket argument (works even if the socket file is gone)."""
    import subprocess as sp

    out = sp.run(["pgrep", "-f", f"input-ipc-server={MPV_SOCKET}"], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def music_playing() -> bool:
    return mpv_command("get_property", "idle-active") is not None or bool(_player_pids())


def mpv_args(urls: list[str]) -> list[str]:
    return ["mpv", "--no-video", "--no-terminal", "--force-window=no", "--idle=no",
            f"--input-ipc-server={MPV_SOCKET}", f"--script-opts=ytdl_hook-ytdl_path={YT_DLP}",
            "--ytdl-format=bestaudio/best", *urls]


def _stop_music() -> None:
    if mpv_command("quit") is None:   # no socket: stop it the hard way
        import os
        import signal

        for pid in _player_pids():
            os.kill(pid, signal.SIGTERM)


def _fmt(results: list[dict], start: int = 0) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        d = r.get("duration")
        length = f" ({int(d) // 60}:{int(d) % 60:02d})" if d else ""
        lines.append(f"{i}. {r['title']}{length} - {r['channel']}" + ("  <- playing" if i == start + 1 else ""))
    return "\n".join(lines)


@tool()
def play_youtube(query: str, kind: Literal["music", "video"] = "music", pick: int = 1) -> str:
    """PLAY a song or video from YouTube (not just search). Music plays in the background;
    a video opens playing in the browser. Use pick=2 for "play the second one" of the last results.

    Args:
        query: what to play, e.g. "arijit singh tum hi ho" or "GTA 6 trailer"; can be empty with pick
        kind: music (audio in the background) or video (watch in the browser)
        pick: which result to play, 1 = top result
    """
    global _last_results
    results = youtube_search(query) if query.strip() else _last_results
    if not results:
        return f"nothing found on YouTube for '{query}'"
    _last_results = results
    i = max(1, min(len(results), pick)) - 1
    chosen = results[i]
    if kind == "video":
        if music_playing():
            mpv_command("set_property", "pause", True)
        subprocess.Popen(["xdg-open", watch_url(chosen["id"])], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        return f"playing the video '{chosen['title']}' in the browser. Results:\n{_fmt(results, i)}"
    _stop_music()
    urls = [watch_url(r["id"]) for r in results[i:]]   # chosen song first, the rest queued for "next"
    MPV_SOCKET.unlink(missing_ok=True)
    subprocess.Popen(mpv_args(urls), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return f"playing '{chosen['title']}' (music, in the background; {len(urls) - 1} more queued). " \
           f"Results:\n{_fmt(results, i)}"


def control_music(action: str) -> str | None:
    """pause/play/next/previous/stop/status for Youru's music player; None if it isn't playing."""
    if not music_playing():
        return None
    if action == "status":
        title = (mpv_command("get_property", "media-title") or {}).get("data", "")
        paused = (mpv_command("get_property", "pause") or {}).get("data", False)
        return f"{'paused' if paused else 'playing'}: {title}"
    commands = {"pause": ("set_property", "pause", True), "play": ("set_property", "pause", False),
                "play_pause": ("cycle", "pause"), "next": ("playlist-next", "force"),
                "previous": ("playlist-prev", "force"), "stop": ("quit",)}
    if action not in commands:
        return None
    if action == "stop":
        _stop_music()
        return "music stopped"
    if mpv_command(*commands[action]) is None and action in ("pause", "play_pause"):
        _stop_music()   # can't reach the player to pause it: stopping is the safe fallback
        return "music stopped"
    return {"stop": "music stopped", "next": "next song", "previous": "previous song"}.get(action, f"music {action}")
