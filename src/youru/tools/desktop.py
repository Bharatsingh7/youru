"""Desktop control on Hyprland (0.56, Lua dispatchers) with PipeWire."""

from __future__ import annotations

import configparser
import difflib
import json
import shutil
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Literal

from . import tool

APP_DIRS = [Path.home() / ".local/share/applications", Path("/usr/share/applications"),
            Path("/var/lib/flatpak/exports/share/applications"),
            Path.home() / ".local/share/flatpak/exports/share/applications"]


def _run(*cmd: str, timeout: float = 5) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"{cmd[0]} failed")
    return r.stdout.strip()


def _spawn(*cmd: str) -> None:
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _hypr(lua: str) -> str:
    return _run("hyprctl", "dispatch", lua)


def desktop_apps() -> dict[str, str]:
    """{display name (lowercase): desktop id} for every launchable app."""
    apps: dict[str, str] = {}
    for d in APP_DIRS:
        for f in sorted(d.glob("*.desktop")) if d.is_dir() else []:
            cp = configparser.ConfigParser(interpolation=None, strict=False)
            try:
                cp.read(f, encoding="utf-8")
                e = cp["Desktop Entry"]
            except Exception:  # noqa: BLE001
                continue
            if e.get("NoDisplay", "false").lower() == "true" or e.get("Type", "Application") != "Application":
                continue
            apps.setdefault(e.get("Name", f.stem).lower(), f.stem)
            apps.setdefault(f.stem.lower(), f.stem)
            for kw in e.get("Keywords", "").split(";"):
                if kw.strip():
                    apps.setdefault(kw.strip().lower(), f.stem)
    return apps


def find_app(name: str) -> str | None:
    apps = desktop_apps()
    q = name.lower().strip()
    if q in apps:
        return apps[q]
    contains = [k for k in apps if q in k]
    if contains:
        return apps[min(contains, key=len)]
    close = difflib.get_close_matches(q, apps.keys(), n=1, cutoff=0.6)
    return apps[close[0]] if close else None


@tool()
def open_app(name: str) -> str:
    """Open (launch) an application by name, e.g. "firefox", "terminal", "vs code", "spotify", "files".
    Websites like YouTube or Gmail open in the browser. Not needed before browser_search.

    Args:
        name: the app's name
    """
    sites = {"youtube": "youtube.com", "gmail": "mail.google.com", "google": "google.com",
             "whatsapp": "web.whatsapp.com", "instagram": "instagram.com", "chatgpt": "chatgpt.com",
             "github": "github.com", "netflix": "netflix.com", "maps": "maps.google.com"}
    if name.lower().strip() in sites:
        return open_url(sites[name.lower().strip()])
    aliases = {"terminal": "kitty", "files": "nautilus", "file manager": "nautilus", "browser": "brave",
               "vs code": "visual studio code", "vscode": "visual studio code", "code": "visual studio code"}
    app = find_app(aliases.get(name.lower().strip(), name))
    if app is None:
        return f"no installed app matches '{name}'"
    _spawn("gtk-launch", app)
    return f"opened {app}"


@tool()
def open_url(url: str) -> str:
    """Open a website or URL in the default browser. Also works for YouTube searches:
    https://www.youtube.com/results?search_query=...

    Args:
        url: the address, e.g. "youtube.com"
    """
    if "://" not in url:
        url = "https://" + url
    _spawn("xdg-open", url)
    return f"opened {url}"


@tool()
def browser_search(query: str, site: Literal["google", "youtube"] = "google") -> str:
    """Open the browser and search for something (one step: opens the browser with the results).
    Use for "open the browser and search X", "search X on Google", "play X on YouTube".

    Args:
        query: what to search for
        site: google or youtube
    """
    q = urllib.parse.quote_plus(query)
    url = (f"https://www.youtube.com/results?search_query={q}" if site == "youtube"
           else f"https://www.google.com/search?q={q}")
    _spawn("xdg-open", url)
    return f"opened the browser with {site} results for '{query}'"


@tool()
def volume(action: Literal["get", "set", "up", "down", "mute", "unmute"], percent: int = 10) -> str:
    """Read or change the speaker volume. Use action "get" to find out the current level.

    Args:
        action: get, set (to percent), up/down (by percent), mute or unmute
        percent: target level for set, or step size for up/down
    """
    sink = "@DEFAULT_AUDIO_SINK@"
    percent = max(0, min(150, percent))
    if action == "set":
        _run("wpctl", "set-volume", sink, f"{percent}%")
    elif action == "up":
        _run("wpctl", "set-volume", "-l", "1.5", sink, f"{percent}%+")
    elif action == "down":
        _run("wpctl", "set-volume", sink, f"{percent}%-")
    elif action in ("mute", "unmute"):
        _run("wpctl", "set-mute", sink, "1" if action == "mute" else "0")
    out = _run("wpctl", "get-volume", sink)  # "Volume: 0.64 [MUTED]"
    level = round(float(out.split()[1]) * 100)
    return f"volume is {level}%" + (" (muted)" if "MUTED" in out else "")


def _ddc_get() -> int:
    out = _run("ddcutil", "getvcp", "10", "--brief", timeout=15)   # "VCP 10 C 100 100"
    return int(out.split()[3])


@tool()
def brightness(action: Literal["get", "set", "up", "down"], percent: int = 10,
               screen: Literal["laptop", "external", "all"] = "laptop") -> str:
    """Read or change screen brightness. Use action "get" to find out the current level.

    Args:
        action: get, set (to percent), up or down (by percent)
        percent: target level for set, or step size for up/down
        screen: laptop (built-in), external (the HDMI monitor) or all
    """
    percent = max(1, min(100, percent))
    out = []
    if screen in ("laptop", "all"):
        arg = {"set": f"{percent}%", "up": f"+{percent}%", "down": f"{percent}%-"}.get(action)
        if arg:
            _run("brightnessctl", "-q", "set", arg)
        cur, mx = _run("brightnessctl", "get"), _run("brightnessctl", "max")
        out.append(f"laptop brightness is {round(int(cur) / int(mx) * 100)}%")
    if screen in ("external", "all"):
        try:
            if action != "get":
                cur = _ddc_get()
                target = {"set": percent, "up": cur + percent, "down": cur - percent}[action]
                _run("ddcutil", "setvcp", "10", str(max(0, min(100, target))), timeout=15)
            out.append(f"external monitor brightness is {_ddc_get()}%")
        except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired):
            out.append("the external monitor did not respond (is it connected, with DDC/CI on?)")
    return ", ".join(out)


@tool()
def media(action: Literal["play", "pause", "play_pause", "next", "previous", "stop", "status"]) -> str:
    """Control what is already playing (pause, resume, next song, stop). To START playing something
    new, use play_youtube instead.

    Args:
        action: play (resume), pause, play_pause, next, previous, stop, or status (what is playing)
    """
    from .media import control_music

    if (done := control_music(action)) is not None:   # Youru's own background music player
        return done
    if action == "status":
        try:
            meta = _run("playerctl", "metadata", "--format", "{{status}}: {{artist}} - {{title}}")
        except RuntimeError:
            return "nothing is playing"
        return meta
    cmd = {"play_pause": "play-pause"}.get(action, action)
    _run("playerctl", cmd)
    return f"media {action} done"


@tool()
def screenshot() -> str:
    """Take a screenshot of the whole screen and save it to ~/Pictures/Screenshots."""
    folder = Path.home() / "Pictures" / "Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"youru-{datetime.now():%Y%m%d-%H%M%S}.png"
    _run("grim", str(path))
    return f"saved screenshot to {path}"


@tool()
def notify(title: str, message: str) -> str:
    """Show a desktop notification.

    Args:
        title: short title
        message: notification text
    """
    _run("notify-send", "-a", "Youru", title, message)
    return "notification shown"


@tool()
def clipboard_get() -> str:
    """Read the text currently in the clipboard."""
    try:
        return _run("wl-paste", "--no-newline") or "the clipboard is empty"
    except RuntimeError:
        return "the clipboard is empty"


@tool()
def clipboard_set(text: str) -> str:
    """Copy text to the clipboard.

    Args:
        text: the text to copy
    """
    subprocess.run(["wl-copy"], input=text, text=True, timeout=5, check=True)
    return "copied to clipboard"


def _clients() -> list[dict]:
    return json.loads(_run("hyprctl", "clients", "-j"))


def _match_window(query: str) -> dict | None:
    q = query.lower()
    wins = [c for c in _clients() if c.get("mapped")]
    for c in wins:
        if q in c["class"].lower() or q in c["title"].lower():
            return c
    names = {f"{c['class']} {c['title']}".lower(): c for c in wins}
    close = difflib.get_close_matches(q, names.keys(), n=1, cutoff=0.3)
    return names[close[0]] if close else None


@tool()
def list_windows() -> str:
    """List open windows (app, title, workspace)."""
    wins = [c for c in _clients() if c.get("mapped")]
    if not wins:
        return "no open windows"
    return "\n".join(f"{c['class']}: {c['title'][:60]} (workspace {c['workspace']['name']})" for c in wins)


@tool()
def focus_window(query: str) -> str:
    """Switch to an open window.

    Args:
        query: part of the app name or window title, e.g. "firefox" or "youtube"
    """
    w = _match_window(query)
    if w is None:
        return f"no open window matches '{query}'"
    _hypr(f"hl.dsp.focus({{ window = 'address:{w['address']}' }})")
    return f"switched to {w['class']}"


@tool()
def switch_workspace(number: int) -> str:
    """Go to a workspace.

    Args:
        number: workspace number 1-10
    """
    _hypr(f"hl.dsp.focus({{ workspace = {int(number)} }})")
    return f"on workspace {number}"


@tool(risk="confirm", describe=lambda a: f"close the {a.get('query')} window")
def close_window(query: str) -> str:
    """Close an open window.

    Args:
        query: part of the app name or window title
    """
    w = _match_window(query)
    if w is None:
        return f"no open window matches '{query}'"
    _hypr(f"hl.dsp.window.close({{ window = 'address:{w['address']}' }})")
    return f"closed {w['class']}"


@tool()
def lock_screen() -> str:
    """Lock the screen."""
    lock = Path.home() / ".local/bin/lock.sh"
    _spawn(str(lock)) if lock.exists() else _run("loginctl", "lock-session")
    return "screen locked"


@tool(risk="confirm", describe=lambda a: "log you out" if a.get("action") == "logout"
      else f"{a.get('action')} the computer")
def power(action: Literal["shutdown", "reboot", "suspend", "logout"]) -> str:
    """Shut down, restart, suspend the computer, or log out of the desktop session.

    Args:
        action: shutdown, reboot, suspend or logout
    """
    if action == "logout":
        _spawn("uwsm", "stop")
        return "logging out"
    cmd = {"shutdown": "poweroff", "reboot": "reboot", "suspend": "suspend"}[action]
    _spawn("systemctl", cmd)
    return f"{action} started"



# ---------------------------------------------------------------- keyboard (wtype, Wayland virtual keyboard)

WTYPE_MODS = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift", "alt": "alt", "altgr": "altgr",
              "super": "logo", "win": "logo", "windows": "logo", "meta": "logo", "logo": "logo", "cmd": "logo"}
WTYPE_KEYS = {"enter": "Return", "return": "Return", "esc": "Escape", "escape": "Escape", "tab": "Tab",
              "space": "space", "backspace": "BackSpace", "delete": "Delete", "del": "Delete", "up": "Up",
              "down": "Down", "left": "Left", "right": "Right", "home": "Home", "end": "End",
              "pageup": "Page_Up", "pagedown": "Page_Down", "insert": "Insert", "print": "Print",
              "printscreen": "Print", "plus": "plus", "minus": "minus", "slash": "slash"}
TERMINALS = {"kitty", "alacritty", "foot", "wezterm", "org.wezfurlong.wezterm", "konsole", "gnome-terminal",
             "xterm", "ghostty", "com.mitchellh.ghostty"}


def wtype_args(keys: str) -> list[str]:
    """"super+d" -> wtype -M logo -k d -m logo ; "ctrl+shift+t" -> ... ; "f5" -> -k F5"""
    parts = [p.strip().lower() for p in keys.replace(" ", "").split("+") if p.strip()]
    if not parts:
        raise ValueError("no keys given")
    *mods, key = parts
    args: list[str] = []
    for m in mods:
        if m not in WTYPE_MODS:
            raise ValueError(f"unknown modifier '{m}' (use ctrl, shift, alt, super)")
        args += ["-M", WTYPE_MODS[m]]
    if key in WTYPE_KEYS:
        name = WTYPE_KEYS[key]
    elif len(key) > 1 and key[0] == "f" and key[1:].isdigit():
        name = key.upper()
    else:
        name = key
    args += ["-k", name]
    for m in reversed(mods):
        args += ["-m", WTYPE_MODS[m]]
    return args


def _focused_class() -> str:
    try:
        return json.loads(_run("hyprctl", "activewindow", "-j")).get("class", "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _typing_risk(args: dict) -> tuple[str, str]:
    if _focused_class() in TERMINALS:
        return "medium", "typing into a terminal can run commands"
    return "low", ""


CLOSING_KEYS = {"super+q", "super+w", "alt+f4", "ctrl+q", "ctrl+w", "ctrl+shift+w", "super+shift+e"}


def _keys_risk(args: dict) -> tuple[str, str]:
    combo = str(args.get("keys", "")).replace(" ", "").lower()
    return ("medium", "this shortcut closes a window or app") if combo in CLOSING_KEYS else ("low", "")


@tool(assess=_keys_risk, describe=lambda a: f"press {a.get('keys')} (it closes something)")
def press_keys(keys: str) -> str:
    """Press a keyboard shortcut or key, e.g. "super+d", "ctrl+shift+t", "alt+tab", "f5", "enter".
    Works for desktop shortcuts and the focused window.

    Args:
        keys: keys joined with +, modifiers first (ctrl, shift, alt, super)
    """
    if shutil.which("wtype") is None:
        return "keyboard control needs wtype: ask the user to run  sudo pacman -S wtype"
    _run("wtype", *wtype_args(keys))
    return f"pressed {keys}"


@tool(assess=_typing_risk, describe=lambda a: f"type \"{str(a.get('text', ''))[:80]}\" into the terminal")
def type_text(text: str) -> str:
    """Type text into the focused window, as if typed on the keyboard.

    Args:
        text: what to type
    """
    if shutil.which("wtype") is None:
        return "keyboard control needs wtype: ask the user to run  sudo pacman -S wtype"
    _run("wtype", "--", text, timeout=30)
    return f"typed {len(text)} characters"
