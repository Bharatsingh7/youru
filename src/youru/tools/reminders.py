"""Timers, alarms and reminders as systemd user timers: they survive restarts and fire even when
`youru assist` isn't running. When one fires it runs `youru alert "<message>"` (sound + notification +
the message spoken in Youru's voice)."""

from __future__ import annotations

import json
import re
import secrets
import shutil
import subprocess
from datetime import datetime, timedelta

from ..config import ROOT
from . import tool

PREFIX = "youru-rem-"


def _alert_cmd(message: str) -> list[str]:
    uv = shutil.which("uv") or "/usr/bin/uv"
    return [uv, "run", "--project", str(ROOT), "youru", "alert", message]


def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "systemd-run failed")
    return r.stdout.strip()


def timer_args(unit: str, description: str, message: str, *, seconds: int | None = None,
               at: datetime | None = None) -> list[str]:
    args = ["systemd-run", "--user", f"--unit={unit}", f"--description=Youru: {description}",
            "--timer-property=AccuracySec=1s", "--collect"]
    if seconds is not None:
        args.append(f"--on-active={seconds}s")
    else:
        args += [f"--on-calendar={at:%Y-%m-%d %H:%M:00}", "--timer-property=Persistent=true"]
    return args + ["--", *_alert_cmd(message)]


def parse_when(at: str, now: datetime | None = None) -> datetime:
    """"2026-10-22 19:00", "19:00" (today, or tomorrow if already past), "7 pm", "7:30am"."""
    now = now or datetime.now()
    s = at.strip().lower().replace(".", ":")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s)
    if not m:
        raise ValueError(f"I couldn't understand the time '{at}' (use YYYY-MM-DD HH:MM)")
    h, mi, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ampm == "pm" and h < 12:
        h += 12
    if ampm == "am" and h == 12:
        h = 0
    when = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    return when if when > now else when + timedelta(days=1)


def _unit() -> str:
    return PREFIX + secrets.token_hex(3)


@tool()
def set_timer(minutes: float, label: str = "Timer") -> str:
    """Start a countdown timer ("10 minute timer", "5 minute ka timer lagao"). It rings and speaks when done.

    Args:
        minutes: how long, in minutes (0.5 = 30 seconds)
        label: what it's for, e.g. "tea"
    """
    seconds = max(5, int(round(minutes * 60)))
    _run(timer_args(_unit(), f"{label} timer ({minutes:g} min)", f"{label} timer is done!", seconds=seconds))
    ends = datetime.now() + timedelta(seconds=seconds)
    return f"timer set for {minutes:g} minutes ({label}), rings at {ends:%I:%M %p}"


@tool()
def set_reminder(at: str, message: str) -> str:
    """Set a reminder or alarm at a time ("remind me at 7 pm to call mom", "kal subah 6 baje uthana").
    Convert words like "kal" (tomorrow) and "subah" (morning) to a real date and time yourself.

    Args:
        at: when, as "YYYY-MM-DD HH:MM" (24-hour), or "HH:MM" for the next such time
        message: what to say, e.g. "Call mom" or "Wake up!"
    """
    when = parse_when(at)
    if when <= datetime.now():
        return f"error: {when:%d %b %I:%M %p} is in the past; ask the user for a future time"
    _run(timer_args(_unit(), message, message, at=when))
    return f"reminder set for {when:%A %d %B, %I:%M %p}: {message}"


def _timers() -> list[dict]:
    out = subprocess.run(["systemctl", "--user", "list-timers", "--all", "--output=json", f"{PREFIX}*"],
                         capture_output=True, text=True, timeout=10).stdout or "[]"
    timers = json.loads(out)
    for t in timers:
        unit = t.get("unit", "")
        desc = subprocess.run(["systemctl", "--user", "show", "-P", "Description", unit],
                              capture_output=True, text=True, timeout=5).stdout.strip()
        t["description"] = desc.removeprefix("Youru: ")
    return timers


@tool()
def list_reminders() -> str:
    """List the timers, alarms and reminders that are set."""
    timers = _timers()
    if not timers:
        return "no timers or reminders are set"
    lines = []
    for t in timers:
        nxt = t.get("next")
        when = datetime.fromtimestamp(nxt / 1e6).strftime("%a %d %b %I:%M %p") if nxt else "?"
        lines.append(f"- {t['description']} at {when}")
    return "\n".join(lines)


@tool()
def cancel_reminder(which: str) -> str:
    """Cancel a timer or reminder.

    Args:
        which: words from its message, e.g. "tea" or "call mom", or "all"
    """
    timers = _timers()
    q = which.lower().strip()
    hits = [t for t in timers if q == "all" or q in t["description"].lower()]
    if not hits:
        return f"no timer or reminder matches '{which}'"
    for t in hits:
        subprocess.run(["systemctl", "--user", "stop", t["unit"]], capture_output=True, timeout=10)
    return "cancelled: " + "; ".join(t["description"] for t in hits)
