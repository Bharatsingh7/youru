"""Deadlines (your deadline-tracker) and quick notes / todo (plain markdown in ~/Documents/Youru)."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from . import tool

NOTES_DIR = Path.home() / "Documents" / "Youru"


def _tracker(*args: str) -> str:
    binary = shutil.which("deadline-tracker") or str(Path.home() / ".local/bin/deadline-tracker")
    r = subprocess.run([binary, *args], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip() or "deadline-tracker failed")
    return r.stdout


def _deadlines() -> list[dict]:
    return json.loads(_tracker("list", "--json") or "[]")


@tool()
def add_deadline(name: str, date: str, time: str = "") -> str:
    """Add a deadline (exam, submission, bill...) to the user's deadline tracker.
    Convert words like "5 November" or "next Monday" to a real date yourself.

    Args:
        name: what is due, e.g. "Physics exam"
        date: YYYY-MM-DD
        time: HH:MM (24-hour), optional
    """
    datetime.strptime(date, "%Y-%m-%d")   # validates; raises a clear error for the model
    args = ["add", "--name", name, "--date", date] + (["--time", time] if time else [])
    _tracker(*args)
    return f"added deadline: {name} on {datetime.strptime(date, '%Y-%m-%d'):%a %d %B}" + (f" {time}" if time else "")


@tool()
def list_deadlines(days: int = 30, include_done: bool = False) -> str:
    """Upcoming deadlines ("what's due this week" -> days=7).

    Args:
        days: how many days ahead to show
        include_done: also show completed ones
    """
    items = [d for d in _deadlines() if include_done or not d.get("completed")]
    soon = [d for d in items if d.get("remaining_seconds", 0) <= days * 86400]
    if not soon:
        return f"nothing due in the next {days} days" + (f" ({len(items)} later)" if items else "")
    soon.sort(key=lambda d: d.get("remaining_seconds", 0))
    return "\n".join(f"- {d['name']}: {d.get('due_str') or d.get('deadline')} ({d.get('remaining', '')}"
                     f"{', overdue' if d.get('remaining_seconds', 0) < 0 else ''})" for d in soon)


def _match(name: str) -> dict | None:
    items = _deadlines()
    names = {d["name"].lower(): d for d in items}
    hit = difflib.get_close_matches(name.lower(), names, n=1, cutoff=0.4)
    return names[hit[0]] if hit else next((d for d in items if name.lower() in d["name"].lower()), None)


@tool()
def complete_deadline(name: str) -> str:
    """Mark a deadline as done.

    Args:
        name: part of its name
    """
    d = _match(name)
    if d is None:
        return f"no deadline matches '{name}'"
    _tracker("complete", str(d["id"]))
    return f"marked done: {d['name']}"


@tool(risk="confirm", describe=lambda a: f"delete the deadline {a.get('name')}")
def delete_deadline(name: str) -> str:
    """Delete a deadline.

    Args:
        name: part of its name
    """
    d = _match(name)
    if d is None:
        return f"no deadline matches '{name}'"
    _tracker("delete", str(d["id"]))
    return f"deleted: {d['name']}"


# ---------------------------------------------------------------- notes and todo

def _file(name: str) -> Path:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return NOTES_DIR / name


@tool()
def add_note(text: str) -> str:
    """Save a quick note ("note likh lo: buy milk", "remember to note this").

    Args:
        text: the note
    """
    with open(_file("notes.md"), "a") as f:
        f.write(f"- {datetime.now():%Y-%m-%d %H:%M} {text.strip()}\n")
    return f"noted: {text.strip()}"


@tool()
def read_notes(last: int = 10) -> str:
    """Read the most recent notes.

    Args:
        last: how many
    """
    p = _file("notes.md")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()] if p.exists() else []
    return "\n".join(lines[-last:]) or "no notes yet"


@tool()
def add_todo(text: str) -> str:
    """Add an item to the to-do list.

    Args:
        text: the task
    """
    with open(_file("todo.md"), "a") as f:
        f.write(f"- [ ] {text.strip()}\n")
    return f"added to your to-do list: {text.strip()}"


@tool()
def list_todo(include_done: bool = False) -> str:
    """Show the to-do list.

    Args:
        include_done: also show finished items
    """
    p = _file("todo.md")
    items = [ln for ln in p.read_text().splitlines() if ln.startswith("- [")] if p.exists() else []
    if not include_done:
        items = [ln for ln in items if ln.startswith("- [ ]")]
    return "\n".join(items) or "your to-do list is empty"


@tool()
def done_todo(text: str) -> str:
    """Tick off a to-do item.

    Args:
        text: part of the item
    """
    p = _file("todo.md")
    lines = p.read_text().splitlines() if p.exists() else []
    open_items = {ln[6:].lower(): i for i, ln in enumerate(lines) if ln.startswith("- [ ] ")}
    hit = difflib.get_close_matches(text.lower(), open_items, n=1, cutoff=0.4) or \
        [k for k in open_items if text.lower() in k][:1]
    if not hit:
        return f"no open to-do item matches '{text}'"
    i = open_items[hit[0]]
    lines[i] = "- [x] " + lines[i][6:]
    p.write_text("\n".join(lines) + "\n")
    return f"done: {lines[i][6:]}"
