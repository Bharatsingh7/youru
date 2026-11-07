"""how_to + memory tools: what Youru knows how to do, and what it remembers about you."""

from __future__ import annotations

from pathlib import Path

import yaml

from .. import memory
from . import tool

CURATED = Path(__file__).resolve().parents[1] / "recipes.yaml"


def _curated() -> list[dict]:
    return yaml.safe_load(CURATED.read_text()) or []


def _score(query: set[str], text: str) -> float:
    w = memory.words(text)
    return len(query & w) / (len(query) or 1)


def find_recipes(task: str, limit: int = 3) -> list[tuple[float, str]]:
    q = memory.words(task)
    found: list[tuple[float, str]] = []
    for r in _curated():
        s = _score(q, r["task"])
        if s > 0:
            cmds = "\n".join(f"  $ {c}" for c in r["commands"])
            note = f"\n  note: {r['note']}" if r.get("note") else ""
            found.append((s, f"- {r['task']}:\n{cmds}{note}"))
    for r in memory.learned_recipes():
        s = _score(q, r["task"]) + 0.2   # learned on this PC: prefer it
        if s > 0.2:
            found.append((s, f"- (learned) {r['task']}:\n  $ {r['command']}"))
    for k in memory.learned_skills():
        s = _score(q, k["trigger"]) + 0.2
        if s > 0.2:
            steps = "; ".join(f"{st['tool']}({st.get('args', '')})" for st in k["steps"])
            found.append((s, f"- (learned skill) {k['trigger']}: {steps}"))
    found.sort(key=lambda x: -x[0])
    return found[:limit]


@tool()
def how_to(task: str) -> str:
    """Look up how to do something on this PC (known commands and learned skills), e.g. "default browser",
    "fan speed", "wifi off", "dark mode", "battery limit". Use it before saying you can't do something.

    Args:
        task: what the user wants, in a few English words
    """
    found = find_recipes(task)
    if not found:
        return (f"no known recipe for '{task}'. Next: web_search 'linux command line how to {task}', "
                "then run_command the best command (check with a read-only command first).")
    return ("Known ways. Now CALL run_command yourself with the right command (replace <...>); do not tell "
            "the user to run it:\n" + "\n".join(t for _, t in found))


@tool()
def remember(fact: str) -> str:
    """Save a lasting fact or preference about the user, e.g. "The user's name is Bharat",
    "The user lives in Beawar, Rajasthan", "The user's favourite singer is Arijit Singh".

    Args:
        fact: one short sentence
    """
    return memory.remember_fact(fact)


@tool()
def recall(about: str = "") -> str:
    """What Youru remembers about the user (optionally only about one topic).

    Args:
        about: topic to look for, or empty for everything
    """
    items = memory.relevant_facts(about, profile=50, matched=20) if about else [f["fact"] for f in memory.facts()]
    return "\n".join(f"- {f}" for f in items) or "I don't remember anything yet."


@tool(risk="confirm", describe=lambda a: f"forget what I remember about {a.get('about')}")
def forget(about: str) -> str:
    """Forget remembered facts about a topic.

    Args:
        about: what to forget, e.g. "my city"
    """
    return memory.forget_fact(about)
