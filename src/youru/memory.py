"""What Youru learns over time: facts about you, corrections, recipes/skills, extra vocabulary.

Plain files in ~/.local/share/youru/ (readable, editable, never sent anywhere):
  memory.jsonl       facts ("The user's name is Bharat")
  corrections.yaml   heard phrase -> what you meant ("officer kholo" -> "browser kholo")
  recipes.yaml       learned task -> shell command that worked
  skills.yaml        learned multi-step tool sequences
  vocab.txt          words speech-to-text should know (names, places)
  pending.yaml       proposals from the nightly self-review that need your OK
Recall is keyword scoring: no extra model, so it costs nothing at runtime.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path

import yaml

HOME = Path.home() / ".local/share/youru"
STOP = set("""a an the is are am was were be to of in on for and or it this that my me i you your please can
could would will do does did what which who how when where kya hai hain ka ki ke ko se mein mera meri mere
aur bhi to ye yeh woh karo kar do""".split())
WORD = re.compile(r"[\wऀ-ॿ']+")


def words(text: str) -> set[str]:
    return {w for w in (x.lower() for x in WORD.findall(text)) if w not in STOP and len(w) > 1}


def _load_yaml(name: str, default):
    p = HOME / name
    try:
        return yaml.safe_load(p.read_text()) or default
    except (FileNotFoundError, yaml.YAMLError):
        return default


def _save_yaml(name: str, data) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / name).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


# ---------------------------------------------------------------- facts

def facts() -> list[dict]:
    p = HOME / "memory.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def remember_fact(fact: str, source: str = "user") -> str:
    fact = " ".join(fact.split()).strip()
    if not fact:
        return "nothing to remember"
    existing = facts()
    for f in existing:   # an update of the same fact replaces it ("my city is X" twice)
        if difflib.SequenceMatcher(None, f["fact"].lower(), fact.lower()).ratio() > 0.8:
            f["fact"], f["time"] = fact, time.time()
            _write_facts(existing)
            return f"updated: {fact}"
    existing.append({"fact": fact, "time": time.time(), "source": source, "uses": 0})
    _write_facts(existing)
    return f"remembered: {fact}"


def forget_fact(about: str) -> str:
    keep, gone = [], []
    q = words(about)
    for f in facts():
        (gone if q and q & words(f["fact"]) else keep).append(f)
    _write_facts(keep)
    return f"forgot {len(gone)} thing(s): " + "; ".join(f["fact"] for f in gone) if gone else "nothing matched"


def _write_facts(items: list[dict]) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "memory.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in items))


ABOUT_ME = re.compile(r"\b(me|my|mine|myself|i|i'm|im|mera|meri|mere|mujhe|mujhko|main|hum|hamara|apna|"
                      r"favou?rite|pasand|usual)\b|मेरा|मेरी|मुझे|मैं", re.I)


def relevant_facts(text: str, profile: int = 4, matched: int = 3) -> list[str]:
    """Facts sharing words with the request, plus a small profile when the request is about the user.
    (Attaching everything to every request distracts a small model.)"""
    items = facts()
    if not items:
        return []
    q = words(text)
    scored = sorted(items, key=lambda f: len(q & words(f["fact"])), reverse=True)
    hits = [f for f in scored if q & words(f["fact"])][:matched]
    base = sorted(items, key=lambda f: (f.get("uses", 0), f["time"]), reverse=True)[:profile] \
        if ABOUT_ME.search(text) else []
    out = []
    for f in hits + base:
        if f["fact"] not in out:
            out.append(f["fact"])
    return out


# ---------------------------------------------------------------- corrections

CORRECTION = re.compile(r"^\s*(no+|nope|nahi+|nahin|galat|wrong|not that|i meant|i said|maine kaha|mera matlab|"
                        r"matlab|नहीं|गलत|मैंने कहा)\b[\s,.:!-]*(i meant|maine kaha|mera matlab)?[\s,:]*(.*)$", re.I)


def correction_target(text: str) -> str | None:
    """"nahi, browser kholo" -> "browser kholo" (None if this isn't a correction)."""
    m = CORRECTION.match(text)
    if not m:
        return None
    rest = m.group(3).strip(" .")
    return rest if len(rest.split()) >= 2 else None


def save_correction(heard: str, meant: str) -> None:
    items = _load_yaml("corrections.yaml", [])
    items = [c for c in items if c["heard"].lower() != heard.lower()]
    items.append({"heard": heard, "meant": meant, "time": int(time.time())})
    _save_yaml("corrections.yaml", items[-300:])


def apply_correction(text: str) -> str | None:
    """What you meant last time you said (nearly) exactly this."""
    best, score = None, 0.0
    for c in _load_yaml("corrections.yaml", []):
        r = difflib.SequenceMatcher(None, c["heard"].lower(), text.lower()).ratio()
        if r > score:
            best, score = c["meant"], r
    return best if score >= 0.85 else None


def vocab() -> list[str]:
    """Words speech-to-text should know: vocab.txt (you can edit it) plus the names in remembered facts
    ("The user studies at AMITY University" -> AMITY, University). Not learned from corrections: those
    are speech-to-text output too, so a misheard word would only reinforce itself."""
    p = HOME / "vocab.txt"
    words_ = [w for w in p.read_text().split("\n") if w.strip()] if p.exists() else []
    for f in facts():
        tokens = WORD.findall(f["fact"])
        words_ += [w for i, w in enumerate(tokens) if i > 1 and w[:1].isupper() and len(w) >= 3 and w.isalpha()
                   and w not in words_]
    return words_[-60:]


def add_vocab(new: list[str]) -> None:
    current = vocab()
    merged = current + [w for w in new if w not in current]
    HOME.mkdir(parents=True, exist_ok=True)
    (HOME / "vocab.txt").write_text("\n".join(merged[-60:]) + "\n")


# ---------------------------------------------------------------- recipes and skills

def save_recipe(task: str, command: str) -> None:
    items = _load_yaml("recipes.yaml", [])
    if any(r["command"] == command for r in items):
        return
    items.append({"task": task, "command": command, "learned": time.strftime("%Y-%m-%d")})
    _save_yaml("recipes.yaml", items[-200:])


def save_skill(trigger: str, steps: list[dict]) -> None:
    items = _load_yaml("skills.yaml", [])
    names = [s["tool"] for s in steps]
    if any([s["tool"] for s in k["steps"]] == names and words(k["trigger"]) == words(trigger) for k in items):
        return
    items.append({"trigger": trigger, "steps": steps, "learned": time.strftime("%Y-%m-%d")})
    _save_yaml("skills.yaml", items[-200:])


def learned_recipes() -> list[dict]:
    return _load_yaml("recipes.yaml", [])


def learned_skills() -> list[dict]:
    return _load_yaml("skills.yaml", [])


def pending() -> list[dict]:
    return _load_yaml("pending.yaml", [])


def set_pending(items: list[dict]) -> None:
    _save_yaml("pending.yaml", items)
