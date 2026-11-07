"""Nightly self-review: learn from the day's failures.

Reads data/sessions/<day>/sessions.jsonl (what you said, meaning, tools, reply), finds what went wrong
(refusals, no tool where one was needed, tool errors, requests you had to repeat) and asks the local LLM
for a command that would have worked. Your own corrections ("nahi, …") are saved straight away; new commands
go to pending.yaml until you approve them (`youru learn --review`). Facts are only learned when you
tell Youru something (the remember tool), never guessed from the log.
"""

from __future__ import annotations

import difflib
import json
import shutil
import shlex
from datetime import date, datetime

import httpx
from rich.console import Console

from . import memory
from .agent import REFUSAL, SEARCH_ONLY, tool_hints
from .config import DATA

console = Console()

ASK = """A voice assistant on the user's Linux PC (CachyOS, Hyprland desktop, ASUS laptop) failed a request.
The user said: {said}
Understood as: {meaning}
Tools it called: {tools}
It replied: {reply}
Problem: {problem}

If ONE bash command would have done this on Arch Linux, answer {{"type": "recipe", "task": "<short English
task>", "command": "<the exact command>"}}. Use only real programs that exist on Arch Linux; no placeholders,
no echo-only commands. If you are not sure, answer {{"type": "none"}}."""

def day_rows(day: date) -> list[dict]:
    f = DATA / "sessions" / day.isoformat() / "sessions.jsonl"
    if not f.exists():
        return []
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("transcript") and r.get("reply") is not None]


def failures(rows: list[dict]) -> list[tuple[dict, str, dict | None]]:
    out = []
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        tools = r.get("tools") or []
        said = r["transcript"]
        if REFUSAL.search(r.get("reply") or ""):
            problem = "it said it couldn't do it"
        elif not tools and any(h not in ("web_search", "how_to") for h in tool_hints(said)):
            problem = "it answered without using a tool, although the request needed one"
        elif any(str(t.get("result", "")).startswith(("error", "exit code 1", "exit code 127")) for t in tools):
            problem = "a tool failed"
        elif nxt and difflib.SequenceMatcher(None, said.lower(), nxt["transcript"].lower()).ratio() > 0.75:
            problem = "the user had to repeat the request"
        elif nxt and memory.correction_target(nxt["transcript"]):
            problem = "the user corrected it"
        else:
            continue
        out.append((r, problem, nxt))
    return out


def ask_llm(url: str, prompt: str, think: bool = False) -> dict:
    r = httpx.post(f"{url}/v1/chat/completions", timeout=180, json={
        "messages": [{"role": "user", "content": prompt}], "max_tokens": 400 if think else 200,
        "temperature": 0.2, "response_format": {"type": "json_object"}, "id_slot": 1,
        "chat_template_kwargs": {"enable_thinking": think}})
    r.raise_for_status()
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError, IndexError):
        return {"type": "none"}


def command_ok(command: str) -> bool:
    """A real, concrete command: its program exists, no placeholders, not just an echo, not dangerous."""
    from .tools.risk import shell

    try:
        head = shlex.split(command)[0]
    except (ValueError, IndexError):
        return False
    if head in ("echo", "printf", "true", ":") or any(ch in command for ch in "<>[]{}") and "|" not in command:
        return False
    if not (head.startswith(("~", "/")) or shutil.which(head)):
        return False
    return shell({"command": command})[0] != "critical"


def review(url: str, day: date | None = None, quiet: bool = False) -> dict:
    from .tools.recipes import find_recipes

    day = day or date.today()
    rows = day_rows(day)
    summary = {"looked_at": 0, "facts": 0, "aliases": 0, "pending": 0}
    pending = memory.pending()
    for row, problem, nxt in failures(rows)[:20]:
        summary["looked_at"] += 1
        if problem == "the user corrected it":   # deterministic: what you actually said next
            meant = memory.correction_target(nxt["transcript"])
            if meant and meant.lower() != row["transcript"].lower():
                memory.save_correction(row["transcript"], meant)
                summary["aliases"] += 1
                if not quiet:
                    console.print(f"[dim]correction:[/] \"{row['transcript']}\" → \"{meant}\"")
            continue
        if problem not in ("it said it couldn't do it", "a tool failed"):
            continue
        task = row.get("meaning") or row["transcript"]
        if (best := find_recipes(task, limit=1)) and best[0][0] >= 0.5:
            continue   # a known recipe covers it already (the failure is from before it existed)
        if any(h not in SEARCH_ONLY | {"run_command"} for h in tool_hints(task)):
            continue   # a dedicated tool handles this now (e.g. play_youtube)
        idea = ask_llm(url, ASK.format(said=row["transcript"], meaning=row.get("meaning") or "-",
                                       tools=", ".join(t["name"] for t in row.get("tools") or []) or "none",
                                       reply=(row.get("reply") or "")[:300], problem=problem), think=True)
        cmd = str(idea.get("command", "")).strip()
        ok = idea.get("type") == "recipe" and idea.get("task") and command_ok(cmd)
        if ok and not any(x.get("command") == cmd for x in pending):
            pending.append({"task": idea["task"], "command": cmd, "why": problem,
                            "from": row["transcript"], "date": day.isoformat()})
            summary["pending"] += 1
        if not quiet:
            console.print(f"[dim]{row['transcript']!r} ({problem}) ->[/] {idea if ok else 'nothing reliable'}")
    # Facts are not extracted here: a small model turns events ("the user asked about the fan") into fake
    # facts. Facts come only from the remember tool, when you tell it something about yourself.
    memory.set_pending(pending)
    return summary


def approve_pending() -> None:
    items = memory.pending()
    if not items:
        console.print("Nothing waiting for approval.")
        return
    keep = []
    for it in items:
        console.print(f"\n[bold]{it['task']}[/]  [dim](from \"{it['from']}\", {it['why']})[/]\n  $ {it['command']}")
        ans = console.input("  Learn this? [y]es / [n]o / [s]kip for now: ").strip().lower()
        if ans.startswith("y"):
            memory.save_recipe(it["task"], it["command"])
        elif not ans.startswith("n"):
            keep.append(it)
    memory.set_pending(keep)


def show_learned() -> None:
    facts = memory.facts()
    console.rule(f"Facts about you ({len(facts)})")
    for f in facts:
        console.print(f"- {f['fact']} [dim]({datetime.fromtimestamp(f['time']):%d %b}, {f.get('source', 'user')})[/]")
    corr = memory._load_yaml("corrections.yaml", [])
    console.rule(f"Corrections ({len(corr)})")
    for c in corr:
        console.print(f"- heard \"{c['heard']}\" → meant \"{c['meant']}\"")
    rec = memory.learned_recipes()
    console.rule(f"Learned commands ({len(rec)})")
    for r in rec:
        console.print(f"- {r['task']}: [cyan]{r['command']}[/] [dim]({r.get('learned', '')})[/]")
    sk = memory.learned_skills()
    console.rule(f"Skills ({len(sk)})")
    for k in sk:
        console.print(f"- {k['trigger']}: " + " → ".join(st["tool"] for st in k["steps"]))
    pend = memory.pending()
    console.rule(f"Waiting for your OK ({len(pend)})")
    for it in pend:
        console.print(f"- {it['task']}: [cyan]{it['command']}[/]   (run: youru learn --review)")
    vocab = memory.vocab()
    if vocab:
        console.rule("Words speech-to-text learned")
        console.print(", ".join(vocab))
    console.print(f"\n[dim]Files: {memory.HOME}[/]")
