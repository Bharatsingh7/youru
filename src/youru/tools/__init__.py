"""Tool registry for the agent.

Tools are plain functions decorated with @tool. The JSON schema the LLM sees is
built from the signature and the docstring ("Args:" lines describe parameters).
Each call gets a risk level: fixed (risk="safe" -> low, risk="confirm" -> medium) or computed per call by
an `assess(args)` function (see risk.py, built on OmniPilot's classifier).
  low -> runs; medium/high -> runs only after the confirmer says yes; critical -> blocked.
"""

from __future__ import annotations

import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass

MAX_RESULT = 4000

_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass
class Tool:
    name: str
    func: Callable[..., str]
    risk: str                                # safe | confirm
    schema: dict
    describe: Callable[[dict], str] | None   # human sentence for the confirmation question
    assess: Callable[[dict], tuple[str, str]] | None = None   # args -> (level, reason)


REGISTRY: dict[str, Tool] = {}


def _param_docs(doc: str) -> tuple[str, dict[str, str]]:
    summary, _, rest = doc.partition("Args:")
    params = {}
    for line in rest.splitlines():
        m = re.match(r"\s*(\w+)\s*:\s*(.+)", line)
        if m:
            params[m.group(1)] = m.group(2).strip()
    return " ".join(summary.split()), params


def _json_type(hint) -> dict:
    origin = typing.get_origin(hint)
    if origin is typing.Literal:
        return {"type": "string", "enum": list(typing.get_args(hint))}
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return _json_type(args[0])
    return {"type": _TYPES.get(hint, "string")}


def tool(risk: str = "safe", describe: Callable[[dict], str] | None = None,
         assess: Callable[[dict], tuple[str, str]] | None = None):
    def wrap(func: Callable[..., str]) -> Callable[..., str]:
        summary, pdocs = _param_docs(inspect.getdoc(func) or "")
        hints = typing.get_type_hints(func)
        props, required = {}, []
        for name, p in inspect.signature(func).parameters.items():
            prop = _json_type(hints.get(name, str))
            if name in pdocs:
                prop["description"] = pdocs[name]
            props[name] = prop
            if p.default is inspect.Parameter.empty:
                required.append(name)
        schema = {"type": "function", "function": {
            "name": func.__name__, "description": summary,
            "parameters": {"type": "object", "properties": props, "required": required}}}
        REGISTRY[func.__name__] = Tool(func.__name__, func, risk, schema, describe, assess)
        return func
    return wrap


def load_all() -> dict[str, Tool]:
    from . import (desktop, digup, files, media, omnipilot, organizer, pyrun, recipes, reminders, screen,  # noqa: F401
                   shell, system, voice, web)

    return REGISTRY


def schemas() -> list[dict]:
    return [t.schema for t in load_all().values()]


def execute(name: str, args: dict | str, confirm: Callable[[str], bool]) -> str:
    """Run a tool call from the LLM. Never raises; errors come back as text."""
    tools = load_all()
    if name not in tools:
        return f"error: unknown tool {name}"
    t = tools[name]
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError as e:
            return f"error: arguments are not valid JSON ({e})"
    level, reason = risk_of(t, args)
    if level == "critical":
        return f"blocked: this is too dangerous to run ({reason}). Tell the user it was blocked."
    if level in ("medium", "high"):
        question = t.describe(args) if t.describe else f"run {name} with {args}"
        if not confirm(question):
            return "cancelled: the user said no. Tell them it was cancelled."
    try:
        out = str(t.func(**args))
    except TypeError as e:
        return f"error: bad arguments for {name}: {e}"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}: {e}"
    return out if len(out) <= MAX_RESULT else out[:MAX_RESULT] + "\n...(truncated)"


def risk_of(t: Tool, args: dict) -> tuple[str, str]:
    if t.assess is not None:
        try:
            return t.assess(args)
        except Exception as e:  # noqa: BLE001  (a broken classifier must not auto-run anything)
            return "high", f"could not assess risk: {e}"
    return ("medium", "needs confirmation") if t.risk == "confirm" else ("low", "")
