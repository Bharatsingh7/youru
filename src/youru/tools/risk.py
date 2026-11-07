"""Risk levels for commands, code and file writes.

OmniPilot's classifier (omnipilot.permission.danger) catches the dangerous patterns: rm -rf, dd to a disk,
sudo, curl | sh, fork bombs... But it calls anything it doesn't recognise "low", including
`echo x > ~/.bashrc` or Python writing a file. Youru auto-runs low risk, so on top of it we add a floor:
only known read-only commands and pure computation stay low; anything else at least asks.
"""

from __future__ import annotations

import re
import shlex
import shutil
from pathlib import Path

from omnipilot.permission.danger import assess as _omni

from ..config import DATA

RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# Commands that only read or report. A command is low risk only if every part of a pipeline is one of these.
READ_ONLY = {
    "ls", "ll", "cat", "head", "tail", "less", "wc", "grep", "rg", "find", "fd", "tree", "du", "df", "free",
    "uname", "whoami", "id", "date", "cal", "uptime", "hostname", "hostnamectl", "ps", "pgrep", "top",
    "which", "whereis", "type", "file", "stat", "lsblk", "lscpu", "lsusb", "lspci", "ip", "ping", "nvidia-smi",
    "sensors", "echo", "printf", "pwd", "env", "printenv", "sort", "uniq", "cut", "awk", "jq", "column",
    "realpath", "dirname", "basename", "md5sum", "sha256sum", "neofetch", "fastfetch", "inxi", "wpctl",
    "brightnessctl", "playerctl", "hyprctl", "journalctl", "pacman", "git", "systemctl",
}
# Subcommands of the tools above that change things.
CHANGES = re.compile(
    r"\bgit\s+(push|commit|reset|checkout|clean|rebase|merge|pull|rm|mv|stash)\b|\bpacman\s+-[^Q\s]*[SRU]|"
    r"\bsystemctl\s+(start|stop|restart|enable|disable|mask|kill|poweroff|reboot|suspend)\b|"
    r"\bfind\b.*\s-(delete|exec|execdir|ok)\b|\bhyprctl\s+(dispatch|keyword|kill|reload)\b|"
    r"\bwpctl\s+set|\bbrightnessctl\s+(s|set)\b|\bplayerctl\s+\w")
# Read-only forms of tools that can also change things.
READ_ONLY_FORMS = re.compile(
    r"^\s*(powerprofilesctl\s+(get|list)|asusctl\s+(profile\s+(get|list)|leds\s+get|battery\s*$|--help)|"
    r"bluetoothctl\s+(show|devices|info|list)|nmcli(\s+-[tf]\S*(\s+\S+)?)*\s+(radio(\s+wifi)?\s*$|"
    r"(dev(ice)?|con(nection)?)(\s+(wifi)?\s*(list|show|status)?)?\s*$|general)|xdg-settings\s+get|"
    r"gsettings\s+(get|list-\S+)|checkupdates|sensors|curl\s+-s\s+https?://\S+\s*$|"
    r"timedatectl(\s+status)?\s*$|localectl(\s+status)?\s*$)", re.I)
SHELL_WORDS = {"cd", "export", "for", "while", "if", "then", "do", "done", "fi", "echo", "printf", "test", "[",
               "true", "false", "read", "set", "unset", "source", "."}
WRITES = re.compile(r"(?<![0-9&])>{1,2}(?!&)|\btee\b|\bsed\s+-i|\btruncate\b")

PY_WRITES = re.compile(
    r"open\([^)]*['\"][wax+]|\.write_(text|bytes)\(|\.(unlink|rmdir|rename|replace|touch|mkdir|chmod)\(|"
    r"\bos\.(remove|unlink|rename|replace|makedirs|mkdir|rmdir|chmod|chown|system|popen|kill)\b|"
    r"\bshutil\.|\bsubprocess\b|\brequests\.(post|put|patch|delete)\b|\bsocket\b")


def _max(*levels: tuple[str, str]) -> tuple[str, str]:
    return max(levels, key=lambda lr: RANK[lr[0]])


def _touches_voice_data(text: str) -> bool:
    return str(DATA) in text or "youru/data" in text


def shell(args: dict) -> tuple[str, str]:
    command = str(args.get("command", ""))
    a = _omni("shell.run_command", {"command": command})
    omni = (a.level, "; ".join(a.reasons))
    if a.level == "critical":
        return omni
    ours = ("low", "read-only command")
    try:
        parts = [shlex.split(p) for p in re.split(r"\|\|?|&&|;", command) if p.strip()]
    except ValueError:
        parts = [[]]
    heads = [p[0] for p in parts if p]
    if heads and all(h not in SHELL_WORDS and not h.startswith(("/", "~", "./")) and shutil.which(h) is None
                     for h in heads):
        return "low", "command not found"   # nothing would run: let it fail fast, no need to ask
    if WRITES.search(command):
        ours = ("medium", "writes to a file")
    elif CHANGES.search(command):
        ours = ("medium", "changes system state")
    else:
        segments = [p for p in re.split(r"\|\|?|&&|;", command) if p.strip()]
        if not all(p and (p[0] in READ_ONLY or READ_ONLY_FORMS.match(seg)) for p, seg in zip(parts, segments)):
            ours = ("medium", "not a known read-only command")
    if _touches_voice_data(command) and not ours[0] == "low":
        ours = ("high", "touches Youru's voice recordings")
    return _max(omni, ours)


def python(args: dict) -> tuple[str, str]:
    code = str(args.get("code", ""))
    a = _omni("python.run_python", {"code": code})
    omni = (a.level, "; ".join(a.reasons))
    if a.level == "critical":
        return omni
    ours = ("medium", "changes files or the system") if PY_WRITES.search(code) else ("low", "computation only")
    if _touches_voice_data(code) and ours[0] != "low":
        ours = ("high", "touches Youru's voice recordings")
    return _max(omni, ours)


def file_write(tool: str, *path_keys: str, existing_is_medium: bool = True):
    """Writing/editing a file in home: low for a new file, medium when it changes an existing one."""
    def assess(args: dict) -> tuple[str, str]:
        omni_args = {k: args.get(k) for k in path_keys}
        a = _omni(tool, omni_args)
        omni = (a.level, "; ".join(a.reasons))
        paths = [Path(str(args.get(k, ""))).expanduser() for k in path_keys if args.get(k)]
        if any(_touches_voice_data(str(p.resolve())) for p in paths):
            return "high", "touches Youru's voice recordings"
        changes_existing = existing_is_medium and any(p.exists() for p in paths)
        ours = ("medium", "changes an existing file") if changes_existing else ("low", "new file")
        return _max(omni, ours)
    return assess
