"""Time, system status and processes."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from datetime import datetime

import psutil

from . import tool


@tool()
def get_time() -> str:
    """Current local date, time and day of the week."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %d %B %Y, %I:%M %p (%Z)")


@tool()
def system_info() -> str:
    """CPU usage and temperature, fan speeds, memory, battery, disk, GPU usage/temperature and uptime."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(os.path.expanduser("~"))
    parts = [f"CPU {psutil.cpu_percent(interval=0.3):.0f}%",
             f"RAM {vm.used / 2**30:.1f}/{vm.total / 2**30:.1f} GB",
             f"home disk free {disk.free / 2**30:.0f} GB"]
    bat = psutil.sensors_battery()
    if bat:
        parts.append(f"battery {bat.percent:.0f}% ({'charging' if bat.power_plugged else 'on battery'})")
    temps = psutil.sensors_temperatures()
    cpu = next((t.current for t in temps.get("coretemp", []) if t.label.startswith("Package")), None)
    if cpu is None and temps.get("coretemp"):
        cpu = max(t.current for t in temps["coretemp"])
    if cpu is not None:
        hottest = max((t.current for t in temps.get("coretemp", [])), default=cpu)
        parts.append(f"CPU temperature {cpu:.0f}°C (hottest core {hottest:.0f}°C, sensor coretemp)")
    fans = psutil.sensors_fans().get("asus") or []
    if fans:   # ASUS: fan1 = CPU, fan2 = GPU
        names = ["CPU fan", "GPU fan"]
        parts.append(", ".join(f"{names[i] if i < 2 else f.label or f'fan{i + 1}'} {f.current} rpm"
                               for i, f in enumerate(fans)))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        u, used, total, temp = [x.strip() for x in gpu.split(",")]
        parts.append(f"GPU {u}% VRAM {used}/{total} MB {temp}°C")
    except Exception:  # noqa: BLE001
        pass
    up = time.time() - psutil.boot_time()
    parts.append(f"uptime {int(up // 3600)}h {int(up % 3600 // 60)}m")
    return ", ".join(parts)


@tool()
def list_processes(name: str = "", top: int = 10) -> str:
    """List running programs, sorted by memory use.

    Args:
        name: only show processes whose name contains this (optional)
        top: how many to show
    """
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info", "cpu_percent"]):
        if name and name.lower() not in (p.info["name"] or "").lower():
            continue
        mem = p.info["memory_info"].rss if p.info["memory_info"] else 0
        procs.append((mem, p.info["pid"], p.info["name"]))
    procs.sort(reverse=True)
    if not procs:
        return "no matching processes"
    return "\n".join(f"{pid} {n} {m / 2**20:.0f} MB" for m, pid, n in procs[:top])


def _targets(name_or_pid: str) -> list[psutil.Process]:
    if name_or_pid.isdigit():
        return [psutil.Process(int(name_or_pid))]
    return [p for p in psutil.process_iter(["name"]) if (p.info["name"] or "").lower() == name_or_pid.lower()]


@tool(risk="confirm", describe=lambda a: f"close the program {a.get('name_or_pid')}")
def kill_process(name_or_pid: str) -> str:
    """Stop a running program (sends SIGTERM).

    Args:
        name_or_pid: exact process name (e.g. "firefox") or a PID
    """
    procs = _targets(name_or_pid)
    if not procs:
        return f"no process named {name_or_pid}"
    for p in procs:
        p.send_signal(signal.SIGTERM)
    return f"stopped {len(procs)} process(es) named {name_or_pid}"
