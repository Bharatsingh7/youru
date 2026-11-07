"""run_python: a persistent Python interpreter in a *separate process*.

OmniPilot's run_python exec()s code inside its own process; here that would let a bad snippet freeze or
corrupt the assistant, and its SIGALRM timeout only works on the main thread. So the code runs in a child
interpreter (same venv, so numpy etc. are available) that keeps variables between calls and is killed and
restarted when it runs past the timeout.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from . import risk, tool

CHILD = r'''
import contextlib, io, json, sys, traceback
ns = {"__name__": "__youru__"}
for line in sys.stdin:
    code = json.loads(line)["code"]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            try:
                value = eval(compile(code, "<youru>", "eval"), ns)   # a bare expression: show its value
                if value is not None:
                    print(repr(value))
            except SyntaxError:
                exec(compile(code, "<youru>", "exec"), ns)
        except SystemExit:
            pass
        except BaseException:
            traceback.print_exc()
    sys.__stdout__.write(json.dumps({"out": out.getvalue(), "err": err.getvalue()}) + "\n")
    sys.__stdout__.flush()
'''


class PythonSession:
    def __init__(self, timeout_s: float = 30):
        self.timeout_s = timeout_s
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def _start(self) -> None:
        self.proc = subprocess.Popen([sys.executable, "-u", "-c", CHILD], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                     cwd=Path.home(), bufsize=1)

    def run(self, code: str) -> str:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            reply: list[str] = []
            reader = threading.Thread(target=lambda: reply.append(self.proc.stdout.readline()), daemon=True)
            reader.start()
            reader.join(self.timeout_s)
            if reader.is_alive() or not reply or not reply[0]:
                self.proc.kill()
                self.proc = None
                return (f"error: the code ran longer than {self.timeout_s:.0f} s and was stopped "
                        "(the Python session was restarted, so earlier variables are gone)")
            r = json.loads(reply[0])
        text = (r["out"] + (("\n" + r["err"]) if r["err"] else "")).strip()
        return text[:3500] or "(ran with no output)"


_session = PythonSession()


@tool(assess=risk.python, describe=lambda a: f"run this Python code: {str(a.get('code', ''))[:200]}")
def run_python(code: str) -> str:
    """Run Python code (variables are kept between calls). Use for calculations, unit conversions,
    dates, text or data processing, and small one-off tools. print() what you want to see.

    Args:
        code: Python source code
    """
    return _session.run(code)
