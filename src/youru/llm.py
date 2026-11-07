"""Runs llama.cpp's llama-server (OpenAI-compatible API) as a background process."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import httpx

from .config import resolve

RUNTIME = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
PID_FILE = RUNTIME / "youru-llm.pid"
LOG_FILE = Path.home() / ".cache" / "youru" / "llama-server.log"


class LlamaServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.url = f"http://127.0.0.1:{cfg.port}"

    @property
    def model_path(self) -> Path:
        return resolve(self.cfg.model)

    def args(self) -> list[str]:
        c = self.cfg
        binary = shutil.which(c.binary) or c.binary
        a = [binary, "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(c.port),
             "-c", str(c.ctx), "-t", str(c.threads), "-fa", "on", "-ub", str(c.get("ubatch", 512)),
             "--jinja", "--reasoning", "off", "--no-ui", "--parallel", str(c.get("parallel", 1))]
        # Thinking is off unless a request asks for it; then it's capped so answers stay quick.
        a += ["--reasoning-budget", str(c.get("think_budget", 128)),
              "--reasoning-budget-message", "Enough thinking, answer briefly now."]
        # Pin to one GPU: by default llama.cpp also spreads layers onto the slow Intel iGPU.
        if c.get("device"):
            a += ["-dev", str(c.device)]
        if str(c.gpu_layers) != "auto":
            a += ["-ngl", str(c.gpu_layers)]
        return a + [str(x) for x in (c.extra_args or [])]

    def warm_up(self, messages: list[dict], tools: list[dict]) -> None:
        """Process the long system prompt + tool list once so later requests reuse it from cache."""
        httpx.post(f"{self.url}/v1/chat/completions", timeout=120, json={
            "messages": messages, "tools": tools, "max_tokens": 1,
            "chat_template_kwargs": {"enable_thinking": False}})

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=1).status_code == 200
        except httpx.HTTPError:
            return False

    def running_model(self) -> str | None:
        try:
            data = httpx.get(f"{self.url}/v1/models", timeout=2).json()["data"]
            return Path(data[0]["id"]).name
        except Exception:  # noqa: BLE001
            return None

    def start(self, timeout: float = 120) -> None:
        if self.healthy():
            return
        if shutil.which(self.cfg.binary) is None:
            raise SystemExit("llama-server not found. Install it: sudo pacman -S llama-cpp ggml-vulkan")
        if not self.model_path.exists():
            raise SystemExit(f"model not found: {self.model_path}")
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_FILE, "w")
        proc = subprocess.Popen(self.args(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        PID_FILE.write_text(str(proc.pid))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise SystemExit(f"llama-server exited (code {proc.returncode}); see {LOG_FILE}")
            if self.healthy():
                return
            time.sleep(0.25)
        raise SystemExit(f"llama-server did not become ready in {timeout:.0f}s; see {LOG_FILE}")

    def stop(self) -> bool:
        if not PID_FILE.exists():
            return False
        try:
            os.killpg(int(PID_FILE.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
        PID_FILE.unlink(missing_ok=True)
        return True

    def complete(self, messages: list[dict], max_tokens: int = 200) -> dict:
        """One non-streaming chat call; the response includes llama.cpp's timings."""
        r = httpx.post(f"{self.url}/v1/chat/completions", timeout=120, json={
            "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        return r.json()

    def bench(self) -> dict:
        """Time to first token and generation speed on a short spoken-style answer."""
        msgs = [{"role": "system", "content": "You are a helpful voice assistant. Answer in 2-3 sentences."},
                {"role": "user", "content": "Explain why the sky is blue."}]
        self.complete(msgs, 8)  # warm-up
        t0 = time.perf_counter()
        first = None
        with httpx.stream("POST", f"{self.url}/v1/chat/completions", timeout=120, json={
                "messages": msgs, "max_tokens": 120, "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and '"content"' in line and first is None:
                    first = time.perf_counter() - t0
        timings = self.complete(msgs, 120).get("timings", {})
        return {"ttft_s": first or float("nan"), "prompt_tok_s": timings.get("prompt_per_second", 0.0),
                "gen_tok_s": timings.get("predicted_per_second", 0.0)}
