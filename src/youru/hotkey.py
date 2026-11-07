"""Unix socket so a Hyprland hotkey can tell a running `youru assist` to start listening."""

from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

SOCKET = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "youru.sock"


class TriggerServer:
    def __init__(self, path: Path = SOCKET):
        self.path = path
        self.event = threading.Event()
        self.path.unlink(missing_ok=True)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.sock.listen(4)
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                if conn.recv(16).strip() == b"listen":
                    self.event.set()
                    conn.sendall(b"ok\n")

    def take(self) -> bool:
        """True once per trigger."""
        if self.event.is_set():
            self.event.clear()
            return True
        return False

    def close(self) -> None:
        self.sock.close()
        self.path.unlink(missing_ok=True)


def send_trigger(path: Path = SOCKET) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(str(path))
            s.sendall(b"listen\n")
            return s.recv(8).startswith(b"ok")
    except OSError:
        return False
