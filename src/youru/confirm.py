"""Yes/no confirmation for risky actions, by voice (English + Hindi) or keyboard."""

from __future__ import annotations

import re

from rich.console import Console

YES = {"yes", "yeah", "yep", "yup", "ok", "okay", "sure", "confirm", "do", "go", "haan", "han", "ha", "haa",
       "ji", "theek", "thik", "kardo", "karo", "हाँ", "हां", "हा", "जी", "ठीक", "करो", "कर", "ओके"}
NO = {"no", "nope", "nah", "cancel", "stop", "dont", "don't", "nahi", "nahin", "nai", "mat", "ruko",
      "नहीं", "नही", "मत", "रुको", "कैंसल"}


def parse_yes_no(text: str) -> bool | None:
    """True for yes, False for no, None if unclear. 'No' wins if both appear."""
    words = set(re.findall(r"[\w'ऀ-ॿ]+", text.lower()))
    if words & NO:
        return False
    if words & YES:
        return True
    return None


class TerminalConfirmer:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def __call__(self, question: str) -> bool:
        ans = self.console.input(f"[bold yellow]Youru wants to {question}. Allow? [y/N][/] ")
        return ans.strip().lower() in ("y", "yes", "haan", "ha")


class VoiceConfirmer:
    """Asks out loud, then listens for one reply without the wake word."""

    def __init__(self, speaker, listen, transcribe, notify=None, tries: int = 2):
        self.speaker = speaker          # Speaker or None
        self.listen = listen            # () -> audio | None
        self.transcribe = transcribe    # audio -> text
        self.notify = notify            # (title, msg) -> None
        self.tries = tries
        self.log: list[str] = []

    def __call__(self, question: str) -> bool:
        prompt = f"Should I {question}? Say yes or no."
        if self.notify:
            self.notify("Youru needs confirmation", prompt)
        for attempt in range(self.tries):
            if self.speaker:
                self.speaker.say(prompt if attempt == 0 else "Please say yes or no.")
            audio = self.listen()
            if audio is None:
                continue
            heard = self.transcribe(audio)
            self.log.append(heard)
            answer = parse_yes_no(heard)
            if answer is not None:
                return answer
        return False
