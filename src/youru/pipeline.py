"""The phase 1 pipeline: VAD + wake word -> capture command -> owner voice check.

    IDLE --wake--> LISTENING --speech ended / timeout--> VERIFY --> accept | uncertain | reject | too_short

Phase 2 plugs in via `Pipeline.on_owner_utterance(callback)`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np

from . import denoise
from .audio import SAMPLE_RATE, RingBuffer, save_wav
from .config import DATA
from .speaker import Embedder, OwnerProfile
from .wakeword import WakeEvent

PROFILE_PATH = DATA / "profile" / "owner.npz"


class State(Enum):
    IDLE = "idle"
    LISTENING = "listening"


@dataclass
class Decision:
    outcome: str                 # accept | uncertain | reject | too_short | no_profile
    t: float
    wake_score: float
    speaker_score: float | None
    speech_s: float
    audio: np.ndarray = field(repr=False)
    latency_ms: float = 0.0      # end of capture -> decision
    transcript: str | None = None
    language: str | None = None
    meaning: str | None = None   # Hindi/Hinglish rewritten to English (agent.normalize)
    tools: list | None = None    # [{"name", "args", "result"}] the agent called
    reply: str | None = None

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        return d


class SpeakerGate:
    """Owner voice check on a captured utterance."""

    def __init__(self, embedder: Embedder, profile: OwnerProfile | None, vad, cfg,
                 profile_path: Path = PROFILE_PATH):
        self.embedder = embedder
        self.profile = profile
        self.vad = vad
        self.cfg = cfg
        self.profile_path = profile_path
        self.noise = denoise.load_noise_profile() if getattr(cfg, "denoise", False) else None

    def check(self, audio: np.ndarray) -> tuple[str, float | None, float]:
        speech = self.vad.speech_only(audio)
        speech_s = len(speech) / SAMPLE_RATE
        if not getattr(self.cfg, "enabled", True):  # voice lock bypassed
            return ("accept" if speech_s >= 0.3 else "too_short"), None, speech_s
        if self.profile is None:
            return "no_profile", None, speech_s
        if speech_s < self.cfg.min_speech_s:
            return "too_short", None, speech_s
        if self.cfg.denoise:
            speech = denoise.reduce(speech, self.noise)
        emb = self.embedder.embed(speech)
        score = self.profile.score(emb)
        if score >= self.cfg.accept:
            outcome = "accept"
            if self.cfg.adapt and score >= self.cfg.accept + self.cfg.adapt_margin:
                self.profile.adapt(emb, self.cfg.adapt_max)
                self.profile.save(self.profile_path)
        elif score >= self.cfg.reject:
            outcome = "uncertain"
        else:
            outcome = "reject"
        return outcome, score, speech_s


class SessionLogger:
    """Saves every triggered utterance; this becomes your personal accent dataset."""

    def __init__(self, root: Path = DATA / "sessions"):
        self.root = root

    def log(self, d: Decision) -> Path:
        now = datetime.now()
        day = self.root / now.strftime("%Y-%m-%d")
        wav = day / f"{now.strftime('%H%M%S_%f')[:-3]}_{d.outcome}.wav"
        save_wav(wav, d.audio)
        with open(day / "sessions.jsonl", "a") as f:
            f.write(json.dumps({"file": wav.name, "time": now.isoformat(), **d.meta()}) + "\n")
        return wav


class Pipeline:
    def __init__(self, vad, wake, gate, listen_cfg, preroll_s: float = 2.5,
                 logger: SessionLogger | None = None):
        self.vad = vad
        self.wake = wake
        self.gate = gate
        self.cfg = listen_cfg
        self.preroll_s = preroll_s
        self.logger = logger
        self.ring = RingBuffer(preroll_s + 0.5)
        self.state = State.IDLE
        self.t = 0.0
        self._owner_cbs: list[Callable[[Decision], None]] = []
        self._event_cbs: list[Callable[[object], None]] = []
        self._reset_capture()

    def on_owner_utterance(self, cb: Callable[[Decision], None]) -> None:
        self._owner_cbs.append(cb)

    def on_event(self, cb: Callable[[object], None]) -> None:
        self._event_cbs.append(cb)

    def _emit(self, ev) -> None:
        for cb in self._event_cbs:
            cb(ev)

    def _reset_capture(self) -> None:
        self.capture: list[np.ndarray] = []
        self.wake_event: WakeEvent | None = None
        self.heard_command = False
        self.gap_seen = False     # a pause after the wake word itself
        self.speech_run = 0.0     # seconds of continuous speech probability
        self.quiet_run = 0.0

    def trigger(self) -> None:
        """Start listening for a command without the wake word (hotkey)."""
        if self.state is State.IDLE:
            self.wake_event = WakeEvent(self.t, 1.0)
            self.capture = []
            self.state = State.LISTENING
            self._emit(self.wake_event)

    def resync(self) -> None:
        """Forget detector state, e.g. after the assistant has been speaking."""
        self.vad.reset()
        self.wake.reset()
        if self.state is State.IDLE:
            self._reset_capture()

    def feed(self, frame: np.ndarray) -> Decision | None:
        self.t += len(frame) / SAMPLE_RATE
        self.ring.append(frame)
        vad_events = self.vad.process(frame)
        for ev in vad_events:
            self._emit(ev)

        if self.state is State.IDLE:
            wake = self.wake.process(frame, self.t, self.vad.speaking)
            if wake is not None:
                self._emit(wake)
                self.state = State.LISTENING
                self.wake_event = wake
                self.capture = [self.ring.last(self.preroll_s)]
            return None

        # LISTENING. Use the raw speech probability, not the VAD's hangover-smoothed
        # `speaking` flag, to tell the wake word's own tail from a real command.
        self.capture.append(frame)
        dt = len(frame) / SAMPLE_RATE
        since_wake = self.t - self.wake_event.t
        if self.vad.prob >= self.vad.threshold:
            self.speech_run, self.quiet_run = self.speech_run + dt, 0.0
        elif self.vad.prob < self.vad.neg_threshold:
            self.speech_run, self.quiet_run = 0.0, self.quiet_run + dt
        if self.quiet_run >= 0.2:
            self.gap_seen = True
        # "Youru ... music chalao" (pause, then speech) or "Youru music chalao" (no pause).
        if (self.gap_seen and self.speech_run >= 0.15) or self.speech_run >= 1.0:
            self.heard_command = True
        ended = self.heard_command and self.quiet_run >= self.vad.min_silence / SAMPLE_RATE
        if ended or (not self.heard_command and since_wake >= self.cfg.command_timeout_s) \
                or since_wake >= self.cfg.max_command_s:
            return self._finish()
        return None

    def _finish(self) -> Decision:
        audio = np.concatenate(self.capture)
        t0 = time.perf_counter()
        outcome, score, speech_s = self.gate.check(audio)
        d = Decision(outcome, self.t, self.wake_event.score, score, speech_s, audio,
                     latency_ms=(time.perf_counter() - t0) * 1000)
        self.state = State.IDLE
        self._reset_capture()
        self.wake.reset()
        if outcome == "accept":
            for cb in self._owner_cbs:
                cb(d)
        if self.logger is not None:
            self.logger.log(d)
        self._emit(d)
        return d

    def run(self, source, stop: Callable[[], bool] = lambda: False) -> None:
        for frame in source.frames():
            self.feed(frame)
            if stop():
                break


def build(cfg, with_logger: bool = True) -> Pipeline:
    """Construct the real pipeline from config.yaml."""
    from . import models
    from .config import resolve
    from .vad import SileroVAD
    from .wakeword import WakeWord

    def make_vad():
        return SileroVAD(resolve(cfg.vad.model), cfg.vad.threshold, cfg.vad.neg_threshold,
                         cfg.vad.min_speech_s, cfg.vad.min_silence_s)

    w = cfg.wakeword
    wake = WakeWord(resolve(w.model), w.threshold, resolve(w.verifier) if w.verifier else None,
                    w.verifier_threshold, w.require_speech, w.refractory_s)
    lock_on = cfg.speaker.get("enabled", True)
    embedder = Embedder(models.ensure(cfg.speaker.model)) if lock_on else None
    profile = OwnerProfile.load(PROFILE_PATH) if lock_on and PROFILE_PATH.exists() else None
    if profile is not None and profile.model_name != embedder.model_name:
        raise SystemExit(f"Profile was enrolled with {profile.model_name} but config uses "
                         f"{embedder.model_name}. Run `youru enroll --rebuild`.")
    gate = SpeakerGate(embedder, profile, make_vad(), cfg.speaker)
    logger = SessionLogger() if with_logger and cfg.logging.save_sessions else None
    return Pipeline(make_vad(), wake, gate, cfg.listen, cfg.audio.preroll_s, logger)
