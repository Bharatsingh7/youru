from types import SimpleNamespace

import numpy as np
import pytest

from youru import models
from youru.audio import FRAME, SAMPLE_RATE, FileSource
from youru.pipeline import Pipeline, SpeakerGate, State
from youru.speaker import Embedder, OwnerProfile
from youru.vad import SileroVAD, SpeechStart
from youru.wakeword import WakeEvent

LISTEN = SimpleNamespace(command_timeout_s=4.0, max_command_s=15.0)
FRAME_S = FRAME / SAMPLE_RATE


class ScriptedVAD:
    """speaking is True inside the given (start, end) second intervals."""

    def __init__(self, spans):
        self.spans = spans
        self.t = 0.0
        self.speaking = False
        self.prob = 0.0
        self.threshold, self.neg_threshold = 0.5, 0.35
        self.min_silence = int(0.8 * SAMPLE_RATE)

    def process(self, frame):
        self.t += len(frame) / SAMPLE_RATE
        now = any(a <= self.t < b for a, b in self.spans)
        events = [SpeechStart(self.t)] if now and not self.speaking else []
        self.speaking = now
        self.prob = 1.0 if now else 0.0
        return events


class ScriptedWake:
    def __init__(self, at):
        self.at = at
        self.score = 0.0
        self.resets = 0
        self.name = "fake"

    def process(self, frame, t, speech):
        if self.at is not None and abs(t - self.at) < FRAME_S / 2:
            return WakeEvent(t, 0.9)
        return None

    def reset(self):
        self.resets += 1


class FixedGate:
    def __init__(self, outcome="accept", score=0.8):
        self.outcome, self.score, self.calls = outcome, score, []

    def check(self, audio):
        self.calls.append(len(audio) / SAMPLE_RATE)
        return self.outcome, self.score, 2.0


def run(pipe, seconds):
    decisions = []
    for _ in range(int(seconds / FRAME_S)):
        d = pipe.feed(np.zeros(FRAME, np.float32))
        if d:
            decisions.append(d)
    return decisions


def test_wake_then_command_then_accept():
    # "Youru" 0.5-1.2 s (wake fires at 1.2), command 2.0-4.0 s
    gate = FixedGate()
    wake = ScriptedWake(1.2)
    pipe = Pipeline(ScriptedVAD([(0.5, 1.2), (2.0, 4.0)]), wake, gate, LISTEN, preroll_s=2.0)
    got = []
    pipe.on_owner_utterance(got.append)
    decisions = run(pipe, 6.0)
    assert [d.outcome for d in decisions] == ["accept"]
    assert got and got[0] is decisions[0]
    # command ends at 4.0 s, plus 0.8 s of silence to be sure it's over
    assert decisions[0].t == pytest.approx(4.8, abs=0.1)
    # pre-roll (up to 1.2 s here) + listening time
    assert gate.calls[0] == pytest.approx(1.2 + 3.6, abs=0.2)
    assert wake.resets == 1 and pipe.state is State.IDLE


def test_command_spoken_straight_after_wake():
    pipe = Pipeline(ScriptedVAD([(0.5, 3.5)]), ScriptedWake(1.2), FixedGate(), LISTEN)
    decisions = run(pipe, 5.0)
    assert len(decisions) == 1 and decisions[0].t == pytest.approx(3.5 + 0.8, abs=0.1)


def test_wake_without_command_times_out():
    pipe = Pipeline(ScriptedVAD([(0.5, 1.2)]), ScriptedWake(1.2), FixedGate(), LISTEN)
    decisions = run(pipe, 7.0)
    assert len(decisions) == 1
    assert decisions[0].t == pytest.approx(1.2 + LISTEN.command_timeout_s, abs=0.1)


def test_endless_speech_is_capped():
    pipe = Pipeline(ScriptedVAD([(0.5, 60)]), ScriptedWake(1.2), FixedGate(), LISTEN)
    decisions = run(pipe, 20.0)
    assert decisions[0].t == pytest.approx(1.2 + LISTEN.max_command_s, abs=0.1)


def test_reject_does_not_reach_owner_callback():
    pipe = Pipeline(ScriptedVAD([(0.5, 1.2), (2.0, 3.0)]), ScriptedWake(1.2), FixedGate("reject", 0.1), LISTEN)
    got = []
    pipe.on_owner_utterance(got.append)
    assert [d.outcome for d in run(pipe, 5.0)] == ["reject"]
    assert got == []


def test_no_wake_no_decision():
    pipe = Pipeline(ScriptedVAD([(0.5, 3.0)]), ScriptedWake(None), FixedGate(), LISTEN)
    assert run(pipe, 5.0) == [] and pipe.state is State.IDLE


# --- real VAD + real speaker model, wake word faked at the start of speech ---

@pytest.fixture(scope="module")
def real_parts(vad_model, clips):
    emb = Embedder(models.ensure("titanet_large"))
    profile = OwnerProfile(emb.model_name, [emb.embed(clips[k]) for k in ("a1", "a2", "a3")])
    cfg = SimpleNamespace(denoise=False, min_speech_s=1.5, accept=0.45, reject=0.3, adapt=False,
                          adapt_margin=0.1, adapt_max=10)
    return emb, profile, cfg


@pytest.mark.parametrize("key,expected", [("a_test", "accept"), ("b1", "reject"), ("c1", "reject")])
def test_end_to_end_from_file(vad_model, clips, real_parts, tmp_path, key, expected):
    emb, profile, cfg = real_parts
    make = lambda: SileroVAD(vad_model, 0.5, 0.35, 0.25, 0.8)
    gate = SpeakerGate(emb, profile, make(), cfg, profile_path=tmp_path / "p.npz")
    pad = np.zeros(SAMPLE_RATE, np.float32)
    audio = np.concatenate([pad, clips[key], pad, pad])
    pipe = Pipeline(make(), ScriptedWake(1.3), gate, LISTEN, preroll_s=2.0)
    decisions = []
    for frame in FileSource(audio).frames():
        if d := pipe.feed(frame):
            decisions.append(d)
    assert [d.outcome for d in decisions] == [expected], [(d.outcome, d.speaker_score) for d in decisions]
