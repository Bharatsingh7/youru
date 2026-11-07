import numpy as np

from youru.audio import RingBuffer, SAMPLE_RATE
from youru.cli import wer
from youru.config import load_config, update_config
from youru.evaluate import _count_triggers, eer


def test_update_config_keeps_comments(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("speaker:\n  accept: 0.55   # recalibrate\n  model: a\nwakeword:\n  threshold: 0.5\n")
    update_config({"speaker": {"accept": 0.612, "model": "b"}, "wakeword": {"threshold": 0.7}}, p)
    text = p.read_text()
    assert "# recalibrate" in text
    cfg = load_config(p)
    assert cfg.speaker.accept == 0.612 and cfg.speaker.model == "b" and cfg.wakeword.threshold == 0.7


def test_ring_buffer_keeps_latest():
    r = RingBuffer(1.0)
    r.append(np.zeros(SAMPLE_RATE, np.float32))
    r.append(np.ones(SAMPLE_RATE // 2, np.float32))
    last = r.last(1.0)
    assert len(last) == SAMPLE_RATE and last[-1] == 1 and last[0] == 0


def test_wer():
    assert wer("music chalao please", "music chalao please") == 0
    assert wer("music chalao please", "music chala do please") == 2 / 3


def test_eer_separable():
    e, t = eer(np.array([0.8, 0.9, 0.7]), np.array([0.1, 0.2, 0.3]))
    assert e == 0 and 0.3 < t <= 0.7


def test_trigger_refractory():
    scores = np.array([0, 0.9, 0.9, 0.9, 0, 0, 0, 0, 0.9])
    assert _count_triggers(scores, 0.5, refractory_frames=3) == 2
