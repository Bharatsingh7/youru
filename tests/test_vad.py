import numpy as np

from youru.audio import SAMPLE_RATE
from youru.vad import SileroVAD, SpeechEnd, SpeechStart


def vad(model):
    return SileroVAD(model, threshold=0.5, min_speech_s=0.25, min_silence_s=0.8)


def test_silence_has_no_speech(vad_model, silence):
    assert vad(vad_model).segments(silence) == []


def test_fan_noise_has_no_speech(vad_model, fan_noise):
    assert vad(vad_model).segments(fan_noise) == []


def test_speech_boundaries(vad_model, clips):
    speech = clips["c1"]
    pad = np.zeros(SAMPLE_RATE, np.float32)
    x = np.concatenate([pad, speech, pad, pad])
    v = vad(vad_model)
    segs = v.segments(x)
    assert segs, "speech not detected"
    # The first segment starts close to where the speech was inserted.
    first_voiced = SAMPLE_RATE + int(np.argmax(np.abs(speech) > 0.02))
    assert abs(segs[0][0] - first_voiced) < 0.15 * SAMPLE_RATE
    # The last segment ends before the trailing silence.
    assert segs[-1][1] <= SAMPLE_RATE + len(speech) + 0.2 * SAMPLE_RATE


def test_speech_in_fan_noise(vad_model, clips, fan_noise):
    speech = clips["a2"]
    x = fan_noise.copy()
    x[SAMPLE_RATE // 2:SAMPLE_RATE // 2 + min(len(speech), len(x) - SAMPLE_RATE // 2)] += \
        speech[: len(x) - SAMPLE_RATE // 2]
    assert vad(vad_model).segments(x)


def test_streaming_events_alternate(vad_model, clips):
    v = vad(vad_model)
    x = np.concatenate([np.zeros(SAMPLE_RATE, np.float32), clips["a1"], np.zeros(2 * SAMPLE_RATE, np.float32)])
    events = []
    for i in range(0, len(x), 1280):
        events += v.process(x[i:i + 1280])
    kinds = [type(e) for e in events]
    assert kinds[0] is SpeechStart and kinds[-1] is SpeechEnd
    assert all(a is not b for a, b in zip(kinds, kinds[1:]))
    assert not v.speaking
