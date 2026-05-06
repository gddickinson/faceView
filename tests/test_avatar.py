"""TalkingAvatar + speech engine + FACS expression tests."""

from __future__ import annotations

import numpy as np

from faceview.vision.avatar import TalkingAvatar
from faceview.vision.expressions import apply_expression, expression_names, get_expression
from faceview.vision.face_state import AU_IDS, FaceState, face_state_to_params
from faceview.vision.sim_face import render_face
from faceview.vision.speech import SpeechEngine, viseme_at
from faceview.vision.visemes import VISEMES, viseme_for_phoneme


# ── FACS expressions ────────────────────────────────────────────────────


def test_bundled_expressions_load():
    names = expression_names()
    assert "neutral" in names and "happy" in names and "surprised" in names
    happy = get_expression("happy")
    assert happy["AU12"] > 0.5  # smile preset must engage corner-pull


def test_apply_expression_sets_aus():
    s = FaceState()
    apply_expression(s, "happy")
    # All AUs zeroed except those in the preset.
    assert s.AU12 > 0.5
    assert s.AU6 > 0.5
    # AUs not in the happy preset should be at zero.
    assert s.AU15 == 0


def test_face_state_to_params_smile_from_au12():
    s = FaceState(AU12=0.9)
    p = face_state_to_params(s)
    assert p.smile > 0.5


def test_face_state_to_params_jaw_from_au26():
    s = FaceState(AU26=0.7)
    p = face_state_to_params(s)
    assert p.jaw_open > 0.5


# ── speech engine ───────────────────────────────────────────────────────


def test_speech_engine_text_to_phonemes_uses_dict():
    eng = SpeechEngine()
    ph = eng.text_to_phonemes("hello world")
    # CMU dict gives a clean tokenization — no unknown letters.
    assert "HH" in ph and "OW1" in ph and "L" in ph
    assert "SIL" in ph  # word boundary marker


def test_speech_engine_falls_back_to_letter_rules_on_unknown_words():
    eng = SpeechEngine()
    ph = eng.text_to_phonemes("zorgblats")  # not in CMU dict
    assert ph  # rule-based fallback produced something


def test_phoneme_to_viseme_mapping_covers_arpabet():
    assert viseme_for_phoneme("M") == "PP"
    assert viseme_for_phoneme("F") == "FF"
    assert viseme_for_phoneme("AA1") == "AA"
    assert viseme_for_phoneme("OW1") == "OH"
    assert viseme_for_phoneme("???") == "REST"


def test_timeline_durations_are_monotonic():
    eng = SpeechEngine()
    seq = eng.generate_au_sequence("Hello world.")
    times = [tv.start_time for tv in seq]
    assert times == sorted(times)
    assert seq[-1].end_time > 0.5


def test_viseme_at_returns_active_segment():
    eng = SpeechEngine()
    seq = eng.generate_au_sequence("Hi.")
    # Mid-utterance must hit a real viseme.
    mid_t = seq[-1].end_time / 2
    tv = viseme_at(seq, mid_t)
    assert tv is not None
    assert tv.viseme in VISEMES


# ── TalkingAvatar end-to-end ────────────────────────────────────────────


def test_avatar_idle_blinks_within_six_seconds(qtbot):
    av = TalkingAvatar(emotion="neutral", seed=1)
    blink_min = 1.0
    for i in range(6 * 30):  # 6 s at 30 Hz
        params = av.tick(i / 30.0)
        blink_min = min(blink_min, params.eye_open)
    # Blink should drive eye_open below 0.5 at some point.
    assert blink_min < 0.5


def test_avatar_say_drives_jaw_motion(qtbot):
    av = TalkingAvatar(emotion="neutral", seed=2)
    av.say("Hello world.")
    seen = []
    for i in range(3 * 30):  # 3 s
        params = av.tick(i / 30.0)
        seen.append(params.jaw_open)
    # While speaking, jaw_open must vary (it goes up for AA and down for PP).
    assert max(seen) - min(seen) > 0.15
    assert max(seen) > 0.25


def test_avatar_renders_to_image_each_tick(qtbot):
    av = TalkingAvatar(emotion="happy", seed=3)
    av.say("Test.")
    frames = []
    for i in range(20):
        params = av.tick(i / 24.0)
        arr = render_face(params, (200, 160))
        frames.append(arr)

    # All frames must be valid uint8 RGB arrays.
    for f in frames:
        assert f.shape == (160, 200, 3)
        assert f.dtype == np.uint8

    # The animation must produce frame variation, not a freeze.
    diff = sum(int(np.abs(a.astype(int) - b.astype(int)).sum())
               for a, b in zip(frames, frames[1:]))
    assert diff > 1_000


def test_avatar_changes_emotion(qtbot):
    av = TalkingAvatar(emotion="neutral", seed=4)
    for _ in range(5):
        av.tick()
    av.set_emotion("sad")
    for _ in range(20):
        params = av.tick()
    assert params.smile < 0  # corner-drop preset must register as a frown


def test_au_ids_are_canonical():
    assert AU_IDS == [
        "AU1", "AU2", "AU4", "AU5", "AU6", "AU9",
        "AU12", "AU15", "AU20", "AU22", "AU25", "AU26",
    ]
