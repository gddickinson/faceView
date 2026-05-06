"""Coarticulation: viseme_blend_at produces continuous AU trajectories."""

from __future__ import annotations

import math

from faceview.vision.speech import (
    SpeechEngine,
    TimedViseme,
    viseme_blend_at,
)


def _line(text: str = "Hello world.") -> list[TimedViseme]:
    eng = SpeechEngine()
    return eng.generate_au_sequence(text)


def test_blend_returns_empty_when_outside_timeline():
    line = _line()
    assert viseme_blend_at(line, -1.0) == {}
    last_end = line[-1].end_time
    assert viseme_blend_at(line, last_end + 1.0) == {}


def test_blend_inside_returns_active_aus():
    line = _line()
    mid = line[len(line) // 2]
    t = (mid.start_time + mid.end_time) / 2.0
    blend = viseme_blend_at(line, t)
    # Mid-slot should produce SOME active AU values (might be empty for REST).
    if mid.au_targets:
        assert blend, f"expected blend at viseme midpoint t={t}"


def test_blend_continuous_across_boundary():
    """Crossing a viseme boundary, no AU should jump by more than the
    sum of (max_au_delta * weight_step). Sample finely and confirm the
    L-infty step between consecutive samples stays small."""
    line = _line()
    if len(line) < 3:
        return  # nothing to cross
    boundary = line[1].end_time
    samples = []
    aus_seen: set[str] = set()
    for i in range(-30, 31):
        t = boundary + i * 0.001  # 1ms steps
        b = viseme_blend_at(line, t)
        samples.append(b)
        aus_seen.update(b.keys())
    for au in aus_seen:
        prev = samples[0].get(au, 0.0)
        for s in samples[1:]:
            cur = s.get(au, 0.0)
            assert abs(cur - prev) < 0.06, (
                f"AU {au} jumped by {abs(cur-prev):.3f} between 1ms samples"
            )
            prev = cur


def test_blend_weights_attack_and_release_have_zero_at_edges():
    # Construct a single viseme to keep the math obvious.
    tv = TimedViseme(
        viseme="AA", start_time=0.5, end_time=0.7,
        au_targets={"AU25": 0.8, "AU26": 0.4},
    )
    # Just before the attack window — no contribution.
    assert viseme_blend_at([tv], 0.5 - 0.040 - 0.001) == {}
    # Just after the release window — no contribution.
    assert viseme_blend_at([tv], 0.7 + 0.060 + 0.001) == {}
    # In the middle of the slot — full contribution.
    full = viseme_blend_at([tv], 0.6)
    assert math.isclose(full["AU25"], 0.8, rel_tol=1e-6)
    assert math.isclose(full["AU26"], 0.4, rel_tol=1e-6)
    # Halfway through the attack — half contribution.
    half_attack = viseme_blend_at([tv], 0.5 - 0.020)
    assert 0.30 < half_attack["AU25"] < 0.50


def test_avatar_speaking_uses_blend(tmp_path):
    """Smoke-check that the avatar's mouth AUs vary smoothly across an
    utterance — proxy for "blend is wired in", not a value check."""
    from faceview.vision.avatar import TalkingAvatar
    avatar = TalkingAvatar(emotion="neutral", seed=7)
    avatar.say("Hello there friend.")
    samples = []
    for i in range(80):
        t = i * 0.030  # 30ms steps
        fp = avatar.tick(t)
        samples.append(fp.jaw_open)
    # Mouth should open at least a little during the utterance.
    assert max(samples) > 0.05
    # And it should *vary* — not be a constant single value.
    assert max(samples) - min(samples) > 0.05
