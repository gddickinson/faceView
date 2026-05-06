"""Persona presets — load, apply, default fallback, avatar integration."""

from __future__ import annotations

import pytest

from faceview.vision.personas import (
    Persona,
    apply_persona,
    list_personas,
    load_persona,
)
from faceview.vision.sim_face import FaceParams


def test_list_personas_includes_default_and_claude():
    names = list_personas()
    assert "default" in names
    assert "claude" in names
    assert len(names) >= 4


def test_load_persona_reads_known_preset():
    p = load_persona("claude")
    assert p.name == "claude"
    assert p.hair_color.startswith("#")
    assert 0 <= p.skin_hue <= 360
    assert p.background.startswith("#")


def test_load_persona_unknown_falls_back_to_default():
    p = load_persona("there_is_no_such_persona")
    assert p.name == "there_is_no_such_persona"  # name preserved
    default = load_persona("default")
    # Falls back to default values for the appearance fields.
    assert p.skin_hue == default.skin_hue
    assert p.hair_color == default.hair_color


def test_apply_persona_overwrites_face_params_appearance():
    fp = FaceParams.neutral()
    p = Persona(name="custom", skin_hue=12.0, hair_color="#ff0000",
                lip_color="#00ff00", background="#0000ff")
    apply_persona(fp, p)
    assert fp.skin_hue == 12.0
    assert fp.hair_color == "#ff0000"
    assert fp.lip_color == "#00ff00"
    assert fp.background == "#0000ff"


def test_avatar_carries_persona_through_tick():
    from faceview.vision.avatar import TalkingAvatar
    avatar = TalkingAvatar(emotion="neutral", persona="claude", seed=1)
    fp = avatar.tick(0.0)
    claude = load_persona("claude")
    assert fp.hair_color == claude.hair_color
    assert fp.lip_color == claude.lip_color
    assert fp.skin_hue == claude.skin_hue


def test_avatar_set_persona_takes_effect_on_next_tick():
    from faceview.vision.avatar import TalkingAvatar
    avatar = TalkingAvatar(emotion="neutral", persona="default", seed=1)
    fp_before = avatar.tick(0.0)
    avatar.set_persona("auburn")
    fp_after = avatar.tick(0.05)
    auburn = load_persona("auburn")
    assert fp_after.hair_color == auburn.hair_color
    assert fp_after.hair_color != fp_before.hair_color
