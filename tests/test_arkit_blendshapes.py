"""ARKit 52-blendshape compatibility layer."""

from __future__ import annotations

from faceview.vision.arkit_blendshapes import (
    ARKIT_BLENDSHAPES, ARKitFrame,
    arkit_to_au_values, au_to_arkit_values,
)


def test_canonical_set_is_52():
    assert len(ARKIT_BLENDSHAPES) == 52
    # All names are camelCase strings.
    for n in ARKIT_BLENDSHAPES:
        assert isinstance(n, str)


def test_arkit_to_au_smile_maps_to_AU12():
    au = arkit_to_au_values({"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0})
    assert au["AU12"] == 1.0


def test_au_to_arkit_jawopen_maps_to_AU26():
    arkit = au_to_arkit_values({"AU26": 1.0})
    assert arkit["jawOpen"] == 1.0
    assert arkit["browInnerUp"] == 0.0


def test_round_trip_AU_through_arkit_preserves_signal():
    src_au = {"AU12": 0.8, "AU26": 0.5, "AU4": 0.3}
    arkit = au_to_arkit_values(src_au)
    back = arkit_to_au_values(arkit)
    # AU values get clamped + sometimes split across L/R, so we just
    # check the dominant ones survive non-zero.
    assert back.get("AU12", 0.0) > 0.5
    assert back.get("AU26", 0.0) > 0.4
    assert back.get("AU4", 0.0) > 0.2


def test_frame_dataclass_round_trip():
    frame = ARKitFrame.from_au_values({"AU12": 1.0})
    au = frame.to_au_values()
    assert au["AU12"] == 1.0


def test_unknown_arkit_shape_ignored():
    au = arkit_to_au_values({"someUnknownShape": 1.0})
    assert au == {}
