"""Anatomical renderer — smoke tests across modes, sizes, and emotions."""

from __future__ import annotations

import numpy as np

from faceview.vision.personas import load_persona
from faceview.vision.sim_face import FaceParams, render_face


def _frame_ok(frame: np.ndarray, size: tuple[int, int]) -> None:
    w, h = size
    assert frame.dtype == np.uint8
    assert frame.shape == (h, w, 3)
    assert frame.max() > 0  # something was drawn


def test_render_face_dispatch_to_anatomical(qtbot):
    p = FaceParams.happy()
    p.render_mode = "anatomical"
    frame = render_face(p, (240, 240))
    _frame_ok(frame, (240, 240))


def test_render_face_dispatch_to_anatomy_overlay(qtbot):
    p = FaceParams.surprised()
    p.render_mode = "anatomy_overlay"
    frame = render_face(p, (320, 320))
    _frame_ok(frame, (320, 320))
    # Overlay in anatomy_overlay should brighten muscles when AUs active —
    # the surprised preset has high AU1/AU2, so the frame should not be
    # pure dark.
    assert frame.mean() > 30


def test_render_face_dispatch_to_wireframe(qtbot):
    p = FaceParams.neutral()
    p.render_mode = "wireframe"
    frame = render_face(p, (200, 200))
    _frame_ok(frame, (200, 200))


def test_render_face_default_stylised_unchanged(qtbot):
    """Default render_mode should still hit the stylised path (no kwargs)."""
    p = FaceParams.happy()
    assert p.render_mode == "stylised"
    frame = render_face(p, (200, 200))
    _frame_ok(frame, (200, 200))


def test_anatomical_persona_drives_dispatch(qtbot):
    """An anatomical persona applied to params should switch modes."""
    from faceview.vision.personas import apply_persona
    p = FaceParams.happy()
    apply_persona(p, load_persona("anatomical"))
    assert p.render_mode == "anatomical"
    frame = render_face(p, (240, 240))
    _frame_ok(frame, (240, 240))


def test_anatomical_render_differs_per_emotion(qtbot):
    """Different FaceParams should produce visibly different anatomical frames."""
    a = FaceParams.happy()
    a.render_mode = "anatomical"
    b = FaceParams.sad()
    b.render_mode = "anatomical"
    fa = render_face(a, (240, 240))
    fb = render_face(b, (240, 240))
    # Pixel-difference sanity check.
    diff = np.abs(fa.astype(int) - fb.astype(int)).mean()
    assert diff > 1.0


def test_anatomical_render_handles_full_au_range(qtbot):
    """Cranking every AU to 1 should not crash."""
    p = FaceParams(
        smile=1.0, jaw_open=1.0, brow_raise=1.0, eye_open=0.5,
        mouth_pucker=1.0, mouth_stretch=1.0, cheek_raise=1.0,
        nose_wrinkle=1.0, upper_lid_raise=1.0,
        inner_brow_raise=1.0, outer_brow_raise=1.0,
        brow_lower=1.0, lip_corner_drop=1.0,
        render_mode="anatomy_overlay",
    )
    frame = render_face(p, (200, 200))
    _frame_ok(frame, (200, 200))
