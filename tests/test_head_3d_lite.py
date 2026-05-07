"""Lite 3D animated head — smoke tests across emotions, sizes, and rotation."""

from __future__ import annotations

import numpy as np
import pytest

from faceview.vision.head_3d_lite import build_3d_template, render_face_3d_lite
from faceview.vision.sim_face import FaceParams, render_face


def test_template_has_face_plus_closure_verts():
    tpl = build_3d_template()
    # 86 face landmarks + ~19 closure verts.
    assert 100 <= len(tpl) <= 120
    names = {v.name for v in tpl}
    assert "chin" in names
    assert "vertex" in names  # back-of-head closure
    assert "neck_front" in names


def test_render_returns_valid_bgr_frame(qtbot):
    p = FaceParams.happy()
    frame = render_face_3d_lite(p, (240, 240))
    assert frame.dtype == np.uint8
    assert frame.shape == (240, 240, 3)
    assert frame.max() > 0


def test_render_face_dispatch_to_head_3d_lite(qtbot):
    p = FaceParams.surprised()
    p.render_mode = "head_3d_lite"
    frame = render_face(p, (240, 240))
    assert frame.shape == (240, 240, 3)


def test_yaw_changes_rendered_pixels(qtbot):
    a = FaceParams.neutral()
    b = FaceParams.neutral()
    b.yaw = 0.6
    fa = render_face_3d_lite(a, (200, 200))
    fb = render_face_3d_lite(b, (200, 200))
    diff = np.abs(fa.astype(int) - fb.astype(int)).mean()
    assert diff > 1.0


def test_emotion_changes_rendered_pixels(qtbot):
    a = render_face_3d_lite(FaceParams.happy(), (200, 200))
    b = render_face_3d_lite(FaceParams.sad(), (200, 200))
    diff = np.abs(a.astype(int) - b.astype(int)).mean()
    assert diff > 0.5


def test_persona_drives_lite_3d_dispatch(qtbot):
    from faceview.vision.personas import apply_persona, load_persona
    p = FaceParams.happy()
    apply_persona(p, load_persona("head_3d_lite"))
    assert p.render_mode == "head_3d_lite"
    frame = render_face(p, (200, 200))
    assert frame.shape == (200, 200, 3)
