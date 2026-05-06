"""Sanity checks for the procedural face renderer."""

from __future__ import annotations

import numpy as np

from faceview.vision.sim_face import FaceParams, render_face


def test_render_returns_bgr_array(qtbot):
    arr = render_face(FaceParams.neutral(), (320, 240))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (240, 320, 3)
    assert arr.dtype == np.uint8


def test_render_is_deterministic(qtbot):
    p = FaceParams(smile=0.5, jaw_open=0.2, brow_raise=0.1)
    a = render_face(p, (160, 160))
    b = render_face(p, (160, 160))
    assert np.array_equal(a, b)


def test_open_mouth_changes_pixels(qtbot):
    closed = render_face(FaceParams.neutral(), (160, 160))
    open_ = render_face(FaceParams(jaw_open=0.6), (160, 160))
    diff = int(np.abs(closed.astype(int) - open_.astype(int)).sum())
    assert diff > 5_000  # more than just antialiasing noise
