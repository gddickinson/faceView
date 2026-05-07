"""Image-warp realistic face renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _texture_present() -> bool:
    from faceview.vision.face_warp import _texture_path
    return _texture_path().exists()


@pytest.mark.skipif(not _texture_present(),
                     reason="neutral face texture not generated yet")
def test_warp_produces_valid_frame(qtbot):
    from faceview.vision.face_warp import render_face_warp
    from faceview.vision.sim_face import FaceParams

    frame = render_face_warp(FaceParams.happy(), (240, 240))
    assert frame.dtype == np.uint8
    assert frame.shape == (240, 240, 3)
    assert frame.mean() > 5.0


@pytest.mark.skipif(not _texture_present(),
                     reason="neutral face texture not generated yet")
def test_warp_dispatch_via_render_mode(qtbot):
    from faceview.vision.sim_face import FaceParams, render_face

    p = FaceParams.surprised()
    p.render_mode = "face_warp_2d"
    frame = render_face(p, (240, 240))
    assert frame.shape == (240, 240, 3)


@pytest.mark.skipif(not _texture_present(),
                     reason="neutral face texture not generated yet")
def test_warp_emotion_changes_pixels(qtbot):
    from faceview.vision.face_warp import render_face_warp
    from faceview.vision.sim_face import FaceParams

    a = render_face_warp(FaceParams.happy(), (240, 240))
    b = render_face_warp(FaceParams.sad(), (240, 240))
    diff = np.abs(a.astype(int) - b.astype(int)).mean()
    assert diff > 0.5


def test_missing_texture_raises_helpful_error(monkeypatch, tmp_path):
    """When the texture doesn't exist the loader should point at the tool."""
    from faceview.core.errors import MissingDependency
    from faceview.vision import face_warp

    monkeypatch.setattr(face_warp, "_texture_path",
                         lambda: tmp_path / "nope.png")
    face_warp._load_texture.cache_clear()
    try:
        with pytest.raises(MissingDependency):
            face_warp._load_texture()
    finally:
        face_warp._load_texture.cache_clear()
