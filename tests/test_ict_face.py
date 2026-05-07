"""ICT-FaceKit blendshape renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _data_present() -> bool:
    from faceview.vision.ict_face import _data_path
    return _data_path().exists()


@pytest.mark.skipif(not _data_present(),
                     reason="ICT face_kit.npz not built")
def test_model_loads_with_arkit_blendshapes():
    from faceview.vision.ict_face import load_ict_model
    m = load_ict_model()
    assert m.vertices.shape[1] == 3
    assert m.triangles.shape[1] == 3
    assert m.deltas.shape[0] == len(m.names)
    # ARKit-style names should be present (some get _L/_R suffix).
    name_set = {n.lower() for n in m.names}
    assert any("jawopen" in n for n in name_set)
    assert any("mouthsmile" in n for n in name_set)


@pytest.mark.skipif(not _data_present(),
                     reason="ICT face_kit.npz not built")
def test_apply_blendshape_changes_geometry():
    from faceview.vision.ict_face import apply_blendshapes, load_ict_model
    m = load_ict_model()
    neutral = m.vertices
    open_jaw = apply_blendshapes(m, {"jawOpen": 1.0})
    diff = np.abs(open_jaw - neutral).sum()
    assert diff > 0


@pytest.mark.skipif(not _data_present(),
                     reason="ICT face_kit.npz not built")
def test_render_returns_valid_frame():
    pytest.importorskip("moderngl")
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.sim_face import FaceParams
    frame = render_face_ict(FaceParams.neutral(), (240, 240))
    assert frame.shape == (240, 240, 3)
    assert frame.dtype == np.uint8
    assert frame.mean() > 5.0


@pytest.mark.skipif(not _data_present(),
                     reason="ICT face_kit.npz not built")
def test_dispatcher_routes_to_ict():
    pytest.importorskip("moderngl")
    from faceview.vision.sim_face import FaceParams, render_face
    p = FaceParams.neutral()
    p.render_mode = "ict_face_3d"
    frame = render_face(p, (200, 200))
    assert frame.shape == (200, 200, 3)
