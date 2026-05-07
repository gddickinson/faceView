"""Decimated BP3D skin mesh — gated on BP3D STLs being copied."""

from __future__ import annotations

import numpy as np
import pytest

from faceview.vision.anatomy_meshes import meshes_available


@pytest.mark.skipif(not meshes_available(), reason="BP3D STLs not present")
def test_decimation_returns_head_only_mesh():
    from faceview.vision.head_decimated import decimated_skin
    decimated_skin.cache_clear()
    verts, tris, normals = decimated_skin(20)
    assert verts.shape[1] == 3
    assert tris.shape[1] == 3
    assert tris.dtype == np.int32
    # Head + neck only — should be way fewer than full body.
    assert 100 < len(verts) < 5000
    assert 200 < len(tris) < 20000


@pytest.mark.skipif(not meshes_available(), reason="BP3D STLs not present")
def test_decimated_renders_valid_frame(qtbot):
    from faceview.vision.head_decimated import (
        decimated_skin, render_face_decimated,
    )
    decimated_skin.cache_clear()
    from faceview.vision.sim_face import FaceParams
    frame = render_face_decimated(FaceParams.neutral(), (240, 240), grid=20)
    assert frame.shape == (240, 240, 3)
    assert frame.dtype == np.uint8
    # Real head should produce non-trivial brightness.
    assert frame.mean() > 5.0


@pytest.mark.skipif(not meshes_available(), reason="BP3D STLs not present")
def test_dispatcher_routes_to_decimated(qtbot):
    from faceview.vision.sim_face import FaceParams, render_face
    p = FaceParams.neutral()
    p.render_mode = "head_decimated_3d"
    frame = render_face(p, (200, 200))
    assert frame.shape == (200, 200, 3)
