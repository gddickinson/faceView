"""Layered anatomy renderer + photo-anatomical bridge."""

from __future__ import annotations

import numpy as np
import pytest

from faceview.vision.faceforge_bridge import faceforge_status
from faceview.vision.sim_face import FaceParams, render_face
from faceview.vision.sim_face_layered import (
    LAYER_NAMES,
    LAYER_PRESETS,
    render_face_layered,
)


@pytest.mark.parametrize("preset", list(LAYER_PRESETS))
def test_layered_preset_renders(qtbot, preset: str):
    p = FaceParams.happy()
    frame = render_face_layered(p, (240, 240), layers=preset)
    assert frame.dtype == np.uint8
    assert frame.shape == (240, 240, 3)
    assert frame.max() > 0


def test_render_face_dispatch_to_layered(qtbot):
    p = FaceParams.surprised()
    p.render_mode = "anatomy_xray"
    frame = render_face(p, (240, 240))
    assert frame.shape == (240, 240, 3)
    # X-ray composes 5 translucent layers — output should not be black.
    assert frame.mean() > 5.0


def test_layer_names_match_presets():
    for preset in LAYER_PRESETS.values():
        for name, _ in preset:
            assert name in LAYER_NAMES, f"unknown layer in preset: {name}"


def test_skull_only_does_not_use_skin(qtbot):
    p = FaceParams.neutral()
    frame_skull = render_face_layered(p, (200, 200), layers="anatomy_skull")
    frame_layers = render_face_layered(p, (200, 200), layers="anatomy_layers")
    diff = np.abs(frame_skull.astype(int) - frame_layers.astype(int)).mean()
    assert diff > 5.0


def test_unknown_layer_raises_value_error(qtbot):
    p = FaceParams.neutral()
    with pytest.raises(ValueError):
        render_face_layered(p, (100, 100), layers="bogus_preset")


def test_explicit_layer_list_accepted(qtbot):
    p = FaceParams.neutral()
    frame = render_face_layered(p, (200, 200), layers=[("skull", 0.7), ("brain", 0.5)])
    assert frame.shape == (200, 200, 3)


def test_faceforge_status_reports_dir(qtbot):
    s = faceforge_status()
    assert "mesh_dir" in s
    assert "mesh_count" in s
    assert "expected_head_neck" in s


def test_faceforge_render_mode_dispatches(qtbot):
    """When meshes are present render through bridge; otherwise raise."""
    from faceview.core.errors import MissingDependency
    from faceview.vision.anatomy_meshes import meshes_available

    p = FaceParams.neutral()
    p.render_mode = "faceforge_3d"
    if meshes_available():
        frame = render_face(p, (200, 200))
        assert frame.shape == (200, 200, 3)
    else:
        with pytest.raises(MissingDependency):
            render_face(p, (200, 200))
