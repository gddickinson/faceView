"""GPU renderer — gated on moderngl + BP3D mesh availability."""

from __future__ import annotations

import numpy as np
import pytest

from faceview.vision.anatomy_meshes import meshes_available


def test_moderngl_optional_import():
    """The bridge should import without GPU; only fail at render time."""
    from faceview.vision import gpu_renderer  # noqa: F401


def test_gpu_available_helper_returns_bool():
    from faceview.vision.gpu_renderer import gpu_available
    assert isinstance(gpu_available(), bool)


@pytest.mark.skipif(not meshes_available(), reason="BP3D STLs not present")
def test_gpu_renders_lifelike_head_when_meshes_present():
    pytest.importorskip("moderngl")
    from faceview.vision.gpu_renderer import render_face_faceforge_gpu
    from faceview.vision.sim_face import FaceParams

    frame = render_face_faceforge_gpu(FaceParams.neutral(), (200, 200),
                                        layer_set="skull_only")
    assert frame.shape == (200, 200, 3)
    assert frame.dtype == np.uint8
    # Skull renders should have non-trivial brightness.
    assert frame.mean() > 5.0
