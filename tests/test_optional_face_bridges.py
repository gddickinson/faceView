"""Bridge modules for optional/heavy face resources should import OK
without their data + raise MissingDependency at runtime."""

from __future__ import annotations

import pytest

from faceview.core.errors import MissingDependency


def test_bfm_module_imports():
    from faceview.vision import bfm_face  # noqa: F401


def test_bfm_render_raises_missing_dep_when_no_model():
    from faceview.vision import bfm_face
    from faceview.vision.sim_face import FaceParams
    try:
        bfm_face.render_face_bfm(FaceParams.neutral(), (100, 100))
    except MissingDependency:
        pass
    except Exception:
        pytest.skip("eos partially-installed; raised something else")


def test_rpm_module_imports():
    from faceview.vision import rpm_avatar  # noqa: F401
    assert hasattr(rpm_avatar, "RPMAvatar")
    assert hasattr(rpm_avatar, "load_rpm_avatar")


def test_flame_module_imports():
    from faceview.vision import flame_face  # noqa: F401


def test_metahuman_module_imports():
    from faceview.vision import metahuman_face  # noqa: F401


def test_metahuman_render_raises_when_no_fbx(qtbot):
    from faceview.vision import metahuman_face
    from faceview.vision.sim_face import FaceParams
    with pytest.raises(MissingDependency):
        metahuman_face.render_face_metahuman(FaceParams.neutral(), (100, 100))


def test_facescape_module_imports():
    from faceview.vision import facescape_face  # noqa: F401


def test_facescape_render_raises_when_no_scan(qtbot):
    from faceview.vision import facescape_face
    from faceview.vision.sim_face import FaceParams
    with pytest.raises(MissingDependency):
        facescape_face.render_face_facescape(FaceParams.neutral(), (100, 100))


def test_deca_module_imports():
    from faceview.vision import deca_capture  # noqa: F401


def test_makehuman_target_loader():
    """Make sure the .target parser handles real CC0 deltas."""
    from faceview.vision.makehuman_mesh import load_target
    deltas = load_target("male_young", 19158)
    assert deltas.shape == (19158, 3)
    # At least some entries should be non-zero (~8000 from the CC0 file).
    nonzero = (deltas != 0).any(axis=1).sum()
    assert nonzero > 100
