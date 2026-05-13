"""Regression test for the body rig — verify that body effects only
move the verts they're supposed to.

After the skeleton-bone voxel relabel + override bake (May 2026), the
label assignment was clean enough that an arm rotation should move
ONLY arm-labelled verts (BPF 4-9) and a leg rotation should move
ONLY leg-labelled verts (BPF 10-15). Anything else is a mislabel
regression.

We sample each effect at u=0.5 (peak) intensity=1.0 and assert
zero unexpected movers (disp > 1.0 unit).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ARM_LABELS = {4, 5, 6, 7, 8, 9}
LEG_LABELS = {10, 11, 12, 13, 14, 15}
DISP_THRESHOLD = 1.0  # units (ICT-frame); rest-pose median NN ≈ 0.74

# Effects to test, grouped by which BPF labels are expected to move.
ARM_ONLY_EFFECTS = [
    "arms_up", "arms_out", "salute", "wave_left", "wave_right",
    "clap", "stretch_up", "arms_crossed", "shrug",
    "point_left", "point_right", "thinking",
    "hands_on_hips",
]
LEG_ONLY_EFFECTS = [
    "kick_left", "kick_right", "squat",
]


def _make_neutral_params(gender: str):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = 0.0
    p._camera_pitch = 0.0
    p._camera_zoom = 0.55
    return p


def _capture_rig_io(render_call):
    """Monkey-patch apply_body_rig_v2 + _apply_manual_overrides to
    grab the rest/posed verts and effective fine labels from a
    single render call."""
    import faceview.vision.body_rig as br
    cap: dict = {}
    orig_rig = br.apply_body_rig_v2
    orig_overrides = br._apply_manual_overrides

    def rig_hook(verts, params, rig):
        cap["rest"] = np.asarray(verts).copy()
        out = orig_rig(verts, params, rig)
        cap["posed"] = np.asarray(out).copy()
        return out

    def overrides_hook(fine_labels, **kw):
        out = orig_overrides(fine_labels, **kw)
        cap["fine_eff"] = np.asarray(out).copy()
        return out

    br.apply_body_rig_v2 = rig_hook
    br._apply_manual_overrides = overrides_hook
    try:
        render_call()
    finally:
        br.apply_body_rig_v2 = orig_rig
        br._apply_manual_overrides = orig_overrides
    return cap


def _capture_posed(gender: str, effect_name: str | None):
    """Render the avatar with one effect at peak. Return rest, posed,
    and effective fine labels."""
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS
    p = _make_neutral_params(gender)
    if effect_name is not None:
        handler = HANDLERS.get(f"pre_{effect_name}") or HANDLERS.get(
            effect_name)
        assert handler is not None, f"no handler for {effect_name}"
        handler(p, 0.5, 1.0)
    cap = _capture_rig_io(
        lambda: render_face_ict(p, size=(360, 640)))
    return cap["rest"], cap["posed"], cap["fine_eff"]


def _check_effect(gender: str, effect_name: str,
                   expected_labels: set[int]) -> int:
    """Return number of unexpected-mover voxels for this effect.
    A vert is an 'unexpected mover' if its disp > DISP_THRESHOLD
    AND its label is not in expected_labels."""
    rest, posed, fine = _capture_posed(gender, effect_name)
    disp = np.linalg.norm(posed - rest, axis=1)
    expected_mask = np.isin(fine, list(expected_labels))
    return int(((~expected_mask) & (disp > DISP_THRESHOLD)).sum())


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("effect", ARM_ONLY_EFFECTS)
def test_arm_effect_only_moves_arms(gender, effect):
    n_wrong = _check_effect(gender, effect, ARM_LABELS)
    assert n_wrong == 0, (
        f"{gender}/{effect}: {n_wrong} non-arm verts moved > "
        f"{DISP_THRESHOLD}u — body rig regression")


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("effect", LEG_ONLY_EFFECTS)
def test_leg_effect_only_moves_legs(gender, effect):
    n_wrong = _check_effect(gender, effect, LEG_LABELS)
    assert n_wrong == 0, (
        f"{gender}/{effect}: {n_wrong} non-leg verts moved > "
        f"{DISP_THRESHOLD}u — body rig regression")


@pytest.mark.parametrize("gender", ["male", "female"])
def test_neutral_pose_has_no_isolated_voxels(gender):
    """At rest, every body vert should have at least one neighbour
    within a few units — flyaway voxels show up as isolated."""
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        pytest.skip("scipy not installed")
    rest, _posed, _fine = _capture_posed(gender, None)
    tree = cKDTree(rest)
    d, _ = tree.query(rest, k=2)
    nn = d[:, 1]
    n_isolated = int((nn > 5.0).sum())
    assert n_isolated == 0, (
        f"{gender}: {n_isolated} isolated voxels at rest "
        f"(nn > 5.0u) — body mesh regression")
