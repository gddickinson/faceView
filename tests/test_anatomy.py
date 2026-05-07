"""Anatomy module — landmarks, muscles, AU-driven deformation."""

from __future__ import annotations

from faceview.vision.anatomy import (
    Muscle,
    deform_landmarks,
    face_params_to_au_values,
    landmark_template,
    landmarks_in_group,
    load_muscles,
    muscle_activation,
)
from faceview.vision.sim_face import FaceParams


def test_landmark_template_has_expected_groups():
    template = landmark_template()
    groups = {lm.group for lm in template}
    assert "face_oval" in groups
    assert "lip_outer_upper" in groups
    assert "lip_outer_lower" in groups
    assert "eye_l_upper" in groups and "eye_r_upper" in groups
    assert "brow_l" in groups and "brow_r" in groups
    assert "nose" in groups


def test_landmarks_in_group_returns_face_oval_count():
    pts = landmarks_in_group("face_oval")
    # 17-20 jawline points roughly clockwise from chin.
    assert 15 <= len(pts) <= 22


def test_muscles_loaded_with_layout():
    muscles = load_muscles()
    # All 43 muscles should resolve through MUSCLE_LAYOUT.
    assert len(muscles) >= 30
    names = {m.name for m in muscles}
    assert "Zygomatic Maj. R" in names
    assert "Zygomatic Maj. L" in names
    assert "Corrugator Sup. R" in names
    assert "Orbicularis Oris" in names


def test_muscle_activation_uses_max_over_au_map():
    m = Muscle(name="t", cx=0.5, cy=0.5, fx=1.0, fy=0.0, radius=0.1,
               au_map={"AU1": 0.7, "AU4": 1.0})
    assert muscle_activation(m, {"AU1": 1.0, "AU4": 0.0}) == 0.7
    assert muscle_activation(m, {"AU1": 0.0, "AU4": 0.5}) == 0.5
    assert muscle_activation(m, {"AU1": 1.0, "AU4": 1.0}) == 1.0
    assert muscle_activation(m, {}) == 0.0


def test_zygomaticus_pulls_lip_corner_up_and_outward():
    """AU12 (smile) should drag the lip corner along the muscle fiber."""
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    deformed = deform_landmarks(base, {"AU12": 1.0})
    idx = {lm.name: i for i, lm in enumerate(template)}
    bx0, by0 = base[idx["lip_corner_l"]]
    dx0, dy0 = deformed[idx["lip_corner_l"]]
    # Up: smaller y
    assert dy0 < by0
    # Outward (left side ⇒ smaller x)
    assert dx0 < bx0


def test_corrugator_pulls_brows_together_and_down():
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    deformed = deform_landmarks(base, {"AU4": 1.0})
    idx = {lm.name: i for i, lm in enumerate(template)}
    # Left brow inner end (brow_l_4) should move right and down.
    bx0, by0 = base[idx["brow_l_4"]]
    dx0, dy0 = deformed[idx["brow_l_4"]]
    assert dx0 > bx0  # toward centre
    assert dy0 > by0  # downward (positive y in screen coords)


def test_face_params_to_au_values_translates_smile():
    happy = FaceParams.happy()
    au = face_params_to_au_values(happy)
    assert au["AU12"] > 0.5
    assert au["AU6"] > 0.0  # cheek raise present in happy preset

    sad = FaceParams.sad()
    au = face_params_to_au_values(sad)
    assert au["AU15"] > 0.5
    assert au["AU12"] == 0.0  # negative smile


def test_neutral_params_produce_no_displacement():
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    deformed = deform_landmarks(base, {})
    for (bx, by), (dx, dy) in zip(base, deformed):
        assert abs(dx - bx) < 1e-9
        assert abs(dy - by) < 1e-9
