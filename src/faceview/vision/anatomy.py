"""Anatomically-grounded landmarks + expression muscles for the 2D renderer.

This module bridges FACS Action Units to 2D vertex displacements via a
muscle catalogue lifted from the faceforge anatomy project (43 named
expression muscles, each with an AU→weight map). Compared to the
hand-rolled smile/jaw knobs of the stylised renderer, this gives:

- Landmarks placed at canonical face proportions (rule of thirds, eye
  spacing, nose-tip height, lip rest).
- Each AU resolves through anatomically-correct muscles to produce
  vertex displacements in the right direction (zygomaticus pulls the
  lip corner up *and outward*, levator labii alaeque nasi pulls the
  upper lip and nasal wing together, mentalis pushes the lower lip up
  via the chin pad, etc.).
- A single :func:`deform_landmarks` call mutates a landmark dict
  according to a :class:`~faceview.vision.sim_face.FaceParams` so the
  renderer can stay focused on drawing.

The muscle catalogue is loaded from
``assets/config/expression_muscles.json`` (trimmed copy of faceforge's
38 STL definitions — names and AU maps only). 2D centroids + fiber
directions are added in :data:`MUSCLE_LAYOUT` here; that table is the
only place anatomical knowledge gets baked in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from faceview.assets import assets_dir


# ── Landmark template ───────────────────────────────────────────────────
#
# Coordinates are in normalised face-box space: x ∈ [0, 1] (left → right
# from the viewer's perspective), y ∈ [0, 1] (top → bottom). The renderer
# scales these to image pixels.
#
# Group names are used by the renderer to look up clusters (e.g. all
# "lip_outer_upper" points form the upper-lip border polyline).

@dataclass
class Landmark:
    name: str
    group: str
    x: float
    y: float


def _arc(group: str, prefix: str, cx: float, cy: float,
         rx: float, ry: float, angles: Iterable[float]) -> list[Landmark]:
    """Helper — emit landmarks along an ellipse at given degrees (math convention)."""
    import math
    out: list[Landmark] = []
    for i, deg in enumerate(angles):
        rad = math.radians(deg)
        out.append(Landmark(
            name=f"{prefix}_{i}",
            group=group,
            x=cx + rx * math.cos(rad),
            y=cy - ry * math.sin(rad),  # screen-y down; flip
        ))
    return out


def _build_template() -> list[Landmark]:
    """Canonical landmark template at BP3D-measured proportions.

    Y positions adjusted (session 8) to match the proportional anatomy
    measured off the BodyParts3D skull mesh — eye line at the vertical
    midpoint of the cranium, nose tip near the lower third, mouth at
    the three-quarter mark. The face is also slightly narrower (head
    ratio ~1:1.45 height:width matches average human anatomy) than
    the original cartoony 1:1.18 oval.
    """
    L: list[Landmark] = []

    # Face oval — narrower / taller silhouette matching BP3D skull.
    jaw_pts = [
        (0.50, 0.96, "chin"),
        (0.43, 0.93, "jaw_l1"), (0.37, 0.88, "jaw_l2"), (0.30, 0.81, "jaw_l3"),
        (0.23, 0.73, "jaw_l4"), (0.18, 0.62, "ear_lower_l"),
        (0.16, 0.50, "ear_upper_l"), (0.16, 0.38, "temple_l"),
        (0.20, 0.24, "forehead_l"), (0.32, 0.13, "hairline_l"),
        (0.50, 0.08, "hairline_top"),
        (0.68, 0.13, "hairline_r"), (0.80, 0.24, "forehead_r"),
        (0.84, 0.38, "temple_r"), (0.84, 0.50, "ear_upper_r"),
        (0.82, 0.62, "ear_lower_r"), (0.77, 0.73, "jaw_r4"),
        (0.70, 0.81, "jaw_r3"), (0.63, 0.88, "jaw_r2"),
        (0.57, 0.93, "jaw_r1"),
    ]
    for x, y, name in jaw_pts:
        L.append(Landmark(name=name, group="face_oval", x=x, y=y))

    # Brows — anatomical mid-height (just above orbit, ~38% of face).
    for side, sx in (("l", 0.32), ("r", 0.68)):
        for i, dx in enumerate([-0.07, -0.035, 0.0, 0.035, 0.07]):
            x = sx + dx
            y = 0.40 + 0.012 * abs(dx) * 30
            L.append(Landmark(name=f"brow_{side}_{i}", group=f"brow_{side}", x=x, y=y))

    # Eyes — at vertical midpoint of head (canonical anatomical rule).
    eye_y_centre = 0.48
    for side, sx in (("l", 0.36), ("r", 0.64)):
        for i, (dx, dy) in enumerate([(-0.055, 0.0), (-0.030, -0.014),
                                       (0.0, -0.020), (0.030, -0.014),
                                       (0.055, 0.0)]):
            L.append(Landmark(name=f"eye_{side}_upper_{i}",
                                group=f"eye_{side}_upper",
                                x=sx + dx, y=eye_y_centre + dy))
        for i, (dx, dy) in enumerate([(-0.055, 0.0), (-0.030, 0.018),
                                       (0.0, 0.022), (0.030, 0.018),
                                       (0.055, 0.0)]):
            L.append(Landmark(name=f"eye_{side}_lower_{i}",
                                group=f"eye_{side}_lower",
                                x=sx + dx, y=eye_y_centre + dy))
        L.append(Landmark(name=f"iris_{side}", group=f"iris_{side}",
                            x=sx, y=eye_y_centre + 0.002))

    # Nose — bridge starts just above eye line; tip at ~64% of face.
    nose_pts = [
        (0.50, 0.42, "nose_root"),
        (0.48, 0.50, "nose_bridge_l"), (0.52, 0.50, "nose_bridge_r"),
        (0.47, 0.58, "nose_dorsum_l"), (0.53, 0.58, "nose_dorsum_r"),
        (0.50, 0.65, "nose_tip"),
        (0.44, 0.66, "nose_alar_l"), (0.56, 0.66, "nose_alar_r"),
        (0.46, 0.68, "nostril_l"), (0.54, 0.68, "nostril_r"),
        (0.50, 0.69, "columella"),
    ]
    for x, y, name in nose_pts:
        L.append(Landmark(name=name, group="nose", x=x, y=y))

    # Philtrum.
    L += [
        Landmark(name="philtrum_l", group="philtrum", x=0.485, y=0.72),
        Landmark(name="philtrum_r", group="philtrum", x=0.515, y=0.72),
    ]

    # Lips — at ~75% of face height.
    upper_lip_pts = [
        (0.40, 0.78, "lip_corner_l"),
        (0.44, 0.755, "lip_upper_l2"),
        (0.475, 0.748, "cupid_l"),
        (0.50, 0.753, "cupid_top"),
        (0.525, 0.748, "cupid_r"),
        (0.56, 0.755, "lip_upper_r2"),
        (0.60, 0.78, "lip_corner_r"),
    ]
    for x, y, name in upper_lip_pts:
        L.append(Landmark(name=name, group="lip_outer_upper", x=x, y=y))

    lower_lip_pts = [
        (0.60, 0.78, "lip_corner_r2"),
        (0.555, 0.815, "lip_lower_r2"),
        (0.50, 0.830, "lip_lower_mid"),
        (0.445, 0.815, "lip_lower_l2"),
        (0.40, 0.78, "lip_corner_l2"),
    ]
    for x, y, name in lower_lip_pts:
        L.append(Landmark(name=name, group="lip_outer_lower", x=x, y=y))

    inner_upper = [(0.46, 0.785, "inner_u_l"), (0.50, 0.783, "inner_u_m"),
                   (0.54, 0.785, "inner_u_r")]
    inner_lower = [(0.46, 0.800, "inner_l_l"), (0.50, 0.805, "inner_l_m"),
                   (0.54, 0.800, "inner_l_r")]
    for x, y, name in inner_upper:
        L.append(Landmark(name=name, group="lip_inner_upper", x=x, y=y))
    for x, y, name in inner_lower:
        L.append(Landmark(name=name, group="lip_inner_lower", x=x, y=y))

    L += [
        Landmark(name="cheek_l", group="cheek", x=0.30, y=0.62),
        Landmark(name="cheek_r", group="cheek", x=0.70, y=0.62),
    ]
    L.append(Landmark(name="glabella", group="glabella", x=0.50, y=0.42))

    return L


@lru_cache(maxsize=1)
def landmark_template() -> list[Landmark]:
    return _build_template()


@lru_cache(maxsize=1)
def landmark_index() -> dict[str, int]:
    return {lm.name: i for i, lm in enumerate(landmark_template())}


def landmarks_in_group(group: str) -> list[Landmark]:
    return [lm for lm in landmark_template() if lm.group == group]


# ── Muscles ─────────────────────────────────────────────────────────────


@dataclass
class Muscle:
    """An anatomical expression muscle in 2D normalised face-box space.

    ``cx, cy`` is the centroid (origin of contraction). ``fx, fy`` is the
    *fiber direction* — the unit vector along which the muscle pulls
    when it contracts. ``radius`` is how far around the centroid
    landmarks are influenced (also normalised). ``au_map`` is the
    weighted contribution of each AU (lifted from faceforge).
    """
    name: str
    cx: float
    cy: float
    fx: float
    fy: float
    radius: float
    au_map: dict[str, float]


# Layout: per faceforge muscle name → (cx, cy, fx, fy, radius). Position
# follows facial anatomy (Frontalis sits over the forehead, Zygomaticus
# Maj. originates at the zygomatic arch and inserts at the lip corner,
# Risorius runs lateral, etc.). Fiber direction encodes the *pull* —
# Zygomaticus Maj. pulls upward-and-outward, so for the L muscle (right
# side of the screen) fx is positive and fy is negative.
#
# Coordinates are in screen space (y increases downward), so a "lift"
# action has fy < 0.
#
# Convention: ``L`` suffix muscles sit on the viewer's LEFT side (small
# x). Their fiber direction with ``fx < 0`` indicates an outward (lateral)
# pull; ``fx > 0`` is medial (toward the midline). ``R`` suffix muscles
# mirror that on the right.
MUSCLE_LAYOUT: dict[str, tuple[float, float, float, float, float]] = {
    # Forehead — Frontalis lifts brows straight up.
    "Frontalis R":          (0.62, 0.22,  0.00, -1.00, 0.16),
    "Frontalis L":          (0.38, 0.22,  0.00, -1.00, 0.16),
    # Orbicularis Oculi — radial sphincter around the eye (fx=fy=0 means
    # ``deform_landmarks`` interprets this as inward radial contraction).
    "Orbic. Oculi Orb. R":  (0.64, 0.40,  0.00,  0.00, 0.12),
    "Orbic. Oculi Orb. L":  (0.36, 0.40,  0.00,  0.00, 0.12),
    "Orbic. Oculi Palp. R": (0.64, 0.40,  0.00,  0.00, 0.07),
    "Orbic. Oculi Palp. L": (0.36, 0.40,  0.00,  0.00, 0.07),
    # Corrugator — pulls inner brow tip toward the midline + down (medial).
    "Corrugator Sup. R":    (0.58, 0.34, -0.70,  0.30, 0.10),
    "Corrugator Sup. L":    (0.42, 0.34,  0.70,  0.30, 0.10),
    # Procerus — pulls glabella down (between brows).
    "Procerus R":           (0.515, 0.34, 0.00,  1.00, 0.06),
    "Procerus L":           (0.485, 0.34, 0.00,  1.00, 0.06),
    # Nasalis / Depr. Septi — bunch nostrils.
    "Nasalis R":            (0.54, 0.55, -0.70,  0.30, 0.07),
    "Nasalis L":            (0.46, 0.55,  0.70,  0.30, 0.07),
    "Depr. Septi Nasi R":   (0.51, 0.62,  0.00,  1.00, 0.05),
    "Depr. Septi Nasi L":   (0.49, 0.62,  0.00,  1.00, 0.05),
    # Zygomaticus Maj. — origin at zygomatic bone, insertion at lip corner.
    # Fiber at insertion points up-and-out (toward origin).
    "Zygomatic Maj. R":     (0.65, 0.70,  0.40, -0.92, 0.18),
    "Zygomatic Maj. L":     (0.35, 0.70, -0.40, -0.92, 0.18),
    "Zygomatic Min. R":     (0.52, 0.65,  0.45, -0.80, 0.13),
    "Zygomatic Min. L":     (0.48, 0.65, -0.45, -0.80, 0.13),
    # Levator labii sup. — pulls upper lip + nasal wing up.
    "Lev. Labii Sup. R":    (0.54, 0.62,  0.10, -1.00, 0.10),
    "Lev. Labii Sup. L":    (0.46, 0.62, -0.10, -1.00, 0.10),
    "Lev. Labii Alae. R":   (0.53, 0.64,  0.30, -0.95, 0.09),
    "Lev. Labii Alae. L":   (0.47, 0.64, -0.30, -0.95, 0.09),
    "Lev. Anguli Oris R":   (0.55, 0.69,  0.20, -0.98, 0.10),
    "Lev. Anguli Oris L":   (0.45, 0.69, -0.20, -0.98, 0.10),
    # Orbicularis Oris — radial sphincter around the lips.
    "Orbicularis Oris":     (0.50, 0.72,  0.00,  0.00, 0.10),
    # Depressor anguli oris — pulls lip corner down-and-out.
    "Depr. Anguli Oris R":  (0.55, 0.78,  0.20,  0.98, 0.12),
    "Depr. Anguli Oris L":  (0.45, 0.78, -0.20,  0.98, 0.12),
    "Depr. Labii Inf. R":   (0.54, 0.77,  0.00,  1.00, 0.10),
    "Depr. Labii Inf. L":   (0.46, 0.77,  0.00,  1.00, 0.10),
    "Mentalis R":           (0.52, 0.85,  0.00, -1.00, 0.08),
    "Mentalis L":           (0.48, 0.85,  0.00, -1.00, 0.08),
    # Risorius — pure lateral pull on lip corner.
    "Risorius R":           (0.62, 0.72,  1.00,  0.00, 0.14),
    "Risorius L":           (0.38, 0.72, -1.00,  0.00, 0.14),
    "Buccinator R":         (0.70, 0.66,  1.00,  0.10, 0.10),
    "Buccinator L":         (0.30, 0.66, -1.00,  0.10, 0.10),
    "Platysma R":           (0.65, 0.90,  0.30,  1.00, 0.12),
    "Platysma L":           (0.35, 0.90, -0.30,  1.00, 0.12),
    "Occipitalis R":        (0.65, 0.16,  0.00, -0.50, 0.10),
    "Occipitalis L":        (0.35, 0.16,  0.00, -0.50, 0.10),
    "Temporoparietalis R":  (0.85, 0.35,  0.00, -1.00, 0.08),
    "Temporoparietalis L":  (0.15, 0.35,  0.00, -1.00, 0.08),
    "Lev. Palpebrae Sup. R":(0.64, 0.39,  0.00, -1.00, 0.06),
    "Lev. Palpebrae Sup. L":(0.36, 0.39,  0.00, -1.00, 0.06),
}


@lru_cache(maxsize=1)
def load_muscles() -> list[Muscle]:
    """Load the trimmed faceforge expression-muscle catalogue with 2D layout.

    Falls back to an empty list if the JSON is missing — keeps the
    renderer importable in test environments without bundled assets.
    """
    path = assets_dir() / "config" / "expression_muscles.json"
    if not path.exists():
        return []
    raw = json.loads(Path(path).read_text())
    out: list[Muscle] = []
    for entry in raw:
        layout = MUSCLE_LAYOUT.get(entry["name"])
        if layout is None:
            continue
        cx, cy, fx, fy, radius = layout
        out.append(Muscle(
            name=entry["name"],
            cx=cx, cy=cy, fx=fx, fy=fy, radius=radius,
            au_map={k: float(v) for k, v in entry["auMap"].items()},
        ))
    return out


def muscle_activation(muscle: Muscle, au_values: dict[str, float]) -> float:
    """Combine AU values via the muscle's au_map. Faceforge uses *max*."""
    a = 0.0
    for au, w in muscle.au_map.items():
        a = max(a, w * float(au_values.get(au, 0.0)))
    return max(0.0, min(1.0, a))


# ── AU → vertex displacement ────────────────────────────────────────────


# Maximum displacement (in normalised face-box units) at full muscle
# activation, per landmark. Tuned to be visible without breaking face
# topology.
MAX_DISPLACEMENT = 0.030


# ── Skeletal jaw model ──────────────────────────────────────────────
#
# Lifted from faceforge (`coordination/simulation.py`):
# ``jaw_angle = AU26 * 0.28 + AU25 * 0.06`` radians, applied as a rigid
# rotation around the temporomandibular joint (TMJ). In our 2D
# face-box space the TMJ sits where the ear-upper landmarks are
# (~y=0.50), and the rotation projects to a downward-and-slightly-
# forward shift of every landmark *below* that line — chin drops
# most because it's farthest from the hinge.
#
# This replaces the old "stretch the lips apart" mouth-open model
# and makes mouth opening read like real jaw motion.

import math as _math

TMJ_Y = 0.50      # TMJ vertical level in face-box [0,1].
JAW_AU26_RAD = 0.28
JAW_AU25_RAD = 0.06


def _jaw_angle(au_values: dict[str, float]) -> float:
    """Same formula as faceforge — radians of mandible rotation."""
    a26 = float(au_values.get("AU26", 0.0))
    a25 = float(au_values.get("AU25", 0.0))
    return a26 * JAW_AU26_RAD + a25 * JAW_AU25_RAD


# Landmark groups that move with the mandible. Upper face (forehead,
# brows, eyes, nose, cheeks above the lip line) stays put.
_JAW_GROUPS = {
    "lip_outer_upper", "lip_outer_lower",
    "lip_inner_upper", "lip_inner_lower",
    "philtrum",
}
# Plus specific face_oval landmarks at the chin / lower jaw.
_JAW_NAMES = {
    "chin", "jaw_l1", "jaw_l2", "jaw_l3", "jaw_r1", "jaw_r2", "jaw_r3",
    "lip_corner_l", "lip_corner_r",
}


def _apply_jaw_rotation(
    base: list[tuple[float, float]],
    template: list[Landmark],
    angle: float,
) -> list[tuple[float, float]]:
    """Rotate lower-face landmarks around the TMJ hinge.

    ``angle`` is in radians; positive = jaw open. The 2D projection
    of a horizontal-axis hinge rotation is a downward translation
    proportional to ``sin(angle) * (y - tmj_y)``. We also apply a
    small forward shrink (``1 - cos(angle)``) to mimic the chin
    moving slightly back as the jaw drops.
    """
    if abs(angle) < 1e-6:
        return base
    sin_a = _math.sin(angle)
    cos_a = _math.cos(angle)
    out: list[tuple[float, float]] = []
    for (x, y), lm in zip(base, template):
        # Skip landmarks above the TMJ line / not in the jaw groups.
        in_jaw = (lm.group in _JAW_GROUPS) or (lm.name in _JAW_NAMES)
        if not in_jaw or y < TMJ_Y:
            out.append((x, y))
            continue
        d = y - TMJ_Y
        new_y = TMJ_Y + d * cos_a + d * sin_a   # rotate + drop
        # No horizontal shift — the hinge axis is horizontal in 2D.
        # (3D would also pull the chin back; in the front view this
        # cancels out into a y-shift only.)
        out.append((x, new_y))
    return out


def deform_landmarks(
    base: list[tuple[float, float]],
    au_values: dict[str, float],
    muscles: list[Muscle] | None = None,
    template: list[Landmark] | None = None,
) -> list[tuple[float, float]]:
    """Skeletal jaw rotation, then muscle contractions.

    Mirrors faceforge's pipeline: (1) jaw rotates around TMJ as a
    rigid bone; (2) expression muscles deform the soft tissue on top
    of the moved skeleton. Without (1), AU26 just stretched the lips
    open — visibly fake. With (1), the chin drops, the lower lip
    follows, and the upper lip stays anchored to the cranium.
    """
    if muscles is None:
        muscles = load_muscles()
    if template is None:
        template = landmark_template()

    # 1. Rigid skeletal jaw rotation.
    jaw_angle = _jaw_angle(au_values)
    rotated = _apply_jaw_rotation(base, template, jaw_angle)

    # 2. Muscle contractions on top of the rotated skeleton.
    out: list[tuple[float, float]] = []
    for x, y in rotated:
        dx = 0.0
        dy = 0.0
        for m in muscles:
            r2 = (x - m.cx) ** 2 + (y - m.cy) ** 2
            if r2 > m.radius * m.radius:
                continue
            falloff = 1.0 - (r2 ** 0.5) / m.radius
            a = muscle_activation(m, au_values)
            if a <= 0.0:
                continue
            scale = MAX_DISPLACEMENT * a * falloff
            # Orbicularis (radial sphincters) contract toward the centroid.
            if m.fx == 0.0 and m.fy == 0.0:
                vx = m.cx - x
                vy = m.cy - y
                norm = (vx * vx + vy * vy) ** 0.5 + 1e-9
                dx += scale * vx / norm
                dy += scale * vy / norm
            else:
                dx += scale * m.fx
                dy += scale * m.fy
        out.append((x + dx, y + dy))
    return out


def face_params_to_au_values(params) -> dict[str, float]:
    """Translate a :class:`FaceParams` into the AU vocabulary muscles use.

    Mirrors the AU-grade fields populated by ``face_state_to_params`` so
    the anatomical renderer doesn't need a separate FaceState. AU12 is
    inferred from positive smile, AU15 from negative smile + lip drop.
    """
    smile = float(getattr(params, "smile", 0.0))
    return {
        "AU1": float(getattr(params, "inner_brow_raise", 0.0)),
        "AU2": float(getattr(params, "outer_brow_raise", 0.0)),
        "AU4": float(getattr(params, "brow_lower", 0.0)),
        "AU5": float(getattr(params, "upper_lid_raise", 0.0)),
        "AU6": float(getattr(params, "cheek_raise", 0.0)),
        "AU9": float(getattr(params, "nose_wrinkle", 0.0)),
        "AU12": max(0.0, smile),
        "AU15": max(0.0, -smile) + float(getattr(params, "lip_corner_drop", 0.0)),
        "AU20": float(getattr(params, "mouth_stretch", 0.0)),
        "AU22": float(getattr(params, "mouth_pucker", 0.0)),
        "AU25": min(1.0, max(0.0, float(getattr(params, "jaw_open", 0.0)) * 1.4)),
        "AU26": min(1.0, float(getattr(params, "jaw_open", 0.0))),
        # Eye blink — closed amount = 1 - eye_open. Without this the
        # ARKit eyeBlink blendshapes never fire and `eye_open` has no
        # visible effect on the ICT renderer.
        "AU45": max(0.0, min(1.0, 1.0 - float(getattr(params, "eye_open", 1.0)))),
        # New mouth shapes:
        "AU17": float(getattr(params, "chin_raise", 0.0)),       # pout / chin raise
        "AU23": float(getattr(params, "lip_tighten", 0.0)),      # lips drawn tight
        "AU24": float(getattr(params, "lip_press", 0.0)),        # lips pressed
        "AU10": float(getattr(params, "upper_lip_raise", 0.0)),  # snarl
        "AU14": float(getattr(params, "dimpler", 0.0)),          # smirk
    }
