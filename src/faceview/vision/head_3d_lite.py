"""Lite 3D animated head — anatomically-positioned low-poly mesh.

Bridges the 2D anatomical renderer and the photo-anatomical BP3D
renderer. Takes the existing 86-landmark template, adds anatomical Z
depth per landmark, closes the silhouette with ~30 back-of-head /
side / scalp points, hand-triangulates ~250 triangles, and renders
the result with the same FACS-driven AU deformation as the 2D
pipeline.

Animation runs in real time on CPU because it's just ~120 vertices —
about 1000× cheaper to rasterise than the 145-mesh BP3D head. AUs
deform the X/Y of each landmark via the muscle layout in
:mod:`vision.anatomy`; Z stays fixed so the head silhouette remains
stable while the face deforms.

Render mode ``head_3d_lite`` routes here through the standard
``render_face`` dispatcher, so the talking-avatar pipeline picks it
up via ``Persona.render_mode`` unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush, QColor, QImage, QPainter, QPainterPath,
)

from faceview.vision.anatomy import (
    deform_landmarks,
    face_params_to_au_values,
    landmark_template,
    load_muscles,
)


# ── Smooth Z function ──────────────────────────────────────────────
#
# Replaces the hand-tuned per-landmark Z values from earlier sessions
# (which produced a faceted, cuboid silhouette) with a continuous
# quadric: an ellipsoidal head dome + small additive feature offsets
# for the nose, lips, ears, and chin. Same input ranges (x, y in
# [0, 1]^2 face-box space) as the 2D landmark template.

import math as _math


_HEAD_RX = 0.45    # half-width of the head dome
_HEAD_RY = 0.55    # half-height (slightly elongated)
_HEAD_RZ = 0.22    # max forward protrusion at face centre
_BACK_DEPTH = -0.32

# Per-group additive Z offsets layered on top of the ellipsoid.
# Kept small so they don't create visible seams between feature
# regions and surrounding skin — the smooth dome carries most of the
# 3D shape.
_GROUP_OFFSETS: dict[str, float] = {
    "face_oval": 0.0,
    "brow_l": 0.0, "brow_r": 0.0,
    "eye_l_upper": -0.005, "eye_r_upper": -0.005,
    "eye_l_lower": 0.0, "eye_r_lower": 0.0,
    "iris_l": 0.0, "iris_r": 0.0,
    "nose": 0.012,
    "philtrum": 0.0,
    "lip_outer_upper": 0.005, "lip_outer_lower": 0.003,
    "lip_inner_upper": 0.0, "lip_inner_lower": 0.0,
    "cheek": 0.0,
    "glabella": 0.0,
}

# Per-landmark fine-tunes — only nose tip + ears + temples need
# anything beyond the smooth dome to read correctly in profile.
_LANDMARK_OFFSETS: dict[str, float] = {
    "nose_tip": 0.045,
    "nose_alar_l": 0.020, "nose_alar_r": 0.020,
    "nostril_l": 0.015, "nostril_r": 0.015,
    "nose_bridge_l": 0.008, "nose_bridge_r": 0.008,
    "nose_dorsum_l": 0.020, "nose_dorsum_r": 0.020,
    "columella": 0.018,
    "chin": 0.005,
    "ear_lower_l": -0.07, "ear_lower_r": -0.07,
    "ear_upper_l": -0.09, "ear_upper_r": -0.09,
    "temple_l": -0.04, "temple_r": -0.04,
    "hairline_top": -0.01,
    "hairline_l": 0.0, "hairline_r": 0.0,
}


def _smooth_z(x: float, y: float, group: str = "", name: str = "") -> float:
    """Continuous head-dome Z + per-feature offset.

    The dome is half an ellipsoid centred at (0.5, 0.5) with axes
    (rx, ry, rz). Inside the ellipse, Z = rz * sqrt(1 - (dx² + dy²));
    outside, Z falls off toward the back-of-head depth so closure
    points remain stable.
    """
    cx, cy = 0.5, 0.50
    dx = (x - cx) / _HEAD_RX
    dy = (y - cy) / _HEAD_RY
    r2 = dx * dx + dy * dy
    if r2 < 1.0:
        base = _HEAD_RZ * _math.sqrt(max(0.0, 1.0 - r2))
    else:
        # Outside the face ellipse — fade toward back depth.
        base = max(_BACK_DEPTH, -0.04 * (r2 - 1.0))
    offset = _LANDMARK_OFFSETS.get(name)
    if offset is None:
        offset = _GROUP_OFFSETS.get(group, 0.0)
    return base + offset


@dataclass
class Vertex:
    name: str
    group: str
    x: float
    y: float
    z: float


def _z_for(name: str, group: str) -> float:
    """Compatibility shim — returns the smooth Z at the landmark's XY."""
    # Look up the landmark's nominal x, y from the 2D template.
    for lm in landmark_template():
        if lm.name == name:
            return _smooth_z(lm.x, lm.y, lm.group, name)
    return 0.0


# ── Back-of-head closure ───────────────────────────────────────────
#
# Adds points on the cranium dome + neck so the head reads as a 3D
# volume from any angle, not just front view.
_CLOSURE_VERTICES: list[Vertex] = [
    # Top of skull (vertex)
    Vertex("vertex", "scalp", 0.50, 0.05, -0.10),
    # Scalp ring (around the top, slightly forward of pure back)
    Vertex("scalp_l", "scalp", 0.20, 0.10, -0.18),
    Vertex("scalp_r", "scalp", 0.80, 0.10, -0.18),
    Vertex("scalp_back", "scalp", 0.50, 0.10, -0.30),
    # Back-of-head
    Vertex("back_top", "back", 0.50, 0.20, -0.32),
    Vertex("back_l", "back", 0.30, 0.30, -0.30),
    Vertex("back_r", "back", 0.70, 0.30, -0.30),
    Vertex("back_mid", "back", 0.50, 0.40, -0.30),
    Vertex("back_lower_l", "back", 0.30, 0.60, -0.25),
    Vertex("back_lower_r", "back", 0.70, 0.60, -0.25),
    Vertex("back_low_mid", "back", 0.50, 0.60, -0.27),
    # Behind the ears
    Vertex("behind_ear_l", "back", 0.10, 0.50, -0.20),
    Vertex("behind_ear_r", "back", 0.90, 0.50, -0.20),
    # Neck back / nape
    Vertex("nape_l", "neck", 0.35, 0.85, -0.20),
    Vertex("nape_r", "neck", 0.65, 0.85, -0.20),
    Vertex("nape_mid", "neck", 0.50, 0.90, -0.22),
    Vertex("neck_l", "neck", 0.32, 0.95, -0.05),
    Vertex("neck_r", "neck", 0.68, 0.95, -0.05),
    Vertex("neck_front", "neck", 0.50, 0.99, 0.0),
]


# Pairs of landmarks to insert midpoints between — densifies the
# mesh in regions where Delaunay otherwise produces large facets.
_MIDPOINT_PAIRS: list[tuple[str, str]] = [
    # Face oval ring (every adjacent pair).
    ("chin", "jaw_l1"), ("jaw_l1", "jaw_l2"), ("jaw_l2", "jaw_l3"),
    ("jaw_l3", "jaw_l4"), ("jaw_l4", "ear_lower_l"),
    ("ear_lower_l", "ear_upper_l"), ("ear_upper_l", "temple_l"),
    ("temple_l", "forehead_l"), ("forehead_l", "hairline_l"),
    ("hairline_l", "hairline_top"), ("hairline_top", "hairline_r"),
    ("hairline_r", "forehead_r"), ("forehead_r", "temple_r"),
    ("temple_r", "ear_upper_r"), ("ear_upper_r", "ear_lower_r"),
    ("ear_lower_r", "jaw_r4"), ("jaw_r4", "jaw_r3"),
    ("jaw_r3", "jaw_r2"), ("jaw_r2", "jaw_r1"), ("jaw_r1", "chin"),
    # Forehead interior — bridge brow tips to forehead and hairline.
    ("brow_l_2", "forehead_l"), ("brow_r_2", "forehead_r"),
    ("brow_l_2", "hairline_l"), ("brow_r_2", "hairline_r"),
    ("glabella", "hairline_top"),
    # Cheek interior — bridge jaw to cheek apple.
    ("cheek_l", "jaw_l3"), ("cheek_r", "jaw_r3"),
    ("cheek_l", "lip_corner_l"), ("cheek_r", "lip_corner_r"),
    ("cheek_l", "ear_lower_l"), ("cheek_r", "ear_lower_r"),
    # Nose-to-cheek bridge.
    ("nose_alar_l", "cheek_l"), ("nose_alar_r", "cheek_r"),
    # Lip-to-chin bridge.
    ("lip_lower_mid", "chin"),
]


def build_3d_template() -> list[Vertex]:
    """Combine the 2D landmark template with closure points + smooth Z.

    Adds midpoint vertices between selected pairs to densify the
    triangulation, which removes the cuboid feel from the original
    sparse mesh.
    """
    out: list[Vertex] = []
    name_to_lm = {}
    for lm in landmark_template():
        v = Vertex(
            name=lm.name, group=lm.group,
            x=lm.x, y=lm.y, z=_smooth_z(lm.x, lm.y, lm.group, lm.name),
        )
        out.append(v)
        name_to_lm[lm.name] = v

    # Midpoint inserts (between named pairs).
    for a, b in _MIDPOINT_PAIRS:
        if a not in name_to_lm or b not in name_to_lm:
            continue
        va, vb = name_to_lm[a], name_to_lm[b]
        mx = (va.x + vb.x) / 2
        my = (va.y + vb.y) / 2
        # Group: prefer the more specific feature group when one is plain skin.
        group = va.group if va.group != "face_oval" else vb.group
        out.append(Vertex(
            name=f"_mid_{a}_{b}",
            group=group,
            x=mx, y=my,
            z=_smooth_z(mx, my, group),
        ))

    # Closure vertices keep their literal Z values (back-of-head /
    # scalp / neck) since they're outside the face ellipse.
    out.extend(_CLOSURE_VERTICES)
    return out


# ── Triangulation ──────────────────────────────────────────────────
#
# Hand-defined triangle list referencing landmarks by name. Building
# the triangulation by name (not index) keeps the table robust against
# template re-orderings.

def _tri(*names: str) -> tuple[str, str, str]:
    return names  # type: ignore[return-value]


# Forehead between brows and hairline.
_FACE_TRIS: list[tuple[str, str, str]] = [
    # Forehead — 6 triangles between brow tips and hairline.
    _tri("brow_l_0", "brow_l_2", "forehead_l"),
    _tri("brow_l_2", "brow_l_4", "hairline_l"),
    _tri("brow_l_2", "hairline_l", "forehead_l"),
    _tri("brow_l_4", "brow_r_0", "hairline_top"),
    _tri("brow_l_4", "hairline_top", "hairline_l"),
    _tri("brow_r_0", "brow_r_2", "hairline_r"),
    _tri("brow_r_2", "brow_r_4", "forehead_r"),
    _tri("brow_r_2", "hairline_r", "hairline_top"),
    _tri("brow_r_2", "forehead_r", "hairline_r"),
    _tri("brow_l_4", "hairline_top", "brow_r_0"),

    # Brows + eyes (upper)
    _tri("brow_l_0", "eye_l_upper_0", "eye_l_upper_2"),
    _tri("brow_l_0", "eye_l_upper_2", "brow_l_2"),
    _tri("brow_l_2", "eye_l_upper_2", "eye_l_upper_4"),
    _tri("brow_l_2", "eye_l_upper_4", "brow_l_4"),
    _tri("brow_r_0", "eye_r_upper_0", "eye_r_upper_2"),
    _tri("brow_r_0", "eye_r_upper_2", "brow_r_2"),
    _tri("brow_r_2", "eye_r_upper_2", "eye_r_upper_4"),
    _tri("brow_r_2", "eye_r_upper_4", "brow_r_4"),

    # Cheeks (between eye-bottom, nose-side, mouth-corner, jaw)
    _tri("eye_l_lower_0", "eye_l_lower_2", "cheek_l"),
    _tri("eye_l_lower_2", "eye_l_lower_4", "cheek_l"),
    _tri("cheek_l", "nose_alar_l", "lip_corner_l"),
    _tri("eye_l_lower_4", "nose_alar_l", "cheek_l"),
    _tri("eye_r_lower_0", "eye_r_lower_2", "cheek_r"),
    _tri("eye_r_lower_2", "eye_r_lower_4", "cheek_r"),
    _tri("cheek_r", "nose_alar_r", "lip_corner_r"),
    _tri("eye_r_lower_4", "nose_alar_r", "cheek_r"),

    # Nose (3D-ish — bridge to dorsum to alar to tip + nostrils)
    _tri("nose_root", "nose_bridge_l", "nose_dorsum_l"),
    _tri("nose_root", "nose_dorsum_r", "nose_bridge_r"),
    _tri("nose_dorsum_l", "nose_alar_l", "nose_tip"),
    _tri("nose_dorsum_r", "nose_tip", "nose_alar_r"),
    _tri("nose_alar_l", "nostril_l", "nose_tip"),
    _tri("nose_alar_r", "nose_tip", "nostril_r"),
    _tri("nostril_l", "columella", "nose_tip"),
    _tri("nostril_r", "nose_tip", "columella"),
    _tri("nose_bridge_l", "nose_root", "glabella"),
    _tri("nose_bridge_r", "glabella", "nose_root"),
    _tri("eye_l_lower_4", "nose_bridge_l", "nose_alar_l"),
    _tri("eye_r_lower_4", "nose_alar_r", "nose_bridge_r"),

    # Philtrum — between nose and upper lip.
    _tri("columella", "philtrum_l", "philtrum_r"),
    _tri("philtrum_l", "cupid_l", "cupid_top"),
    _tri("philtrum_l", "cupid_top", "philtrum_r"),
    _tri("philtrum_r", "cupid_top", "cupid_r"),
    _tri("nose_alar_l", "columella", "philtrum_l"),
    _tri("nose_alar_r", "philtrum_r", "columella"),
    _tri("nose_alar_l", "philtrum_l", "lip_upper_l2"),
    _tri("nose_alar_r", "lip_upper_r2", "philtrum_r"),

    # Upper lip
    _tri("lip_corner_l", "lip_upper_l2", "philtrum_l"),
    _tri("lip_corner_l", "philtrum_l", "cupid_l"),
    _tri("lip_corner_r", "cupid_r", "philtrum_r"),
    _tri("lip_corner_r", "philtrum_r", "lip_upper_r2"),

    # Lower lip
    _tri("lip_corner_l", "lip_lower_l2", "lip_lower_mid"),
    _tri("lip_corner_l", "lip_lower_mid", "lip_corner_r"),
    _tri("lip_corner_r", "lip_lower_mid", "lip_lower_r2"),

    # Chin
    _tri("lip_lower_l2", "chin", "lip_lower_mid"),
    _tri("lip_lower_mid", "chin", "lip_lower_r2"),
    _tri("lip_corner_l", "lip_lower_l2", "jaw_l1"),
    _tri("lip_corner_r", "jaw_r1", "lip_lower_r2"),
    _tri("lip_lower_l2", "jaw_l1", "chin"),
    _tri("jaw_l1", "jaw_r1", "chin"),
    _tri("lip_lower_r2", "chin", "jaw_r1"),

    # Jawline / cheek-to-jaw
    _tri("cheek_l", "lip_corner_l", "jaw_l2"),
    _tri("cheek_r", "jaw_r2", "lip_corner_r"),
    _tri("cheek_l", "jaw_l2", "jaw_l3"),
    _tri("cheek_r", "jaw_r3", "jaw_r2"),
    _tri("lip_corner_l", "jaw_l1", "jaw_l2"),
    _tri("lip_corner_r", "jaw_r2", "jaw_r1"),

    # Side face / temple to ear
    _tri("temple_l", "cheek_l", "ear_upper_l"),
    _tri("ear_upper_l", "cheek_l", "ear_lower_l"),
    _tri("ear_lower_l", "cheek_l", "jaw_l3"),
    _tri("ear_lower_l", "jaw_l3", "jaw_l4"),
    _tri("temple_r", "ear_upper_r", "cheek_r"),
    _tri("ear_upper_r", "ear_lower_r", "cheek_r"),
    _tri("ear_lower_r", "jaw_l3" if False else "jaw_r3", "cheek_r"),
    _tri("ear_lower_r", "jaw_r4", "jaw_r3"),

    # Temple to forehead
    _tri("temple_l", "forehead_l", "cheek_l"),
    _tri("forehead_l", "brow_l_0", "cheek_l"),
    _tri("brow_l_0", "eye_l_upper_0", "cheek_l"),
    _tri("eye_l_upper_0", "eye_l_lower_0", "cheek_l"),
    _tri("temple_r", "cheek_r", "forehead_r"),
    _tri("forehead_r", "cheek_r", "brow_r_4"),
    _tri("brow_r_4", "cheek_r", "eye_r_upper_4"),
    _tri("eye_r_upper_4", "cheek_r", "eye_r_lower_4"),
]

# Back of head + scalp (closes the silhouette so rotating shows volume).
_BACK_TRIS: list[tuple[str, str, str]] = [
    # Top of scalp dome.
    _tri("hairline_top", "vertex", "hairline_l"),
    _tri("hairline_top", "hairline_r", "vertex"),
    _tri("hairline_l", "vertex", "scalp_l"),
    _tri("hairline_r", "scalp_r", "vertex"),
    _tri("vertex", "scalp_l", "scalp_back"),
    _tri("vertex", "scalp_back", "scalp_r"),
    # Back of head.
    _tri("scalp_l", "back_l", "scalp_back"),
    _tri("scalp_r", "scalp_back", "back_r"),
    _tri("scalp_back", "back_l", "back_top"),
    _tri("scalp_back", "back_top", "back_r"),
    _tri("back_l", "back_top", "back_mid"),
    _tri("back_r", "back_mid", "back_top"),
    _tri("back_l", "back_mid", "back_lower_l"),
    _tri("back_r", "back_lower_r", "back_mid"),
    _tri("back_lower_l", "back_low_mid", "back_lower_r"),
    _tri("back_lower_l", "back_mid", "back_low_mid"),
    _tri("back_lower_r", "back_low_mid", "back_mid"),
    # Sides linking temple/ear to back-of-head.
    _tri("temple_l", "scalp_l", "hairline_l"),
    _tri("temple_l", "behind_ear_l", "scalp_l"),
    _tri("temple_l", "ear_upper_l", "behind_ear_l"),
    _tri("ear_upper_l", "ear_lower_l", "behind_ear_l"),
    _tri("scalp_l", "behind_ear_l", "back_l"),
    _tri("behind_ear_l", "back_lower_l", "back_l"),
    _tri("behind_ear_l", "ear_lower_l", "back_lower_l"),
    _tri("temple_r", "hairline_r", "scalp_r"),
    _tri("temple_r", "scalp_r", "behind_ear_r"),
    _tri("temple_r", "behind_ear_r", "ear_upper_r"),
    _tri("ear_upper_r", "behind_ear_r", "ear_lower_r"),
    _tri("scalp_r", "back_r", "behind_ear_r"),
    _tri("behind_ear_r", "back_r", "back_lower_r"),
    _tri("behind_ear_r", "back_lower_r", "ear_lower_r"),
    # Back to nape (neck transition).
    _tri("back_lower_l", "nape_l", "back_low_mid"),
    _tri("back_low_mid", "nape_mid", "nape_l"),
    _tri("back_low_mid", "nape_r", "nape_mid"),
    _tri("back_lower_r", "back_low_mid", "nape_r"),
    _tri("ear_lower_l", "jaw_l4", "back_lower_l"),
    _tri("back_lower_l", "jaw_l4", "nape_l"),
    _tri("ear_lower_r", "back_lower_r", "jaw_r4"),
    _tri("back_lower_r", "nape_r", "jaw_r4"),
    # Nape to neck-front, jaw to neck.
    _tri("nape_l", "jaw_l4", "neck_l"),
    _tri("nape_l", "neck_l", "nape_mid"),
    _tri("nape_mid", "neck_l", "neck_front"),
    _tri("nape_mid", "neck_front", "neck_r"),
    _tri("nape_mid", "neck_r", "nape_r"),
    _tri("nape_r", "neck_r", "jaw_r4"),
    _tri("jaw_l4", "jaw_l3", "neck_l"),
    _tri("jaw_l3", "jaw_l2", "neck_l"),
    _tri("jaw_l2", "jaw_l1", "neck_l"),
    _tri("jaw_l1", "neck_front", "neck_l"),
    _tri("jaw_l1", "chin", "neck_front"),
    _tri("jaw_r1", "neck_r", "neck_front"),
    _tri("jaw_r1", "neck_front", "chin"),
    _tri("jaw_r2", "jaw_r1", "neck_r"),
    _tri("jaw_r3", "jaw_r2", "neck_r"),
    _tri("jaw_r4", "jaw_r3", "neck_r"),
]


_TRIANGLES = _FACE_TRIS + _BACK_TRIS


# ── group → material ────────────────────────────────────────────────


def _material_for(group: str, persona) -> tuple[QColor, float]:
    """Per-group base color + shininess for the lite renderer."""
    # Skin-toned default.
    skin_hex = getattr(persona, "skin_hue", 28.0) if persona else 28.0
    skin = QColor.fromHsvF((float(skin_hex) % 360) / 360.0, 0.46, 0.92)
    if group in {"lip_outer_upper", "lip_outer_lower",
                  "lip_inner_upper", "lip_inner_lower", "philtrum"}:
        return QColor(getattr(persona, "lip_color", "#a44a4a")), 12.0
    if group in {"scalp", "back"}:
        return QColor(getattr(persona, "hair_color", "#2c1810")), 4.0
    if group in {"iris_l", "iris_r"}:
        return QColor(80, 60, 38), 30.0
    if group == "neck":
        return skin.darker(112), 6.0
    return skin, 8.0


# ── projection & rendering ──────────────────────────────────────────


def _rotate_project(verts: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    cp_, sp_ = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]], dtype=np.float64)
    ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]], dtype=np.float64)
    return verts @ (ry @ rx).T


def _build_triangulation(verts_xy: np.ndarray, n_face: int) -> np.ndarray:
    """3D-aware triangulation over the front face.

    Front-of-face triangles via 2D Delaunay on the projected XY (the
    face is roughly convex from the front). ``n_face`` is the number
    of front-face vertices (template + midpoint inserts), excluding
    the back-of-head closure points which get hand-tris.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        return np.array([])
    face_xy = verts_xy[:n_face]
    tri = Delaunay(face_xy)
    return tri.simplices


def _subdivide(verts: np.ndarray, triangles: np.ndarray,
                groups: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """One pass of edge-midpoint subdivision with group inheritance.

    Each triangle (a, b, c) gets midpoints (mab, mbc, mca) and
    becomes 4 smaller triangles. A midpoint inherits its parents'
    group if they agree, otherwise defaults to ``face_oval`` so it
    blends into skin.
    """
    edge_idx: dict[tuple[int, int], int] = {}
    new_verts: list[np.ndarray] = []
    new_groups: list[str] = list(groups)
    next_idx = len(verts)

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key in edge_idx:
            return edge_idx[key]
        nonlocal next_idx
        new_verts.append((verts[a] + verts[b]) / 2.0)
        new_groups.append(groups[a] if groups[a] == groups[b] else "face_oval")
        edge_idx[key] = next_idx
        next_idx += 1
        return next_idx - 1

    new_tris: list[tuple[int, int, int]] = []
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        mab = midpoint(a, b)
        mbc = midpoint(b, c)
        mca = midpoint(c, a)
        new_tris.append((a, mab, mca))
        new_tris.append((mab, b, mbc))
        new_tris.append((mca, mbc, c))
        new_tris.append((mab, mbc, mca))

    if new_verts:
        all_verts = np.vstack([verts, np.array(new_verts, dtype=verts.dtype)])
    else:
        all_verts = verts
    return all_verts, np.array(new_tris, dtype=np.int32), new_groups


def render_face_3d_lite(params, size=(640, 480)) -> np.ndarray:
    w, h = size
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(getattr(params, "background", "#0a0d12")))

    template = build_3d_template()
    name_to_idx = {v.name: i for i, v in enumerate(template)}

    base_xy = [(v.x, v.y) for v in template]
    au_values = face_params_to_au_values(params)
    n_face = len(landmark_template())
    deformed_face = deform_landmarks(base_xy[:n_face], au_values, muscles=load_muscles())
    deformed_xy = list(deformed_face) + base_xy[n_face:]

    verts = np.array(
        [[deformed_xy[i][0] - 0.5, deformed_xy[i][1] - 0.5, template[i].z]
         for i in range(len(template))],
        dtype=np.float64,
    )
    verts_xy_2d = np.array(
        [[deformed_xy[i][0], deformed_xy[i][1]] for i in range(len(template))],
        dtype=np.float64,
    )

    # Front-face Delaunay over template + midpoint inserts (everything
    # except the back-of-head closure verts). The number of front-face
    # vertices = len(template) - len(_CLOSURE_VERTICES).
    n_face = len(template) - len(_CLOSURE_VERTICES)
    front_tris = _build_triangulation(verts_xy_2d, n_face)
    back_tris: list[tuple[int, int, int]] = []
    for tri in _BACK_TRIS:
        try:
            back_tris.append(tuple(name_to_idx[n] for n in tri))  # type: ignore[arg-type]
        except KeyError:
            continue
    if len(front_tris):
        triangles = np.vstack([front_tris, np.array(back_tris, dtype=np.int32)])
    else:
        triangles = np.array(back_tris, dtype=np.int32)

    # Subdivide once. Each triangle → 4 smaller; midpoint vertices
    # inherit their parents' group when agreed (so lips stay lip-
    # coloured), otherwise fall back to skin.
    base_groups = [v.group for v in template]
    verts, triangles, all_groups = _subdivide(verts, triangles, base_groups)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    rotated = _rotate_project(verts, yaw, pitch)
    scale = min(w, h) * 0.85
    sx = rotated[:, 0] * scale + w / 2
    sy = rotated[:, 1] * scale + h / 2

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # No stroke — pen at >0 width creates visible mesh edges.
    p.setPen(Qt.PenStyle.NoPen)

    light = np.array([-0.4, -0.3, -1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    view = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    half = light + view
    half /= np.linalg.norm(half)

    from collections import Counter

    # ── Per-vertex normals ────────────────────────────────────────
    # Average the geometric normals of all incident triangles. Used
    # for smooth Phong-style shading — neighbouring triangles share
    # vertex normals, so the shading transitions continuously
    # instead of jumping at every triangle boundary.
    n_verts = len(rotated)
    vert_normals = np.zeros((n_verts, 3), dtype=np.float64)
    vert_normal_count = np.zeros(n_verts, dtype=np.int32)

    tri_data: list[tuple[int, int, int, np.ndarray, float]] = []
    for tri in triangles:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        v0 = rotated[i0]; v1 = rotated[i1]; v2 = rotated[i2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        n_ = np.cross(edge1, edge2)
        nl = np.linalg.norm(n_)
        if nl < 1e-9:
            continue
        n_ /= nl
        avg_z = (v0[2] + v1[2] + v2[2]) / 3.0
        tri_data.append((i0, i1, i2, n_, float(avg_z)))
        vert_normals[i0] += n_; vert_normal_count[i0] += 1
        vert_normals[i1] += n_; vert_normal_count[i1] += 1
        vert_normals[i2] += n_; vert_normal_count[i2] += 1

    # Normalise per-vertex normals.
    mask = vert_normal_count > 0
    vert_normals[mask] /= vert_normal_count[mask, None]
    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    np.divide(vert_normals, np.maximum(norms, 1e-9), out=vert_normals)

    # ── Per-vertex shading ────────────────────────────────────────
    # Vectorised Phong: diffuse + specular per vertex.
    n_dot_l = np.abs(vert_normals @ light)
    n_dot_h = np.abs(vert_normals @ half)
    diff_v = np.maximum(0.0, n_dot_l)
    spec_v = np.power(np.clip(n_dot_h, 0.0, 1.0), 18) * 0.45
    shade_v = np.clip(0.30 + 0.65 * diff_v + spec_v, 0.0, 1.6)

    # ── Triangle render ───────────────────────────────────────────
    feature_groups = {
        "lip_outer_upper", "lip_outer_lower",
        "lip_inner_upper", "lip_inner_lower",
        "iris_l", "iris_r",
        "eye_l_upper", "eye_l_lower",
        "eye_r_upper", "eye_r_lower",
        "scalp", "back", "neck",
    }

    rendered = []
    for i0, i1, i2, n_, avg_z in tri_data:
        groups = [all_groups[i0], all_groups[i1], all_groups[i2]]
        if groups[0] == groups[1] == groups[2] and groups[0] in feature_groups:
            group = groups[0]
        else:
            group = "face_oval"
        # Triangle shade = mean of its three vertex shades. Smaller
        # facets + averaged shading approximates Gouraud cheaply.
        tri_shade = float((shade_v[i0] + shade_v[i1] + shade_v[i2]) / 3.0)
        rendered.append((avg_z, (i0, i1, i2), tri_shade, group))

    # Z-sort painter's algorithm: smaller z (further) drawn first.
    rendered.sort(key=lambda r: r[0])

    for avg_z, (i0, i1, i2), shade, group in rendered:
        col, _shin = _material_for(group, params)
        r = min(255, int(col.red() * shade))
        g = min(255, int(col.green() * shade))
        b = min(255, int(col.blue() * shade))
        p.setBrush(QBrush(QColor(r, g, b)))
        path = QPainterPath()
        path.moveTo(QPointF(sx[i0], sy[i0]))
        path.lineTo(QPointF(sx[i1], sy[i1]))
        path.lineTo(QPointF(sx[i2], sy[i2]))
        path.closeSubpath()
        p.drawPath(path)

    p.end()
    return _qimage_to_bgr(img)


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return arr[:, :, ::-1].copy()
