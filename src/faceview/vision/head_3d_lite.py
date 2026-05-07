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


# ── Z-depth catalogue ───────────────────────────────────────────────
#
# Hand-tuned Z values per landmark group (positive = toward viewer).
# Tuned against a profile-view photo so the silhouette reads correctly.

_GROUP_Z: dict[str, float] = {
    "face_oval": 0.0,        # the oval is on the front-facing skin plane
    "brow_l": 0.04, "brow_r": 0.04,
    "eye_l_upper": 0.02, "eye_r_upper": 0.02,
    "eye_l_lower": 0.02, "eye_r_lower": 0.02,
    "iris_l": 0.04, "iris_r": 0.04,
    "nose": 0.06,
    "philtrum": 0.05,
    "lip_outer_upper": 0.05, "lip_outer_lower": 0.04,
    "lip_inner_upper": 0.03, "lip_inner_lower": 0.03,
    "cheek": 0.03,
    "glabella": 0.04,
}

# Specific overrides for individual landmarks where the group default
# is too coarse (e.g. nose tip protrudes farther than nose bridge).
_LANDMARK_Z: dict[str, float] = {
    "nose_root": 0.04,
    "nose_bridge_l": 0.05, "nose_bridge_r": 0.05,
    "nose_dorsum_l": 0.07, "nose_dorsum_r": 0.07,
    "nose_tip": 0.10,
    "nose_alar_l": 0.06, "nose_alar_r": 0.06,
    "nostril_l": 0.05, "nostril_r": 0.05,
    "columella": 0.04,
    "chin": 0.04, "jaw_l1": 0.02, "jaw_r1": 0.02,
    "jaw_l2": -0.02, "jaw_r2": -0.02,
    "jaw_l3": -0.05, "jaw_r3": -0.05,
    "jaw_l4": -0.08, "jaw_r4": -0.08,
    "ear_lower_l": -0.12, "ear_lower_r": -0.12,
    "ear_upper_l": -0.14, "ear_upper_r": -0.14,
    "temple_l": -0.10, "temple_r": -0.10,
    "forehead_l": -0.02, "forehead_r": -0.02,
    "hairline_l": -0.04, "hairline_r": -0.04,
    "hairline_top": -0.06,
    "lip_corner_l": 0.02, "lip_corner_r": 0.02,
}


@dataclass
class Vertex:
    name: str
    group: str
    x: float
    y: float
    z: float


def _z_for(name: str, group: str) -> float:
    if name in _LANDMARK_Z:
        return _LANDMARK_Z[name]
    return _GROUP_Z.get(group, 0.0)


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


def build_3d_template() -> list[Vertex]:
    """Combine the 2D landmark template with closure points + Z values."""
    out: list[Vertex] = []
    for lm in landmark_template():
        out.append(Vertex(
            name=lm.name, group=lm.group,
            x=lm.x, y=lm.y, z=_z_for(lm.name, lm.group),
        ))
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


def _build_triangulation(verts_xy: np.ndarray) -> np.ndarray:
    """3D-aware triangulation.

    Front-of-face triangles via 2D Delaunay on the projected XY (the
    face is roughly convex from the front). Back-of-head triangles
    via the hand-defined ``_BACK_TRIS`` list. The two halves are
    stitched at the temple/ear/jaw seam.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError:
        # Fallback: only the hand-defined triangulation. Will leave gaps.
        return np.array([])

    # Face vertices = the 86 template entries (indices 0..85). We
    # triangulate just those by 2D position.
    face_xy = verts_xy[:86]
    tri = Delaunay(face_xy)
    return tri.simplices  # (M, 3) ndarray of indices into face_xy


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

    # Build triangulation: Delaunay on the front face + hand-defined back.
    front_tris = _build_triangulation(verts_xy_2d)
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

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    rotated = _rotate_project(verts, yaw, pitch)
    scale = min(w, h) * 0.9
    sx = rotated[:, 0] * scale + w / 2
    sy = rotated[:, 1] * scale + h / 2

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)

    light = np.array([-0.4, -0.3, -1.0], dtype=np.float64)
    light /= np.linalg.norm(light)
    view = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    half = light + view
    half /= np.linalg.norm(half)

    from collections import Counter

    rendered = []
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
        # Use absolute value for shading — winding order is mixed
        # between Delaunay output and the hand-coded back triangulation.
        diff = abs(float(n_ @ light))
        spec = abs(float(n_ @ half))
        spec = spec ** 18 * 0.45
        avg_z = (v0[2] + v1[2] + v2[2]) / 3.0
        groups = [template[i0].group, template[i1].group, template[i2].group]
        # Only assign feature color when *all three* vertices share the
        # same feature group; otherwise the triangle spans skin and
        # should be coloured as skin. This stops Delaunay-spanning
        # triangles from painting big lip-coloured wedges across cheeks.
        feature_groups = {
            "lip_outer_upper", "lip_outer_lower",
            "lip_inner_upper", "lip_inner_lower",
            "iris_l", "iris_r",
            "eye_l_upper", "eye_l_lower",
            "eye_r_upper", "eye_r_lower",
            "scalp", "back", "neck",
        }
        if groups[0] == groups[1] == groups[2] and groups[0] in feature_groups:
            group = groups[0]
        else:
            group = "face_oval"  # default skin
        rendered.append((float(avg_z), [i0, i1, i2], diff, spec, group))

    # Z-sort painter's algorithm: smaller z (further) drawn first.
    rendered.sort(key=lambda r: r[0])

    for avg_z, idx, diff, spec, group in rendered:
        col, _shin = _material_for(group, params)
        shade = 0.30 + 0.65 * diff + spec
        r = min(255, int(col.red() * shade))
        g = min(255, int(col.green() * shade))
        b = min(255, int(col.blue() * shade))
        p.setBrush(QBrush(QColor(r, g, b)))
        i0, i1, i2 = idx
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
