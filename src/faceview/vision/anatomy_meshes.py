"""BodyParts3D STL mesh loading + 2D projection rendering.

Loads BP3D `.stl` files (binary format) into NumPy arrays and renders
them via QPainter as Z-sorted lambert-shaded triangles. Designed to
work with the **head + neck** subset of BodyParts3D (~115 meshes,
~120 MB on disk). The meshes are *not* committed to the repo — run
``python -m tools.copy_anatomy_meshes /path/to/bodyparts3D/stl`` once
to populate ``assets/anatomy_meshes/``.

This is the foundation of the photo-anatomical render mode. With
real meshes it produces a recognisable medical-illustration head;
without them, every entry-point raises :class:`MissingDependency`
with the install hint.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


# ── FMA codes for the head + neck subset ──────────────────────────────
#
# 12 expression-muscle pairs (left/right) handled by ``expression_muscles.json``
# remain valid here — same FMA IDs. We add the skull bones, mandible,
# cervical vertebrae C1-C7, face features (eyes / ears / nose cartilages),
# jaw muscles, neck muscles, and stylised brain regions.

HEAD_NECK_FMAS = {
    # ── Skull bones ───────────────────────────────────────
    "FMA46565": "skull",
    "FMA53672": "neurocranium",
    "FMA53673": "viscerocranium",
    "FMA52801": "basicranium",
    "FMA52748": "mandible",
    "FMA52747": "zygomatic_bone",
    # ── Cervical vertebrae C1-C7 ─────────────────────────
    "FMA12519": "atlas_c1",
    "FMA12520": "axis_c2",
    "FMA12521": "c3",
    "FMA12522": "c4",
    "FMA12523": "c5",
    "FMA12524": "c6",
    "FMA12525": "c7",
    # ── 43 expression muscles (face) ─────────────────────
    "FMA46759": "frontalis_r",
    "FMA46760": "frontalis_l",
    "FMA46782": "orbic_oculi_orb_r",
    "FMA46783": "orbic_oculi_orb_l",
    "FMA46785": "orbic_oculi_palp_r",
    "FMA46786": "orbic_oculi_palp_l",
    "FMA46796": "corrugator_r",
    "FMA46797": "corrugator_l",
    "FMA55610": "procerus_r",
    "FMA55611": "procerus_l",
    "FMA55606": "nasalis_r",
    "FMA55607": "nasalis_l",
    "FMA55608": "depr_septi_r",
    "FMA55609": "depr_septi_l",
    "FMA46810": "zygomatic_maj_r",
    # …additional muscle FMAs follow the catalogue but are loaded
    # dynamically — the 43-row list lives in
    # assets/config/expression_muscles.json with their codes as
    # known to faceforge.
}


# ── STL loader ──────────────────────────────────────────────────────


@dataclass
class TriMesh:
    """Triangle mesh — vertices (Nx3), triangles (Mx3 index), per-tri normals (Mx3)."""
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    centroid: np.ndarray
    bbox: tuple[np.ndarray, np.ndarray]


def _read_binary_stl(path: Path) -> TriMesh:
    """Parse a binary STL file. Returns a :class:`TriMesh`."""
    with path.open("rb") as f:
        header = f.read(80)  # noqa: F841
        n_tri = struct.unpack("<I", f.read(4))[0]
        # Each triangle: normal (3f) + vert0 (3f) + vert1 (3f) + vert2 (3f) + attr (2)
        rec_dtype = np.dtype([
            ("normal", "<3f4"),
            ("v0", "<3f4"),
            ("v1", "<3f4"),
            ("v2", "<3f4"),
            ("attr", "<u2"),
        ])
        data = np.frombuffer(f.read(rec_dtype.itemsize * n_tri), dtype=rec_dtype)

    if len(data) == 0:
        return TriMesh(
            name=path.stem,
            vertices=np.zeros((0, 3), dtype=np.float32),
            triangles=np.zeros((0, 3), dtype=np.int32),
            normals=np.zeros((0, 3), dtype=np.float32),
            centroid=np.zeros(3, dtype=np.float32),
            bbox=(np.zeros(3), np.zeros(3)),
        )

    verts = np.empty((len(data) * 3, 3), dtype=np.float32)
    verts[0::3] = data["v0"]
    verts[1::3] = data["v1"]
    verts[2::3] = data["v2"]
    tris = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
    normals = data["normal"].copy()
    # Replace (0,0,0) "auto" normals with computed ones.
    bad = np.linalg.norm(normals, axis=1) < 1e-6
    if bad.any():
        v0 = verts[tris[bad, 0]]
        v1 = verts[tris[bad, 1]]
        v2 = verts[tris[bad, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        normals[bad] = n

    return TriMesh(
        name=path.stem,
        vertices=verts,
        triangles=tris,
        normals=normals,
        centroid=verts.mean(axis=0),
        bbox=(verts.min(axis=0), verts.max(axis=0)),
    )


# ── Asset path + availability ───────────────────────────────────────


def mesh_dir() -> Path:
    return assets_dir() / "anatomy_meshes"


def meshes_available() -> bool:
    d = mesh_dir()
    if not d.is_dir():
        return False
    return any(d.glob("*.stl"))


def list_available_meshes() -> list[str]:
    d = mesh_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.stl"))


@lru_cache(maxsize=128)
def load_mesh(name: str) -> TriMesh:
    """Load and cache a single STL mesh by name (without extension)."""
    path = mesh_dir() / f"{name}.stl"
    if not path.exists():
        raise MissingDependency(
            f"anatomy mesh {name}",
            install_hint=(
                "Run `python -m tools.copy_anatomy_meshes "
                "/path/to/bodyparts3D/stl` to populate "
                f"{mesh_dir()} from a local BodyParts3D download."
            ),
        )
    return _read_binary_stl(path)


# ── 2D projection + rendering ───────────────────────────────────────


def project_orthographic(
    verts: np.ndarray,
    yaw: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Apply yaw/pitch rotation, return (N, 3) where x,y is screen, z is depth.

    Includes the canonical BodyParts3D → screen reorientation: BP3D
    uses ``+Z`` up and ``+Y`` forward (anatomical/medical convention),
    we want ``+Y`` up and ``-Z`` toward the camera (screen
    convention). That's a fixed −90° rotation around X applied first.
    """
    # Fixed BP3D → screen reorientation.
    # BP3D: +Z up, +Y anatomical-forward, +X right.
    # Screen: +Y up, -Z toward camera, +X right.
    # Rotation chain: -90° around X (Z up → Y up), then 180° around Y so
    # the face points toward the camera (BP3D +Y becomes screen -Z).
    rx0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    ry180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    R = ry @ rx @ ry180 @ rx0
    return verts @ R.T


def render_meshes(
    meshes: list[TriMesh],
    size: tuple[int, int],
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    light_dir: tuple[float, float, float] = (-0.4, -0.5, -0.7),
    bg_color: tuple[int, int, int] = (10, 12, 16),
    base_colour: tuple[int, int, int] = (220, 210, 195),
    materials: list | None = None,
    ambient: float = 0.30,
    specular_strength: float = 0.35,
) -> np.ndarray:
    """Render meshes with optional per-mesh materials.

    ``materials`` is an optional list aligned with ``meshes`` providing
    ``MeshSpec`` records (color / opacity / shininess). When omitted,
    all meshes share ``base_colour`` and ``shininess=6``. Lighting is
    Phong (ambient + diffuse + specular) with a single key light. The
    rasteriser is CPU-only Z-sorted polygon paint via QPainter — there
    is no per-pixel shading, only per-triangle.
    """
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QPainter, QPainterPath,
    )

    w, h = size
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(*bg_color))

    if not meshes:
        return _qimage_to_bgr(img)

    # Choose the scale reference: prefer "bone" meshes if any are present
    # (so the head fills the frame even when a full-body skin mesh is in
    # the list). Otherwise use everything.
    if materials is not None and any(s.category == "bone" for s in materials):
        ref_meshes = [m for m, s in zip(meshes, materials) if s.category == "bone"]
    else:
        ref_meshes = meshes
    vmin = np.min([m.vertices.min(axis=0) for m in ref_meshes], axis=0)
    vmax = np.max([m.vertices.max(axis=0) for m in ref_meshes], axis=0)
    centre = (vmin + vmax) / 2.0
    span = float(np.linalg.norm(vmax - vmin))
    scale = 0.85 * min(w, h) / max(span, 1e-6)

    light = np.asarray(light_dir, dtype=np.float32)
    light /= max(1e-9, np.linalg.norm(light))
    view = np.array([0, 0, -1], dtype=np.float32)
    half = light + view
    half /= max(1e-9, np.linalg.norm(half))

    # Build a flat list of (avg_z, pts2d, color, alpha, draw_order).
    all_tris: list[tuple[float, int, np.ndarray, QColor]] = []

    for mi, m in enumerate(meshes):
        spec = materials[mi] if materials is not None else None
        if spec is not None:
            col_r, col_g, col_b = spec.color
            opacity = spec.opacity
            shininess = max(1.0, float(spec.shininess))
            order = spec.draw_order
        else:
            col_r, col_g, col_b = base_colour
            opacity = 1.0
            shininess = 6.0
            order = 100

        v = m.vertices - centre
        v3 = project_orthographic(v, yaw=yaw, pitch=pitch)
        normals = project_orthographic(m.normals, yaw=yaw, pitch=pitch)
        x = v3[:, 0] * scale + w / 2
        y = -v3[:, 1] * scale + h / 2
        z = v3[:, 2]
        # Vectorised shading.
        n_dot_l = np.abs(normals @ light)
        diffuse = np.maximum(0.0, n_dot_l)
        n_dot_h = np.abs(normals @ half)
        specular = np.power(np.clip(n_dot_h, 0.0, 1.0), shininess) * specular_strength
        shade = np.clip(ambient + diffuse * (1.0 - ambient) + specular, 0.0, 1.6)

        for ti in range(len(m.triangles)):
            i0, i1, i2 = m.triangles[ti]
            avg_z = (z[i0] + z[i1] + z[i2]) / 3.0
            sh = float(shade[ti])
            r = min(255, int(col_r * sh))
            g = min(255, int(col_g * sh))
            b = min(255, int(col_b * sh))
            qc = QColor(r, g, b)
            qc.setAlphaF(opacity)
            pts2d = np.array([[x[i0], y[i0]], [x[i1], y[i1]], [x[i2], y[i2]]])
            all_tris.append((float(avg_z), int(order), pts2d, qc))

    # Sort: primarily by draw_order (bones first), secondary by z (back to front).
    all_tris.sort(key=lambda t: (t[1], t[0]), reverse=False)
    # Want back-to-front *within* each draw_order: re-sort within groups.
    # Simpler — sort by (order, -z) so larger z (back) draws first.
    all_tris.sort(key=lambda t: (t[1], -t[0]))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setPen(Qt.PenStyle.NoPen)
    for _avg_z, _order, pts2d, qc in all_tris:
        p.setBrush(QBrush(qc))
        path = QPainterPath()
        path.moveTo(QPointF(pts2d[0, 0], pts2d[0, 1]))
        path.lineTo(QPointF(pts2d[1, 0], pts2d[1, 1]))
        path.lineTo(QPointF(pts2d[2, 0], pts2d[2, 1]))
        path.closeSubpath()
        p.drawPath(path)
    p.end()
    return _qimage_to_bgr(img)


def _qimage_to_bgr(img):
    from PySide6.QtGui import QImage
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return arr[:, :, ::-1].copy()
