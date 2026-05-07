"""MakeHuman base mesh — proper human topology, CC0-licensed.

Loads the MakeHuman community base mesh (`assets/data/makehuman/base.obj`,
~19K vertices, ~18K triangles, designed for character animation, with
proper feature rings around eyes / mouth / nose). Decimates via vertex
clustering and renders through the same QPainter / GPU pipeline.

The MakeHuman mesh is a higher-quality starting point than the BP3D
skin mesh for character work because:

- Topology designed for animation (proper edge loops around features)
- Released CC0 (no attribution required, freely modifiable)
- Standard format used across MakeHuman / Blender / Unreal / Unity
- Ships at a reasonable polygon count (BP3D is ~30K, MakeHuman ~19K)

This module is the bridge: load → reorient → decimate → render.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _mesh_path() -> Path:
    return assets_dir() / "data" / "makehuman" / "base.obj"


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Wavefront OBJ — return (verts, tris)."""
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                # OBJ faces can have v/vt/vn entries — take just v.
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                # Triangulate fan if quad/poly.
                for i in range(1, len(idxs) - 1):
                    tris.append((idxs[0], idxs[i], idxs[i + 1]))
    return (np.array(verts, dtype=np.float32),
            np.array(tris, dtype=np.int32))


@lru_cache(maxsize=4)
def load_makehuman_head(grid: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Load the MakeHuman base, crop to head + neck, decimate.

    Returns ``(verts, tris)`` in screen-coord space.
    """
    path = _mesh_path()
    if not path.exists():
        raise MissingDependency(
            "MakeHuman base mesh", "gpu",
            hint=(
                "Bundle assets/data/makehuman/base.obj from the "
                "MakeHuman community CC0 distribution."
            ),
        )
    verts, tris = _parse_obj(path)

    # MakeHuman convention: +Y up (head at +Y), +Z forward (face at +Z).
    # Crop to head + upper neck FIRST (in original Y-up space), then
    # flip Y so screen-Y goes down.
    body_y_min = verts[:, 1].min()
    body_y_max = verts[:, 1].max()
    body_h = body_y_max - body_y_min
    # Head is at LARGE Y (top of figure in MH coords).
    keep_min_y = body_y_max - 0.16 * body_h    # head + neck
    keep_max_y = body_y_max + 0.005 * body_h
    # Crop in MakeHuman's native +Y-up space.
    vert_mask = (verts[:, 1] >= keep_min_y) & (verts[:, 1] <= keep_max_y)
    tri_mask = (vert_mask[tris[:, 0]] & vert_mask[tris[:, 1]]
                & vert_mask[tris[:, 2]])
    cropped_tris = tris[tri_mask]

    # Vertex-cluster decimation.
    used = np.unique(cropped_tris)
    v_subset = verts[used]
    vmin = v_subset.min(axis=0)
    vmax = v_subset.max(axis=0)
    span = np.maximum(vmax - vmin, 1e-6)
    cell = np.floor((v_subset - vmin) / span * grid).astype(np.int32)
    cell = np.clip(cell, 0, grid - 1)
    cell_id = cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]
    unique_cells, inv = np.unique(cell_id, return_inverse=True)
    n_new = len(unique_cells)
    counts = np.bincount(inv)
    sums = np.zeros((n_new, 3), dtype=np.float64)
    np.add.at(sums, inv, v_subset.astype(np.float64))
    new_verts = (sums / counts[:, None]).astype(np.float32)

    # Remap triangles.
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[used] = np.arange(len(used))
    raw_tris = remap[cropped_tris]
    new_tris = inv[raw_tris]
    keep = (
        (new_tris[:, 0] != new_tris[:, 1]) &
        (new_tris[:, 1] != new_tris[:, 2]) &
        (new_tris[:, 0] != new_tris[:, 2])
    )
    final_tris = new_tris[keep].astype(np.int32)
    # Flip Y so screen-Y goes down for the renderer (apply at the
    # end so the head ends up upright in image coords).
    new_verts[:, 1] = -new_verts[:, 1]
    return new_verts, final_tris


def _project(verts: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    return verts @ (ry @ rx).T


def render_face_makehuman(
    params,
    size: tuple[int, int] = (480, 480),
    *,
    grid: int = 24,
) -> np.ndarray:
    """Render the decimated MakeHuman head via QPainter Z-sort."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QPainter, QPainterPath,
    )

    verts, tris = load_makehuman_head(grid)
    if len(tris) == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    rotated = _project(verts, yaw, pitch)
    vmin = rotated.min(axis=0)
    vmax = rotated.max(axis=0)
    centre = (vmin + vmax) / 2.0
    span = float(np.linalg.norm(vmax - vmin))
    w, h = size
    scale = 0.85 * min(w, h) / max(span, 1e-6)

    sx = (rotated[:, 0] - centre[0]) * scale + w / 2
    sy = (rotated[:, 1] - centre[1]) * scale + h / 2

    v0 = rotated[tris[:, 0]]
    v1 = rotated[tris[:, 1]]
    v2 = rotated[tris[:, 2]]
    n_ = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n_, axis=1, keepdims=True)
    n_norm = np.divide(n_, np.maximum(nl, 1e-9))
    avg_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0

    light = np.array([-0.4, -0.4, -1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    diff = np.abs(n_norm @ light)
    shade = np.clip(0.32 + 0.62 * diff, 0.0, 1.4)

    z_median = np.median(avg_z)
    front_mask = avg_z > z_median - (avg_z.max() - avg_z.min()) * 0.05
    order = np.where(front_mask)[0]
    order = order[np.argsort(avg_z[order])]

    bg = QColor(getattr(params, "background", "#0a0d12"))
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(bg)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    p.setPen(Qt.PenStyle.NoPen)

    skin_r, skin_g, skin_b = 220, 188, 165
    for ti in order:
        s = float(shade[ti])
        col = QColor(int(skin_r * s), int(skin_g * s), int(skin_b * s))
        p.setBrush(QBrush(col))
        i0, i1, i2 = tris[ti]
        path = QPainterPath()
        path.moveTo(QPointF(sx[i0], sy[i0]))
        path.lineTo(QPointF(sx[i1], sy[i1]))
        path.lineTo(QPointF(sx[i2], sy[i2]))
        path.closeSubpath()
        p.drawPath(path)
    p.end()

    arr = img.convertToFormat(QImage.Format.Format_RGB888)
    ptr = arr.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    a = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return a[:, :, ::-1].copy()
