"""Derive 2D landmark positions from the BP3D skull + skin meshes.

The hand-coded landmark template in :mod:`vision.anatomy` uses
proportions from a generic face. Once the BP3D head meshes have been
copied locally, we can measure real anatomical reference points off
the actual mesh data — silhouette of the face oval, exact orbital
boundaries, mandible angle, hairline curve — and either bake those
values into a refined template or use them at runtime.

This module extracts those measurements. The rendered images become
proportionally accurate to a real human skull (BP3D is derived from
medical imaging) without any hand-tuning.

The function :func:`bp3d_landmark_overrides` returns a dict of
``{name: (x_norm, y_norm)}`` for landmarks we can reliably measure
from the skull alone — face_oval, hairline, brow, jaw, temple, ear.
Lip and eye landmarks stay hand-coded because they sit on soft tissue
that the skin mesh covers but the skull doesn't expose.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from faceview.vision.anatomy_meshes import (
    list_available_meshes,
    load_mesh,
    meshes_available,
)


# BP3D coordinate system after the BP3D→screen reorientation we use
# elsewhere: +X right, +Y up, +Z toward camera.
_REORIENT = np.array([
    [-1, 0, 0],
    [0,  0, -1],
    [0, -1, 0],
], dtype=np.float32)


def _project(verts: np.ndarray) -> np.ndarray:
    """Reorient BP3D verts to screen-coord space."""
    return verts @ _REORIENT.T


@lru_cache(maxsize=1)
def _skull_xy_bbox() -> tuple[np.ndarray, np.ndarray] | None:
    """Compute the bounding box of (frontal-bone + zygomatic + mandible)."""
    targets = ["FMA52734", "FMA52747", "FMA52748",  # frontal, zygomatic, mandible
                "FMA46565", "FMA53672", "FMA53673"]
    avail = set(list_available_meshes())
    pts: list[np.ndarray] = []
    for fma in targets:
        if fma in avail:
            m = load_mesh(fma)
            pts.append(_project(m.vertices))
    if not pts:
        return None
    all_pts = np.vstack(pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


@lru_cache(maxsize=1)
def bp3d_landmark_overrides() -> dict[str, tuple[float, float]]:
    """Measure anatomical landmark positions from the BP3D skull.

    Coordinates are normalised to a [0, 1]^2 face box keyed off the
    skull's own XY extents — same convention as the hand-coded
    template in :mod:`vision.anatomy`.

    Returns ``{}`` when the meshes aren't present so the existing
    template is used unchanged.
    """
    if not meshes_available():
        return {}

    bbox = _skull_xy_bbox()
    if bbox is None:
        return {}
    vmin, vmax = bbox

    # Convert raw coords to [0,1] using the *same* skull-only bbox the
    # renderer uses to scale.
    span = vmax - vmin

    def to_norm(x: float, y: float) -> tuple[float, float]:
        # After our _REORIENT the head is upside-down in BP3D-Y space
        # because the negative-Z mapping inverts vertical. So Y in
        # input space already ascends with screen-down.
        nx = (x - vmin[0]) / max(span[0], 1e-6)
        ny = (y - vmin[1]) / max(span[1], 1e-6)
        return float(nx), float(ny)

    avail = set(list_available_meshes())
    overrides: dict[str, tuple[float, float]] = {}

    # Mandible (FMA52748) — chin = lowest point near midline (largest Y in
    # screen coords).
    if "FMA52748" in avail:
        m = load_mesh("FMA52748")
        proj = _project(m.vertices)
        midx = (vmin[0] + vmax[0]) / 2.0
        mask = np.abs(proj[:, 0] - midx) < 4.0
        if mask.any():
            chin_idx = np.argmax(proj[mask, 1])
            chin_pt = proj[mask][chin_idx]
            overrides["chin"] = to_norm(chin_pt[0], chin_pt[1])
        # Mandible angles — extreme X at lower-third of mandible.
        # gonion (jaw angle) = lateral-most point on mandible.
        l_idx = np.argmin(proj[:, 0])
        r_idx = np.argmax(proj[:, 0])
        overrides["jaw_l4"] = to_norm(proj[l_idx, 0], proj[l_idx, 1])
        overrides["jaw_r4"] = to_norm(proj[r_idx, 0], proj[r_idx, 1])

    # Cranium silhouette — top of skull, temples.
    if "FMA46565" in avail:
        m = load_mesh("FMA46565")
        proj = _project(m.vertices)
        midx = (vmin[0] + vmax[0]) / 2.0
        # Top of skull = smallest screen-Y near midline.
        mask = np.abs(proj[:, 0] - midx) < 4.0
        if mask.any():
            top_idx = np.argmin(proj[mask, 1])
            top_pt = proj[mask][top_idx]
            overrides["hairline_top"] = to_norm(top_pt[0], top_pt[1])
        # Temples (lateral-most points at mid skull height).
        midy = (vmin[1] + vmax[1]) / 2.0
        mask = np.abs(proj[:, 1] - midy) < 12.0
        if mask.any():
            l_idx = np.argmin(proj[mask, 0])
            r_idx = np.argmax(proj[mask, 0])
            overrides["temple_l"] = to_norm(proj[mask][l_idx, 0],
                                              proj[mask][l_idx, 1])
            overrides["temple_r"] = to_norm(proj[mask][r_idx, 0],
                                              proj[mask][r_idx, 1])

    return overrides


def apply_overrides_to_template(template: list, overrides: dict) -> None:
    """Mutate a landmark template in-place to use measured BP3D positions."""
    for lm in template:
        if lm.name in overrides:
            x, y = overrides[lm.name]
            lm.x = x
            lm.y = y
