"""Multi-angle texture atlas for face_warp — rotation support.

Builds on :mod:`vision.face_warp` by loading textures at five yaw
angles (-45°, -22°, 0°, +22°, +45°) and blending the two nearest
to the requested ``params.yaw`` per frame. Each texture is warped
independently via the existing FACS-driven landmark pipeline, then
cross-faded by the yaw-distance weight.

Result: photo-real face that **rotates** AND **deforms with FACS**
— combining the strengths of ``face_warp_2d`` (lifelike texture +
expression animation) and ``faceforge_3d_gpu`` (real 3D rotation).

Render mode: ``face_warp_3d``.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency
from faceview.vision.anatomy import (
    deform_landmarks,
    face_params_to_au_values,
    landmark_template,
    load_muscles,
)


# Atlas yaw angles in radians, matching the rendered files.
ATLAS_YAWS_RAD = [
    math.radians(-45),
    math.radians(-22),
    0.0,
    math.radians(+22),
    math.radians(+45),
]
ATLAS_FILES = [
    "face_yawn045.png", "face_yawn022.png", "face_yawp000.png",
    "face_yawp022.png", "face_yawp045.png",
]


def _atlas_dir() -> Path:
    return assets_dir() / "data" / "atlas"


@lru_cache(maxsize=1)
def _load_atlas() -> list[np.ndarray]:
    out: list[np.ndarray] = []
    d = _atlas_dir()
    for name in ATLAS_FILES:
        p = d / name
        if not p.exists():
            raise MissingDependency(
                "face atlas textures", "gpu",
                hint=(
                    "Generate the atlas via "
                    "`python -m tools.render_face_atlas` "
                    "(requires moderngl + BP3D STLs)."
                ),
            )
        from PIL import Image
        img = Image.open(p).convert("RGB")
        out.append(np.asarray(img, dtype=np.uint8))
    return out


def _bilinear_sample(tex: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample a texture at fractional coordinates."""
    h, w = tex.shape[:2]
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)
    a = tex[y0, x0].astype(np.float32)
    b = tex[y0, x1].astype(np.float32)
    c = tex[y1, x0].astype(np.float32)
    d = tex[y1, x1].astype(np.float32)
    return (a * (1 - wx)[:, None] * (1 - wy)[:, None]
            + b * wx[:, None] * (1 - wy)[:, None]
            + c * (1 - wx)[:, None] * wy[:, None]
            + d * wx[:, None] * wy[:, None])


def _pick_neighbours(yaw_rad: float) -> tuple[int, int, float]:
    """Find the two atlas indices straddling ``yaw_rad`` and the blend weight.

    Returns ``(idx_a, idx_b, t)`` where the rendered output is
    ``(1 - t) * atlas[idx_a] + t * atlas[idx_b]``.
    """
    yaws = ATLAS_YAWS_RAD
    yaw = max(yaws[0], min(yaws[-1], yaw_rad))   # clamp to atlas range
    for i in range(len(yaws) - 1):
        if yaws[i] <= yaw <= yaws[i + 1]:
            span = yaws[i + 1] - yaws[i]
            t = (yaw - yaws[i]) / max(span, 1e-9)
            return i, i + 1, float(t)
    return 0, 1, 0.0


def _warp_one(
    tex: np.ndarray,
    src_xy: np.ndarray,
    tgt_xy: np.ndarray,
    out_size: tuple[int, int],
) -> np.ndarray:
    """Reverse-mapping piecewise-affine warp via Delaunay + barycentric."""
    from scipy.spatial import Delaunay

    out_w, out_h = out_size
    tex_h, tex_w = tex.shape[:2]
    tri = Delaunay(tgt_xy)

    yy, xx = np.indices((out_h, out_w))
    out_norm = np.column_stack([
        xx.ravel().astype(np.float32) / out_w,
        yy.ravel().astype(np.float32) / out_h,
    ])
    simplex_ids = tri.find_simplex(out_norm)
    valid = simplex_ids >= 0

    transforms = tri.transform[simplex_ids]
    diffs = out_norm - transforms[:, 2, :]
    b01 = np.einsum("pij,pj->pi", transforms[:, :2, :], diffs)
    b2 = 1.0 - b01.sum(axis=1, keepdims=True)
    bary = np.concatenate([b01, b2], axis=1)
    src_verts = src_xy[tri.simplices[simplex_ids]]
    src_norm = np.einsum("pi,pij->pj", bary, src_verts)
    src_px_x = src_norm[:, 0] * tex_w
    src_px_y = src_norm[:, 1] * tex_h

    flat = np.zeros((out_h * out_w, 3), dtype=np.float32)
    flat[valid] = _bilinear_sample(tex, src_px_x[valid], src_px_y[valid])
    return flat.reshape(out_h, out_w, 3), valid.reshape(out_h, out_w)


def render_face_warp_atlas(
    params,
    size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Render a yaw-aware photo-real face by blending two atlas textures."""
    atlas = _load_atlas()

    # Convert FaceParams.yaw [-1, 1] to radians using same scale as
    # the GPU renderer (×0.6).
    yaw_rad = float(getattr(params, "yaw", 0.0)) * 0.6
    idx_a, idx_b, t = _pick_neighbours(yaw_rad)

    # Same source landmarks for both — only the texture pixels differ.
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    au_values = face_params_to_au_values(params)
    deformed = deform_landmarks(base, au_values, muscles=load_muscles())

    src_xy = np.array(base, dtype=np.float32)
    tgt_xy = np.array(deformed, dtype=np.float32)

    rgb_a, valid_a = _warp_one(atlas[idx_a], src_xy, tgt_xy, size)
    rgb_b, valid_b = _warp_one(atlas[idx_b], src_xy, tgt_xy, size)
    blended = (1 - t) * rgb_a + t * rgb_b
    valid = valid_a & valid_b

    bg_rgb = _hex_to_rgb(getattr(params, "background", "#0a0d12"))
    bg = np.array(bg_rgb, dtype=np.float32)
    out = np.where(valid[..., None], blended, bg).astype(np.uint8)
    return out[:, :, ::-1].copy()  # RGB → BGR


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
