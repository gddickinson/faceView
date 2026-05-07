"""Image-warp realistic face renderer.

Approach: render the BP3D photo-anatomical head once at neutral pose
through the GPU pipeline (saved as ``assets/data/neutral_face.png``),
then at runtime warp that texture per-frame using the FACS-driven
landmark deformation from :mod:`vision.anatomy`.

The warp is piecewise-affine: Delaunay over the deformed (target)
landmarks, barycentric reverse-mapping per output pixel, bilinear
sampling from the source texture. Pure NumPy + scipy — no OpenCV
required.

Why this works for realistic-looking animation:
- The source pixels come from a real medical-imaging mesh, so skin
  tone, lip shape, eye sockets, ear cartilage all read as real.
- The warp deforms small triangles, preserving local appearance while
  letting features slide in response to AU activations.

Limitations: the texture is locked to a single viewing angle (front),
so this mode doesn't rotate. Pair with ``faceforge_3d_gpu`` for
rotation; ``face_warp_2d`` for in-place expression animation.

Render mode: ``face_warp_2d``.
"""

from __future__ import annotations

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


def _texture_path() -> Path:
    return assets_dir() / "data" / "neutral_face.png"


@lru_cache(maxsize=1)
def _load_texture() -> np.ndarray:
    """Load the cached neutral face texture as RGB uint8."""
    path = _texture_path()
    if not path.exists():
        raise MissingDependency(
            "neutral_face.png", "gpu",
            hint=(
                "Generate the texture once via "
                "`python -m tools.render_neutral_face_texture` "
                "(requires moderngl + BP3D STLs in assets/anatomy_meshes/)."
            ),
        )
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _bilinear_sample(tex: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample ``tex[ys, xs]`` for fractional ys, xs."""
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
    out = (a * (1 - wx)[:, None] * (1 - wy)[:, None]
           + b * wx[:, None] * (1 - wy)[:, None]
           + c * (1 - wx)[:, None] * wy[:, None]
           + d * wx[:, None] * wy[:, None])
    return out.astype(np.uint8)


def render_face_warp(
    params,
    size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """Render a photo-real animated face by warping a neutral texture.

    Returns a BGR uint8 array. Raises :class:`MissingDependency` if
    the neutral face texture or scipy aren't available.
    """
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise MissingDependency("scipy", "gpu") from exc

    tex = _load_texture()                       # (H_tex, W_tex, 3) RGB
    tex_h, tex_w = tex.shape[:2]
    out_h, out_w = size[1], size[0]

    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]    # source landmarks in [0,1]

    # Deformed (target) landmarks in [0,1] per the AU pipeline.
    au_values = face_params_to_au_values(params)
    deformed = deform_landmarks(base, au_values, muscles=load_muscles())

    src_xy = np.array(base, dtype=np.float32)        # (N, 2)
    tgt_xy = np.array(deformed, dtype=np.float32)    # (N, 2)

    # Build Delaunay over the target landmarks.
    tri = Delaunay(tgt_xy)
    if len(tri.simplices) == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # For each output pixel, find which target triangle contains it,
    # then compute barycentric coords and map to source pixel.
    yy, xx = np.indices((out_h, out_w))
    out_norm = np.column_stack([
        xx.ravel().astype(np.float32) / out_w,
        yy.ravel().astype(np.float32) / out_h,
    ])  # (out_h*out_w, 2)

    simplex_ids = tri.find_simplex(out_norm)
    valid = simplex_ids >= 0

    # Vectorised barycentric via Delaunay's transform table.
    # tri.transform[i] is a (3, 2) matrix where rows 0..1 are the
    # affine that maps a point to barycentric (b0, b1) and row 2 is
    # the simplex's reference vertex.
    transforms = tri.transform[simplex_ids]                  # (P, 3, 2)
    diffs = out_norm - transforms[:, 2, :]                   # (P, 2)
    b01 = np.einsum("pij,pj->pi", transforms[:, :2, :], diffs)
    b2 = 1.0 - b01.sum(axis=1, keepdims=True)
    bary = np.concatenate([b01, b2], axis=1)                 # (P, 3)

    # Source landmark vertices for each pixel's simplex.
    src_verts = src_xy[tri.simplices[simplex_ids]]           # (P, 3, 2)
    src_norm = np.einsum("pi,pij->pj", bary, src_verts)      # (P, 2)

    # Map normalised source coords to texture pixel coords.
    src_px_x = src_norm[:, 0] * tex_w
    src_px_y = src_norm[:, 1] * tex_h

    # Bilinear sample.
    rgb_flat = np.zeros((out_h * out_w, 3), dtype=np.uint8)
    rgb_flat[valid] = _bilinear_sample(tex, src_px_x[valid], src_px_y[valid])

    # Background fill for invalid pixels (outside any triangle).
    bg_hex = getattr(params, "background", "#0a0d12")
    bg_rgb = _hex_to_rgb(bg_hex)
    rgb_flat[~valid] = bg_rgb

    out = rgb_flat.reshape(out_h, out_w, 3)
    return out[:, :, ::-1].copy()  # RGB → BGR


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
