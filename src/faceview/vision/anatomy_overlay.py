"""Render BP3D anatomy meshes as transient overlays.

Used by the skull_flash / brain_flash anatomy effects to composite
real BP3D skull or brain pixels over the ICT face for a brief
flash. Approximate alignment is fine — these are < 1 second
overlays driven by a sin envelope, so any sub-pixel mismatch
washes out.

Caches:
- ``_anatomy_renderer`` shares the ICT moderngl context (avoids
  the moderngl-5 multi-context limitation).
- ``_overlay_cache`` keyed on (layer, w, h, yaw_q, pitch_q) so we
  don't re-rasterise the BP3D mesh every frame at the same pose.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

import cv2
import numpy as np


_overlay_cache: dict[tuple, np.ndarray] = {}


@lru_cache(maxsize=1)
def _anatomy_renderer():
    """BP3D _GpuRenderer sharing the ICT moderngl context."""
    from faceview.vision.gpu_renderer import _GpuRenderer
    from faceview.vision.ict_face import _ensure_renderer
    ict = _ensure_renderer()
    return _GpuRenderer(ctx=ict.ctx)


@lru_cache(maxsize=8)
def _layer_specs(layer: str) -> list:
    """Resolve the BP3D mesh spec list for a named layer."""
    from faceview.vision.anatomy_catalog import MeshSpec, specs_for_layer_set
    from faceview.vision.anatomy_meshes import list_available_meshes
    avail = set(list_available_meshes())
    if layer == "skull_only":
        return [s for s in specs_for_layer_set("skull_only")
                 if s.fma in avail and s.category == "bone"]
    if layer == "brain":
        return _brain_specs(avail)
    raise ValueError(f"unknown anatomy layer: {layer}")


def _brain_specs(avail: set[str]) -> list:
    """Build a MeshSpec list from assets/config/anatomy/brain.json.

    BP3D brain meshes (81 in total: cerebellum, pons, medulla, ~22
    cerebral gyri, corpus callosum, optic chiasm, etc.) — provides
    the actual brain anatomy for the brain_flash effect rather than
    a procedural cartoon overlay.
    """
    import json
    from faceview.assets import assets_dir
    from faceview.vision.anatomy_catalog import MeshSpec

    path = assets_dir() / "config" / "anatomy" / "brain.json"
    if not path.exists():
        return []
    try:
        defs = json.loads(path.read_text())
    except Exception:
        return []
    specs = []
    for i, d in enumerate(defs):
        fma = d.get("stl")
        if not fma or fma not in avail:
            continue
        col = int(d.get("color", 13413034))
        rgb = ((col >> 16) & 0xff, (col >> 8) & 0xff, col & 0xff)
        specs.append(MeshSpec(
            fma=fma, name=d.get("name", fma), category="brain",
            color=rgb, opacity=1.0, shininess=4.0, draw_order=i,
        ))
    return specs


def render_anatomy_overlay(
    layer: str,
    size: tuple[int, int],
    yaw: float = 0.0, pitch: float = 0.0,
    bg_rgb: tuple[int, int, int] = (0, 0, 0),
) -> Optional[np.ndarray]:
    """Render a BP3D anatomy layer at the given pose. Returns BGR
    image or None if BP3D meshes aren't on disk.

    Cached per (layer, size, quantised pose) so repeated calls
    during a single flash are free.
    """
    try:
        from faceview.vision.anatomy_meshes import meshes_available
        if not meshes_available():
            return None
    except Exception:
        return None

    # Quantise pose to ~5-degree buckets for cache reuse.
    yq = round(yaw / 0.087) * 0.087
    pq = round(pitch / 0.087) * 0.087
    key = (layer, size, round(yq, 3), round(pq, 3))
    if key in _overlay_cache:
        return _overlay_cache[key]

    try:
        specs = _layer_specs(layer)
        if not specs:
            return None
        rend = _anatomy_renderer()
        img = rend.render(specs, size, yaw=yq, pitch=pq, bg=bg_rgb)
    except Exception:
        return None

    # Cap cache size at ~16 entries.
    if len(_overlay_cache) > 16:
        _overlay_cache.pop(next(iter(_overlay_cache)))
    _overlay_cache[key] = img
    return img


def fit_overlay_to_head(overlay: np.ndarray, target_bgr: np.ndarray,
                          bg_rgb: tuple[int, int, int],
                          *, scale_factor: float = 1.0,
                          ) -> np.ndarray:
    """Crop+rescale BP3D overlay so its head footprint matches the
    target ICT image's head footprint. Approximate fit via bbox of
    foreground pixels.

    ``scale_factor`` shrinks (<1.0) or enlarges (>1.0) the overlay
    relative to the bbox-match. For skull/brain anatomy that should
    sit *inside* the skin envelope, pass ~0.78–0.85 so the bone
    outline reads as recessed under the skin.
    """
    h, w, _ = target_bgr.shape
    bg = np.array(bg_rgb, dtype=np.float32)

    def fg_bbox(img):
        diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :],
                                axis=2)
        ys, xs = np.where(diff > 25.0)
        if not len(xs):
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    a_box = fg_bbox(overlay)
    i_box = fg_bbox(target_bgr)
    if a_box is None or i_box is None:
        return cv2.resize(overlay, (w, h))

    ax0, ay0, ax1, ay1 = a_box
    ix0, iy0, ix1, iy1 = i_box
    a_w = max(1, ax1 - ax0 + 1)
    i_w = max(1, ix1 - ix0 + 1)
    scale = (i_w / a_w) * float(scale_factor)
    a_crop = overlay[ay0:ay1 + 1, ax0:ax1 + 1]
    new_w = max(1, int(a_crop.shape[1] * scale))
    new_h = max(1, int(a_crop.shape[0] * scale))
    a_resized = cv2.resize(a_crop, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
    icx = (ix0 + ix1) // 2
    icy = (iy0 + iy1) // 2
    tx = icx - new_w // 2
    ty = icy - new_h // 2
    out = np.zeros_like(target_bgr)
    out[:, :] = bg.astype(np.uint8)
    sx0 = max(0, -tx)
    sy0 = max(0, -ty)
    dx0 = max(0, tx)
    dy0 = max(0, ty)
    cw = max(0, min(new_w - sx0, w - dx0))
    ch = max(0, min(new_h - sy0, h - dy0))
    if cw > 0 and ch > 0:
        out[dy0:dy0 + ch, dx0:dx0 + cw] = a_resized[sy0:sy0 + ch,
                                                       sx0:sx0 + cw]
    return out
