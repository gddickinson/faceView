"""Dominant-color analysis (no neural net).

A tiny k-means over a downscaled crop produces 1–3 representative
colours plus their share of the region. Useful when the LLM is asked
*"what colour is the user's shirt?"* / *"is the background warm or
cool?"*. Each call is < 5 ms — there is no model load.
"""

from __future__ import annotations

import numpy as np

from faceview.core.logger import get_logger


log = get_logger("color")


# Hue ranges (OpenCV H ∈ [0, 180]) for saturated colours.
_HUE_NAMES: list[tuple[str, int, int]] = [
    ("red",     0,   10),
    ("orange",  10,  20),
    ("yellow",  20,  35),
    ("green",   35,  85),
    ("teal",    85,  100),
    ("blue",    100, 130),
    ("purple",  130, 160),
    ("pink",    160, 170),
    ("red",     170, 180),
]


def _name_color_bgr(b: int, g: int, r: int) -> str:
    """Coarse colour name for a single BGR pixel — fast hand-tuned buckets."""
    try:
        import cv2  # type: ignore
    except ImportError:
        return "?"
    px = np.array([[[b, g, r]]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), hsv[1] / 255.0, hsv[2] / 255.0
    # Achromatic checks come first.
    if v < 0.12:
        return "black"
    if s < 0.18:
        return "white" if v > 0.85 else ("grey" if v > 0.30 else "dark grey")
    # Brown: warm hues, low saturation, low value.
    if 5 < h < 25 and s < 0.55 and v < 0.45:
        return "brown"
    # Hue-only lookup for chromatic pixels.
    for name, h_lo, h_hi in _HUE_NAMES:
        if h_lo <= h <= h_hi:
            return name
    return "neutral"


def describe_color(
    frame: np.ndarray,
    region: str = "full",
    k: int = 3,
) -> str:
    """Return one short sentence describing the dominant colours."""
    from faceview.llm.vision_tool import _crop_to_region  # lazy: cycle
    try:
        import cv2  # type: ignore
    except ImportError:
        return "OpenCV isn't available — can't analyse colours."
    if frame is None:
        return "No camera frame is available right now."
    crop = _crop_to_region(frame, region)
    # Downscale for speed.
    h, w = crop.shape[:2]
    target = 96
    scale = target / max(h, w)
    if scale < 1.0:
        crop = cv2.resize(
            crop, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    flat = crop.reshape(-1, 3).astype(np.float32)
    k = max(1, min(5, k))
    # k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _ret, labels, centres = cv2.kmeans(
        flat, k, None, criteria, 1, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.ravel()
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    parts: list[str] = []
    total = float(counts.sum())
    for idx in order:
        if counts[idx] / total < 0.10:
            break
        b, g, r = (int(v) for v in centres[idx])
        name = _name_color_bgr(b, g, r)
        share = counts[idx] / total
        parts.append(f"{name} ({share:.0%})")
        if len(parts) >= 3:
            break
    log.info("color.described", region=region, colours=parts)
    return f"Dominant colours in the {region} region: " + ", ".join(parts)
