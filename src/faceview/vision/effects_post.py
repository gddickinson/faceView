"""PostFX effects — modify the rendered BGR image after the renderer.

Pixel-space effects: lighting, colour shifts, scanlines, smoke,
comic shock-lines, anatomy flashes (skull / brain composited),
glitch, hologram interference, vignettes.

Each function takes ``(bgr, u, intensity)`` and returns a new BGR
uint8 array. ``u ∈ [0, 1]`` is normalised time into the effect's
duration; ``intensity ∈ [0, 1]`` is the trigger amplitude.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


# ── Colour / lighting ─────────────────────────────────────────────


def post_color_pulse(bgr: np.ndarray, u: float, intensity: float,
                      tint: tuple[float, float, float] = (0.3, 0.3, 1.4)
                      ) -> np.ndarray:
    """Multiply channels by a pulsing tint (sin envelope)."""
    env = math.sin(u * math.pi) * intensity
    factor = np.array([1 + (tint[0] - 1) * env,
                        1 + (tint[1] - 1) * env,
                        1 + (tint[2] - 1) * env], dtype=np.float32)
    out = bgr.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def post_red_flash(bgr, u, intensity):
    return post_color_pulse(bgr, u, intensity, tint=(0.4, 0.4, 1.6))


def post_blue_flash(bgr, u, intensity):
    return post_color_pulse(bgr, u, intensity, tint=(1.6, 0.6, 0.3))


def post_green_flash(bgr, u, intensity):
    return post_color_pulse(bgr, u, intensity, tint=(0.4, 1.5, 0.4))


def post_strobe(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    """High-frequency on/off flash overlay."""
    on = math.sin(u * math.pi * 12) > 0
    if not on:
        return bgr
    overlay = np.full_like(bgr, int(220 * intensity))
    return cv2.addWeighted(bgr, 1 - 0.5 * intensity, overlay, 0.5 * intensity, 0)


def post_neon_flicker(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    rng = np.random.default_rng(int(u * 10000))
    factor = 1.0 + (rng.random() - 0.5) * 0.4 * intensity
    out = bgr.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def post_fade_to_black(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    factor = 1.0 - intensity * u
    return (bgr.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)


def post_halo_burst(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    luma = bgr.max(axis=2).astype(np.float32)
    bright = (luma > 180).astype(np.uint8)[:, :, None] * bgr
    sigma = max(8.0, 24.0 * math.sin(u * math.pi))
    blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=sigma, sigmaY=sigma)
    boost = blurred.astype(np.float32) * intensity * math.sin(u * math.pi)
    return np.clip(bgr.astype(np.float32) + boost, 0, 255).astype(np.uint8)


def post_vignette(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    ys = np.linspace(-1, 1, h, dtype=np.float32)
    xs = np.linspace(-1, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(xx * xx + yy * yy)
    mask = 1.0 - np.clip(r * 0.9, 0, 1) * intensity
    return (bgr.astype(np.float32) * mask[:, :, None]).clip(0, 255).astype(np.uint8)


# ── Sci-fi ────────────────────────────────────────────────────────


def post_scanlines(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, _, _ = bgr.shape
    mask = np.ones((h, 1, 1), dtype=np.float32)
    mask[::2] = 1.0 - 0.4 * intensity
    out = bgr.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def post_pixelate(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    blocks = max(1, int(80 - 70 * intensity * math.sin(u * math.pi)))
    small = cv2.resize(bgr, (max(1, w // blocks), max(1, h // blocks)),
                        interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def post_glitch(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    rng = np.random.default_rng(int(u * 9999))
    out = bgr.copy()
    n_slices = int(8 * intensity) + 2
    for _ in range(n_slices):
        y0 = rng.integers(0, h)
        y1 = min(h, y0 + rng.integers(2, 18))
        dx = int(rng.integers(-22, 22) * intensity)
        if dx == 0:
            continue
        out[y0:y1] = np.roll(out[y0:y1], dx, axis=1)
    if intensity > 0.4:
        b, g, r = cv2.split(out)
        b = np.roll(b, int(3 * intensity), axis=1)
        r = np.roll(r, int(-3 * intensity), axis=1)
        out = cv2.merge([b, g, r])
    return out


def post_hologram(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    tinted = post_color_pulse(bgr, 0.5, intensity * 0.5,
                                 tint=(1.6, 1.3, 0.4))
    interference = np.zeros_like(tinted, dtype=np.float32)
    band_y = int((u * 4 % 1.0) * h)
    cv2.line(interference, (0, band_y), (w, band_y),
              (180, 220, 255), 8)
    out = tinted.astype(np.float32) + interference * intensity
    return np.clip(out, 0, 255).astype(np.uint8)


def post_vaporwave(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    dx = int(6 * intensity)
    r = np.roll(r, dx, axis=1)
    b = np.roll(b, -dx, axis=1)
    out = cv2.merge([b, g, r])
    h, w, _ = out.shape
    grid = np.zeros_like(out)
    spacing = 24
    for y in range(0, h, spacing):
        cv2.line(grid, (0, y), (w, y), (200, 80, 180), 1)
    for x in range(0, w, spacing):
        cv2.line(grid, (x, 0), (x, h), (200, 80, 180), 1)
    return cv2.addWeighted(out, 1.0, grid, 0.18 * intensity, 0)


def post_invert_colors(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    inv = 255 - bgr
    a = intensity
    return np.clip(bgr.astype(np.float32) * (1 - a)
                    + inv.astype(np.float32) * a, 0, 255).astype(np.uint8)


def post_chromatic_aberration(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    b, g, r = cv2.split(bgr)
    dx = int(6 * intensity)
    r = np.roll(r, dx, axis=1)
    b = np.roll(b, -dx, axis=1)
    return cv2.merge([b, g, r])


# ── Smoke / particles ─────────────────────────────────────────────


def post_smoke_rise(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    rng = np.random.default_rng(42)
    smoke = np.zeros((h, w), dtype=np.float32)
    for _ in range(60):
        cx = int(rng.integers(int(w * 0.25), int(w * 0.75)))
        cy = int(h * (1.0 - u * 1.2 + rng.random() * 0.4)) % h
        radius = int(20 + rng.random() * 35)
        cv2.circle(smoke, (cx, cy), radius, 0.4, -1, cv2.LINE_AA)
    smoke = cv2.GaussianBlur(smoke, (0, 0), sigmaX=18.0)
    smoke *= intensity
    smoke3 = np.stack([smoke, smoke * 0.95, smoke * 0.85], axis=-1) * 200
    return np.clip(bgr.astype(np.float32) + smoke3, 0, 255).astype(np.uint8)


def post_sparkle_burst(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    rng = np.random.default_rng(int(u * 100))
    n = int(40 * intensity * math.sin(u * math.pi))
    out = bgr.copy()
    for _ in range(max(0, n)):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        r = int(rng.integers(2, 6))
        col = (255, 240, 200)
        cv2.circle(out, (cx, cy), r, col, -1, cv2.LINE_AA)
        cv2.line(out, (cx - r * 2, cy), (cx + r * 2, cy), col, 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - r * 2), (cx, cy + r * 2), col, 1, cv2.LINE_AA)
    return out


def post_electric_arcs(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    rng = np.random.default_rng(int(u * 1000))
    out = bgr.copy()
    cx, cy = w // 2, int(h * 0.45)
    for _ in range(int(6 * intensity)):
        angle = rng.random() * 2 * math.pi
        length = int(rng.integers(40, 120))
        x, y = cx, cy
        pts = [(x, y)]
        for _ in range(6):
            angle += (rng.random() - 0.5) * 0.6
            x += int(math.cos(angle) * length / 6)
            y += int(math.sin(angle) * length / 6)
            pts.append((x, y))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(out, a, b, (255, 240, 180), 2, cv2.LINE_AA)
    return out


# ── Anatomy flashes ───────────────────────────────────────────────


def _real_anatomy_overlay(bgr: np.ndarray, layer: str
                              ) -> np.ndarray | None:
    """Try to render the real BP3D anatomy mesh aligned to bgr's
    head footprint. Returns None if BP3D meshes aren't on disk."""
    try:
        from faceview.vision.anatomy_overlay import (
            fit_overlay_to_head, render_anatomy_overlay,
        )
    except Exception:
        return None
    h, w, _ = bgr.shape
    raw = render_anatomy_overlay(layer, (w, h), bg_rgb=(0, 8, 16))
    if raw is None:
        return None
    return fit_overlay_to_head(raw, bgr, (0, 8, 16))


def post_skull_flash(bgr: np.ndarray, u: float, intensity: float, *,
                      features: dict | None = None) -> np.ndarray:
    """Brief skull flash. Uses the real BP3D skull mesh if available,
    falls back to a luma-mask bone-tint stand-in."""
    skull = _real_anatomy_overlay(bgr, "skull_only")
    a = intensity * math.sin(u * math.pi)
    if skull is None:
        # Fallback: tint bright luma regions bone-white.
        luma = bgr.max(axis=2).astype(np.float32)
        head_mask = (luma > np.percentile(luma, 70)).astype(np.float32)
        skull_f = np.stack([
            head_mask * 240, head_mask * 230, head_mask * 200,
        ], axis=-1).astype(np.float32)
        return np.clip(bgr.astype(np.float32) * (1 - a)
                        + skull_f * a, 0, 255).astype(np.uint8)
    # Real skull: cool-tint slightly to read against any palette,
    # then alpha-blend over the ICT face.
    skull_f = skull.astype(np.float32) * np.array([1.20, 1.05, 0.80],
                                                       dtype=np.float32)
    skull_f = np.clip(skull_f, 0, 255)
    return np.clip(bgr.astype(np.float32) * (1 - a)
                    + skull_f * a, 0, 255).astype(np.uint8)


def post_brain_flash(bgr: np.ndarray, u: float, intensity: float, *,
                      features: dict | None = None) -> np.ndarray:
    """Brief brain flash. Uses the BP3D skull as a base, restricts to
    the upper head region, and tints pink to read as brain."""
    a = intensity * math.sin(u * math.pi) * 0.85
    brain = _real_anatomy_overlay(bgr, "brain")
    h, w, _ = bgr.shape
    if brain is None:
        # Fallback path.
        luma = bgr.max(axis=2).astype(np.float32)
        head_mask = (luma > np.percentile(luma, 60)).astype(np.float32)
        yy = np.arange(h)[:, None] / h
        head_mask *= (yy < 0.5).astype(np.float32).repeat(w, axis=1)
        brain_f = np.stack([
            head_mask * 140, head_mask * 100, head_mask * 220,
        ], axis=-1).astype(np.float32)
        out = bgr.astype(np.float32) * (1 - a) + brain_f * a
    else:
        # Real anatomy: pink tint + restrict to upper-half y so it
        # reads as the brain region, not the full skull.
        yy = np.arange(h)[:, None] / h
        upper_mask = (yy < 0.55).astype(np.float32).repeat(w, axis=1)
        pink = brain.astype(np.float32) * np.array([0.85, 0.60, 1.20],
                                                        dtype=np.float32)
        pink = np.clip(pink, 0, 255) * upper_mask[:, :, None]
        out = bgr.astype(np.float32) * (1 - a) + pink * a
    # Add gyri-like low-frequency noise for texture.
    rng = np.random.default_rng(17)
    noise = rng.random((h // 8, w // 8)).astype(np.float32) * 50
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    luma = bgr.max(axis=2)
    head_mask_simple = (luma > 80).astype(np.float32)
    out += noise[:, :, None] * head_mask_simple[:, :, None] * a * 0.25
    return np.clip(out, 0, 255).astype(np.uint8)


def post_xray_flash(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    inv = 255 - bgr
    a = intensity * math.sin(u * math.pi)
    return np.clip(bgr.astype(np.float32) * (1 - a)
                    + inv.astype(np.float32) * a, 0, 255).astype(np.uint8)


# ── Comic + emotional ─────────────────────────────────────────────


def post_shock_lines(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    cx, cy = w // 2, int(h * 0.22)
    rng = np.random.default_rng(7)
    n = int(18 * intensity)
    radius = int(120 + 200 * math.sin(u * math.pi))
    for i in range(n):
        a = (i / max(1, n)) * 2 * math.pi + rng.random() * 0.05
        x1 = cx + int(math.cos(a) * 60)
        y1 = cy + int(math.sin(a) * 60)
        x2 = cx + int(math.cos(a) * radius)
        y2 = cy + int(math.sin(a) * radius)
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)
    return out


def _eye_centres(features: dict | None, h: int, w: int
                   ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve (eye_L, eye_R) pixel centres from features dict;
    fall back to fractional defaults if absent."""
    if features and "eye_L" in features and "eye_R" in features:
        return ((int(features["eye_L"][0]), int(features["eye_L"][1])),
                (int(features["eye_R"][0]), int(features["eye_R"][1])))
    return (int(w * 0.42), int(h * 0.42)), (int(w * 0.58), int(h * 0.42))


def _cheek_centres(features: dict | None, h: int, w: int
                     ) -> tuple[tuple[int, int], tuple[int, int]]:
    if features and "cheek_L" in features and "cheek_R" in features:
        return ((int(features["cheek_L"][0]), int(features["cheek_L"][1])),
                (int(features["cheek_R"][0]), int(features["cheek_R"][1])))
    return (int(w * 0.40), int(h * 0.50)), (int(w * 0.60), int(h * 0.50))


def _temple(features: dict | None, h: int, w: int, side: str
              ) -> tuple[int, int]:
    key = "brow_L" if side == "left" else "brow_R"
    if features and key in features:
        x, y = features[key]
        # Move outward + slightly down to read as "temple".
        offset = -25 if side == "left" else +25
        return int(x + offset), int(y + 6)
    return (int(w * 0.30 if side == "left" else w * 0.70), int(h * 0.30))


def _forehead(features: dict | None, h: int, w: int) -> tuple[int, int]:
    if features and "forehead" in features:
        x, y = features["forehead"]
        return int(x), int(y)
    return int(w * 0.5), int(h * 0.10)


def post_sweat_drop(bgr: np.ndarray, u: float, intensity: float, *,
                     features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    # Anchor at temple — slides downward over u.
    tx, ty = _temple(features, h, w, "left")
    drop_x = tx
    drop_y = ty + int(h * 0.4 * u)
    r = int(14 * intensity)
    if r < 2:
        return out
    cv2.ellipse(out, (drop_x, drop_y), (r, int(r * 1.4)), 0, 0, 360,
                 (220, 200, 80), -1, cv2.LINE_AA)
    cv2.circle(out, (drop_x - r // 3, drop_y - r // 2),
                max(1, r // 3), (255, 255, 255), -1, cv2.LINE_AA)
    return out


def post_heart_eyes(bgr: np.ndarray, u: float, intensity: float, *,
                     features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    eye_l, eye_r = _eye_centres(features, h, w)
    size = int(18 * intensity)
    if size < 4:
        return out
    for cx, cy in (eye_l, eye_r):
        cy = cy - int(40 * u)  # float upward
        cv2.circle(out, (cx - size // 3, cy), size // 2,
                    (90, 90, 230), -1, cv2.LINE_AA)
        cv2.circle(out, (cx + size // 3, cy), size // 2,
                    (90, 90, 230), -1, cv2.LINE_AA)
        pts = np.array([[cx - size, cy], [cx + size, cy],
                          [cx, cy + size]], dtype=np.int32)
        cv2.fillPoly(out, [pts], (90, 90, 230), cv2.LINE_AA)
    return out


def post_tears(bgr: np.ndarray, u: float, intensity: float, *,
                features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    eye_l, eye_r = _eye_centres(features, h, w)
    for ex, ey in (eye_l, eye_r):
        end_y = ey + int(h * 0.45 * u)
        cv2.line(out, (ex, ey), (ex, end_y),
                  (255, 220, 180), 3, cv2.LINE_AA)
        cv2.circle(out, (ex, end_y), int(6 * intensity),
                    (255, 230, 200), -1, cv2.LINE_AA)
    return out


def post_anger_steam(bgr: np.ndarray, u: float, intensity: float, *,
                      features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    rng = np.random.default_rng(11)
    tl = _temple(features, h, w, "left")
    tr = _temple(features, h, w, "right")
    for cx, cy0 in (tl, tr):
        for k in range(5):
            cy = cy0 - (k * 18) - int(30 * u)
            r = int((10 + k * 4) * intensity)
            cv2.circle(out, (cx + int(rng.integers(-4, 4)), cy), r,
                        (220, 220, 240), 2, cv2.LINE_AA)
    return out


def post_blush_extreme(bgr: np.ndarray, u: float, intensity: float, *,
                        features: dict | None = None) -> np.ndarray:
    a = intensity * math.sin(u * math.pi)
    if a <= 0:
        return bgr
    h, w, _ = bgr.shape
    cl, cr = _cheek_centres(features, h, w)
    overlay = bgr.copy()
    for cx, cy in (cl, cr):
        cv2.ellipse(overlay, (cx, cy), (38, 22), 0, 0, 360,
                     (80, 80, 240), -1, cv2.LINE_AA)
    return cv2.addWeighted(bgr, 1.0, overlay, a * 0.4, 0)


def post_exclamation(bgr: np.ndarray, u: float, intensity: float, *,
                       features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    fx, fy = _forehead(features, h, w)
    y = fy + int(8 * math.sin(u * math.pi * 4)) - 30
    sz = int(36 * intensity)
    if sz < 6:
        return out
    cv2.putText(out, "!", (fx - sz // 4, y + sz),
                  cv2.FONT_HERSHEY_TRIPLEX, sz / 18,
                  (60, 220, 240), max(2, sz // 8), cv2.LINE_AA)
    return out


def post_question_mark(bgr: np.ndarray, u: float, intensity: float, *,
                         features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    fx, fy = _forehead(features, h, w)
    y = fy + int(6 * math.sin(u * math.pi * 3)) - 26
    sz = int(36 * intensity)
    if sz < 6:
        return out
    cv2.putText(out, "?", (fx - sz // 4, y + sz),
                  cv2.FONT_HERSHEY_TRIPLEX, sz / 18,
                  (255, 200, 80), max(2, sz // 8), cv2.LINE_AA)
    return out


def post_tongue_out(bgr: np.ndarray, u: float, intensity: float, *,
                      features: dict | None = None) -> np.ndarray:
    """Draw a pink tongue sticking out of the mouth.

    Pairs with the PreFX of the same name that opens the jaw.
    The tongue wags side-to-side over the duration; length grows
    then retracts via the half-sine envelope.
    """
    h, w, _ = bgr.shape
    out = bgr.copy()
    if features and "mouth" in features:
        mx, my = int(features["mouth"][0]), int(features["mouth"][1])
    elif features and "chin" in features:
        cx, cy = int(features["chin"][0]), int(features["chin"][1])
        mx, my = cx, cy - int(h * 0.04)
    else:
        mx, my = int(w * 0.5), int(h * 0.55)

    env = math.sin(u * math.pi)
    if env < 0.04:
        return out
    length = int((28 + 28 * env) * intensity)
    width = int((20 + 4 * env) * intensity)
    if length < 6 or width < 4:
        return out
    # Wag side-to-side.
    wag = int(8 * math.sin(u * math.pi * 6))
    cx = mx + wag
    cy = my + length // 2

    # Tongue body — pink ellipse below the lip line.
    cv2.ellipse(out, (cx, cy), (width, length), 0, 0, 360,
                 (100, 90, 220), -1, cv2.LINE_AA)
    # Slight darker outline.
    cv2.ellipse(out, (cx, cy), (width, length), 0, 0, 360,
                 (60, 40, 160), 1, cv2.LINE_AA)
    # Centre groove (the median sulcus).
    cv2.line(out, (cx, cy - length + 4), (cx, cy + length - 4),
              (70, 55, 180), 1, cv2.LINE_AA)
    # Highlight near the tip.
    tip_y = cy + length - 3
    cv2.circle(out, (cx - width // 3, cy - length // 4),
                max(2, width // 5), (180, 160, 240), -1, cv2.LINE_AA)
    return out


def post_dark_pupils(bgr: np.ndarray, u: float, intensity: float, *,
                      features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    eye_l, eye_r = _eye_centres(features, h, w)
    r = int(14 + 10 * intensity * math.sin(u * math.pi))
    if r < 4:
        return out
    for ex, ey in (eye_l, eye_r):
        cv2.circle(out, (ex, ey), r, (0, 0, 0), -1, cv2.LINE_AA)
    return out


def post_vein_show(bgr: np.ndarray, u: float, intensity: float, *,
                    features: dict | None = None) -> np.ndarray:
    h, w, _ = bgr.shape
    out = bgr.copy()
    rng = np.random.default_rng(13)
    fx, fy = _forehead(features, h, w)
    fy = fy + 20  # closer to actual forehead skin
    for _ in range(int(4 * intensity)):
        x, y = fx + rng.integers(-40, 40), fy + rng.integers(-10, 10)
        for _ in range(5):
            nx = x + rng.integers(-12, 12)
            ny = y + rng.integers(-4, 4)
            cv2.line(out, (x, y), (nx, ny),
                      (60, 60, 180), 1, cv2.LINE_AA)
            x, y = nx, ny
    return out


def post_grayscale(bgr: np.ndarray, u: float, intensity: float) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g3 = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    a = intensity
    return cv2.addWeighted(bgr, 1 - a, g3, a, 0)


HANDLERS = {
    "red_flash":            post_red_flash,
    "blue_flash":           post_blue_flash,
    "green_flash":          post_green_flash,
    "color_pulse":          post_color_pulse,
    "strobe":               post_strobe,
    "neon_flicker":         post_neon_flicker,
    "fade_to_black":        post_fade_to_black,
    "halo_burst":           post_halo_burst,
    "scanlines":            post_scanlines,
    "pixelate":             post_pixelate,
    "glitch":               post_glitch,
    "hologram":             post_hologram,
    "vignette":             post_vignette,
    "smoke_rise":           post_smoke_rise,
    "sparkle_burst":        post_sparkle_burst,
    "electric_arcs":        post_electric_arcs,
    "shock_lines":          post_shock_lines,
    "sweat_drop":           post_sweat_drop,
    "heart_eyes":           post_heart_eyes,
    "tears":                post_tears,
    "anger_steam":          post_anger_steam,
    "blush_extreme":        post_blush_extreme,
    "exclamation":          post_exclamation,
    "question_mark":        post_question_mark,
    "skull_flash":          post_skull_flash,
    "brain_flash":          post_brain_flash,
    "xray_flash":           post_xray_flash,
    "vein_show":            post_vein_show,
    "vaporwave":            post_vaporwave,
    "dark_pupils":          post_dark_pupils,
    "tongue_out":           post_tongue_out,
    "invert_colors":        post_invert_colors,
    "grayscale":            post_grayscale,
    "chromatic_aberration": post_chromatic_aberration,
}
