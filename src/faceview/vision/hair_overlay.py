"""Procedural hair styles drawn over the rendered face.

The avatar's ICT head is bald by default. This module composites a
hair shape on top — multiple styles selectable from the GUI:

  none / short_cap / side_part / fringe / long_straight /
  curly_afro / mohawk / ponytail / wild_spikes / buzz

Each style is a function ``(bgr, hair_color, head_bbox) → bgr``.
Anchored on the head bbox detected from the rendered image plus
optionally the forehead/crown feature pixels for hairline placement.
A subtle gloss highlight + per-pixel jitter is added for depth.

The styles are pure cv2 / numpy — no external mesh assets needed.
The trade-off: the hair is camera-locked (doesn't 3D-rotate with
the head). Acceptable for the small head sway range we render at.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


# ── helpers ───────────────────────────────────────────────────────


def _hex_to_bgr(hex_color: str, fallback=(40, 30, 20)) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    except (ValueError, IndexError):
        return fallback


def _shade(color: tuple[int, int, int], factor: float
             ) -> tuple[int, int, int]:
    return tuple(int(min(255, max(0, c * factor))) for c in color)


def _head_top_outline(bgr: np.ndarray, bg_thr: int = 30
                        ) -> np.ndarray | None:
    """Top silhouette per column of the rendered head.

    Returns an array of length W where each entry is the topmost
    foreground pixel y for that column (or H if column empty).
    """
    h, w, _ = bgr.shape
    luma = bgr.max(axis=2)
    fg = luma > bg_thr
    if not fg.any():
        return None
    top_y = np.full(w, h, dtype=np.int32)
    for x in range(w):
        col = np.where(fg[:, x])[0]
        if len(col):
            top_y[x] = col[0]
    return top_y


# ── individual styles ─────────────────────────────────────────────


def hair_short_cap(bgr: np.ndarray, color: tuple[int, int, int],
                     head_bbox: tuple[int, int, int, int],
                     forehead_y: int | None = None) -> np.ndarray:
    """Tight close-cropped cap following the head outline."""
    out = bgr.copy()
    h, w, _ = bgr.shape
    x0, y0, x1, y1 = head_bbox
    head_h = y1 - y0
    head_w = x1 - x0
    cy = y0 + int(head_h * 0.18)
    cx = (x0 + x1) // 2

    top_y = _head_top_outline(bgr)
    if top_y is None:
        return out

    # Cap envelope: from top of head down to ~0.30 of head height.
    cap_bottom = y0 + int(head_h * 0.32)
    pts: list[tuple[int, int]] = []
    for x in range(x0, x1 + 1):
        if x < 0 or x >= w:
            continue
        ty = int(top_y[x])
        if ty < cap_bottom:
            pts.append((x, ty))
    if len(pts) < 5:
        return out
    # Add bottom edge: arc from right side back to left along
    # cap_bottom y, with a slight curve so it follows the brow.
    right_x = pts[-1][0]
    left_x = pts[0][0]
    arc_pts = []
    arc_w = right_x - left_x
    for x in range(right_x, left_x - 1, -2):
        # Concave arc at brow level.
        t = abs(x - cx) / max(1, arc_w / 2)
        y = cap_bottom - int((1 - t * t) * head_h * 0.05)
        arc_pts.append((x, y))
    poly = np.array(pts + arc_pts, dtype=np.int32)
    cv2.fillPoly(out, [poly], color, cv2.LINE_AA)
    # Subtle gloss highlight along the top.
    light = _shade(color, 1.4)
    for k, (x, y) in enumerate(pts):
        if k % 6 == 0:
            cv2.circle(out, (x, y + 3), 2, light, -1, cv2.LINE_AA)
    return out


def hair_fringe(bgr: np.ndarray, color: tuple[int, int, int],
                  head_bbox: tuple[int, int, int, int],
                  forehead_y: int | None = None) -> np.ndarray:
    """Cap + bangs falling onto the forehead."""
    out = hair_short_cap(bgr, color, head_bbox, forehead_y)
    h, w, _ = bgr.shape
    x0, y0, x1, y1 = head_bbox
    cx = (x0 + x1) // 2
    head_w = x1 - x0
    head_h = y1 - y0
    # Forehead default if not supplied.
    fy = (forehead_y if forehead_y is not None
            else y0 + int(head_h * 0.32))
    # Triangular fringe with jagged bottom.
    n = 7
    rng = np.random.default_rng(11)
    pts = []
    cap_y = y0 + int(head_h * 0.30)
    for i in range(n + 1):
        x = cx - head_w // 3 + int(i * (head_w * 2 / 3) / n)
        y = fy + int(rng.integers(-3, 12))
        pts.append((x, y))
    pts += [(cx + head_w // 3, cap_y), (cx - head_w // 3, cap_y)]
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)
    return out


def hair_side_part(bgr: np.ndarray, color: tuple[int, int, int],
                     head_bbox: tuple[int, int, int, int],
                     forehead_y: int | None = None) -> np.ndarray:
    """Parted on the screen-left, longer / fuller on the right."""
    out = hair_short_cap(bgr, color, head_bbox, forehead_y)
    h, w, _ = bgr.shape
    x0, y0, x1, y1 = head_bbox
    cx = (x0 + x1) // 2
    head_w = x1 - x0
    head_h = y1 - y0
    # Wave of hair sweeping right from the part.
    rng = np.random.default_rng(7)
    part_x = cx - head_w // 8
    pts = [(part_x, y0 + int(head_h * 0.10))]
    for i in range(1, 12):
        x = part_x + int(i * (head_w * 0.55) / 11)
        y = (y0 + int(head_h * 0.10)
              + int(math.sin(i / 11 * math.pi) * head_h * 0.18)
              + int(rng.integers(-2, 2)))
        pts.append((x, y))
    pts.append((x1 - 4, y0 + int(head_h * 0.32)))
    pts.append((part_x, y0 + int(head_h * 0.32)))
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)],
                  _shade(color, 0.92), cv2.LINE_AA)
    return out


def hair_long_straight(bgr: np.ndarray, color: tuple[int, int, int],
                          head_bbox: tuple[int, int, int, int],
                          forehead_y: int | None = None) -> np.ndarray:
    """Long straight hair down past shoulders."""
    out = hair_short_cap(bgr, color, head_bbox, forehead_y)
    h, w, _ = bgr.shape
    x0, y0, x1, y1 = head_bbox
    head_w = x1 - x0
    head_h = y1 - y0
    cx = (x0 + x1) // 2
    # Two side falls of hair.
    for sign, jitter_seed in ((-1, 3), (+1, 5)):
        rng = np.random.default_rng(jitter_seed)
        outer_x = cx + sign * (head_w // 2 - 4)
        inner_x = cx + sign * (head_w // 4)
        top_y = y0 + int(head_h * 0.25)
        bottom_y = min(h - 1, y1 + int(head_h * 0.20))
        outline = []
        for k in range(8):
            t = k / 7
            x = outer_x + sign * int(rng.integers(-3, 4))
            y = top_y + int(t * (bottom_y - top_y))
            outline.append((x, y))
        # Inner edge sweep back up.
        for k in range(8):
            t = k / 7
            x = inner_x + sign * int(rng.integers(-2, 3))
            y = bottom_y - int(t * (bottom_y - top_y))
            outline.append((x, y))
        poly = np.array(outline, dtype=np.int32)
        cv2.fillPoly(out, [poly], color, cv2.LINE_AA)
        # Streak highlights.
        light = _shade(color, 1.35)
        for k in range(0, 8, 2):
            t = k / 7
            x = (outer_x + inner_x) // 2 + sign * 4
            y = top_y + int(t * (bottom_y - top_y))
            cv2.line(out, (x, y), (x + sign * 6, y + 4),
                      light, 1, cv2.LINE_AA)
    return out


def hair_curly_afro(bgr: np.ndarray, color: tuple[int, int, int],
                      head_bbox: tuple[int, int, int, int],
                      forehead_y: int | None = None) -> np.ndarray:
    """Round dome of tight curls."""
    out = bgr.copy()
    x0, y0, x1, y1 = head_bbox
    cx = (x0 + x1) // 2
    cy = y0 + int((y1 - y0) * 0.10)
    head_w = x1 - x0
    radius = int(head_w * 0.55)
    rng = np.random.default_rng(13)
    # Big base dome.
    cv2.circle(out, (cx, cy + 8), radius, color, -1, cv2.LINE_AA)
    # Random curl bumps along the top.
    light = _shade(color, 1.35)
    dark = _shade(color, 0.7)
    n_curls = 80
    for _ in range(n_curls):
        angle = rng.random() * 2 * math.pi
        r_jitter = radius + rng.integers(-4, 12)
        bx = cx + int(math.cos(angle) * r_jitter)
        by = cy + 8 + int(math.sin(angle) * r_jitter * 0.8)
        if by > y0 + (y1 - y0) * 0.45:
            continue
        sz = int(rng.integers(6, 12))
        cv2.circle(out, (bx, by), sz, color, -1, cv2.LINE_AA)
        # Highlight + shadow on each curl.
        cv2.circle(out, (bx - sz // 3, by - sz // 3),
                    max(2, sz // 3), light, -1, cv2.LINE_AA)
        cv2.circle(out, (bx + sz // 3, by + sz // 3),
                    max(2, sz // 4), dark, -1, cv2.LINE_AA)
    return out


def hair_mohawk(bgr: np.ndarray, color: tuple[int, int, int],
                  head_bbox: tuple[int, int, int, int],
                  forehead_y: int | None = None) -> np.ndarray:
    """Central strip of hair, sides bald."""
    out = bgr.copy()
    x0, y0, x1, y1 = head_bbox
    cx = (x0 + x1) // 2
    head_h = y1 - y0
    head_w = x1 - x0
    # Strip from forehead up through crown, narrower at ends.
    strip_w = max(8, head_w // 6)
    n = 10
    pts_top = []
    pts_bottom = []
    for i in range(n + 1):
        t = i / n
        # Spikes pattern: inverted V at top.
        spike_y = y0 - int(head_h * 0.18 * math.sin(t * math.pi))
        x = cx + int((t - 0.5) * strip_w * 1.1)
        pts_top.append((x, spike_y + y0 // 8 + int(head_h * 0.04)))
    for i in range(n, -1, -1):
        t = i / n
        x = cx + int((t - 0.5) * strip_w)
        pts_bottom.append((x, y0 + int(head_h * 0.32)))
    pts = pts_top + pts_bottom
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)
    # Spike accents.
    light = _shade(color, 1.5)
    rng = np.random.default_rng(19)
    for _ in range(12):
        sx = cx + rng.integers(-strip_w // 2, strip_w // 2)
        sy_top = y0 - int(head_h * 0.18) + rng.integers(0, 8)
        sy_bot = y0 + int(head_h * 0.05)
        cv2.line(out, (sx, sy_top), (sx, sy_bot),
                  light, 1, cv2.LINE_AA)
    return out


def hair_ponytail(bgr: np.ndarray, color: tuple[int, int, int],
                    head_bbox: tuple[int, int, int, int],
                    forehead_y: int | None = None) -> np.ndarray:
    """Slick-back cap with a ponytail visible on one side."""
    out = hair_short_cap(bgr, color, head_bbox, forehead_y)
    x0, y0, x1, y1 = head_bbox
    head_h = y1 - y0
    head_w = x1 - x0
    cx = (x0 + x1) // 2
    # Pony tail trailing off to the right, behind the head.
    pty_x = x1 + 4
    pty_y0 = y0 + int(head_h * 0.20)
    pty_y1 = y0 + int(head_h * 0.65)
    pts = []
    rng = np.random.default_rng(23)
    for k in range(8):
        t = k / 7
        x = pty_x + int(t * head_w * 0.10) + rng.integers(-2, 3)
        y = pty_y0 + int(t * (pty_y1 - pty_y0))
        pts.append((x, y))
    for k in range(8):
        t = k / 7
        x = pty_x - int(head_w * 0.05) + rng.integers(-2, 3)
        y = pty_y1 - int(t * (pty_y1 - pty_y0))
        pts.append((x, y))
    cv2.fillPoly(out, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)
    # Hair tie band.
    cv2.line(out, (pty_x - 5, pty_y0 - 2),
              (pty_x + 5, pty_y0 - 2), _shade(color, 0.5),
              3, cv2.LINE_AA)
    return out


def hair_wild_spikes(bgr: np.ndarray, color: tuple[int, int, int],
                       head_bbox: tuple[int, int, int, int],
                       forehead_y: int | None = None) -> np.ndarray:
    """Anime spiky-mop style — many random outward spikes."""
    out = hair_short_cap(bgr, color, head_bbox, forehead_y)
    x0, y0, x1, y1 = head_bbox
    cx = (x0 + x1) // 2
    head_h = y1 - y0
    head_w = x1 - x0
    rng = np.random.default_rng(29)
    crown_y = y0 + int(head_h * 0.10)
    # ~30 spikes radiating outward from the crown area.
    for _ in range(30):
        angle_deg = rng.uniform(-100, -80) + rng.uniform(-50, 50)
        # Bias upward (270° = up).
        angle = math.radians(angle_deg)
        # Origin near crown.
        x0p = cx + rng.integers(-head_w // 4, head_w // 4)
        y0p = crown_y + rng.integers(0, 8)
        length = int(head_h * 0.18) + rng.integers(0, head_h // 8)
        x1p = x0p + int(math.cos(angle) * length)
        y1p = y0p + int(math.sin(angle) * length)
        thickness = max(2, rng.integers(3, 7))
        cv2.line(out, (x0p, y0p), (x1p, y1p), color,
                  thickness, cv2.LINE_AA)
        # Highlight tip.
        cv2.circle(out, (x1p, y1p), 2, _shade(color, 1.3),
                    -1, cv2.LINE_AA)
    return out


def hair_buzz(bgr: np.ndarray, color: tuple[int, int, int],
                head_bbox: tuple[int, int, int, int],
                forehead_y: int | None = None) -> np.ndarray:
    """Shaved-head stubble — speckled pattern over the head outline."""
    out = bgr.copy()
    h, w, _ = bgr.shape
    luma = bgr.max(axis=2)
    head_mask = (luma > 30).astype(np.uint8)
    if not head_mask.any():
        return out
    x0, y0, x1, y1 = head_bbox
    head_h = y1 - y0
    # Restrict to upper 40 % of head.
    upper = np.zeros_like(head_mask)
    upper[y0:y0 + int(head_h * 0.40)] = head_mask[y0:y0 + int(head_h * 0.40)]
    rng = np.random.default_rng(31)
    ys, xs = np.where(upper > 0)
    if not len(xs):
        return out
    n = min(len(xs), 600)
    pick = rng.choice(len(xs), n, replace=False)
    for i in pick:
        x, y = xs[i], ys[i]
        cv2.circle(out, (int(x), int(y)), 1, color, -1)
    return out


# ── registry ──────────────────────────────────────────────────────


STYLES = {
    "none":          None,
    "short_cap":     hair_short_cap,
    "fringe":        hair_fringe,
    "side_part":     hair_side_part,
    "long_straight": hair_long_straight,
    "curly_afro":    hair_curly_afro,
    "mohawk":        hair_mohawk,
    "ponytail":      hair_ponytail,
    "wild_spikes":   hair_wild_spikes,
    "buzz":          hair_buzz,
}


def list_styles() -> list[str]:
    return list(STYLES.keys())


def apply_hair(bgr: np.ndarray, style: str, color_hex: str,
                head_bbox: tuple[int, int, int, int] | None,
                forehead_y: int | None = None) -> np.ndarray:
    if style == "none" or style not in STYLES:
        return bgr
    handler = STYLES[style]
    if handler is None or head_bbox is None:
        return bgr
    color = _hex_to_bgr(color_hex)
    try:
        return handler(bgr, color, head_bbox, forehead_y)
    except Exception:
        return bgr
