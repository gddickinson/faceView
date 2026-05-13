"""Assess each cascade mode by counting cyan/red pixels in
four quadrants of the side-view overlay:

  above-ear × back-of-head   ← motion HERE is wanted (cranium rotation)
  above-ear × front-of-face  ← motion HERE is wanted (cranium rotation)
  below-ear × back-of-neck   ← motion HERE is the bug (user's complaint)
  below-ear × front-of-face  ← motion HERE is the bug (user's complaint)

Output: ranked table + annotated overlays with the four quadrants
labelled and pixel counts shown per quadrant.

User criterion: minimise total below-ear motion pixels, with non-zero
above-ear motion pixels. Best modes have motion CONFINED to above
the bottom of the ear.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# Per-mode list — match the names in the saved overlay PNGs.
MODES = [
    ("legacy_no_anchor",           "BUG (legacy)"),
    ("flex_anchor_-0.30",           "flex_anchored"),
    ("cranium_only",                "cranium_only (face distorts)"),
    ("head_block_short_neck",       "head_block_short_neck"),
    ("head_block_neck_stretch",     "head_block_neck_stretch ★"),
    ("head_block_long_neck",        "head_block_long_neck"),
]

DOCS = "/Users/george/claude_test/faceView/docs"


def classify_pixels(img: np.ndarray, erode_thin_edges: int = 3):
    """Return cyan_mask, red_mask after eroding thin edge artifacts.

    Anti-aliased edges of the stationary body produce 1-2 pixel
    wide cyan/red bands between renders due to floating-point AA
    variation. Eroding by 1 pixel removes those false positives
    while preserving the real motion zones (which are thicker).
    """
    r = img[..., 0].astype(np.int32)
    g = img[..., 1].astype(np.int32)
    b = img[..., 2].astype(np.int32)
    bright = (r + g + b) >= 30
    cyan_mask = bright & (b > r * 1.6) & (g > r * 1.4)
    red_mask  = bright & (r > g * 1.5) & (r > b * 1.5)
    def _erode(m):
        # 4-connectivity erosion: pixel survives only if all 4
        # cardinal neighbours are also True.
        out = m.copy()
        out[:-1, :] &= m[1:, :]
        out[1:, :]  &= m[:-1, :]
        out[:, :-1] &= m[:, 1:]
        out[:, 1:]  &= m[:, :-1]
        return out

    for _ in range(erode_thin_edges):
        cyan_mask = _erode(cyan_mask)
        red_mask  = _erode(red_mask)
    return cyan_mask, red_mask


def find_ear_y_and_head_x(img, skip_top=50):
    """Auto-detect ear Y and head-back/front X using the rest pose.

    skip_top: ignore the topmost `skip_top` pixels (avoids the title
    label drawn at the top of each panel).
    """
    fg = img.mean(axis=2) > 30.0
    fg[:skip_top] = False  # mask out title bar
    rows = fg.any(axis=1)
    if not rows.any():
        return None
    top_y = int(np.argmax(rows))  # top of avatar
    widths = fg.sum(axis=1)
    h = img.shape[0]
    # Scan widths downward from top of head. Identify the local
    # maximum width (top of head), then the local minimum (neck).
    region = widths[top_y:]
    if len(region) < 50:
        return None
    smoothed = np.convolve(region, np.ones(9)/9, mode="same")
    # Find first significant width peak (head)
    peak_thresh = float(smoothed.max()) * 0.5
    # Walk forward until smoothed drops below peak_thresh after rising
    peak_idx = int(np.argmax(smoothed[: min(len(smoothed), 200)]))
    # Find first local minimum after the peak by scanning forward.
    if peak_idx + 30 < len(smoothed):
        after = smoothed[peak_idx + 30:]
        # Take first index where width is < 70% of peak
        below = np.where(after < smoothed[peak_idx] * 0.7)[0]
        if len(below) > 0:
            neck_idx = peak_idx + 30 + int(below[0])
        else:
            neck_idx = peak_idx + int(np.argmin(after)) + 30
    else:
        neck_idx = peak_idx + 10
    neck_y_abs = top_y + neck_idx
    head_height = max(20, neck_y_abs - top_y)
    chin_y_abs = top_y + int(head_height * 0.92)
    ear_bottom_y = top_y + int(head_height * 0.55)
    # Front / back X at the head-mid Y (around top + head_height*0.5)
    mid_y = top_y + head_height // 2
    if 0 <= mid_y < h:
        row = fg[mid_y]
        if row.any():
            front_x = int(row.size - 1 - np.argmax(row[::-1]))
            back_x = int(np.argmax(row))
            center_x = (front_x + back_x) // 2
        else:
            center_x = img.shape[1] // 2
    else:
        center_x = img.shape[1] // 2
    return {
        "top_y": top_y,
        "ear_bottom_y": ear_bottom_y,
        "chin_y": chin_y_abs,
        "neck_y": neck_y_abs,
        "center_x": center_x,
        "head_height": head_height,
    }


def assess_overlay(overlay_img, ear_y, center_x):
    """Count cyan and red pixels in four quadrants. The overlay's
    middle column (the REST shot) is excluded — only DOWN and UP
    overlay panels are analysed."""
    cyan, red = classify_pixels(overlay_img)
    # Skip title label area so its red/cyan text doesn't count
    cyan[:50] = False
    red[:50] = False
    h, w = overlay_img.shape[:2]
    # Build quadrant masks (using image-coord Y where 0 is top)
    ys, xs = np.indices((h, w))
    above_ear = ys < ear_y
    below_ear = ys >= ear_y
    # In this side view, the face/chin is on the LEFT of the panel
    # (camera_yaw=+90 with chin facing -X). So front = x < center_x.
    front = xs < center_x
    back  = xs >= center_x

    def cnt(mask):
        return int(mask.sum())

    return {
        "above_back_cyan":  cnt(cyan & above_ear & back),
        "above_back_red":   cnt(red  & above_ear & back),
        "above_front_cyan": cnt(cyan & above_ear & front),
        "above_front_red":  cnt(red  & above_ear & front),
        "below_back_cyan":  cnt(cyan & below_ear & back),
        "below_back_red":   cnt(red  & below_ear & back),
        "below_front_cyan": cnt(cyan & below_ear & front),
        "below_front_red":  cnt(red  & below_ear & front),
    }


def split_overlay(img):
    """Composite overlay rows are [rest | down | up] horizontally
    concatenated. Split into the three panels (each panel is
    img.width // 3 wide)."""
    h, w = img.shape[:2]
    panel_w = w // 3
    return {
        "rest": img[:, 0:panel_w],
        "down": img[:, panel_w:2*panel_w],
        "up":   img[:, 2*panel_w:3*panel_w],
    }


def annotate_quadrants(panel, ear_y, center_x, counts, title):
    pil = Image.fromarray(panel).convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    drw = ImageDraw.Draw(overlay)
    drw.line([(0, ear_y), (pil.width, ear_y)],
             fill=(255, 230, 80, 220), width=2)
    drw.line([(center_x, 0), (center_x, pil.height)],
             fill=(255, 230, 80, 220), width=2)
    try:
        font_big = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
        font_t = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        font_big = font_t = ImageFont.load_default()
    # Labels: above-back, above-front, below-back, below-front
    text_quads = [
        (10, 10, "above-back",
         counts["above_back_cyan"] + counts["above_back_red"], (255, 200, 200)),
        (center_x + 10, 10, "above-front",
         counts["above_front_cyan"] + counts["above_front_red"], (255, 200, 200)),
        (10, ear_y + 10, "below-back (BAD)",
         counts["below_back_cyan"] + counts["below_back_red"], (255, 100, 100)),
        (center_x + 10, ear_y + 10, "below-front (BAD)",
         counts["below_front_cyan"] + counts["below_front_red"], (255, 100, 100)),
    ]
    for x, y, lab, n, c in text_quads:
        drw.rectangle((x - 4, y - 2, x + 165, y + 35),
                      fill=(0, 0, 0, 200))
        drw.text((x, y), lab, fill=c, font=font_big)
        drw.text((x, y + 16), f"motion px: {n}", fill=(220, 220, 220),
                 font=font_big)
    drw.rectangle((0, pil.height - 28, pil.width, pil.height),
                  fill=(0, 0, 0, 230))
    drw.text((6, pil.height - 24), title, fill=(255, 240, 180), font=font_t)
    out = Image.alpha_composite(pil, overlay)
    return np.asarray(out.convert("RGB"))


def find_head_box(rest_panel):
    """Find tight bounding box of the AVATAR HEAD only.

    Scan widths from the top of the avatar; pick the FIRST minimum
    after the first peak, but require the minimum to be less than 60%
    of the peak (otherwise keep looking). This avoids the chest/torso
    width region that's wider than the head.
    """
    fg = rest_panel.mean(axis=2) > 30.0
    fg[:50] = False  # skip title bar
    rows = fg.any(axis=1)
    if not rows.any():
        return None
    top_y = int(np.argmax(rows))
    widths = fg.sum(axis=1)
    smoothed = np.convolve(widths, np.ones(7)/7, mode="same")
    # Find HEAD peak first — local max in top portion of image
    region_end = min(rest_panel.shape[0], top_y + 250)
    sm_head = smoothed[top_y:region_end]
    if len(sm_head) < 30:
        return None
    head_peak_idx = int(np.argmax(sm_head))
    head_peak_val = float(sm_head[head_peak_idx])
    head_peak_y = top_y + head_peak_idx
    # Find first index AFTER head peak where smoothed drops below 50% of peak
    after = smoothed[head_peak_y + 10:]
    if len(after) < 5:
        return None
    below_50 = np.where(after < head_peak_val * 0.5)[0]
    if len(below_50) > 0:
        neck_y = head_peak_y + 10 + int(below_50[0])
    else:
        neck_y = head_peak_y + 80
    head_height = max(40, neck_y - top_y)
    ear_bottom_y = top_y + int(head_height * 0.62)
    chin_y = top_y + int(head_height * 0.90)
    # Front/back X at chin level
    if 0 <= chin_y < fg.shape[0] and fg[chin_y].any():
        row_fg = fg[chin_y]
        back_x = int(np.argmax(row_fg))
        front_x = int(row_fg.size - 1 - np.argmax(row_fg[::-1]))
        center_x = (back_x + front_x) // 2
    else:
        center_x = rest_panel.shape[1] // 2
    return {
        "top_y": top_y,
        "neck_y": neck_y,
        "head_height": head_height,
        "ear_bottom_y": ear_bottom_y,
        "chin_y": chin_y,
        "center_x": center_x,
    }


def main():
    rows = []
    results = []
    # Use the rest panel of curve_back_pivot to find ear Y (rest pose
    # is identical across modes).
    ref_path = f"{DOCS}/nod_overlay_curve_back_pivot.png"
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    ref_panels = split_overlay(ref)
    # Hand-tuned positions verified against the rest panel — the
    # avatar's anatomy at the cropped resolution is consistent
    # across modes since rest pose doesn't depend on cascade settings.
    ear_y = 180     # bottom of the ear at this image scale
    center_x = 175  # head's vertical centerline X
    print(f"ear_y={ear_y}  center_x={center_x}  (hand-tuned)")

    for mode, friendly in MODES:
        path = f"{DOCS}/nod_overlay_{mode}.png"
        if not os.path.exists(path):
            print(f"skipping {mode}: file missing")
            continue
        img = np.asarray(Image.open(path).convert("RGB"))
        panels = split_overlay(img)
        down_counts = assess_overlay(panels["down"], ear_y, center_x)
        up_counts   = assess_overlay(panels["up"],   ear_y, center_x)

        # Aggregate scores
        below_motion = sum(
            (down_counts[k] + up_counts[k])
            for k in down_counts if k.startswith("below_"))
        above_motion = sum(
            (down_counts[k] + up_counts[k])
            for k in down_counts if k.startswith("above_"))
        total = below_motion + above_motion
        confined_pct = (100.0 * above_motion / max(1, total))
        results.append({
            "name": mode,
            "friendly": friendly,
            "below_total": below_motion,
            "above_total": above_motion,
            "confined_pct": confined_pct,
            "down": down_counts,
            "up": up_counts,
        })

        # Annotate down + up panels
        annot_down = annotate_quadrants(
            panels["down"], ear_y, center_x, down_counts,
            f"{friendly} | DOWN -22.9°")
        annot_up = annotate_quadrants(
            panels["up"], ear_y, center_x, up_counts,
            f"{friendly} | UP +22.9°")
        row = np.hstack([annot_down, annot_up])
        rows.append(row)

    # Composite grid
    out = np.vstack(rows)
    out_path = f"{DOCS}/nod_quadrant_assessment.png"
    Image.fromarray(out).save(out_path)
    print(f"wrote {out_path}")

    # Print ranking
    results.sort(key=lambda r: r["below_total"])
    print()
    print(f"{'mode':<32} {'below_px':>9} {'above_px':>9} "
          f"{'%confined':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['friendly']:<32} "
              f"{r['below_total']:>9} "
              f"{r['above_total']:>9} "
              f"{r['confined_pct']:>9.1f}%")

    with open(f"{DOCS}/nod_quadrant_assessment.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {DOCS}/nod_quadrant_assessment.json")


if __name__ == "__main__":
    main()
