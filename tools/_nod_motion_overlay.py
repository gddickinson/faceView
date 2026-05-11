"""Generate side-view motion-overlay images per FACEVIEW_NOD_MODE.

For each mode:
  - Render rest pose (pitch=0) and extreme nod pose (pitch=±1)
  - Compute per-pixel difference
  - Composite: rest pose as desaturated grayscale base, RED overlay
    with intensity proportional to motion at that pixel.
  - Stack down-nod and up-nod side by side for each mode.

Output: /tmp/nod_motion_overlays.png with one row per mode showing
       [rest | down-overlay | up-overlay].
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Modes to visualise — pick the most informative set.
MODES = [
    ("legacy_no_anchor",          "BUG (legacy)"),
    ("flex_anchor_-0.30",          "flex_anchored (old)"),
    ("cranium_only",               "cranium_only (distorts face)"),
    ("head_block_short_neck",      "head_block_short_neck"),
    ("head_block_neck_stretch",    "head_block_neck_stretch ★"),
    ("head_block_long_neck",       "head_block_long_neck"),
]


def _params(pitch_slider, gender="male"):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = float(np.deg2rad(90.0))  # side view
    p._camera_zoom = 1.0
    p.pitch = pitch_slider
    return p


def install_override():
    import faceview.vision.ict_face as ict

    def _resolve_override():
        cfg = json.loads(os.environ["_FACEVIEW_NOD_OVERRIDE"])
        return (tuple(cfg["pitch"]), tuple(cfg["yaw"]),
                float(cfg["fade"]), cfg["anchor"],
                float(cfg.get("anchor_fade_band", 0.15)),
                float(cfg.get("pivot_z_offset", 0.0)),
                cfg.get("single_pivot_y_norm"))

    ict._resolve_nod_mode = _resolve_override


def render_with_mode(mode_name, pitch, size=(720, 960)):
    sys.path.insert(0, "/Users/george/claude_test/faceView/tools")
    from _neck_base_sweep import CONFIGS
    cfg = next(c for c in CONFIGS if c.name == mode_name)
    os.environ["_FACEVIEW_NOD_OVERRIDE"] = json.dumps({
        "pitch": list(cfg.pitch), "yaw": list(cfg.yaw),
        "fade": cfg.fade, "anchor": cfg.anchor,
        "pivot_z_offset": cfg.pivot_z_offset,
        "anchor_fade_band": cfg.anchor_fade_band,
        "single_pivot_y_norm": cfg.single_pivot_y_norm,
    })
    install_override()
    import faceview.vision.ict_face as ict
    return ict.render_face_ict(_params(pitch), size=size)


def make_motion_overlay(rest_img, posed_img, _gain=6.0):
    """Per-pixel motion intensity map:

    - GREY base: desaturated rest pose (visible reference)
    - RED overlay: per-pixel L1 magnitude. Brighter = more motion.
      Uses both silhouette change AND interior shading change as
      proxies for motion. Apparent "motion into chest" really means
      the head TRAVELED OVER the chest in 2D screen space — the
      chest itself doesn't move.
    """
    rest = rest_img.astype(np.float32)
    pose = posed_img.astype(np.float32)
    diff = np.abs(rest - pose).sum(axis=2)
    # Threshold out aliasing noise then amplify
    diff = np.clip((diff - 8.0) * _gain, 0.0, 255.0)
    alpha = diff / 255.0
    gray = (rest[..., 0]*0.3 + rest[..., 1]*0.59 + rest[..., 2]*0.11)
    base = np.stack([gray, gray, gray], axis=2) * 0.50
    red = np.full_like(base, 0.0)
    red[..., 0] = 255.0
    red[..., 1] = 50.0
    red[..., 2] = 50.0
    out = base * (1.0 - alpha[..., None]) + red * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_silhouette_diff(rest_img, posed_img, dilate=0):
    """Symmetric cyan-rest / red-pitched diff overlay.

    For each pixel:
      - foreground in BOTH poses → neutral grey (no motion here)
      - only rest fg            → CYAN (rest silhouette only)
      - only pitched fg         → RED (pitched silhouette only)
      - background in both      → black

    Uses explicit masks instead of alpha stacking so overlapping
    pixels don't drift toward the second-painted colour.
    """
    rest = rest_img.astype(np.float32)
    pose = posed_img.astype(np.float32)
    rest_lum = rest.mean(axis=2)
    pose_lum = pose.mean(axis=2)
    rest_fg = rest_lum > 30.0
    pose_fg = pose_lum > 30.0
    both_fg = rest_fg & pose_fg
    only_rest = rest_fg & ~pose_fg
    only_pose = pose_fg & ~rest_fg
    out = np.zeros_like(rest)
    # Stationary skin: faint desaturated rest as a backdrop
    if both_fg.any():
        gray = (rest[..., 0]*0.3 + rest[..., 1]*0.59 + rest[..., 2]*0.11)
        gray3 = np.stack([gray, gray, gray], axis=2) * 0.50
        out[both_fg] = gray3[both_fg]
    # Pure cyan where rest only
    out[only_rest, 0] = 40
    out[only_rest, 1] = 200
    out[only_rest, 2] = 230
    # Pure red where pose only
    out[only_pose, 0] = 240
    out[only_pose, 1] = 60
    out[only_pose, 2] = 60
    return np.clip(out, 0, 255).astype(np.uint8)


def label(img, title, sub=None):
    pil = Image.fromarray(img).convert("RGB")
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        font_s = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        font = font_s = ImageFont.load_default()
    drw.rectangle((0, 0, pil.width, 30 + (24 if sub else 0)),
                  fill=(0, 0, 0))
    drw.text((8, 4), title, fill=(255, 240, 180), font=font)
    if sub:
        drw.text((8, 30), sub, fill=(180, 220, 240), font=font_s)
    return np.asarray(pil)


def crop_neck(img):
    h, w = img.shape[:2]
    return img[int(h*0.05):int(h*0.55), int(w*0.20):int(w*0.80)]


def load_metrics():
    with open("/tmp/neck_sweep.json") as f:
        data = json.load(f)
    return {r["name"]: r for r in data}


def main():
    metrics = load_metrics()
    rows = []
    for mode, friendly in MODES:
        print(f"rendering {mode} ...", flush=True)
        rest = crop_neck(render_with_mode(mode, 0.0))
        down = crop_neck(render_with_mode(mode, -1.0))
        up   = crop_neck(render_with_mode(mode, +1.0))
        try:
            down_overlay = make_silhouette_diff(rest, down)
            up_overlay   = make_silhouette_diff(rest, up)
        except ImportError:
            down_overlay = make_motion_overlay(rest, down)
            up_overlay   = make_motion_overlay(rest, up)

        m = metrics.get(mode, {})
        chin_dz = (abs(m.get("chin_p+1.0_dz_mean", 0))
                   + abs(m.get("chin_p-1.0_dz_mean", 0))) / 2.0
        chin_dy = (abs(m.get("chin_p+1.0_dy_mean", 0))
                   + abs(m.get("chin_p-1.0_dy_mean", 0))) / 2.0
        body_base = m.get("avg_body_BASE_NECK_mean", 0)
        sub = (f"chin_dz={chin_dz:.2f}u  chin_dy={chin_dy:.2f}u  "
               f"body_BASE={body_base:.4f}u")

        # Per-mode individual file (full-size, easier to inspect).
        row = np.hstack([
            label(rest,        f"{friendly}  REST", sub),
            label(down_overlay, "DOWN -22.9°  (cyan=rest, red=pitched)"),
            label(up_overlay,   "UP +22.9°  (cyan=rest, red=pitched)"),
        ])
        individual_path = f"/tmp/nod_overlay_{mode}.png"
        Image.fromarray(row).save(individual_path)
        print(f"  wrote {individual_path}")

        rows.append(row)

    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/nod_motion_overlays.png")
    print(f"wrote /tmp/nod_motion_overlays.png shape={out.shape}")


if __name__ == "__main__":
    main()
