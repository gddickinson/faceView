"""Parameter sweep over cervical-cascade configurations. For each
config, render the avatar at pitch=+1.0 and measure displacement of
TRACKED base-of-neck vertices (locked by ID at rest pose).

Goal: find the config(s) that minimise base motion while preserving
visible upper-neck flex.

Tracked vert sets (body mesh, IDs frozen at rest pose):
  - BASE     = verts in y_norm ∈ [-0.50, -0.30]  (C7..C5 band:
               clavicle + neck-base — should be ~0)
  - JUNCTION = verts in y_norm ∈ [-0.30, -0.20]  (C5..C4 band:
               transition zone)
  - UPPER    = verts in y_norm ∈ [-0.20, +0.00]  (C3..C1 band:
               should flex VISIBLY)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Configurations to sweep
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    pitch: tuple
    yaw: tuple
    fade: float
    anchor: float | None  # y_norm anchor threshold, None = off
    pivot_z_offset: float = 0.0  # head_h units, negative = back of neck
    anchor_fade_band: float = 0.15  # head_h units
    single_pivot_y_norm: float | None = None  # head_h units

    def env_install(self):
        os.environ["_FACEVIEW_NOD_OVERRIDE"] = json.dumps({
            "pitch": list(self.pitch),
            "yaw": list(self.yaw),
            "fade": self.fade,
            "anchor": self.anchor,
            "pivot_z_offset": self.pivot_z_offset,
            "anchor_fade_band": self.anchor_fade_band,
            "single_pivot_y_norm": self.single_pivot_y_norm,
        })


# 12 anchor-aware configs spanning bend distributions + anchor depths.
# Each pitch tuple sums to ≈1.0 in deltas so the chin rotates the
# full angle.
CONFIGS: list[Config] = [
    # --- legacy + variations ---
    Config("legacy_no_anchor",
        pitch=(1.00, 0.98, 0.85, 0.55, 0.25, 0.10, 0.04, 0.015, 0.005,
                 0.002, 0.0, 0.0),
        yaw=(1.00, 0.98, 0.55, 0.30, 0.15, 0.08, 0.03, 0.01, 0.003,
                 0.001, 0.0, 0.0),
        fade=1.5, anchor=None),
    Config("legacy_anchor_-0.30",
        pitch=(1.00, 0.98, 0.85, 0.55, 0.25, 0.10, 0.04, 0.015, 0.005,
                 0.002, 0.0, 0.0),
        yaw=(1.00, 0.98, 0.55, 0.30, 0.15, 0.08, 0.03, 0.01, 0.003,
                 0.001, 0.0, 0.0),
        fade=1.5, anchor=-0.30),
    Config("legacy_anchor_-0.20",
        pitch=(1.00, 0.98, 0.85, 0.55, 0.25, 0.10, 0.04, 0.015, 0.005,
                 0.002, 0.0, 0.0),
        yaw=(1.00, 0.98, 0.55, 0.30, 0.15, 0.08, 0.03, 0.01, 0.003,
                 0.001, 0.0, 0.0),
        fade=1.5, anchor=-0.20),
    # --- sharp profiles (current sharper / sharp_anchored / flex_anchored) ---
    Config("sharper_no_anchor",
        pitch=(1.00, 0.95, 0.65, 0.20, 0.05, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.40, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=None),
    Config("sharp_anchor_-0.25",
        pitch=(1.00, 0.95, 0.65, 0.20, 0.05, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.40, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.25),
    Config("flex_anchor_-0.30",
        pitch=(1.00, 0.95, 0.65, 0.30, 0.08, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.45, 0.20, 0.04, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.5, anchor=-0.30),
    # --- extremes: bend concentrated at the very top ---
    Config("top_only_no_anchor",
        # All rotation at the SKULL-C1 / C1-C2 disc, near-zero below.
        pitch=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=None),
    Config("top_only_anchor_-0.10",
        pitch=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=-0.10),
    Config("top_only_anchor_-0.05",
        # Anchor pulled UP almost to the chin — extreme isolation.
        pitch=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.50, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=-0.05),
    # --- single-pivot rigid head (no cascade) ---
    Config("single_pivot_C1",
        # All rotation happens at C1-C2 disc only. Above = full skull
        # rotation, below = nothing. Anchor as belt-and-braces.
        pitch=(1.00, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.4, anchor=-0.05),
    # --- balanced approach: medium flex + strict anchor ---
    Config("balanced_anchor_-0.20",
        pitch=(1.00, 0.95, 0.55, 0.15, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.35, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.8, anchor=-0.20),
    Config("balanced_anchor_-0.15",
        pitch=(1.00, 0.95, 0.55, 0.15, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.35, 0.08, 0.01, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.8, anchor=-0.15),
    # --- new variants for tighter base + visible flex ---
    Config("top_pivot_anchor_-0.15",
        # Single C1-C2 pivot, anchor at -0.15 (clamps everything
        # below chin band aggressively).
        pitch=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.5, anchor=-0.15),
    Config("top_pivot_anchor_-0.10",
        pitch=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.5, anchor=-0.10),
    Config("top_pivot_anchor_-0.20",
        pitch=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.80, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.5, anchor=-0.20),
    Config("upper_cervical_anchor_-0.15",
        # Lets C1-C3 take all the rotation, anchor strict at -0.15.
        pitch=(1.00, 0.90, 0.40, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.90, 0.30, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=-0.15),
    Config("upper_cervical_anchor_-0.20",
        pitch=(1.00, 0.90, 0.40, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.90, 0.30, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=-0.20),
    Config("upper_cervical_anchor_-0.10",
        pitch=(1.00, 0.90, 0.40, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.90, 0.30, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.6, anchor=-0.10),
    # --- back-of-neck pivot family with TIGHT anchors ---
    Config("curve_back_pivot",
        pitch=(1.00, 0.92, 0.80, 0.62, 0.40, 0.22, 0.08, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.92, 0.70, 0.45, 0.25, 0.12, 0.04, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.25,
        pivot_z_offset=-0.20, anchor_fade_band=0.10),
    Config("low_pivot_block",
        pitch=(1.00, 0.99, 0.97, 0.93, 0.80, 0.50, 0.15, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.99, 0.95, 0.85, 0.65, 0.35, 0.10, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.8, anchor=-0.22,
        pivot_z_offset=-0.25, anchor_fade_band=0.08),
    Config("neck_curve_strict",
        pitch=(1.00, 0.88, 0.72, 0.52, 0.30, 0.12, 0.03, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.88, 0.62, 0.38, 0.18, 0.06, 0.02, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.22,
        pivot_z_offset=-0.18, anchor_fade_band=0.10),
    # Centerline pivot version (NO back offset) for A/B
    Config("curve_centerline_pivot",
        pitch=(1.00, 0.92, 0.80, 0.62, 0.40, 0.22, 0.08, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.92, 0.70, 0.45, 0.25, 0.12, 0.04, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.25,
        pivot_z_offset=0.0, anchor_fade_band=0.10),
    # Sweep different pivot Z depths with same cumul
    Config("curve_back_-0.15",
        pitch=(1.00, 0.92, 0.80, 0.62, 0.40, 0.22, 0.08, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.92, 0.70, 0.45, 0.25, 0.12, 0.04, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.25,
        pivot_z_offset=-0.15, anchor_fade_band=0.10),
    Config("curve_back_-0.30",
        pitch=(1.00, 0.92, 0.80, 0.62, 0.40, 0.22, 0.08, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.92, 0.70, 0.45, 0.25, 0.12, 0.04, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0, anchor=-0.25,
        pivot_z_offset=-0.30, anchor_fade_band=0.10),
    Config("low_block_-0.30",
        # low_pivot_block with even deeper Z pivot
        pitch=(1.00, 0.99, 0.97, 0.93, 0.80, 0.50, 0.15, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.99, 0.95, 0.85, 0.65, 0.35, 0.10, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.8, anchor=-0.22,
        pivot_z_offset=-0.30, anchor_fade_band=0.08),
    # --- cranium-only modes (single rotation at ear level) ---
    Config("cranium_only",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=+0.28, pivot_z_offset=-0.20,
        anchor_fade_band=0.04, single_pivot_y_norm=+0.30),
    Config("cranium_high_pivot",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=+0.38, pivot_z_offset=-0.20,
        anchor_fade_band=0.04, single_pivot_y_norm=+0.40),
    Config("cranium_soft_seam",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=+0.20, pivot_z_offset=-0.20,
        anchor_fade_band=0.10, single_pivot_y_norm=+0.30),
    Config("cranium_pivot_0.25",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=+0.23, pivot_z_offset=-0.15,
        anchor_fade_band=0.06, single_pivot_y_norm=+0.25),
    Config("cranium_pivot_0.20",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=+0.18, pivot_z_offset=-0.15,
        anchor_fade_band=0.06, single_pivot_y_norm=+0.20),
    # --- head moves as a block, neck stretches ---
    Config("head_block_neck_stretch",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=-0.30, pivot_z_offset=-0.20,
        anchor_fade_band=0.20, single_pivot_y_norm=+0.30),
    Config("head_block_short_neck",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=-0.22, pivot_z_offset=-0.20,
        anchor_fade_band=0.12, single_pivot_y_norm=+0.30),
    Config("head_block_long_neck",
        pitch=(1.0,)*12, yaw=(1.0,)*12, fade=0.5,
        anchor=-0.50, pivot_z_offset=-0.20,
        anchor_fade_band=0.40, single_pivot_y_norm=+0.30),
]


# ---------------------------------------------------------------------------
# Cascade override — patches _resolve_nod_mode at runtime
# ---------------------------------------------------------------------------

def install_cascade_override():
    """Monkey-patch _resolve_nod_mode to read from _FACEVIEW_NOD_OVERRIDE."""
    import faceview.vision.ict_face as ict

    def _resolve_override():
        cfg = json.loads(os.environ["_FACEVIEW_NOD_OVERRIDE"])
        return (tuple(cfg["pitch"]), tuple(cfg["yaw"]),
                float(cfg["fade"]), cfg["anchor"],
                float(cfg.get("anchor_fade_band", 0.15)),
                float(cfg.get("pivot_z_offset", 0.0)),
                cfg.get("single_pivot_y_norm"))

    ict._resolve_nod_mode = _resolve_override


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _params(pitch_slider: float, gender: str = "male"):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_zoom = 0.55
    p.pitch = pitch_slider
    return p


def capture_all_calls(pitch_slider, gender="male"):
    """Run a single render; capture before/after for ALL cascade
    calls (ICT head + body mesh). Returns dict by mesh name."""
    import faceview.vision.ict_face as ict
    calls = []
    orig = ict._apply_cervical_cascade

    def hook(verts, yaw, p_in, roll, chin_y, head_h, pivot_z=0.0):
        before = verts.copy()
        out = orig(verts, yaw, p_in, roll, chin_y, head_h, pivot_z=pivot_z)
        calls.append((before, out, chin_y, head_h))
        return out

    ict._apply_cervical_cascade = hook
    try:
        ict.render_face_ict(_params(pitch_slider, gender), size=(360, 640))
    finally:
        ict._apply_cervical_cascade = orig
    if len(calls) < 2:
        return None
    # ICT head mesh = LARGER one (~26k verts); body = SMALLER (~7k).
    sizes = [c[0].shape[0] for c in calls]
    head_idx = int(np.argmax(sizes))
    body_idx = int(np.argmin(sizes))
    return {"head": calls[head_idx], "body": calls[body_idx]}


def find_tracked_ids(rest_verts, chin_y, head_h):
    """Return dict of tracked vert ID arrays per band, FROZEN at
    rest pose. We use these same IDs across every cascade config."""
    y_norm = (rest_verts[:, 1] - chin_y) / head_h
    return {
        "ABOVE_CHIN":  np.where((y_norm >= +0.00) & (y_norm < +0.05))[0],
        "UPPER_NECK":  np.where((y_norm >= -0.10) & (y_norm < +0.00))[0],
        "MID_NECK":    np.where((y_norm >= -0.20) & (y_norm < -0.10))[0],
        "LOWER_NECK":  np.where((y_norm >= -0.30) & (y_norm < -0.20))[0],
        "BASE_NECK":   np.where((y_norm >= -0.40) & (y_norm < -0.30))[0],
        "CLAVICLE":    np.where((y_norm >= -0.55) & (y_norm < -0.40))[0],
        "UPPER_TORSO": np.where((y_norm >= -1.00) & (y_norm < -0.55))[0],
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    install_cascade_override()

    # First pass: install legacy, capture rest pose, identify tracked IDs.
    CONFIGS[0].env_install()
    calls = capture_all_calls(0.0)
    if calls is None:
        print("FAIL: no cascade calls captured")
        return
    rest_body, _, chin_y, head_h = calls["body"]
    rest_head, _, _, _ = calls["head"]
    tracked_body = find_tracked_ids(rest_body, chin_y, head_h)
    tracked_head = find_tracked_ids(rest_head, chin_y, head_h)
    print(f"head_h={head_h:.3f}  chin_y={chin_y:.3f}")
    print("tracked BODY-mesh vert counts per band:")
    for name, ids in tracked_body.items():
        print(f"  {name:<12} {len(ids):>5}")
    print("tracked ICT-mesh vert counts per band:")
    for name, ids in tracked_head.items():
        print(f"  {name:<12} {len(ids):>5}")
    print()

    pitches = [+1.0, -1.0]

    # Sweep both meshes — capture total disp AND per-axis components
    # so we can see chin's front-to-back (Z) sweep separately.
    results = []
    chin_track = None  # IDs of chin verts for arc measurement
    for cfg in CONFIGS:
        cfg.env_install()
        row = {"name": cfg.name, "anchor": cfg.anchor, "fade": cfg.fade,
               "pivot_z_offset": cfg.pivot_z_offset}
        for pitch in pitches:
            calls = capture_all_calls(pitch)
            if calls is None:
                continue
            for mesh_name, tracked in (("body", tracked_body),
                                         ("head", tracked_head)):
                v0, v1, _, _ = calls[mesh_name]
                delta = v1 - v0
                disp = np.linalg.norm(delta, axis=1)
                for band, ids in tracked.items():
                    if len(ids) == 0:
                        continue
                    d = disp[ids]
                    key = f"{mesh_name}_p{pitch:+.1f}_{band}"
                    row[key + "_mean"] = float(d.mean())
                    row[key + "_max"] = float(d.max())
                    # per-axis means (signed so direction shows)
                    dx = delta[ids, 0]
                    dy = delta[ids, 1]
                    dz = delta[ids, 2]
                    row[key + "_dz_mean"] = float(dz.mean())
                    row[key + "_dy_mean"] = float(dy.mean())
            # ICT chin verts: top-most Y in head mesh (mean of top 50)
            v0_head, v1_head, chin_y_, head_h_ = calls["head"]
            y_norm_head = (v0_head[:, 1] - chin_y_) / head_h_
            chin_mask = (y_norm_head >= -0.02) & (y_norm_head <= +0.05)
            if chin_mask.any():
                cd = (v1_head - v0_head)[chin_mask]
                row[f"chin_p{pitch:+.1f}_dz_mean"] = float(cd[:, 2].mean())
                row[f"chin_p{pitch:+.1f}_dy_mean"] = float(cd[:, 1].mean())
                row[f"chin_p{pitch:+.1f}_disp_mean"] = float(
                    np.linalg.norm(cd, axis=1).mean())
        results.append(row)

    # Average across pitches for ranking
    for row in results:
        for mesh_name, tracked in (("body", tracked_body),
                                     ("head", tracked_head)):
            for band in tracked:
                pmean = [row.get(f"{mesh_name}_p+1.0_{band}_mean", 0.0),
                         row.get(f"{mesh_name}_p-1.0_{band}_mean", 0.0)]
                pmax = [row.get(f"{mesh_name}_p+1.0_{band}_max", 0.0),
                        row.get(f"{mesh_name}_p-1.0_{band}_max", 0.0)]
                row[f"avg_{mesh_name}_{band}_mean"] = float(np.mean(pmean))
                row[f"avg_{mesh_name}_{band}_max"] = float(np.max(pmax))

    # Rank by COMBINED base motion (body neck-base + ICT mesh lower
    # collar region, which is the visible junction).
    def combined_base(r):
        return (r["avg_body_BASE_NECK_mean"]
                + r.get("avg_head_BASE_NECK_mean", 0.0)
                + r.get("avg_head_UPPER_TORSO_mean", 0.0))

    results.sort(key=combined_base)

    print(f"{'config':<28} {'anc':>5} {'piv_z':>6}  "
          f"{'body_BASE':>9} {'head_BASE':>9}  "
          f"{'chin_disp':>9} {'chin_dz':>8} {'chin_dy':>8}  "
          f"{'body_UN':>8}")
    print("-" * 115)
    for row in results:
        anc = f"{row['anchor']:+.2f}" if row["anchor"] is not None else " off"
        pz = f"{row.get('pivot_z_offset',0.0):+.2f}"
        chin_d = (abs(row.get("chin_p+1.0_disp_mean", 0))
                  + abs(row.get("chin_p-1.0_disp_mean", 0))) / 2.0
        chin_dz = (abs(row.get("chin_p+1.0_dz_mean", 0))
                   + abs(row.get("chin_p-1.0_dz_mean", 0))) / 2.0
        chin_dy = (abs(row.get("chin_p+1.0_dy_mean", 0))
                   + abs(row.get("chin_p-1.0_dy_mean", 0))) / 2.0
        print(f"{row['name']:<28} {anc:>5} {pz:>6}  "
              f"{row['avg_body_BASE_NECK_mean']:>9.5f}"
              f" {row.get('avg_head_BASE_NECK_mean',0):>9.5f}  "
              f"{chin_d:>9.3f} {chin_dz:>8.3f} {chin_dy:>8.3f}  "
              f"{row.get('avg_body_UPPER_NECK_mean',0):>8.4f}")

    # Save raw results for downstream tools
    with open("/tmp/neck_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote /tmp/neck_sweep.json")


if __name__ == "__main__":
    main()
