"""Drive each actual body effect handler at peak intensity, detect
flyaway voxels, and reclassify via rest-pose nearest-neighbour
voting (same logic as ``isolated_voxel_relabel`` but exercises the
REAL multi-joint effects from ``faceview.vision.effects_pre`` rather
than synthetic single-joint test poses).

Some effects request shoulder rolls past 90° (e.g. ``stretch_up``
asks for 2.10 rad of roll). Combined with body-pitch propagation
they expose mis-labelled voxels that a pure-single-joint test
misses.

Outputs (under ``docs/effect_relabel/<gender>/``):
  - eff_<name>.png        per-effect diagnostic
  - grid_effects.png      composite grid

With ``--apply``: writes new high-confidence reassignments to
``src/faceview/assets/body_label_overrides_<gender>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw  # noqa: E402

from tools.extreme_pose_relabel import (  # noqa: E402
    _make_neutral_params, _capture_rig_io,
    _project_with_camera, _label_image,
    _knn_majority_label, _PALETTE_COLORS,
)
from tools.isolated_voxel_relabel import _nn_distance, BPF_NAMES  # noqa: E402


# Effects to exercise. Skip eye/mouth/head-only effects — they
# don't drive the body rig and aren't useful for flyaway detection.
BODY_EFFECTS = [
    "body_bow", "body_lean_back", "body_twist_left", "body_twist_right",
    "body_lean_left", "body_lean_right",
    "wave_left", "wave_right",
    "arms_up", "arms_out",
    "kick_left", "kick_right",
    "squat",
    "shrug", "arms_crossed", "hands_on_hips",
    "point_left", "point_right", "thinking",
    "clap", "stretch_up", "salute", "curtsy",
    "lunge_left", "lunge_right", "jump",
]


def _peak_render(gender: str, effect_name: str, size=(360, 640)):
    """Render the effect at u=0.5 (peak) and intensity=1.0."""
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS

    handler = HANDLERS.get(f"pre_{effect_name}") or HANDLERS.get(effect_name)
    if handler is None:
        return None

    p = _make_neutral_params(gender)
    handler(p, 0.5, 1.0)
    return _capture_rig_io(lambda: render_face_ict(p, size=tuple(size)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--nn-threshold", type=float, default=4.0)
    ap.add_argument("--isolation-ratio", type=float, default=3.0)
    ap.add_argument("--min-disp", type=float, default=2.0)
    ap.add_argument("--min-poses", type=int, default=1)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--min-agreement", type=int, default=5)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--effects", nargs="*", default=None,
                     help="Subset of effect names; defaults to the full "
                          "body-effect list.")
    ap.add_argument("--size", type=int, nargs=2, default=(360, 640))
    args = ap.parse_args()

    from faceview.assets import assets_dir
    out_dir = Path(f"docs/effect_relabel/{args.gender}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Baseline rest-pose NN distances.
    from faceview.vision.ict_face import render_face_ict
    p = _make_neutral_params(args.gender)
    cap = _capture_rig_io(lambda: render_face_ict(p, size=tuple(args.size)))
    rest = cap["rest"]
    fine_eff_ref = cap["fine_eff"]
    n_verts = len(rest)
    rest_nn = _nn_distance(rest)
    print(f"Loaded labels: {n_verts} verts")
    print(f"Rest NN-dist  median={np.median(rest_nn):.3f}  "
            f"99%={np.percentile(rest_nn, 99):.3f}")

    # ── Static rest-pose label-island detector ───────────────────
    # For each vert, find its k nearest REST-POSE neighbours.
    # Verts whose current label disagrees with the strong majority
    # of their k-NN are systematic mislabels (e.g. 600 hand_R-tagged
    # voxels physically at the left foot — their nearest rest-pose
    # neighbours are all foot_L). This catches the bulk of bad
    # labels in a single pass without needing extreme poses.
    print("Computing rest-pose k-NN label-island detector …")
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(rest)
        _, idx = tree.query(rest, k=args.knn_k + 1)
        nbr_idx = idx[:, 1:]  # drop self
    except ImportError:
        nbr_idx = None
    label_island = np.zeros(n_verts, dtype=bool)
    if nbr_idx is not None:
        for i in range(n_verts):
            own = int(fine_eff_ref[i])
            nbrs = fine_eff_ref[nbr_idx[i]]
            same = int((nbrs == own).sum())
            # Mislabel if FEWER than half of nearest rest neighbours
            # share its label. (Strong signal — the vert is alone
            # in a sea of other-labelled voxels at rest pose.)
            if same < args.knn_k // 2:
                label_island[i] = True
    n_island = int(label_island.sum())
    print(f"  Rest-pose label islands: {n_island}")

    effects = args.effects or BODY_EFFECTS
    isolation_count = np.zeros(n_verts, dtype=np.int32)
    pose_records: dict[str, dict] = {}

    for ename in effects:
        cap = _peak_render(args.gender, ename, size=tuple(args.size))
        if cap is None:
            print(f"  skip '{ename}' (handler not found)")
            continue
        posed = cap["posed"]
        if len(posed) != n_verts:
            print(f"  skip '{ename}' (vert count mismatch)")
            continue
        nn = _nn_distance(posed)
        disp = np.linalg.norm(posed - rest, axis=1)
        # Two detectors combined:
        #
        # 1. Geometric isolation — vert ended up far from any other
        #    vert in the posed mesh. Catches lone flyaways.
        iso_geom = (nn > args.nn_threshold) | (
            nn > args.isolation_ratio * np.maximum(rest_nn, 1e-3))
        # 2. Per-label outlier displacement — vert moved much more
        #    than the median for its label. Catches CLUSTERS of
        #    mis-labeled voxels (which travel together so iso_geom
        #    misses them).
        per_label_outlier = np.zeros(n_verts, dtype=bool)
        for lid in range(16):
            m = fine_eff_ref == lid
            if int(m.sum()) < 5:
                continue
            label_disp = disp[m]
            med = float(np.median(label_disp))
            iqr = float(np.percentile(label_disp, 75) -
                          np.percentile(label_disp, 25))
            # If the label's typical disp is essentially 0, treat any
            # movement >= 5 units as clearly anomalous. Otherwise
            # flag verts > median + 3 × IQR.
            if med < 1.0 and iqr < 1.0:
                thresh = 5.0
            else:
                thresh = med + 3.0 * max(iqr, 1.0)
            per_label_outlier |= m & (disp > thresh)
        moved = disp > args.min_disp
        is_iso = (iso_geom & moved) | per_label_outlier
        n_iso = int(is_iso.sum())
        print(f"  {ename:18s}  iso={n_iso:3d}  max_nn={nn.max():.2f}  "
                f"max_disp={disp.max():.1f}")
        isolation_count += is_iso.astype(np.int32)
        pose_records[ename] = dict(
            rgb=cap["rgb"], posed=posed, isolated=is_iso, nn=nn, disp=disp,
            centre=cap["centre"], scale=cap["scale"],
            yaw=cap["render_yaw"], pitch=cap["render_pitch"],
        )

    # Aggregate offenders & reassign via rest k-NN.
    # Combine effect-driven detector with static rest-pose island.
    offender_mask = (isolation_count >= args.min_poses) | label_island
    offender_idx = np.where(offender_mask)[0]
    print(f"\nFlyaway offenders: "
            f"{int((isolation_count >= args.min_poses).sum())} from "
            f"effects + {n_island} rest-pose islands = "
            f"{int(offender_mask.sum())} total")

    transitions: Counter = Counter()
    suggested = fine_eff_ref.copy()
    n_changed = 0
    if len(offender_idx) > 0:
        pool_mask = ~offender_mask
        new_lab, confident = _knn_majority_label(
            rest, fine_eff_ref, offender_idx,
            pool_mask=pool_mask, k=args.knn_k,
            min_agreement=args.min_agreement)
        for j, vi in enumerate(offender_idx):
            if not confident[j]:
                continue
            old = int(fine_eff_ref[vi])
            new = int(new_lab[j])
            if old != new:
                suggested[vi] = new
                transitions[(old, new)] += 1
        n_changed = sum(transitions.values())
        n_skip = int((~confident).sum())
        if n_skip:
            print(f"  Skipped {n_skip} low-confidence reassignments")

    print(f"\nHigh-confidence reassignments: {n_changed}")
    for (a, b), n in sorted(transitions.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {BPF_NAMES.get(a):10s} → {BPF_NAMES.get(b):10s}: {n}")

    # Render per-effect diagnostic.
    cells = []
    for ename, rec in pose_records.items():
        rgb = rec["rgb"]
        h, w = rgb.shape[:2]
        pix = _project_with_camera(rec["posed"],
                                          centre=rec["centre"],
                                          scale=rec["scale"],
                                          yaw=rec["yaw"], pitch=rec["pitch"],
                                          size=(w, h))
        img = Image.fromarray(rgb.astype(np.uint8))
        drw = ImageDraw.Draw(img, "RGBA")
        for vi in np.where(rec["isolated"])[0]:
            px, py, _ = pix[vi]
            if not (0 <= px < w and 0 <= py < h):
                continue
            is_off = bool(offender_mask[vi])
            is_changed = is_off and int(suggested[vi]) != int(fine_eff_ref[vi])
            if is_changed:
                color = (255, 0, 255, 240)
            elif is_off:
                color = (255, 140, 40, 220)
            else:
                color = (240, 220, 80, 170)
            drw.line((px - 4, py, px + 4, py), fill=color, width=1)
            drw.line((px, py - 4, px, py + 4), fill=color, width=1)
            if is_changed:
                sl = int(suggested[vi])
                ring = _PALETTE_COLORS.get(sl, (255, 255, 255))
                drw.ellipse((px - 3, py - 3, px + 3, py + 3),
                              outline=ring + (255,), width=1)
        labeled = _label_image(
            np.asarray(img),
            f"{ename}  iso={int(rec['isolated'].sum())}")
        Image.fromarray(labeled).save(out_dir / f"eff_{ename}.png")
        cells.append(labeled)

    cols = 6
    rows = (len(cells) + cols - 1) // cols
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)
    row_imgs = [np.hstack(cells[i * cols:(i + 1) * cols])
                  for i in range(rows)]
    grid = np.vstack(row_imgs)
    Image.fromarray(grid).save(out_dir / "grid_effects.png")
    print(f"\nWrote {out_dir / 'grid_effects.png'}")

    if args.apply and n_changed > 0:
        ov_path = (assets_dir()
                     / f"body_label_overrides_{args.gender}.json")
        existing: dict[str, int] = {}
        if ov_path.exists():
            try:
                data = json.loads(ov_path.read_text())
                for k, v in data.items():
                    if not k.startswith("_"):
                        existing[k] = int(v)
            except Exception:
                pass
        new_overrides = {str(int(vi)): int(suggested[vi])
                         for vi in offender_idx
                         if int(suggested[vi]) != int(fine_eff_ref[vi])}
        merged = {**existing, **new_overrides}
        out_doc = {
            "_comment": ("Effect-driven flyaway reassignments via "
                         "effect_flyaway_relabel. Each entry is a vert "
                         "that ended up isolated from the body during a "
                         "real body-effect pose, reassigned by rest-pose "
                         "k-NN majority vote among well-behaved voxels."),
            "_legend": {str(k): v for k, v in BPF_NAMES.items()},
            **{k: int(v) for k, v in merged.items()},
        }
        ov_path.write_text(json.dumps(out_doc, indent=2))
        print(f"Wrote {len(new_overrides)} new + {len(existing)} "
                f"existing = {len(merged)} overrides → {ov_path.name}")
    elif args.apply:
        print("No reassignments to apply.")


if __name__ == "__main__":
    main()
