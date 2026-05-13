"""Identify voxels that fly away from the body during extreme poses
and reclassify them via rest-pose nearest-neighbour voting.

A "flyaway" voxel is one whose nearest neighbour in the POSED state
is far away — either absolutely (>``--nn-threshold`` BP3D units) or
much further than in the rest pose (>``--isolation-ratio`` × the
vert's rest NN distance). These are voxels that the rig moved
along with a chain but whose rest-pose neighbours did NOT come
along — i.e. they're geometrically isolated from the body in the
posed image.

For each flyaway candidate, the new label is decided by a k-NN
majority vote among well-behaved (non-flyaway) voxels at their
REST positions. Only reassignments where ≥``--min-agreement`` of
``--knn-k`` neighbours agree are applied.

Outputs:
  docs/extreme_pose_relabel/<gender>/iso_<pose>.png   per-pose diagnostic
  docs/extreme_pose_relabel/<gender>/grid_isolated.png   combined grid

With ``--apply``, writes high-confidence reassignments to
``src/faceview/assets/body_label_overrides.json`` (applied AFTER
runtime label-cleanup so they actually stick).
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
    POSES, _make_neutral_params, _capture_rig_io,
    _project_with_camera, _label_image,
    _knn_majority_label, _PALETTE_COLORS,
)


BPF_NAMES = {
    0: "neck", 1: "chest", 2: "abdomen", 3: "pelvis",
    4: "u_arm_L", 5: "u_arm_R", 6: "fore_L", 7: "fore_R",
    8: "hand_L", 9: "hand_R", 10: "thigh_L", 11: "thigh_R",
    12: "shin_L", 13: "shin_R", 14: "foot_L", 15: "foot_R",
}


def _nn_distance(verts: np.ndarray) -> np.ndarray:
    """For each vert, return distance to its nearest OTHER vert."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts)
        dists, _ = tree.query(verts, k=2)
        return dists[:, 1]
    except ImportError:
        n = len(verts)
        nn = np.full(n, np.inf)
        for i in range(n):
            d = np.linalg.norm(verts - verts[i], axis=1)
            d[i] = np.inf
            nn[i] = d.min()
        return nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--nn-threshold", type=float, default=4.0,
                     help="Vert is flyaway if its posed NN-distance > "
                          "this absolute threshold (ICT-frame units; "
                          "rest pose median ≈ 0.74, 99%% ≈ 2.78).")
    ap.add_argument("--isolation-ratio", type=float, default=3.0,
                     help="OR flyaway if posed NN > this × rest NN.")
    ap.add_argument("--min-disp", type=float, default=2.0,
                     help="Vert must have moved at least this many units "
                          "(rules out stationary verts whose neighbours "
                          "flew off and left them looking isolated).")
    ap.add_argument("--min-poses", type=int, default=1,
                     help="Voxel must be flyaway in this many poses.")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--min-agreement", type=int, default=7,
                     help="Reassign only if ≥this k-NN neighbours agree.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--size", type=int, nargs=2, default=(360, 640))
    args = ap.parse_args()

    from faceview.assets import assets_dir
    labels_path = assets_dir() / f"body_part_labels_{args.gender}.npz"
    fine_npz = np.asarray(np.load(labels_path)["labels"], dtype=np.int32)
    n_verts = len(fine_npz)
    print(f"Loaded labels: {n_verts} verts")

    out_dir = Path(f"docs/extreme_pose_relabel/{args.gender}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render rest pose once to get baseline NN distances.
    from faceview.vision.ict_face import render_face_ict
    p = _make_neutral_params(args.gender)
    cap = _capture_rig_io(lambda: render_face_ict(p, size=tuple(args.size)))
    rest = cap["rest"]
    fine_eff_ref = cap.get("fine_eff", fine_npz)
    if len(rest) != n_verts:
        raise RuntimeError(
            f"vert count mismatch: rig={len(rest)} npz={n_verts}")
    rest_nn = _nn_distance(rest)
    print(f"Rest NN-dist:  median={np.median(rest_nn):.4f}  "
            f"95%={np.percentile(rest_nn, 95):.4f}  "
            f"99%={np.percentile(rest_nn, 99):.4f}")

    # Per-pose flyaway detection.
    isolation_count = np.zeros(n_verts, dtype=np.int32)
    pose_records: dict[str, dict] = {}
    for pname, (joint_kwargs, _expected) in POSES.items():
        print(f"  → pose '{pname}' …")
        p = _make_neutral_params(args.gender)
        for k, v in joint_kwargs.items():
            setattr(p, k, float(v))
        cap = _capture_rig_io(
            lambda: render_face_ict(p, size=tuple(args.size)))
        posed = cap["posed"]
        posed_nn = _nn_distance(posed)
        disp = np.linalg.norm(posed - rest, axis=1)
        is_iso_geom = (posed_nn > args.nn_threshold) | (
            posed_nn > args.isolation_ratio * np.maximum(rest_nn, 1e-3))
        moved = disp > args.min_disp
        # Real flyaway = vert moved AND ended up isolated.
        # (Stationary chest verts whose adjacent arm flew off would
        # also have high posed_nn but didn't move themselves.)
        is_iso = is_iso_geom & moved
        n_iso = int(is_iso.sum())
        print(f"    isolated: {n_iso}  (max nn={posed_nn.max():.3f})")
        isolation_count += is_iso.astype(np.int32)
        pose_records[pname] = dict(
            rgb=cap["rgb"], posed=posed, isolated=is_iso,
            centre=cap["centre"], scale=cap["scale"],
            yaw=cap["render_yaw"], pitch=cap["render_pitch"],
        )

    # Aggregate across poses.
    offender_mask = isolation_count >= args.min_poses
    offender_idx = np.where(offender_mask)[0]
    print(f"\nFlyaway offenders (isolated in ≥{args.min_poses} poses): "
            f"{int(offender_mask.sum())}")

    # k-NN reassignment on REST positions among non-offender pool.
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
        n_unconfident = int((~confident).sum())
        if n_unconfident:
            print(f"  Skipped {n_unconfident} low-confidence "
                    f"reassignments (< {args.min_agreement}/"
                    f"{args.knn_k} neighbours agreed)")

    print(f"\nHigh-confidence reassignments: {n_changed}")
    for (a, b), n in sorted(transitions.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {BPF_NAMES.get(a):10s} → {BPF_NAMES.get(b):10s}: {n}")

    # Per-pose diagnostic renders.
    cells = []
    pose_names = list(pose_records.keys())
    for pname in pose_names:
        rec = pose_records[pname]
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
            is_changed = is_off and (int(suggested[vi])
                                            != int(fine_eff_ref[vi]))
            # Color code:
            #   magenta = will be reassigned (high-conf)
            #   orange  = flagged offender but no confident new label
            #   yellow  = isolated in only one pose
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
            f"{pname}  iso={int(rec['isolated'].sum())}")
        Image.fromarray(labeled).save(out_dir / f"iso_{pname}.png")
        cells.append(labeled)

    cols = 5
    rows = (len(cells) + cols - 1) // cols
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)
    row_imgs = [np.hstack(cells[i * cols:(i + 1) * cols])
                  for i in range(rows)]
    grid = np.vstack(row_imgs)
    Image.fromarray(grid).save(out_dir / "grid_isolated.png")
    print(f"\nWrote {out_dir / 'grid_isolated.png'}")

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
            "_comment": ("Isolated-voxel reassignments via "
                         "isolated_voxel_relabel. Each entry is a "
                         "flyaway voxel reassigned to its rest-pose "
                         "k-NN majority label."),
            "_legend": {str(k): v for k, v in BPF_NAMES.items()},
            **{k: int(v) for k, v in merged.items()},
        }
        ov_path.write_text(json.dumps(out_doc, indent=2))
        print(f"Wrote {len(new_overrides)} new + {len(existing)} "
                f"existing = {len(merged)} overrides → {ov_path.name}")
    elif args.apply:
        print("No high-confidence reassignments to apply.")


if __name__ == "__main__":
    main()
