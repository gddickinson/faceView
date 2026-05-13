"""Compare pose-offender masks BEFORE vs AFTER applying overrides.

Runs the same extreme-pose analysis twice (once with overrides
absent, once with them present) and reports per-pose:

  - fixed    = wrong before, ok now (good)
  - new_bad  = ok before, wrong now (BAD — what we want to find)
  - still_bad = wrong in both

For new_bad voxels, also reports:
  - direct  = the voxel is one of the 33 reassigned ones
  - cascade = the voxel shares a triangle with a reassigned vert
  - other   = unrelated (topology/filter side-effect)

Renders each pose with cross-overlays colored:
  - magenta crosses = NEW BAD (good→bad regression)
  - dim green = fixed (was wrong before, ok now)

Outputs to docs/extreme_pose_relabel/<gender>/diff_<pose>.png
and grid_diff.png.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# Reuse the pose catalog + helpers from the relabel tool.
from tools.extreme_pose_relabel import (  # noqa: E402
    POSES, _make_neutral_params, _capture_rig_io,
    _project_with_camera, _classify_motion, _label_image,
)


def _render_records(gender: str, fine_npz: np.ndarray, size=(360, 640)):
    """Render all poses and capture per-pose wrong_mover/wrong_stayer
    masks using the rig's EFFECTIVE labels."""
    from faceview.vision.ict_face import render_face_ict
    recs = {}
    fine_eff_ref = None
    for pname, (joint_kwargs, expected) in POSES.items():
        p = _make_neutral_params(gender)
        for k, v in joint_kwargs.items():
            setattr(p, k, float(v))
        cap = _capture_rig_io(lambda: render_face_ict(p, size=size))
        fine_eff = cap.get("fine_eff", fine_npz)
        if fine_eff_ref is None:
            fine_eff_ref = fine_eff
            # Spot-check: print the labels for the override indices
            # so we can see if they actually changed in this pass.
            import json as _json
            ov_path_dbg = (Path(
                __import__("faceview.assets",
                              fromlist=["assets_dir"]).assets_dir())
                / "body_label_overrides.json")
            try:
                if ov_path_dbg.exists():
                    ov_data = _json.loads(ov_path_dbg.read_text())
                    sample_keys = [k for k in list(ov_data.keys())[:5]
                                    if not k.startswith("_")]
                    print(f"    [debug] overrides file present "
                            f"{ov_path_dbg.name}; first 5 verts:")
                    for k in sample_keys:
                        vi = int(k)
                        print(f"      vert {vi}: npz={fine_npz[vi]}  "
                                f"eff={fine_eff[vi]}  wanted={ov_data[k]}")
                else:
                    print(f"    [debug] overrides file ABSENT during this pass")
            except Exception as e:
                print(f"    [debug] err: {e}")
        disp = np.linalg.norm(cap["posed"] - cap["rest"], axis=1)
        wm, ws = _classify_motion(disp, expected, fine_eff,
                                       move_thresh=0.15, stay_thresh=0.012)
        recs[pname] = dict(
            rgb=cap["rgb"], posed=cap["posed"], wm=wm, ws=ws,
            expected=expected,
            centre=cap["centre"], scale=cap["scale"],
            yaw=cap["render_yaw"], pitch=cap["render_pitch"],
        )
    recs["_fine_eff"] = fine_eff_ref
    return recs


def _draw_diff(rgb: np.ndarray, posed: np.ndarray,
                 fixed_mask: np.ndarray, new_bad_mask: np.ndarray,
                 direct_mask: np.ndarray, *, centre, scale,
                 yaw, pitch) -> np.ndarray:
    h, w = rgb.shape[:2]
    pix = _project_with_camera(posed, centre=centre, scale=scale,
                                  yaw=yaw, pitch=pitch, size=(w, h))
    img = Image.fromarray(rgb.astype(np.uint8))
    drw = ImageDraw.Draw(img, "RGBA")
    # Green: fixed (small dim crosses)
    for vi in np.where(fixed_mask)[0]:
        px, py, _ = pix[vi]
        if not (0 <= px < w and 0 <= py < h):
            continue
        drw.line((px - 3, py, px + 3, py), fill=(60, 220, 100, 180), width=1)
        drw.line((px, py - 3, px, py + 3), fill=(60, 220, 100, 180), width=1)
    # Magenta: newly broken (clearly visible)
    for vi in np.where(new_bad_mask)[0]:
        px, py, _ = pix[vi]
        if not (0 <= px < w and 0 <= py < h):
            continue
        is_direct = bool(direct_mask[vi])
        color = (255, 0, 255, 255) if is_direct else (255, 100, 200, 235)
        drw.line((px - 5, py, px + 5, py), fill=color, width=2)
        drw.line((px, py - 5, px, py + 5), fill=color, width=2)
        if is_direct:
            drw.ellipse((px - 4, py - 4, px + 4, py + 4),
                          outline=color, width=1)
    return np.asarray(img)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    args = ap.parse_args()

    from faceview.assets import assets_dir
    out_dir = Path(f"docs/extreme_pose_relabel/{args.gender}")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = assets_dir() / f"body_part_labels_{args.gender}.npz"
    overrides_path = assets_dir() / "body_label_overrides.json"

    # Load current state.
    fine_npz = np.asarray(np.load(labels_path)["labels"], dtype=np.int32)
    print(f"Labels: {len(fine_npz)} verts")
    if not overrides_path.exists():
        raise FileNotFoundError(
            f"{overrides_path} not present — run extreme_pose_relabel "
            f"with --apply first to generate it.")
    overrides = json.loads(overrides_path.read_text())
    override_indices = np.array([int(k) for k in overrides
                                       if not k.startswith("_")], dtype=np.int32)
    print(f"Overrides: {len(override_indices)} verts")

    # PASS 1: render WITHOUT overrides — temporarily move the file aside.
    # IMPORTANT: clear the rig-state LRU cache between passes, otherwise
    # the second pass reuses the first pass's cached masks/weights and
    # the override changes never reach the rig.
    import faceview.vision.body_rig as br
    stash = overrides_path.parent / (overrides_path.name + ".tmp_stash")
    shutil.move(overrides_path, stash)
    print(f"Stashed: {overrides_path.exists()=} {stash.exists()=}")
    try:
        print("Pass 1: WITHOUT overrides …")
        br._cached_rig_state.cache_clear()
        recs_before = _render_records(args.gender, fine_npz)
    finally:
        shutil.move(stash, overrides_path)

    # PASS 2: render WITH overrides.
    print("Pass 2: WITH overrides …")
    br._cached_rig_state.cache_clear()
    recs_after = _render_records(args.gender, fine_npz)

    # Sanity check: how many labels actually differ between the two
    # passes' effective labels?
    fb = recs_before.pop("_fine_eff")
    fa = recs_after.pop("_fine_eff")
    if fb is not None and fa is not None:
        differing = int((fb != fa).sum())
        print(f"\nfine_eff differs at {differing} verts between passes "
                f"(expected ≈ {len(override_indices)})")
        if differing == 0:
            print("WARN: Pass 1 effective labels match Pass 2 — "
                    "override stash didn't take effect.")

    # Build adjacency for "cascade" classification (verts that share
    # a triangle with a reassigned vert). Pull tris from the raw OBJ
    # then clamp to the in-rig vertex count.
    from faceview.vision.body_3d import _load_body_obj
    _, tris = _load_body_obj(args.gender)
    n = len(fine_npz)
    adj_sets: list[set[int]] = [set() for _ in range(n)]
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        if a >= n or b >= n or c >= n:
            continue
        adj_sets[a].add(b); adj_sets[a].add(c)
        adj_sets[b].add(a); adj_sets[b].add(c)
        adj_sets[c].add(a); adj_sets[c].add(b)
    direct_mask = np.zeros(n, dtype=bool)
    direct_mask[override_indices] = True
    cascade_mask = np.zeros(n, dtype=bool)
    for vi in override_indices:
        for nb in adj_sets[int(vi)]:
            cascade_mask[nb] = True
    cascade_mask &= ~direct_mask

    # Aggregate stats.
    totals = Counter()
    cells = []
    pose_names = [n for n in POSES.keys() if n in recs_before]
    for pname in pose_names:
        rb = recs_before[pname]
        ra = recs_after[pname]
        bad_b = rb["wm"] | rb["ws"]
        bad_a = ra["wm"] | ra["ws"]
        fixed = bad_b & ~bad_a
        new_bad = ~bad_b & bad_a
        still_bad = bad_b & bad_a
        # Categorize new_bad.
        nb_direct = int((new_bad & direct_mask).sum())
        nb_cascade = int((new_bad & cascade_mask).sum())
        nb_other = int(new_bad.sum()) - nb_direct - nb_cascade
        totals["fixed"] += int(fixed.sum())
        totals["new_bad"] += int(new_bad.sum())
        totals["still_bad"] += int(still_bad.sum())
        totals["new_bad_direct"] += nb_direct
        totals["new_bad_cascade"] += nb_cascade
        totals["new_bad_other"] += nb_other
        print(f"  {pname:16s}  fixed={int(fixed.sum()):4d}  "
                f"new_bad={int(new_bad.sum()):4d}  "
                f"(direct={nb_direct} cascade={nb_cascade} "
                f"other={nb_other})  still_bad={int(still_bad.sum()):4d}")

        # Draw diff overlay on AFTER pose render.
        img = _draw_diff(
            ra["rgb"], ra["posed"], fixed, new_bad, direct_mask,
            centre=ra["centre"], scale=ra["scale"],
            yaw=ra["yaw"], pitch=ra["pitch"])
        labeled = _label_image(
            img, f"{pname}  new={int(new_bad.sum())} fixed={int(fixed.sum())}")
        Image.fromarray(labeled).save(out_dir / f"diff_{pname}.png")
        cells.append(labeled)

    cols = 5
    rows = (len(cells) + cols - 1) // cols
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)
    row_imgs = [np.hstack(cells[i * cols:(i + 1) * cols]) for i in range(rows)]
    grid = np.vstack(row_imgs)
    Image.fromarray(grid).save(out_dir / "grid_diff.png")

    print("\nTotal across poses:")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out_dir / 'grid_diff.png'}")


if __name__ == "__main__":
    main()
