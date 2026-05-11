"""Drive the body rig into extreme test poses, detect voxels whose
motion disagrees with their body-part label, render diagnostic
images, and (optionally) reassign offenders by k-NN majority vote
on rest-pose positions of well-behaved voxels.

For each pose we know which BPF labels SHOULD move (the targeted
chain) and which should NOT. A voxel is flagged as:

  • wrong-mover  → moved a lot, but its label says it shouldn't
  • wrong-stayer → barely moved, but its label says it should

A voxel is reassigned if it is wrong in at least ``--min-votes``
distinct poses. Reassignment looks at the K nearest rest-position
neighbours among voxels that were CONSISTENT across all poses and
takes the majority label.

Outputs (under ``docs/extreme_pose_relabel/<gender>/``):
  - extreme_<pose>.png    rendered avatar with cross-overlays
  - summary.txt           per-label transition counts

If --apply is given, the corrected labels are written to
``src/faceview/assets/body_part_labels_<gender>.npz`` (the original
file is backed up to ``.npz.bak`` next to it the first time).

Run:
    PYTHONPATH=src python -m tools.extreme_pose_relabel --gender male
    PYTHONPATH=src python -m tools.extreme_pose_relabel --gender male --apply
"""
from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

# ── Pose catalogue ────────────────────────────────────────────────
# Each pose: dict of FaceParams attribute → angle, plus the set of
# BPF label ids that SHOULD move under that pose.
BPF = dict(
    NECK=0, CHEST=1, ABDOMEN=2, PELVIS=3,
    U_ARM_L=4, U_ARM_R=5, FORE_L=6, FORE_R=7,
    HAND_L=8, HAND_R=9,
    THIGH_L=10, THIGH_R=11, SHIN_L=12, SHIN_R=13,
    FOOT_L=14, FOOT_R=15,
)
ARM_L = {BPF["U_ARM_L"], BPF["FORE_L"], BPF["HAND_L"]}
ARM_R = {BPF["U_ARM_R"], BPF["FORE_R"], BPF["HAND_R"]}
FORE_L = {BPF["FORE_L"], BPF["HAND_L"]}
FORE_R = {BPF["FORE_R"], BPF["HAND_R"]}
LEG_L = {BPF["THIGH_L"], BPF["SHIN_L"], BPF["FOOT_L"]}
LEG_R = {BPF["THIGH_R"], BPF["SHIN_R"], BPF["FOOT_R"]}
SHIN_L = {BPF["SHIN_L"], BPF["FOOT_L"]}
SHIN_R = {BPF["SHIN_R"], BPF["FOOT_R"]}

# pose name → (joint params, expected-moving label set)
POSES = {
    "arm_L_up":     ({"l_shoulder_pitch": -1.45}, ARM_L),
    "arm_R_up":     ({"r_shoulder_pitch": -1.45}, ARM_R),
    "arm_L_side":   ({"l_shoulder_roll": -1.45}, ARM_L),
    "arm_R_side":   ({"r_shoulder_roll":  1.45}, ARM_R),
    # NOTE: poses below are pure single-joint rotations so that
    # `expected` cleanly describes the labels that SHOULD move.
    # Don't add a shoulder pitch to an elbow test — it would rotate
    # the upper-arm too, and those (correctly-moving) verts would
    # show up as "wrong-movers" because their label isn't in the
    # leaf joint's chain.
    "elbow_L_bent": ({"l_elbow_pitch": -1.85}, FORE_L),
    "elbow_R_bent": ({"r_elbow_pitch": -1.85}, FORE_R),
    "leg_L_lift":   ({"l_hip_pitch": -0.85}, LEG_L),
    "leg_R_lift":   ({"r_hip_pitch": -0.85}, LEG_R),
    "knee_L_bend":  ({"l_knee_pitch": 1.55}, SHIN_L),
    "knee_R_bend":  ({"r_knee_pitch": 1.55}, SHIN_R),
}

# Palette used when drawing the suggested-label rings.
_PALETTE_COLORS = {
    0: (255, 230, 90),    # neck — yellow
    1: (60, 160, 230),    # chest — blue
    2: (235, 130, 50),    # abdomen — orange
    3: (160, 90, 200),    # pelvis — purple
    4: (210, 50, 110),    # u_arm_L — magenta
    5: (60, 180, 150),    # u_arm_R — teal
    6: (220, 90, 150),    # fore_L — pink
    7: (110, 200, 200),   # fore_R — cyan
    8: (255, 100, 200),   # hand_L
    9: (160, 230, 230),   # hand_R
    10: (110, 60, 170),   # thigh_L
    11: (160, 200, 70),   # thigh_R
    12: (50, 130, 70),    # shin_L
    13: (220, 170, 200),  # shin_R
    14: (40, 60, 180),    # foot_L
    15: (240, 130, 130),  # foot_R
}


def _make_neutral_params(gender: str):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = 0.0
    p._camera_pitch = 0.0
    p._camera_zoom = 0.55
    return p


def _capture_rig_io(render_call):
    """Run a render() call and capture (rest_verts, posed_verts, rgb,
    centre, scale, size, fine_eff) in a single pass. We monkey-patch
    ``apply_body_rig_v2`` (lifted from ``body_rig`` since ``ict_face``
    does a local import) to grab the verts, ``build_rig_state`` to
    grab the EFFECTIVE labels (post-cleanup + overrides), and the
    renderer's ``render`` method to grab the camera bbox so screen-
    space projections match the rendered image exactly."""
    import faceview.vision.body_rig as br
    import faceview.vision.ict_face as ift

    cap = {}
    orig_rig = br.apply_body_rig_v2
    orig_overrides = br._apply_manual_overrides

    def rig_hook(verts, params, rig):
        cap["rest"] = np.asarray(verts).copy()
        out = orig_rig(verts, params, rig)
        cap["posed"] = np.asarray(out).copy()
        cap["rig"] = rig
        return out

    def overrides_hook(fine_labels, **kw):
        out = orig_overrides(fine_labels, **kw)
        cap["fine_eff"] = np.asarray(out).copy()
        return out

    br.apply_body_rig_v2 = rig_hook
    br._apply_manual_overrides = overrides_hook
    rend = ift._ensure_renderer()
    orig_render = rend.render

    def render_hook(*args, **kwargs):
        cap["centre"] = np.asarray(kwargs["centre"]).copy()
        cap["scale"] = float(kwargs["scale"])
        cap["render_yaw"] = float(kwargs.get("yaw", 0.0))
        cap["render_pitch"] = float(kwargs.get("pitch", 0.0))
        cap["render_size"] = tuple(kwargs["size"])
        # Capture the FINAL verts handed to the renderer (post
        # apply_body_rig_v2 + _apply_neck_rotation +
        # _apply_cervical_cascade). The "posed" capture above only
        # sees the state right after apply_body_rig_v2.
        cap["final_verts"] = np.asarray(kwargs["verts"]).copy()
        return orig_render(*args, **kwargs)

    rend.render = render_hook
    try:
        cap["rgb"] = render_call()
    finally:
        br.apply_body_rig_v2 = orig_rig
        br._apply_manual_overrides = orig_overrides
        rend.render = orig_render
    return cap


def _project_with_camera(verts: np.ndarray, *, centre: np.ndarray,
                            scale: float, yaw: float, pitch: float,
                            size: tuple[int, int]) -> np.ndarray:
    """Replicate the ICTRenderer projection: model = ry @ rx @ flipZ
    @ scale_aspect @ T(-centre); then NDC → pixels with Y flipped."""
    w, h = size
    aspect = float(h) / float(w) if w > 0 else 1.0
    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]])
    flip = np.diag([1.0, 1.0, -1.0])
    S = np.diag([scale * aspect, scale, scale])
    M = ry @ rx @ flip @ S
    v = (verts - centre) @ M.T
    px = (v[:, 0] + 1.0) * w * 0.5
    py = (1.0 - v[:, 1]) * h * 0.5
    return np.column_stack([px, py, v[:, 2]])


def _render_pose(gender: str, pose_params: dict, size=(360, 640)) -> dict:
    """Run a full render and return capture dict with rgb, rest, posed,
    centre, scale, render_yaw/pitch/size."""
    from faceview.vision.ict_face import render_face_ict
    p = _make_neutral_params(gender)
    for k, v in pose_params.items():
        setattr(p, k, float(v))
    return _capture_rig_io(lambda: render_face_ict(p, size=size))


def _classify_motion(disp: np.ndarray, expected: set[int],
                       fine: np.ndarray, *,
                       move_thresh: float, stay_thresh: float):
    """Return per-vert (wrong_mover_mask, wrong_stayer_mask)."""
    moved = disp > move_thresh
    stayed = disp < stay_thresh
    expected_arr = np.zeros(len(fine), dtype=bool)
    for lid in expected:
        expected_arr |= (fine == lid)
    wrong_mover = moved & ~expected_arr
    wrong_stayer = stayed & expected_arr
    return wrong_mover, wrong_stayer


def _knn_majority_label(rest_verts: np.ndarray, fine: np.ndarray,
                          target_idx: np.ndarray, *,
                          pool_mask: np.ndarray, k: int = 10,
                          min_agreement: int = 7) -> tuple[
                              np.ndarray, np.ndarray]:
    """For each vert in target_idx, pick the majority label among its
    ``k`` nearest rest-position neighbours drawn from
    ``rest_verts[pool_mask]``. Returns (new_labels, confident_mask).
    A reassignment is "confident" iff the majority label appears
    ≥``min_agreement`` times among the k neighbours."""
    pool_pts = rest_verts[pool_mask]
    pool_lab = fine[pool_mask]
    n = len(target_idx)
    new_labels = np.empty(n, dtype=fine.dtype)
    confident = np.zeros(n, dtype=bool)
    if len(pool_pts) == 0:
        new_labels[:] = fine[target_idx]
        return new_labels, confident
    for i, vi in enumerate(target_idx):
        d = np.linalg.norm(pool_pts - rest_verts[vi], axis=1)
        nn = np.argsort(d)[:k]
        nn_labels = pool_lab[nn]
        cnt = Counter(int(x) for x in nn_labels)
        best, votes = cnt.most_common(1)[0]
        new_labels[i] = best
        confident[i] = (votes >= min_agreement)
    return new_labels, confident


def _label_image(img: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    w = max(int(drw.textlength(text, font=font)), 60)
    drw.rectangle((4, 4, w + 14, 22), fill=(0, 0, 0))
    drw.text((10, 6), text, fill=(255, 235, 180), font=font)
    return np.asarray(pil)


def _draw_overlay(rgb: np.ndarray, posed: np.ndarray,
                    wrong_mover: np.ndarray, wrong_stayer: np.ndarray,
                    suggested_labels: np.ndarray | None,
                    *, centre, scale, yaw, pitch) -> np.ndarray:
    h, w = rgb.shape[:2]
    pix = _project_with_camera(posed, centre=centre, scale=scale,
                                  yaw=yaw, pitch=pitch, size=(w, h))
    img = Image.fromarray(rgb.astype(np.uint8))
    drw = ImageDraw.Draw(img, "RGBA")
    # crosses: red for wrong-movers (moved when shouldn't),
    #          cyan for wrong-stayers (didn't move when should)
    for kind, mask, color in [
        ("mover", wrong_mover, (255, 60, 60, 235)),
        ("stayer", wrong_stayer, (90, 220, 255, 235)),
    ]:
        idxs = np.where(mask)[0]
        for vi in idxs:
            px, py, _ = pix[vi]
            if not (0 <= px < w and 0 <= py < h):
                continue
            drw.line((px - 4, py, px + 4, py), fill=color, width=1)
            drw.line((px, py - 4, px, py + 4), fill=color, width=1)
            if suggested_labels is not None:
                sl = int(suggested_labels[vi])
                if sl in _PALETTE_COLORS:
                    r, g, b = _PALETTE_COLORS[sl]
                    drw.ellipse((px - 3, py - 3, px + 3, py + 3),
                                  outline=(r, g, b, 255), width=1)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--out-dir", default=None,
                     help="Default: docs/extreme_pose_relabel/<gender>")
    ap.add_argument("--move-thresh", type=float, default=0.15,
                     help="Disp > this counts as 'moved' (BP3D units). "
                          "Bump up to ignore seam-smoothing leakage.")
    ap.add_argument("--stay-thresh", type=float, default=0.012,
                     help="Disp < this counts as 'stayed'")
    ap.add_argument("--min-votes", type=int, default=2,
                     help="Voxel must be wrong in this many distinct poses")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--min-agreement", type=int, default=7,
                     help="Reassign only if ≥this neighbours of k agree")
    ap.add_argument("--stayers-only", action="store_true",
                     help="Only reassign wrong-stayers (most reliable signal). "
                          "Skips wrong-movers entirely.")
    ap.add_argument("--apply", action="store_true",
                     help="Write reassigned labels back to the NPZ")
    ap.add_argument("--size", type=int, nargs=2, default=(360, 640))
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(f"docs/extreme_pose_relabel/{args.gender}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing labels.
    from faceview.assets import assets_dir
    labels_path = assets_dir() / f"body_part_labels_{args.gender}.npz"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    npz = np.load(labels_path)
    fine = np.asarray(npz["labels"], dtype=np.int32)
    n_verts = len(fine)
    print(f"Loaded labels: {n_verts} verts from {labels_path}")

    # Render each pose, capture verts, classify, draw overlay.
    pose_records: dict[str, dict] = {}
    rest_verts_ref: np.ndarray | None = None
    fine_eff_ref: np.ndarray | None = None

    for pname, (joint_kwargs, expected) in POSES.items():
        print(f"  → pose '{pname}' …")
        cap = _render_pose(args.gender, joint_kwargs, size=tuple(args.size))
        rest, posed = cap["rest"], cap["posed"]
        if len(rest) != n_verts:
            raise RuntimeError(
                f"vert count mismatch: rig={len(rest)} npz={n_verts}")
        if rest_verts_ref is None:
            rest_verts_ref = rest
        # Use the EFFECTIVE labels the rig actually used (after
        # mode-filter, stray cleanup, manual overrides). Falls back
        # to NPZ labels if the hook didn't fire (e.g. no rig).
        fine_eff = cap.get("fine_eff", fine)
        if fine_eff_ref is None:
            fine_eff_ref = fine_eff
        disp = np.linalg.norm(posed - rest, axis=1)
        wrong_mover, wrong_stayer = _classify_motion(
            disp, expected, fine_eff,
            move_thresh=args.move_thresh, stay_thresh=args.stay_thresh)
        pose_records[pname] = dict(
            rgb=cap["rgb"], posed=posed, disp=disp,
            wrong_mover=wrong_mover, wrong_stayer=wrong_stayer,
            expected=expected,
            centre=cap["centre"], scale=cap["scale"],
            yaw=cap["render_yaw"], pitch=cap["render_pitch"],
        )
        print(f"    wrong_mover={int(wrong_mover.sum())}  "
                f"wrong_stayer={int(wrong_stayer.sum())}")

    # Aggregate offences across poses.
    n_wrong = np.zeros(n_verts, dtype=np.int32)
    for rec in pose_records.values():
        if not args.stayers_only:
            n_wrong += rec["wrong_mover"].astype(np.int32)
        n_wrong += rec["wrong_stayer"].astype(np.int32)
    offender_mask = n_wrong >= args.min_votes
    offender_idx = np.where(offender_mask)[0]
    consistent_mask = ~offender_mask
    mode = "stayers-only" if args.stayers_only else "movers+stayers"
    print(f"\n{int(offender_mask.sum())} offenders ({mode}, wrong in ≥"
            f"{args.min_votes} poses)")

    # Reassign by k-NN at REST pose using only consistent voxels.
    transitions: Counter = Counter()
    n_changed = 0
    fine_for_knn = fine_eff_ref if fine_eff_ref is not None else fine
    suggested = fine_for_knn.copy()
    if len(offender_idx) > 0:
        new_lab, confident = _knn_majority_label(
            rest_verts_ref, fine_for_knn, offender_idx,
            pool_mask=consistent_mask, k=args.knn_k,
            min_agreement=args.min_agreement)
        n_unconfident = int((~confident).sum())
        if n_unconfident:
            print(f"  Skipped {n_unconfident} low-confidence reassignments")
        # Apply only confident reassignments.
        for j, vi in enumerate(offender_idx):
            if not confident[j]:
                continue
            old = int(fine_for_knn[vi])
            new = int(new_lab[j])
            if old != new:
                suggested[vi] = new
                transitions[(old, new)] += 1
        n_changed = sum(transitions.values())
    print(f"k-NN reassignment changed {n_changed} labels")
    BPF_NAMES = {
        0: "neck", 1: "chest", 2: "abdomen", 3: "pelvis",
        4: "u_arm_L", 5: "u_arm_R", 6: "fore_L", 7: "fore_R",
        8: "hand_L", 9: "hand_R", 10: "thigh_L", 11: "thigh_R",
        12: "shin_L", 13: "shin_R", 14: "foot_L", 15: "foot_R",
    }
    if transitions:
        print("Top transitions:")
        for (a, b), n in sorted(transitions.items(),
                                  key=lambda kv: -kv[1])[:15]:
            print(f"  {BPF_NAMES.get(a):10s} → {BPF_NAMES.get(b):10s}: {n}")

    def _overlay(rec):
        return _draw_overlay(
            rec["rgb"], rec["posed"],
            rec["wrong_mover"], rec["wrong_stayer"],
            suggested if len(offender_idx) else None,
            centre=rec["centre"], scale=rec["scale"],
            yaw=rec["yaw"], pitch=rec["pitch"])

    # Draw + save overlays for each pose with the SUGGESTED labels.
    for pname, rec in pose_records.items():
        out_img = _overlay(rec)
        out_img = _label_image(
            out_img,
            f"{pname}  m={int(rec['wrong_mover'].sum())} "
            f"s={int(rec['wrong_stayer'].sum())}")
        Image.fromarray(out_img).save(out_dir / f"extreme_{pname}.png")

    # Composite grid.
    pose_names = list(pose_records.keys())
    cols = 5
    rows = (len(pose_names) + cols - 1) // cols
    cells = []
    for pname in pose_names:
        rec = pose_records[pname]
        out_img = _overlay(rec)
        cells.append(_label_image(out_img, pname))
    blank = np.zeros_like(cells[0])
    while len(cells) < rows * cols:
        cells.append(blank)
    row_imgs = [np.hstack(cells[i * cols:(i + 1) * cols])
                  for i in range(rows)]
    grid = np.vstack(row_imgs)
    Image.fromarray(grid).save(out_dir / "grid.png")

    # Summary.
    summary = [
        f"gender: {args.gender}",
        f"verts: {n_verts}",
        f"move_thresh: {args.move_thresh}  stay_thresh: {args.stay_thresh}",
        f"min_votes: {args.min_votes}  knn_k: {args.knn_k}",
        f"offenders: {int(offender_mask.sum())}",
        f"reassigned: {n_changed}",
        "",
        "transitions:",
    ]
    BPF_NAMES = {
        0: "neck", 1: "chest", 2: "abdomen", 3: "pelvis",
        4: "u_arm_L", 5: "u_arm_R", 6: "fore_L", 7: "fore_R",
        8: "hand_L", 9: "hand_R", 10: "thigh_L", 11: "thigh_R",
        12: "shin_L", 13: "shin_R", 14: "foot_L", 15: "foot_R",
    }
    for (a, b), n in sorted(transitions.items(), key=lambda kv: -kv[1]):
        summary.append(f"  {BPF_NAMES.get(a):10s} → "
                          f"{BPF_NAMES.get(b):10s}: {n}")
    (out_dir / "summary.txt").write_text("\n".join(summary))
    print(f"\nWrote outputs to {out_dir}")

    if args.apply and n_changed > 0:
        # Write changes to body_label_overrides.json — this is loaded
        # AFTER the runtime mode-filter and stray-component cleanup
        # in build_rig_state, so isolated reassignments don't get
        # smoothed away. (Modifying the NPZ directly causes the
        # runtime cleanup to revert ~half of the changes and produce
        # cascading new artefacts — confirmed empirically.)
        import json
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
        # Build override map only for the verts we changed.
        new_overrides = {str(int(vi)): int(suggested[vi])
                         for vi in offender_idx
                         if int(suggested[vi]) != int(fine_for_knn[vi])}
        merged = {**existing, **new_overrides}
        BPF_NAMES_FULL = {str(k): v for k, v in BPF_NAMES.items()}
        out_doc = {
            "_comment": ("Conservative wrong-stayer reassignments via "
                         "extreme_pose_relabel. k-NN majority vote on "
                         "rest-pose positions of well-behaved voxels."),
            "_legend": BPF_NAMES_FULL,
            **{k: int(v) for k, v in merged.items()},
        }
        ov_path.write_text(json.dumps(out_doc, indent=2))
        print(f"Wrote {len(new_overrides)} new + {len(existing)} "
                f"existing = {len(merged)} overrides → {ov_path.name}")
    elif args.apply:
        print("No reassignments — nothing to apply.")


if __name__ == "__main__":
    main()
