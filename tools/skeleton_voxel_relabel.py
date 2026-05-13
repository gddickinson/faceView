"""Skeleton-bone voxel relabel.

Uses the rig's actual joint pivots (shoulder/elbow/wrist/hip/knee/ankle
plus a torso spine) as ground-truth landmarks to detect mis-labeled
voxels. Two complementary detectors:

  A. Rest-pose bone-distance (static).
     For each vert, measure rest distance to every bone segment.
     If the bone closest to the vert is NOT its current label, the
     vert is geometrically attached to the wrong limb chain. This
     catches whole mislabeled clusters that survive the rest-pose
     k-NN island detector (because the cluster's neighbours all
     share the same wrong label).

  B. Per-pose bone-following (dynamic).
     For each effect, compare each vert's distance to its OWNING
     bone before vs. after the pose. Under the correct label the
     bone moves *with* the vert so the distance is preserved (the
     rig applies the bone's rigid transform). If posed_dist
     diverges from rest_dist, the wrong bone moved the vert.

Outputs (under ``docs/skeleton_relabel/<gender>/``):
  - bone_distance_map.png  rest verts colored by current-label bone
                           distance (red = far from owning bone)
  - per_effect_<name>.png  per-pose bone-following diagnostic
  - grid_bones.png         composite

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
from tools.effect_flyaway_relabel import BODY_EFFECTS, _peak_render  # noqa: E402
from tools.isolated_voxel_relabel import BPF_NAMES  # noqa: E402


# Map BPF label → (pivot_a, pivot_b) defining the bone segment.
# Torso labels use a synthesized spine — see _build_torso_spine().
_LIMB_BONES = {
    4:  ("shoulder_L", "elbow_L"),    # u_arm_L
    5:  ("shoulder_R", "elbow_R"),    # u_arm_R
    6:  ("elbow_L", "wrist_L"),       # fore_L
    7:  ("elbow_R", "wrist_R"),       # fore_R
    10: ("hip_L", "knee_L"),          # thigh_L
    11: ("hip_R", "knee_R"),          # thigh_R
    12: ("knee_L", "ankle_L"),        # shin_L
    13: ("knee_R", "ankle_R"),        # shin_R
}
# Tip joints: hand / foot are short stubs extending past wrist / ankle
# in the direction of the upstream bone.
_TIP_BONES = {
    8:  ("wrist_L", "elbow_L", 1.4),   # hand_L: 1.4 × (wrist - elbow) past wrist
    9:  ("wrist_R", "elbow_R", 1.4),
    14: ("ankle_L", "knee_L", 0.8),    # foot_L: 0.8 × (ankle - knee) past ankle
    15: ("ankle_R", "knee_R", 0.8),
}


def _segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray
                       ) -> np.ndarray:
    """Return distance from each row of ``pts`` to the line segment ab."""
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-9:
        return np.linalg.norm(pts - a, axis=1)
    ap = pts - a
    t = np.clip(ap @ ab / ab_len2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    return np.linalg.norm(pts - proj, axis=1)


def _build_bones(pivots: dict, rest: np.ndarray, fine: np.ndarray
                  ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Return BPF_label → (segment_start, segment_end) for all 16 labels.

    Torso bones (neck/chest/abdomen/pelvis) are derived from skeleton
    landmarks + rest verts of each label so we have *some* anchor.
    """
    bones: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lid, (k1, k2) in _LIMB_BONES.items():
        if k1 in pivots and k2 in pivots:
            bones[lid] = (np.asarray(pivots[k1], dtype=np.float32),
                          np.asarray(pivots[k2], dtype=np.float32))
    for lid, (root, upstream, frac) in _TIP_BONES.items():
        if root in pivots and upstream in pivots:
            r = np.asarray(pivots[root], dtype=np.float32)
            u = np.asarray(pivots[upstream], dtype=np.float32)
            direction = r - u  # upstream → root
            tip = r + frac * direction
            bones[lid] = (r, tip)

    # Torso bones — use the rest-position centroid OF VERTS currently
    # labeled as that part for the segment endpoints (a single point
    # collapsed to a degenerate segment is fine, _segment_distance
    # handles it). This relies on the bulk of torso labels being
    # correct; the tool's purpose is fixing limb mis-labels, not torso.
    for lid in (0, 1, 2, 3):
        m = fine == lid
        if int(m.sum()) >= 5:
            c = rest[m].mean(axis=0).astype(np.float32)
            bones[lid] = (c, c)
    return bones


def _bone_distance_to_label(verts: np.ndarray,
                              bones: dict[int, tuple[np.ndarray, np.ndarray]],
                              label: int) -> np.ndarray:
    if label not in bones:
        return np.full(len(verts), np.inf, dtype=np.float32)
    a, b = bones[label]
    return _segment_distance(verts, a, b)


def _closest_bone_label(verts: np.ndarray,
                          bones: dict[int, tuple[np.ndarray, np.ndarray]],
                          candidate_labels: list[int]) -> np.ndarray:
    """For each vert, return the BPF label whose bone is closest."""
    n = len(verts)
    best_lab = np.zeros(n, dtype=np.int32)
    best_d = np.full(n, np.inf, dtype=np.float32)
    for lid in candidate_labels:
        if lid not in bones:
            continue
        a, b = bones[lid]
        d = _segment_distance(verts, a, b)
        better = d < best_d
        best_d[better] = d[better]
        best_lab[better] = lid
    return best_lab


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gender", default="male",
                     choices=["male", "female"])
    ap.add_argument("--rest-margin", type=float, default=1.5,
                     help="Flag a vert if its current bone is at least "
                          "this many units FARTHER than the nearest "
                          "limb bone. Larger = more conservative.")
    ap.add_argument("--rest-ratio", type=float, default=1.5,
                     help="Flag if dist(current_bone) > ratio × "
                          "dist(closest_limb_bone).")
    ap.add_argument("--strong-ratio", type=float, default=5.0,
                     help="If dist(current_bone) > ratio × "
                          "dist(closest_limb_bone), the bone-distance "
                          "pick is strong enough to BYPASS the k-NN "
                          "veto. Catches internally-consistent "
                          "mislabeled clusters whose neighbours all "
                          "share the same wrong label.")
    ap.add_argument("--pose-margin", type=float, default=3.0,
                     help="In a posed render, flag a vert if its "
                          "distance to its owning bone changed by more "
                          "than this many units vs. the rest pose. "
                          "Rigid-skin distance should be preserved.")
    ap.add_argument("--min-poses", type=int, default=2,
                     help="Vert must fail bone-following in this "
                          "many poses (default 2 — single-pose false "
                          "positives from torso bend are common).")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--min-agreement", type=int, default=5)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--effects", nargs="*", default=None)
    ap.add_argument("--size", type=int, nargs=2, default=(360, 640))
    return ap


def _to_seg_endpoints(bones: dict[int, tuple[np.ndarray, np.ndarray]]
                       ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    keys = sorted(bones.keys())
    aa = np.stack([bones[k][0] for k in keys])
    bb = np.stack([bones[k][1] for k in keys])
    return aa, bb, keys


def main():
    args = _build_arg_parser().parse_args()

    from faceview.assets import assets_dir
    from faceview.vision.ict_face import render_face_ict

    out_dir = Path(f"docs/skeleton_relabel/{args.gender}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Rest capture: verts, effective labels, pivots ──────────────
    p = _make_neutral_params(args.gender)
    cap = _capture_rig_io(lambda: render_face_ict(p, size=tuple(args.size)))
    rest = cap["rest"]
    fine_eff = cap["fine_eff"]
    rig = cap["rig"]
    # Snapshot rest pivots — the RigState is cached + mutated in-place,
    # so subsequent posed renders would otherwise rewrite this dict.
    pivots = {k: np.asarray(v, dtype=np.float32).copy()
              for k, v in rig.pivots.items()}
    # Keep an UNCORRECTED snapshot too: the dynamic bone-following
    # check compares posed pivots (from the rig, which uses the
    # uncorrected values) against rest. If we mirror-correct rest,
    # the check sees a fake "motion" and over-flags. So compare
    # posed_pivots against this raw snapshot for motion detection.
    raw_pivots = {k: v.copy() for k, v in pivots.items()}
    n_verts = len(rest)

    # ── Mirror-correct broken side pivots ──────────────────────────
    # The skeleton fitter for some meshes places one side's wrist
    # at the elbow (BP region detection fails). For our bone-distance
    # ground truth, mirror the LONGER (more reasonable) side's
    # joints if there's a clear asymmetry. We check |x| of each
    # joint; if a pair (L, R) differs by >30%, replace the shorter
    # one with the mirror of the longer.
    pair_keys = [("shoulder_L", "shoulder_R"), ("elbow_L", "elbow_R"),
                  ("wrist_L", "wrist_R"), ("hip_L", "hip_R"),
                  ("knee_L", "knee_R"), ("ankle_L", "ankle_R")]
    body_axis = 0  # X is the L-R bilateral axis
    for kL, kR in pair_keys:
        if kL not in pivots or kR not in pivots:
            continue
        L = pivots[kL]; R = pivots[kR]
        xL = abs(float(L[body_axis]))
        xR = abs(float(R[body_axis]))
        # Compare arm-length signature: distance from shoulder to wrist
        # along the chain. Easier: compare absolute joint X — if one
        # is much smaller than the other, it's likely been collapsed
        # toward the body centre. Use Y match too: wrist Y should be
        # far below shoulder Y; if a "wrist" Y is near elbow Y, it's
        # not really a wrist position.
        if xR < 0.6 * xL:
            print(f"  Mirroring {kL} → {kR}: |xR|={xR:.1f} << |xL|={xL:.1f}")
            mirrored = L.copy()
            mirrored[body_axis] = -L[body_axis]
            pivots[kR] = mirrored
        elif xL < 0.6 * xR:
            print(f"  Mirroring {kR} → {kL}: |xL|={xL:.1f} << |xR|={xR:.1f}")
            mirrored = R.copy()
            mirrored[body_axis] = -R[body_axis]
            pivots[kL] = mirrored
    print(f"Loaded labels: {n_verts} verts; "
            f"{len(pivots)} joint pivots: {sorted(pivots.keys())}")

    # ── Build bones ────────────────────────────────────────────────
    bones = _build_bones(pivots, rest, fine_eff)
    print(f"Built {len(bones)} bone segments")

    # ── A0. Cross-side anatomical sanity check ─────────────────────
    # Subject's L is +X, R is -X. Any vert labeled with a side
    # (4-15) whose rest position is on the WRONG side is
    # systematically mislabeled (e.g. hand_R-tagged verts at +X are
    # on the subject's left hand). The mirror label is the immediate
    # answer.
    side_R_labels = {5, 7, 9, 11, 13, 15}
    side_L_labels = {4, 6, 8, 10, 12, 14}
    side_pair = {5: 4, 7: 6, 9: 8, 11: 10, 13: 12, 15: 14,
                  4: 5, 6: 7, 8: 9, 10: 11, 12: 13, 14: 15}
    SIDE_TOLERANCE = 3.0  # units; allow small overlap at the midline
    cross_side = np.zeros(n_verts, dtype=bool)
    for lid in side_R_labels:
        m = (fine_eff == lid) & (rest[:, 0] > SIDE_TOLERANCE)
        cross_side |= m
    for lid in side_L_labels:
        m = (fine_eff == lid) & (rest[:, 0] < -SIDE_TOLERANCE)
        cross_side |= m
    cross_side_count = int(cross_side.sum())
    print(f"Cross-side mislabels: {cross_side_count}")

    # ── A. Static rest-pose bone-distance detector ─────────────────
    # For each vert, compute distance to its current label's bone AND
    # distance to its closest limb-bone. If the closest-limb-bone is
    # much nearer than the current-label-bone, the vert is anatomically
    # attached to the wrong chain.
    print("\nComputing rest-pose bone distances …")
    limb_label_ids = list(_LIMB_BONES.keys()) + list(_TIP_BONES.keys())
    closest_limb = _closest_bone_label(rest, bones, limb_label_ids)

    # Distance to each vert's current-label bone (limb verts only;
    # torso verts use degenerate point bones so the comparison isn't
    # meaningful).
    dist_to_own = np.full(n_verts, np.inf, dtype=np.float32)
    dist_to_closest = np.full(n_verts, np.inf, dtype=np.float32)
    for lid in limb_label_ids:
        m = fine_eff == lid
        if int(m.sum()) == 0:
            continue
        dist_to_own[m] = _bone_distance_to_label(rest[m], bones, lid)
    for lid in limb_label_ids:
        m = closest_limb == lid
        if int(m.sum()) == 0:
            continue
        dist_to_closest[m] = _bone_distance_to_label(rest[m], bones, lid)

    is_limb = np.isin(fine_eff, list(limb_label_ids))
    rest_offender = (
        is_limb
        & (closest_limb != fine_eff)
        & (dist_to_own - dist_to_closest > args.rest_margin)
        & (dist_to_own > args.rest_ratio
                * np.maximum(dist_to_closest, 1e-3)))
    n_rest = int(rest_offender.sum())
    print(f"Rest-pose bone-distance offenders: {n_rest}")

    # Per-transition breakdown
    rest_trans: Counter = Counter()
    for vi in np.where(rest_offender)[0]:
        rest_trans[(int(fine_eff[vi]), int(closest_limb[vi]))] += 1
    print("  Top rest transitions:")
    for (a, b), n in sorted(rest_trans.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {BPF_NAMES.get(a):10s} → {BPF_NAMES.get(b):10s}: {n}")

    # ── B. Per-pose bone-following detector ────────────────────────
    # Run each effect at u=0.5. For each vert, recompute its distance
    # to its OWNING bone in the posed frame and see if it changed by
    # more than pose-margin units. Rigid-skin transform should keep
    # this distance constant — divergence = wrong bone moved the vert.
    print("\nRunning per-effect bone-following test …")
    effects = args.effects or BODY_EFFECTS
    pose_records: dict[str, dict] = {}
    follow_fail_count = np.zeros(n_verts, dtype=np.int32)

    # Cache rest dist-to-own for each label
    rest_dist_own = np.full(n_verts, np.nan, dtype=np.float32)
    for lid in bones:
        m = fine_eff == lid
        if int(m.sum()) == 0:
            continue
        rest_dist_own[m] = _bone_distance_to_label(rest[m], bones, lid)

    for ename in effects:
        cap2 = _peak_render(args.gender, ename, size=tuple(args.size))
        if cap2 is None:
            continue
        posed = cap2["posed"]
        rig_p = cap2["rig"]
        # Rebuild bones using POSED pivots from the captured rig.
        # NOTE: the rig captured here is the SAME RigState (its
        # `pivots` attribute reflects the live skeleton after rig
        # rotations are propagated for the current frame). If pivots
        # aren't updated in-place, fall back to rest pivots — bone-
        # following will then trivially register motion as "wrong",
        # so skip in that case.
        posed_pivots = rig_p.pivots
        # Heuristic: detect if any pivot moved vs RAW rest pivots
        # (NOT the mirror-corrected ones — the mirror correction is a
        # diagnostic-only change to our snapshot and isn't applied to
        # the rig).
        moved_pivots = any(
            float(np.linalg.norm(np.asarray(posed_pivots[k])
                                       - raw_pivots[k])) > 0.1
            for k in raw_pivots if k in posed_pivots)
        if not moved_pivots:
            # Rig doesn't propagate pivots in-place — synthesise
            # posed pivots by applying the rig transform to rest
            # pivots: we just use the rest pivots and rely on the
            # rest-pose detector instead.
            pose_records[ename] = dict(
                rgb=cap2["rgb"], posed=posed, fail=np.zeros(n_verts, bool),
                centre=cap2["centre"], scale=cap2["scale"],
                yaw=cap2["render_yaw"], pitch=cap2["render_pitch"],
                skipped="static pivots")
            continue
        posed_bones = _build_bones(posed_pivots, posed, fine_eff)
        posed_dist_own = np.full(n_verts, np.nan, dtype=np.float32)
        for lid in posed_bones:
            m = fine_eff == lid
            if int(m.sum()) == 0:
                continue
            posed_dist_own[m] = _bone_distance_to_label(
                posed[m], posed_bones, lid)
        delta = np.abs(posed_dist_own - rest_dist_own)
        # Limb-vert only; torso bones are degenerate
        is_limb_eff = np.isin(fine_eff, list(limb_label_ids))
        fail = is_limb_eff & (delta > args.pose_margin)
        follow_fail_count += fail.astype(np.int32)
        print(f"  {ename:18s}  fail={int(fail.sum()):4d}  "
                f"max_delta={float(np.nanmax(delta)):.1f}")
        pose_records[ename] = dict(
            rgb=cap2["rgb"], posed=posed, fail=fail, delta=delta,
            centre=cap2["centre"], scale=cap2["scale"],
            yaw=cap2["render_yaw"], pitch=cap2["render_pitch"])

    follow_offender = follow_fail_count >= args.min_poses
    print(f"\nBone-following offenders "
            f"(fail in ≥{args.min_poses} poses): "
            f"{int(follow_offender.sum())}")

    # ── Combine + reassign ─────────────────────────────────────────
    offender_mask = rest_offender | follow_offender | cross_side
    offender_idx = np.where(offender_mask)[0]
    print(f"\nCombined offenders: {len(offender_idx)} "
            f"(rest={n_rest}, follow={int(follow_offender.sum())})")

    # Reassign each offender to the label whose bone is closest at
    # rest. (This is what the rest-pose detector already computed for
    # rest_offenders, but we also include follow_offenders here.)
    suggested = fine_eff.copy()
    transitions: Counter = Counter()
    # Strong-confidence mask: bone-distance ratio dominates k-NN.
    strong_mask = np.zeros(n_verts, dtype=bool)
    for vi in offender_idx:
        own_d = float(dist_to_own[vi])
        cls_d = float(dist_to_closest[vi])
        if cls_d > 1e-6 and own_d / cls_d >= args.strong_ratio:
            strong_mask[vi] = True

    for vi in offender_idx:
        old = int(fine_eff[vi])
        if cross_side[vi]:
            # Cross-side: vert is on the wrong side of midline for
            # its label. Only mirror if the mirrored bone is ALSO
            # actually nearby — otherwise the vert is probably a
            # torso/seam vert that happens to be tagged with a
            # side label, and the closest-limb pick is more
            # informative. Verify by checking that distance to the
            # mirrored bone is comparable to distance to the
            # closest-limb pick.
            mirror = side_pair.get(old)
            if mirror is not None and mirror in bones:
                a, b = bones[mirror]
                d_mirror = float(_segment_distance(rest[vi:vi+1], a, b)[0])
                d_closest = float(dist_to_closest[vi])
                if d_mirror <= d_closest * 1.3:
                    new_lab = mirror
                    strong_mask[vi] = True
                else:
                    new_lab = int(closest_limb[vi])
            else:
                new_lab = int(closest_limb[vi])
        else:
            new_lab = int(closest_limb[vi])
        if new_lab == old:
            continue
        suggested[vi] = new_lab
        transitions[(old, new_lab)] += 1

    # Safety: where bone-distance ratio is NOT extreme, let k-NN
    # veto / re-route the reassignment.
    if len(offender_idx) > 0:
        knn_lab, knn_conf = _knn_majority_label(
            rest, fine_eff, offender_idx,
            pool_mask=~offender_mask,
            k=args.knn_k, min_agreement=args.min_agreement)
        n_strong_kept = 0
        for j, vi in enumerate(offender_idx):
            if suggested[vi] == fine_eff[vi]:
                continue
            if strong_mask[vi]:
                # Bone-distance is overwhelming — ignore k-NN.
                n_strong_kept += 1
                continue
            if knn_conf[j] and int(knn_lab[j]) != int(suggested[vi]):
                old = int(fine_eff[vi])
                new = int(knn_lab[j])
                prev = int(suggested[vi])
                if (old, prev) in transitions:
                    transitions[(old, prev)] -= 1
                    if transitions[(old, prev)] <= 0:
                        del transitions[(old, prev)]
                if new == old:
                    suggested[vi] = old
                else:
                    suggested[vi] = new
                    transitions[(old, new)] += 1
        print(f"  Strong-ratio reassignments (bypass k-NN): "
                f"{n_strong_kept}")
    n_changed = int((suggested != fine_eff).sum())
    print(f"\nReassignments: {n_changed}")
    for (a, b), n in sorted(transitions.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {BPF_NAMES.get(a):10s} → {BPF_NAMES.get(b):10s}: {n}")

    # ── Diagnostic: rest bone-distance heat map ────────────────────
    h, w = args.size[1], args.size[0]
    pix = _project_with_camera(
        rest, centre=cap["centre"], scale=cap["scale"],
        yaw=cap["render_yaw"], pitch=cap["render_pitch"], size=(w, h))
    img = Image.fromarray(cap["rgb"].astype(np.uint8))
    drw = ImageDraw.Draw(img, "RGBA")
    for vi in np.where(rest_offender)[0]:
        px, py, _ = pix[vi]
        if not (0 <= px < w and 0 <= py < h):
            continue
        ring = _PALETTE_COLORS.get(int(closest_limb[vi]), (255, 255, 255))
        drw.line((px - 4, py, px + 4, py), fill=(255, 0, 255, 230), width=1)
        drw.line((px, py - 4, px, py + 4), fill=(255, 0, 255, 230), width=1)
        drw.ellipse((px - 3, py - 3, px + 3, py + 3),
                       outline=ring + (255,), width=1)
    Image.fromarray(_label_image(
        np.asarray(img),
        f"rest bone-distance  off={n_rest}")
    ).save(out_dir / "bone_distance_map.png")

    # ── Per-pose diagnostic ────────────────────────────────────────
    cells = []
    for ename, rec in pose_records.items():
        rgb = rec["rgb"]; h, w = rgb.shape[:2]
        pix = _project_with_camera(
            rec["posed"], centre=rec["centre"], scale=rec["scale"],
            yaw=rec["yaw"], pitch=rec["pitch"], size=(w, h))
        img = Image.fromarray(rgb.astype(np.uint8))
        drw = ImageDraw.Draw(img, "RGBA")
        skipped = rec.get("skipped")
        if not skipped:
            for vi in np.where(rec["fail"])[0]:
                px, py, _ = pix[vi]
                if not (0 <= px < w and 0 <= py < h):
                    continue
                color = (255, 0, 255, 230) if offender_mask[vi] else (
                    255, 140, 40, 200)
                drw.line((px - 4, py, px + 4, py), fill=color, width=1)
                drw.line((px, py - 4, px, py + 4), fill=color, width=1)
        title = f"{ename}  fail={int(rec['fail'].sum())}"
        if skipped:
            title += f"  ({skipped})"
        cells.append(_label_image(np.asarray(img), title))
    if cells:
        cols = 5
        rows = (len(cells) + cols - 1) // cols
        blank = np.zeros_like(cells[0])
        while len(cells) < rows * cols:
            cells.append(blank)
        rows_img = [np.hstack(cells[i*cols:(i+1)*cols]) for i in range(rows)]
        Image.fromarray(np.vstack(rows_img)).save(out_dir / "grid_bones.png")
        print(f"Wrote {out_dir / 'grid_bones.png'}")

    # ── Apply ──────────────────────────────────────────────────────
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
                         for vi in np.where(suggested != fine_eff)[0]}
        merged = {**existing, **new_overrides}
        out_doc = {
            "_comment": ("Skeleton-bone voxel relabel — verts whose "
                         "anatomical bone (rest distance) differs from "
                         "their current label, reassigned to the bone "
                         "they're closest to."),
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
