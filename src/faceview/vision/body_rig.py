"""Improved body rig — uses painted BPF region labels and the
fitted skeleton's joint positions for natural-looking limb / neck /
body rotations.

Replaces the older heuristic rig that used Vitruvian Y-bands and
X-thresholds for skinning weights, which leaked rotations onto
adjacent regions (torso lifts with shoulder, hand on leg, etc.).

Public entry points:

* :func:`build_rig_state` — pre-compute per-vertex group masks +
  joint pivots for a body mesh; cached so each render frame doesn't
  rebuild adjacency.
* :func:`apply_body_rig_v2` — apply a hierarchy of rotations
  (body → arm chain → leg chain) to body verts using the masks +
  pivots. Body rotation propagates to limb pivots, AND each parent
  rotation propagates to its child joint pivots (so the elbow stays
  attached to the rotated upper arm, etc.).

Joint rotations are clamped to anatomically realistic ranges so
extreme parameter values can't tear the mesh.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


# ── Anatomically realistic joint rotation limits (radians) ─────────
# These bracket each Euler component to natural ROM. The skin rig
# is approximate; clamping here keeps artistic poses inside what
# the mesh can absorb without tearing.
_JOINT_LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    # Single-bone rotation skinning fundamentally can't deform a
    # rigid body mesh through extreme angles (90°+) without seam
    # tearing. We clamp to ANATOMICALLY VALID + RIG-FRIENDLY ranges
    # — the seam triangles in this mesh look acceptable up to
    # ~50° rotation and break down past ~70°.
    "shoulder": {"pitch": (-0.90, 0.90), "roll": (-0.90, 0.90),
                  "yaw":   (-0.40, 0.40)},
    # Elbow flex up to ~110° (still natural) — past this the
    # forearm/hand seam stretches visibly.
    "elbow":    {"pitch": (-1.90, 0.05), "roll": (-0.12, 0.12),
                  "yaw":   (-0.12, 0.12)},
    "wrist":    {"pitch": (-0.60, 0.60), "roll": (-0.50, 0.50),
                  "yaw":   (-0.30, 0.30)},
    "hip":      {"pitch": (-0.90, 0.45), "roll": (-0.45, 0.35),
                  "yaw":   (-0.35, 0.35)},
    "knee":     {"pitch": (-0.05, 1.70), "roll": (-0.08, 0.08),
                  "yaw":   (-0.08, 0.08)},
    "ankle":    {"pitch": (-0.40, 0.40), "roll": (-0.25, 0.25),
                  "yaw":   (-0.15, 0.15)},
}


def _clamp_joint(joint: str, yaw: float, pitch: float, roll: float
                   ) -> tuple[float, float, float]:
    lims = _JOINT_LIMITS.get(joint)
    if lims is None:
        return yaw, pitch, roll
    yl, yh = lims.get("yaw",   (-3.14, 3.14))
    pl, ph = lims.get("pitch", (-3.14, 3.14))
    rl, rh = lims.get("roll",  (-3.14, 3.14))
    return (max(yl, min(yh, yaw)),
              max(pl, min(ph, pitch)),
              max(rl, min(rh, roll)))


# ── BPF group composites for hierarchical limb chains ──────────────


def _arm_chain_masks(fine: np.ndarray, side: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (full_arm, below_elbow, hand_only) hard masks (bool)
    for the given side. ``full_arm`` is everything the shoulder
    moves; ``below_elbow`` is what the elbow moves; ``hand_only``
    is what the wrist moves."""
    from faceview.vision.body_3d import (
        BPF_UPPER_ARM_L, BPF_UPPER_ARM_R,
        BPF_FOREARM_L, BPF_FOREARM_R,
        BPF_HAND_L, BPF_HAND_R,
    )
    if side == "L":
        ua, fa, hd = BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L
    else:
        ua, fa, hd = BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R
    full = (fine == ua) | (fine == fa) | (fine == hd)
    below_elbow = (fine == fa) | (fine == hd)
    hand = (fine == hd)
    return full, below_elbow, hand


def _leg_chain_masks(fine: np.ndarray, side: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from faceview.vision.body_3d import (
        BPF_THIGH_L, BPF_THIGH_R, BPF_SHIN_L, BPF_SHIN_R,
        BPF_FOOT_L, BPF_FOOT_R,
    )
    if side == "L":
        th, sh, ft = BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L
    else:
        th, sh, ft = BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R
    full = (fine == th) | (fine == sh) | (fine == ft)
    below_knee = (fine == sh) | (fine == ft)
    foot = (fine == ft)
    return full, below_knee, foot


def _torso_mask(fine: np.ndarray) -> np.ndarray:
    """Verts that the body's hip-pivot rotation affects: torso +
    arms + head/neck (everything above the hip line). Legs stay put
    so a body-bow looks natural (legs stay grounded)."""
    from faceview.vision.body_3d import (
        BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN,
        BPF_UPPER_ARM_L, BPF_UPPER_ARM_R, BPF_FOREARM_L, BPF_FOREARM_R,
        BPF_HAND_L, BPF_HAND_R,
    )
    return ((fine == BPF_NECK) | (fine == BPF_CHEST)
              | (fine == BPF_ABDOMEN) | (fine == BPF_PELVIS_SKIN)
              | (fine == BPF_UPPER_ARM_L) | (fine == BPF_UPPER_ARM_R)
              | (fine == BPF_FOREARM_L) | (fine == BPF_FOREARM_R)
              | (fine == BPF_HAND_L) | (fine == BPF_HAND_R))


# ── Mesh-adjacency boundary smoothing ──────────────────────────────


def _build_adjacency(tris: np.ndarray, n: int
                       ) -> list[list[int]]:
    """1-ring vertex adjacency from the triangle list."""
    adj: list[set[int]] = [set() for _ in range(n)]
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        adj[a].add(b); adj[a].add(c)
        adj[b].add(a); adj[b].add(c)
        adj[c].add(a); adj[c].add(b)
    return [list(s) for s in adj]


def _smooth_labels_mode(fine: np.ndarray, adj: list[list[int]],
                          n_iters: int = 2) -> np.ndarray:
    """Mode-filter pass: each vert adopts the most common label
    among itself + its 1-ring neighbours. Smooths classification
    noise at body-part boundaries (stray verts assigned to a
    neighbouring region get pulled back into the majority cluster).
    Repeated ``n_iters`` times.
    """
    out = fine.astype(np.int32, copy=True)
    n = len(out)
    for _ in range(max(0, n_iters)):
        new_labels = out.copy()
        for i in range(n):
            neighbours = adj[i]
            if not neighbours:
                continue
            # Count labels including self; bincount with the
            # observed range is faster than collections.Counter.
            counts: dict[int, int] = {}
            counts[int(out[i])] = 1
            for j in neighbours:
                lab = int(out[j])
                counts[lab] = counts.get(lab, 0) + 1
            best_lab = int(out[i])
            best_n = counts[best_lab]
            for lab, c in counts.items():
                if c > best_n:
                    best_n = c
                    best_lab = lab
            new_labels[i] = best_lab
        if np.array_equal(new_labels, out):
            break
        out = new_labels
    return out


def _bilateral_fade(mask: np.ndarray, adj: list[list[int]],
                      ring_weights_in: tuple[float, ...] = (0.92, 0.97),
                      ring_weights_out: tuple[float, ...] = (0.30, 0.10),
                      allowed_out: np.ndarray | None = None
                     ) -> np.ndarray:
    """Symmetric multi-ring fade across the limb/torso boundary.

    The crucial visible artifact at extreme arm rotations is the
    seam-spanning triangles between weight=1.0 arm verts and
    weight=0.0 torso verts — when one vert moves 80 units and its
    triangle partner stays put, the triangle becomes a long thin
    sliver (the "stretched skin"). A smooth bilateral fade trades
    a tiny torso-side movement for huge reduction in those slivers.

    For each ring N from the boundary:
    * In-mask ring N: weight = ring_weights_in[N]  (close to 1.0)
    * Out-of-mask ring N: weight = ring_weights_out[N]  (close to 0)

    Triangles spanning the boundary now connect verts whose weights
    differ by ~0.30 instead of 1.00 — far less stretch.
    """
    n = len(mask)
    out = np.zeros(n, dtype=np.float32)
    out[mask] = 1.0
    # Identify in-mask and out-of-mask seam rings. The OUT-seam is
    # additionally restricted to verts in ``allowed_out`` (labels
    # that are anatomically valid neighbours of this region) so the
    # fade doesn't leak into far body parts that just happen to be
    # mesh-adjacent — e.g. arm verts at the side of the body are
    # adjacent to thigh verts, but a shoulder rotation should never
    # move the thigh.
    seam_in = np.zeros(n, dtype=bool)
    seam_out = np.zeros(n, dtype=bool)
    for i in range(n):
        if mask[i]:
            for j in adj[i]:
                if not mask[j]:
                    seam_in[i] = True
                    break
        else:
            if allowed_out is not None and not allowed_out[i]:
                continue
            for j in adj[i]:
                if mask[j]:
                    seam_out[i] = True
                    break

    # Apply ring 0 weights.
    out[seam_in] = ring_weights_in[0]
    out[seam_out] = ring_weights_out[0]

    # Walk subsequent rings.
    for ring_n in range(1, max(len(ring_weights_in),
                                  len(ring_weights_out))):
        # In-mask ring N: in-mask vert with neighbour in ring N-1
        # in-side ring (and itself not yet in any ring).
        new_in = np.zeros(n, dtype=bool)
        new_out = np.zeros(n, dtype=bool)
        if ring_n < len(ring_weights_in):
            for i in range(n):
                if not mask[i] or seam_in[i]:
                    continue
                # already at deeper ring weight or default 1.0?
                # Detect by comparing current weight to default.
                if abs(out[i] - 1.0) > 1e-6:
                    continue
                for j in adj[i]:
                    if seam_in[j]:
                        new_in[i] = True
                        break
            out[new_in] = ring_weights_in[ring_n]
            seam_in = seam_in | new_in
        if ring_n < len(ring_weights_out):
            for i in range(n):
                if mask[i] or seam_out[i]:
                    continue
                if allowed_out is not None and not allowed_out[i]:
                    continue
                if abs(out[i] - 0.0) > 1e-6:
                    continue
                for j in adj[i]:
                    if seam_out[j]:
                        new_out[i] = True
                        break
            out[new_out] = ring_weights_out[ring_n]
            seam_out = seam_out | new_out
    return out


def _soft_weight(mask: np.ndarray, adj: list[list[int]],
                   mode: str = "bilateral",
                   allowed_out: np.ndarray | None = None
                  ) -> np.ndarray:
    """Skinning weight for a region mask.

    Out-of-mask verts always get 0.0 (the torso never moves with
    arm rotations).

    ``mode`` controls how IN-mask verts near the seam are graded so
    the in-region triangles between deep-mask (weight 1.0) and
    seam (lower weight) don't visibly distort:

    * ``"hard"`` — every in-mask vert gets 1.0. Sharp seam, but
      tightest within-region rigidity.
    * ``"inner_1ring"`` — seam-ring verts (in-mask, adjacent to
      out-of-mask) get 0.7. Single-ring discontinuity — the
      previous default but causes some triangle stretch.
    * ``"graded_3ring"`` (default) — seam-ring 0.85, second ring
      0.95, deeper 1.0. Spreads the seam discontinuity over 3
      rings so each adjacent triangle pair has only a small
      weight gap, dramatically reducing visible stretch.
    """
    if mode == "bilateral":
        return _bilateral_fade(mask, adj, allowed_out=allowed_out)

    n = len(mask)
    out = np.zeros(n, dtype=np.float32)
    out[mask] = 1.0
    if mode == "hard":
        return out

    # Identify seam-ring (1-ring): in-mask verts with ≥1 out-of-mask
    # neighbour.
    seam_ring = np.zeros(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in adj[i]:
            if not mask[j]:
                seam_ring[i] = True
                break

    if mode == "inner_1ring":
        out[seam_ring] = 0.7
        return out

    # graded_3ring: seam=0.85, ring2=0.95
    second_ring = np.zeros(n, dtype=bool)
    for i in range(n):
        if not mask[i] or seam_ring[i]:
            continue
        for j in adj[i]:
            if seam_ring[j]:
                second_ring[i] = True
                break
    out[second_ring] = 0.95
    out[seam_ring] = 0.85
    return out


# ── Rig state cache ────────────────────────────────────────────────


@dataclass
class RigState:
    """Pre-computed masks + pivot dict for a particular body mesh."""
    n_verts: int
    weights: dict[str, np.ndarray]   # group_name → per-vert weight
    pivots: dict[str, np.ndarray]    # joint_name → 3D position
    # Adjacency + seam-ring index used by the post-rotation seam
    # smoother. Built once with the masks; cheap to keep around.
    adj: list[list[int]] | None = None
    seam_indices: dict[str, np.ndarray] | None = None


@lru_cache(maxsize=4)
def _cached_rig_state(verts_hash: int, n_verts: int,
                        body_morph_key: int) -> RigState | None:
    """Cache key — keyed by body morph (rounded to int x100) and
    vert count, since the masks + adjacency are expensive to build."""
    return None  # populated by build_rig_state(); _cached_rig_state
                  # only exists so build_rig_state can store via
                  # functools.lru_cache via an outer wrapper.


def reclassify_via_stretch_test(
    verts: np.ndarray, tris: np.ndarray, fine_labels: np.ndarray,
    pivots: dict[str, np.ndarray], masks: dict[str, np.ndarray],
    n_passes: int = 3, edge_grow_threshold: float = 2.0,
) -> np.ndarray:
    """Use trial rotations to identify mis-labeled voxels and
    reclassify them.

    Logic: for each joint rotation trial, run the rotation. For
    every triangle whose longest edge grows by more than the
    threshold, look at its three verts:
        * If 2 verts moved with the group and 1 stayed → the stayed
          vert is most likely mis-labeled (geometrically belongs
          with the moving group but its own label put it in the
          static region). Re-label it to match the moving verts.
        * If 1 vert moved and 2 stayed → the moving vert is the
          mis-classified one (probably a stray island; should join
          the static neighbours).

    Iterate until convergence or n_passes reached. After that, the
    labels are physically self-consistent: the label-mask boundary
    runs along edges that DON'T stretch under joint rotation.
    """
    new_labels = fine_labels.copy()
    n_v = len(verts)

    # Trial rotations: (mask_key, in_group_labels, joint_pivot_key, angle, axis)
    # We pick a single representative rotation per joint that
    # exercises the largest moving cluster.
    from faceview.vision.body_3d import (
        BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L,
        BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R,
        BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L,
        BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R,
    )
    trials = [
        ("arm_L_full", "shoulder_L",
            (BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L), 0.85, "roll"),
        ("arm_R_full", "shoulder_R",
            (BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R), -0.85, "roll"),
        ("arm_L_belowel", "elbow_L",
            (BPF_FOREARM_L, BPF_HAND_L), -1.20, "pitch"),
        ("arm_R_belowel", "elbow_R",
            (BPF_FOREARM_R, BPF_HAND_R), -1.20, "pitch"),
        ("leg_L_full", "hip_L",
            (BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L), -0.70, "pitch"),
        ("leg_R_full", "hip_R",
            (BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R), -0.70, "pitch"),
    ]

    for pass_n in range(n_passes):
        # Rebuild masks for this iteration's labels.
        local_masks = {}
        for side in ("L", "R"):
            arm = ((new_labels == (BPF_UPPER_ARM_L if side == "L"
                                      else BPF_UPPER_ARM_R))
                    | (new_labels == (BPF_FOREARM_L if side == "L"
                                          else BPF_FOREARM_R))
                    | (new_labels == (BPF_HAND_L if side == "L"
                                          else BPF_HAND_R)))
            fore = ((new_labels == (BPF_FOREARM_L if side == "L"
                                       else BPF_FOREARM_R))
                     | (new_labels == (BPF_HAND_L if side == "L"
                                           else BPF_HAND_R)))
            leg = ((new_labels == (BPF_THIGH_L if side == "L"
                                       else BPF_THIGH_R))
                    | (new_labels == (BPF_SHIN_L if side == "L"
                                          else BPF_SHIN_R))
                    | (new_labels == (BPF_FOOT_L if side == "L"
                                          else BPF_FOOT_R)))
            local_masks[f"arm_{side}_full"]    = arm.astype(np.float32)
            local_masks[f"arm_{side}_belowel"] = fore.astype(np.float32)
            local_masks[f"leg_{side}_full"]    = leg.astype(np.float32)

        any_change = False
        for mask_key, pivot_key, group_labels, angle, axis in trials:
            anchor = pivots.get(pivot_key)
            w = local_masks.get(mask_key)
            if anchor is None or w is None:
                continue
            yaw = angle if axis == "yaw" else 0.0
            pitch = angle if axis == "pitch" else 0.0
            roll = angle if axis == "roll" else 0.0
            R = _rmat(yaw, pitch, roll)
            diff = verts - anchor
            rotated = (diff @ R.T) + anchor
            wcol = w[:, None]
            v_test = (verts * (1.0 - wcol)
                       + rotated * wcol).astype(np.float32)

            # Per-tri longest edge growth
            e1p = verts[tris[:, 1]] - verts[tris[:, 0]]
            e2p = verts[tris[:, 2]] - verts[tris[:, 1]]
            e3p = verts[tris[:, 0]] - verts[tris[:, 2]]
            e1q = v_test[tris[:, 1]] - v_test[tris[:, 0]]
            e2q = v_test[tris[:, 2]] - v_test[tris[:, 1]]
            e3q = v_test[tris[:, 0]] - v_test[tris[:, 2]]
            lp = np.maximum(np.maximum(
                np.linalg.norm(e1p, axis=1),
                np.linalg.norm(e2p, axis=1)),
                np.linalg.norm(e3p, axis=1))
            lq = np.maximum(np.maximum(
                np.linalg.norm(e1q, axis=1),
                np.linalg.norm(e2q, axis=1)),
                np.linalg.norm(e3q, axis=1))
            growth = lq / np.maximum(lp, 1e-6)
            bad_tris = np.where(growth > edge_grow_threshold)[0]

            if len(bad_tris) == 0:
                continue

            # For each bad triangle, count which verts are in the
            # moving group. If 2 are in and 1 is out, the OUT vert
            # is mis-classified — re-label it to a label from
            # ``group_labels`` (pick the most common label among
            # its neighbours that are in the group).
            for ti in bad_tris:
                vs = tris[ti]
                in_group = [bool(w[v] > 0.5) for v in vs]
                in_count = sum(in_group)
                if in_count == 2:
                    # The single out-of-group vert is suspect.
                    out_vi = int(vs[in_group.index(False)])
                    # Pick the dominant label among its in-group
                    # vert-neighbours (in this triangle).
                    neighbour_labels = [int(new_labels[int(vs[k])])
                                            for k in range(3)
                                            if in_group[k]]
                    if neighbour_labels:
                        # Pick most common, but it must be in the
                        # group's allowed labels.
                        from collections import Counter
                        cnt = Counter(neighbour_labels)
                        for lab, _ in cnt.most_common():
                            if lab in group_labels:
                                if int(new_labels[out_vi]) != lab:
                                    new_labels[out_vi] = lab
                                    any_change = True
                                break
                elif in_count == 1:
                    # The single in-group vert is the suspect (lone
                    # island). Re-label it to its out-of-group
                    # neighbours in this triangle.
                    in_vi = int(vs[in_group.index(True)])
                    neighbour_labels = [int(new_labels[int(vs[k])])
                                            for k in range(3)
                                            if not in_group[k]]
                    if neighbour_labels:
                        from collections import Counter
                        cnt = Counter(neighbour_labels)
                        new_lab = cnt.most_common(1)[0][0]
                        if int(new_labels[in_vi]) != new_lab:
                            new_labels[in_vi] = new_lab
                            any_change = True
        if not any_change:
            break
    return new_labels


def _is_intra_chain_triangle(
    l0: int, l1: int, l2: int
) -> bool:
    """A triangle is INTRA-CHAIN if all three of its vert labels
    belong to the same skeletal chain (arm, leg, torso). These
    triangles are legitimate seam triangles that connect adjacent
    bone segments and MUST be preserved — removing them creates
    visible holes in the silhouette (floating hands, gapped
    shoulders, etc.).
    """
    from faceview.vision.body_3d import (
        BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN,
        BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L,
        BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R,
        BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L,
        BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R,
    )
    chains = (
        # Torso + adjacent (chest connects to either arm, hip area
        # connects to either thigh; we keep these in the same
        # protected chain so legitimate shoulder/hip seams survive).
        {BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN,
          BPF_UPPER_ARM_L, BPF_UPPER_ARM_R,
          BPF_THIGH_L, BPF_THIGH_R},
        # Left arm chain
        {BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L},
        # Right arm chain
        {BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R},
        # Left leg chain
        {BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L},
        # Right leg chain
        {BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R},
    )
    s = {l0, l1, l2}
    for chain in chains:
        if s.issubset(chain):
            return True
    return False


def filter_empirical_bad_triangles(
    verts: np.ndarray, tris: np.ndarray, fine_labels: np.ndarray,
    pivots: dict[str, np.ndarray] | None = None,
    masks: dict[str, np.ndarray] | None = None,
    edge_grow_max: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical bad-triangle detection.

    Runs trial rotations on each joint at the maximum anatomical
    angle. Any triangle whose longest edge grows by more than
    ``edge_grow_max`` (default 3×) is flagged as bad and stripped
    from the mesh.

    This catches phantom triangles that the label-pair filter
    missed (e.g. legitimate-by-labels but geometrically broken
    triangles) AND dynamic stretches caused by mis-classified
    verts on either side of a true anatomical seam.

    Returns (kept_tris, removed_mask).
    """
    if pivots is None or masks is None:
        return tris, np.zeros(len(tris), dtype=bool)

    n_tris = len(tris)
    bad = np.zeros(n_tris, dtype=bool)

    # Pre-compute intra-chain protection: triangles whose labels are
    # all in one skeletal chain are legitimate seams and immune to
    # the empirical filter. Removing them creates visible holes
    # (floating hands, gapped shoulders, etc.).
    l0 = fine_labels[tris[:, 0]]
    l1 = fine_labels[tris[:, 1]]
    l2 = fine_labels[tris[:, 2]]
    protected = np.zeros(n_tris, dtype=bool)
    for i in range(n_tris):
        if _is_intra_chain_triangle(int(l0[i]), int(l1[i]),
                                          int(l2[i])):
            protected[i] = True

    def _R(yaw: float, pitch: float, roll: float) -> np.ndarray:
        return _rmat(yaw, pitch, roll)

    def _edge_growth(verts_post: np.ndarray) -> np.ndarray:
        e1_pre = verts[tris[:, 1]] - verts[tris[:, 0]]
        e2_pre = verts[tris[:, 2]] - verts[tris[:, 1]]
        e3_pre = verts[tris[:, 0]] - verts[tris[:, 2]]
        e1_post = verts_post[tris[:, 1]] - verts_post[tris[:, 0]]
        e2_post = verts_post[tris[:, 2]] - verts_post[tris[:, 1]]
        e3_post = verts_post[tris[:, 0]] - verts_post[tris[:, 2]]
        len_pre = np.maximum(np.maximum(
            np.linalg.norm(e1_pre, axis=1),
            np.linalg.norm(e2_pre, axis=1)),
            np.linalg.norm(e3_pre, axis=1))
        len_post = np.maximum(np.maximum(
            np.linalg.norm(e1_post, axis=1),
            np.linalg.norm(e2_post, axis=1)),
            np.linalg.norm(e3_post, axis=1))
        return len_post / np.maximum(len_pre, 1e-6)

    # Trial rotations CALIBRATED to runtime joint clamps so we
    # only filter triangles that misbehave at REALISTIC rotation
    # magnitudes. Using larger trials over-removes legitimate seam
    # triangles and leaves visible holes in the body silhouette.
    trials: list[tuple[str, str, float]] = [
        ("shoulder_L", "arm_L_full",    0.85),
        ("shoulder_R", "arm_R_full",    0.85),
        ("elbow_L",    "arm_L_belowel", -1.50),
        ("elbow_R",    "arm_R_belowel", -1.50),
        ("hip_L",      "leg_L_full",   -0.70),
        ("hip_R",      "leg_R_full",   -0.70),
        ("knee_L",     "leg_L_belowkn", 1.20),
        ("knee_R",     "leg_R_belowkn", 1.20),
    ]

    for joint, mask_key, angle in trials:
        anchor = pivots.get(joint)
        w = masks.get(mask_key)
        if anchor is None or w is None:
            continue
        # Apply pure roll rotation around the joint axis
        for axis_idx in (0, 1, 2):  # try yaw, pitch, roll axes
            yaw = angle if axis_idx == 0 else 0.0
            pitch = angle if axis_idx == 1 else 0.0
            roll = angle if axis_idx == 2 else 0.0
            R = _R(yaw, pitch, roll)
            diff = verts - anchor
            rotated = (diff @ R.T) + anchor
            wcol = w[:, None]
            v_test = (verts * (1.0 - wcol) + rotated * wcol)
            growth = _edge_growth(v_test.astype(np.float32))
            bad |= growth > edge_grow_max
    # Never filter protected intra-chain triangles.
    bad &= ~protected
    return tris[~bad], bad


def _connected_components_per_label(
    fine_labels: np.ndarray, adj: list[list[int]]
) -> dict[int, list[set[int]]]:
    """Per-label connected components in mesh adjacency. Returns
    ``{label_id: [component_set, ...]}``."""
    n = len(fine_labels)
    visited = np.zeros(n, dtype=bool)
    components: dict[int, list[set[int]]] = {}
    for v in range(n):
        if visited[v]:
            continue
        lab = int(fine_labels[v])
        comp: set[int] = set()
        stack = [v]
        while stack:
            u = stack.pop()
            if visited[u] or int(fine_labels[u]) != lab:
                continue
            visited[u] = True
            comp.add(u)
            for w in adj[u]:
                if not visited[w] and int(fine_labels[w]) == lab:
                    stack.append(w)
        components.setdefault(lab, []).append(comp)
    return components


def cleanup_spatial_outliers(
    fine_labels: np.ndarray, verts: np.ndarray,
    z_score_threshold: float = 2.5
) -> np.ndarray:
    """Per-label spatial outlier reclassification.

    For each label, compute the 3D centroid and per-axis std-dev
    of its main vert cluster. Any vert whose Mahalanobis-style
    z-score (max over axes) exceeds the threshold is a spatial
    outlier — its 3D position disagrees with its label.

    Reassign each spatial outlier to the label whose centroid
    is CLOSEST to its 3D position. This catches the scenario
    where a vert at the foot is labeled as arm: connected-component
    analysis misses it (the mesh has an arm→foot edge somewhere)
    but the 3D position clearly doesn't match the arm cluster.
    """
    out = fine_labels.copy()
    n_labels = 16
    centroids = np.zeros((n_labels, 3), dtype=np.float32)
    stds = np.zeros((n_labels, 3), dtype=np.float32)
    sizes = np.zeros(n_labels, dtype=np.int32)
    for lid in range(n_labels):
        m = out == lid
        sizes[lid] = int(m.sum())
        if sizes[lid] > 5:
            pts = verts[m]
            centroids[lid] = pts.mean(axis=0)
            stds[lid] = pts.std(axis=0)

    n_change = 0
    for i in range(len(out)):
        own = int(out[i])
        if sizes[own] <= 5:
            continue
        # Per-axis z-score from own label's centroid.
        std = stds[own]
        std_safe = np.maximum(std, 1.0)  # avoid div by ~0
        diff = np.abs(verts[i] - centroids[own])
        z = float((diff / std_safe).max())
        if z <= z_score_threshold:
            continue
        # Find closest non-own label.
        best_lab = own
        best_d = float("inf")
        for lid in range(n_labels):
            if lid == own or sizes[lid] <= 5:
                continue
            d = float(np.linalg.norm(verts[i] - centroids[lid]))
            if d < best_d:
                best_d = d
                best_lab = lid
        own_d = float(np.linalg.norm(verts[i] - centroids[own]))
        # Only reassign if the new label is meaningfully closer.
        if best_lab != own and best_d < own_d * 0.7:
            out[i] = best_lab
            n_change += 1
    if n_change:
        print(f"[body_rig] spatial outlier cleanup: "
                f"reassigned {n_change} verts")
    return out


def cleanup_stray_components(
    fine_labels: np.ndarray, adj: list[list[int]],
    verts: np.ndarray | None = None,
    min_size_ratio: float = 0.20,
    max_dist_ratio: float = 0.6,
) -> np.ndarray:
    """Reassign label-strays back to the body.

    For each label, find the connected components in mesh
    adjacency. The LARGEST component is the canonical cluster for
    that label. A smaller component is reassigned (to its border
    neighbours' dominant label) if EITHER:
        * Its size < ``min_size_ratio`` × largest, OR
        * Its centroid is more than ``max_dist_ratio`` × main-
          cluster's diameter away from the main cluster's centroid
          (spatial isolation check — catches small disconnected
          patches geographically separated from the body).

    The combination catches: small accidental marks (size check)
    AND larger but spatially isolated patches (distance check).

    Iterates so cascading corrections converge.
    """
    out = fine_labels.copy()
    n_pass = 4
    for _ in range(n_pass):
        components = _connected_components_per_label(out, adj)
        any_change = False
        for lab, comps in components.items():
            if len(comps) <= 1:
                continue
            # Find largest component (the canonical cluster).
            comps_sorted = sorted(comps, key=len, reverse=True)
            biggest = comps_sorted[0]
            biggest_size = len(biggest)
            threshold_size = max(2, int(biggest_size * min_size_ratio))

            # Compute main cluster centroid + diameter for spatial
            # check.
            if verts is not None:
                main_pts = verts[list(biggest)]
                main_centre = main_pts.mean(axis=0)
                main_diameter = float(
                    np.linalg.norm(main_pts.max(axis=0)
                                       - main_pts.min(axis=0)))
            else:
                main_centre = None
                main_diameter = 0.0

            for comp in comps_sorted[1:]:
                stray = len(comp) < threshold_size
                if not stray and main_centre is not None and \
                        main_diameter > 1e-3:
                    pts = verts[list(comp)]
                    c = pts.mean(axis=0)
                    dist = float(np.linalg.norm(c - main_centre))
                    if dist > max_dist_ratio * main_diameter:
                        stray = True
                if not stray:
                    continue
                from collections import Counter
                border_labels: Counter = Counter()
                for v in comp:
                    for w in adj[v]:
                        if w not in comp:
                            border_labels[int(out[w])] += 1
                if not border_labels:
                    continue
                new_lab = border_labels.most_common(1)[0][0]
                if new_lab == lab:
                    continue
                for v in comp:
                    out[v] = new_lab
                any_change = True
        if not any_change:
            break
    return out


def _apply_manual_overrides(fine_labels: np.ndarray,
                                body_morph: float = 1.0) -> np.ndarray:
    """Load manual vert→label overrides and apply them on top of the
    auto-classified labels. Silently no-op if the file is missing or
    malformed.

    The loader picks a gender-specific file based on ``body_morph``:
    morph ≥ 0 → ``body_label_overrides_male.json``;
    morph <  0 → ``body_label_overrides_female.json``. This keeps
    male and female reassignments separate, since the two meshes
    have different vertex counts and topology — a vert index that's
    a hand on one mesh might be a foot on the other.

    File format:
    ```json
    {
      "_comment": "Map vert_idx (str) → BPF label id (int)",
      "_legend": {"0": "neck", "1": "chest", "4": "u_arm_L", ...},
      "152": 4,
      "983": 8
    }
    ```
    """
    import json
    from pathlib import Path
    try:
        from faceview.assets import assets_dir
    except ImportError:
        return fine_labels
    suffix = "male" if body_morph >= 0 else "female"
    path = assets_dir() / f"body_label_overrides_{suffix}.json"
    # Legacy fallback — the original single-file format. Only loaded
    # if no gender-specific file exists, for back-compat.
    if not path.exists():
        path = assets_dir() / "body_label_overrides.json"
    if not path.exists():
        return fine_labels
    try:
        data = json.loads(path.read_text())
    except Exception:
        return fine_labels
    out = fine_labels.copy()
    n_applied = 0
    for k, v in data.items():
        if k.startswith("_"):
            continue
        try:
            vi = int(k)
            lab = int(v)
        except (TypeError, ValueError):
            continue
        if 0 <= vi < len(out) and 0 <= lab <= 15:
            out[vi] = lab
            n_applied += 1
    if n_applied:
        print(f"[body_rig] applied {n_applied} manual label overrides")
    return out


def filter_phantom_triangles(
    tris: np.ndarray, fine_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove triangles whose verts span anatomically-disconnected
    BPF regions — e.g. hand-to-thigh bridges that the body OBJ
    shell has between arms hanging at the sides and the outer
    thighs in T-pose. When the arm rotates these phantoms tear
    into long sail-shaped slivers (the visible "stretched skin"
    artifact).

    Returns (kept_tris, removed_mask).
    """
    # Anatomical chain adjacency. Two labels can validly share a
    # triangle only if connected here.
    from faceview.vision.body_3d import (
        BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN,
        BPF_UPPER_ARM_L, BPF_UPPER_ARM_R,
        BPF_FOREARM_L, BPF_FOREARM_R,
        BPF_HAND_L, BPF_HAND_R,
        BPF_THIGH_L, BPF_THIGH_R,
        BPF_SHIN_L, BPF_SHIN_R,
        BPF_FOOT_L, BPF_FOOT_R,
    )
    valid_pairs: set[tuple[int, int]] = set()
    # All torso parts inter-connect (chest/abdomen/neck/pelvis are
    # one continuous shell with mesh triangles between any pair).
    torso = {BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN}
    for a in torso:
        for b in torso:
            valid_pairs.add((a, b))
    # Hip is bilateral — abdomen and pelvis both connect to either
    # thigh through the natural mesh continuity.
    for hip in (BPF_ABDOMEN, BPF_PELVIS_SKIN):
        for thigh in (BPF_THIGH_L, BPF_THIGH_R):
            valid_pairs.add((hip, thigh))
            valid_pairs.add((thigh, hip))
    # Shoulder area: chest + neck adjacent to either upper arm.
    for sh_t in (BPF_CHEST, BPF_NECK):
        for sh_a in (BPF_UPPER_ARM_L, BPF_UPPER_ARM_R):
            valid_pairs.add((sh_t, sh_a))
            valid_pairs.add((sh_a, sh_t))
    # Limb chains (intra-limb connectivity only).
    for chain in (
        (BPF_UPPER_ARM_L, BPF_FOREARM_L, BPF_HAND_L),
        (BPF_UPPER_ARM_R, BPF_FOREARM_R, BPF_HAND_R),
        (BPF_THIGH_L, BPF_SHIN_L, BPF_FOOT_L),
        (BPF_THIGH_R, BPF_SHIN_R, BPF_FOOT_R),
    ):
        for a, b in zip(chain, chain[1:]):
            valid_pairs.add((a, b))
            valid_pairs.add((b, a))
        # Same-limb endpoints can also share a triangle
        # (e.g. upper_arm and hand at the elbow corner via the
        # forearm being the shared neighbour) — narrow case but
        # allowed.
        valid_pairs.add((chain[0], chain[2]))
        valid_pairs.add((chain[2], chain[0]))
    # A label is always compatible with itself.
    for lid in range(16):
        valid_pairs.add((lid, lid))

    l0 = fine_labels[tris[:, 0]]
    l1 = fine_labels[tris[:, 1]]
    l2 = fine_labels[tris[:, 2]]
    valid_arr = np.array(sorted(valid_pairs), dtype=np.int32)
    valid_set = set(valid_pairs)

    keep = np.ones(len(tris), dtype=bool)
    for i in range(len(tris)):
        a, b, c = int(l0[i]), int(l1[i]), int(l2[i])
        if ((a, b) not in valid_set
                or (b, c) not in valid_set
                or (a, c) not in valid_set):
            keep[i] = False
    return tris[keep], ~keep


def build_rig_state(body_verts: np.ndarray, body_tris: np.ndarray,
                      fine_labels: np.ndarray | None,
                      body_morph: float = 1.0) -> RigState | None:
    """Compute or fetch the rig state for this body mesh."""
    if len(body_verts) == 0 or fine_labels is None:
        return None
    from faceview.vision.skeleton_landmarks import limb_landmarks

    skin = limb_landmarks(body_morph=body_morph)
    adj = _build_adjacency(body_tris, len(body_verts))

    # Smooth label classification noise at body-part boundaries
    # (stray verts that the threshold classifier mis-assigned).
    fine_labels = _smooth_labels_mode(fine_labels, adj, n_iters=2)

    # Stray cleanup passes — opt-in via env var. Painted labels
    # are usually clean enough; running these passes added small
    # numbers of corrections in some cases but also occasionally
    # destabilised legitimate clusters. Default OFF for the
    # painted-label path; set FACEVIEW_RIG_CLEANUP=1 to enable.
    if os.environ.get("FACEVIEW_RIG_CLEANUP", "").strip() in (
            "1", "true", "yes"):
        fine_labels = cleanup_stray_components(
            fine_labels, adj, verts=body_verts,
            min_size_ratio=0.20, max_dist_ratio=0.6)
        fine_labels = cleanup_spatial_outliers(
            fine_labels, body_verts, z_score_threshold=3.0)

    # Apply manual label overrides if a JSON file with explicit
    # vert-index→label mappings is present. This lets the user
    # correct any voxel the auto-classifier got wrong by editing a
    # simple file (no code changes needed). Format:
    #     {"7": 4, "152": 4, "983": 8}
    # (vert indices as strings since JSON keys must be strings;
    #  values are BPF label ids 0..15)
    fine_labels = _apply_manual_overrides(fine_labels, body_morph=body_morph)

    # Build pivots first (they're label-independent — based on
    # skeleton landmarks only).
    pivots: dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        chain = skin.get(f"arm_{side}")
        if chain:
            pivots[f"shoulder_{side}"] = np.asarray(chain["shoulder"],
                                                          dtype=np.float32)
            pivots[f"elbow_{side}"]    = np.asarray(chain["elbow"],
                                                          dtype=np.float32)
            pivots[f"wrist_{side}"]    = np.asarray(chain["wrist"],
                                                          dtype=np.float32)
        chain = skin.get(f"leg_{side}")
        if chain:
            pivots[f"hip_{side}"]   = np.asarray(chain["hip"],
                                                       dtype=np.float32)
            pivots[f"knee_{side}"]  = np.asarray(chain["knee"],
                                                       dtype=np.float32)
            pivots[f"ankle_{side}"] = np.asarray(chain["ankle"],
                                                       dtype=np.float32)
    if "hip_L" in pivots and "hip_R" in pivots:
        pivots["body_root"] = (
            (pivots["hip_L"] + pivots["hip_R"]) * 0.5).astype(np.float32)

    # Stretch-test reclassification — run trial rotations and
    # re-label any vert that doesn't move with the cluster it's
    # geometrically part of. This is the "label cleanup" step
    # that catches mis-classified voxels the static label-smoother
    # missed (stretches reveal them dynamically).
    def _build_initial_masks(labs: np.ndarray) -> dict[str, np.ndarray]:
        m: dict[str, np.ndarray] = {}
        for s in ("L", "R"):
            arm, fore, _hand = _arm_chain_masks(labs, s)
            leg, _shin, _foot = _leg_chain_masks(labs, s)
            m[f"arm_{s}_full"]    = arm.astype(np.float32)
            m[f"arm_{s}_belowel"] = fore.astype(np.float32)
            m[f"leg_{s}_full"]    = leg.astype(np.float32)
        return m
    # Disabled — with painted labels + stray + spatial cleanup
    # already correcting bad voxels, the stretch-test
    # reclassification was over-eager and reassigning legitimate
    # seam verts. Keep the function around in case the user wants
    # it for future experiments via env-var opt-in.
    if os.environ.get("FACEVIEW_RIG_STRETCH_RELABEL", "").strip() in (
            "1", "true", "yes"):
        _initial_masks = _build_initial_masks(fine_labels)
        fine_labels = reclassify_via_stretch_test(
            body_verts, body_tris, fine_labels,
            pivots=pivots, masks=_initial_masks,
            n_passes=4, edge_grow_threshold=2.0)

    # Anatomically-valid out-fade neighbours per group. These cap
    # the bilateral fade so a shoulder rotation can only fade into
    # NECK/CHEST verts at the seam (never thigh, leg, or other arm).
    from faceview.vision.body_3d import (
        BPF_NECK, BPF_CHEST, BPF_ABDOMEN, BPF_PELVIS_SKIN,
    )
    arm_out_labels   = {BPF_NECK, BPF_CHEST}
    leg_out_labels   = {BPF_PELVIS_SKIN, BPF_ABDOMEN}
    no_out_labels: set[int] = set()  # below-elbow / below-knee /
                                       # hand / foot: no out fade

    def _allowed(labels: set[int]) -> np.ndarray:
        m = np.zeros(len(fine_labels), dtype=bool)
        for lid in labels:
            m |= (fine_labels == lid)
        return m

    # Graded seam weights by default — seam-ring 0.85, second ring
    # 0.95, deeper 1.0. Smooths the shoulder/armpit transition
    # without violating the regression invariant (non-arm/leg
    # labelled verts still get weight 0.0).
    # tests/test_body_rig_regression.py enforces that arm/leg
    # rotations only move arm/leg labels.
    # Set FACEVIEW_RIG_WEIGHT_MODE=hard to revert to the old
    # binary seam (sharper shoulder edges).
    weight_mode = os.environ.get("FACEVIEW_RIG_WEIGHT_MODE",
                                       "graded_3ring").strip() or "graded_3ring"
    weights: dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        arm, fore, hand = _arm_chain_masks(fine_labels, side)
        weights[f"arm_{side}_full"]    = _soft_weight(arm,  adj,
                                                          mode=weight_mode)
        weights[f"arm_{side}_belowel"] = _soft_weight(fore, adj,
                                                          mode=weight_mode)
        weights[f"arm_{side}_hand"]    = _soft_weight(hand, adj,
                                                          mode=weight_mode)
        leg, shin, foot = _leg_chain_masks(fine_labels, side)
        weights[f"leg_{side}_full"]    = _soft_weight(leg,  adj,
                                                          mode=weight_mode)
        weights[f"leg_{side}_belowkn"] = _soft_weight(shin, adj,
                                                          mode=weight_mode)
        weights[f"leg_{side}_foot"]    = _soft_weight(foot, adj,
                                                          mode=weight_mode)
    weights["torso"] = _soft_weight(_torso_mask(fine_labels), adj,
                                          mode=weight_mode)

    # Pre-compute seam-ring vert indices per group so the
    # post-rotation seam smoother can find them quickly.
    seam_indices: dict[str, np.ndarray] = {}
    for key, w in weights.items():
        in_mask = w > 1e-3
        seam = np.zeros(len(body_verts), dtype=bool)
        for i in range(len(body_verts)):
            if not in_mask[i]:
                continue
            for j in adj[i]:
                if not in_mask[j]:
                    seam[i] = True
                    break
        seam_indices[key] = np.where(seam)[0].astype(np.int32)

    return RigState(n_verts=len(body_verts),
                    weights=weights, pivots=pivots,
                    adj=adj, seam_indices=seam_indices)


# ── Rotation primitive ────────────────────────────────────────────


def _rmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    cr_, sr_ = float(np.cos(roll)), float(np.sin(roll))
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]],
                    dtype=np.float32)
    return Ry @ Rx @ Rz


def _rotate_weighted(verts: np.ndarray, pivot: np.ndarray,
                       R: np.ndarray, weight: np.ndarray
                      ) -> np.ndarray:
    if not (weight > 1e-3).any():
        return verts
    diff = verts - pivot
    rotated = (diff @ R.T) + pivot
    w = weight[:, None]
    return (verts * (1.0 - w) + rotated * w).astype(np.float32)


def _smooth_seam(verts: np.ndarray, seam_idx: np.ndarray,
                   adj: list[list[int]] | None,
                   in_mask: np.ndarray | None = None,
                   blend: float = 0.35) -> np.ndarray:
    """Laplacian-style seam smoothing — restricted to in-mask
    neighbours so seam-arm verts smooth WITHIN the arm cluster
    rather than getting yanked toward the torso. Result: the seam
    transitions from rigid in-arm rotation to a slightly relaxed
    seam ring, reducing the visible triangle stretch right at the
    shoulder/hip boundary without polluting in-arm rigidity.
    """
    if adj is None or len(seam_idx) == 0 or blend <= 0.0:
        return verts
    out = verts.copy()
    for i in seam_idx:
        ii = int(i)
        nbrs = adj[ii]
        if not nbrs:
            continue
        # Only average IN-MASK neighbours — keeps the seam vert
        # bound to other rotated arm verts, not torso verts.
        if in_mask is not None:
            nbrs_in = [j for j in nbrs if in_mask[j]]
            if not nbrs_in:
                continue
            nbrs = nbrs_in
        avg = np.zeros(3, dtype=np.float32)
        for j in nbrs:
            avg += verts[j]
        avg /= float(len(nbrs))
        out[ii] = (1.0 - blend) * verts[ii] + blend * avg
    return out.astype(np.float32)


# ── Public rig entry ───────────────────────────────────────────────


def _propagate_pivots(pivots: dict[str, np.ndarray],
                        anchor: np.ndarray, R: np.ndarray,
                        children: tuple[str, ...]) -> None:
    """Rotate the named child pivots around ``anchor`` in place so
    they follow a parent-bone rotation. After the elbow joint moves
    with shoulder rotation, the forearm/elbow rotation must pivot
    around the new elbow position — otherwise the forearm stretches.
    """
    for k in children:
        v = pivots.get(k)
        if v is None:
            continue
        pivots[k] = ((v - anchor) @ R.T + anchor).astype(np.float32)


def apply_body_rig_v2(body_verts: np.ndarray, params,
                        rig: RigState | None) -> np.ndarray:
    """Apply hierarchical rotations using painted BPF masks + 3D
    skeleton joint pivots. Order: body root → per-side shoulder →
    elbow → wrist; same for hips → knees → ankles. Each parent
    rotation propagates to ITS CHILD pivot positions so child joints
    stay attached to the rotated parent bone.
    """
    if rig is None or len(body_verts) == 0:
        return body_verts
    out = body_verts
    # Mutable copy of pivots — we update child positions after each
    # parent rotation so the chain stays connected.
    pivots = {k: v.copy() for k, v in rig.pivots.items()}

    # NOTE: body rotation (body_yaw / body_pitch / body_roll) is
    # applied later in the render pipeline by ``_apply_neck_rotation``
    # in ict_face.py — that function rotates BOTH the body verts and
    # the ICT head mesh together so the head stays attached to the
    # neck. We deliberately don't apply body rotation here; instead
    # we just propagate it to the limb pivots so this frame's limb
    # rotations operate around the joints' POST-body-rotation
    # positions.
    body_yaw = float(getattr(params, "body_yaw", 0.0))
    body_pitch = float(getattr(params, "body_pitch", 0.0))
    body_roll = float(getattr(params, "body_roll", 0.0))
    body_yaw   = max(-0.60, min(0.60, body_yaw))
    body_pitch = max(-0.40, min(0.65, body_pitch))
    body_roll  = max(-0.40, min(0.40, body_roll))
    if (abs(body_yaw) > 1e-3 or abs(body_pitch) > 1e-3
            or abs(body_roll) > 1e-3) and "body_root" in pivots:
        Rb = _rmat(body_yaw, body_pitch, body_roll)
        anchor = pivots["body_root"]
        # Pivot propagation only — no vertex rotation here.
        _propagate_pivots(pivots, anchor, Rb,
                            ("shoulder_L", "shoulder_R",
                             "elbow_L", "elbow_R",
                             "wrist_L", "wrist_R"))

    # ── Arm chain per side ─────────────────────────────────
    for side in ("L", "R"):
        # shoulder rotation moves the entire arm; elbow + wrist
        # pivots must follow.
        prefix = f"{side.lower()}_shoulder"
        yaw, pitch, roll = _clamp_joint(
            "shoulder",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"shoulder_{side}")
        w = rig.weights.get(f"arm_{side}_full")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)
            seam = (rig.seam_indices or {}).get(f"arm_{side}_full")
            if seam is not None:
                out = _smooth_seam(out, seam, rig.adj,
                                       in_mask=(w > 1e-3))
            _propagate_pivots(pivots, anchor, R,
                                (f"elbow_{side}", f"wrist_{side}"))

        prefix = f"{side.lower()}_elbow"
        yaw, pitch, roll = _clamp_joint(
            "elbow",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"elbow_{side}")
        w = rig.weights.get(f"arm_{side}_belowel")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)
            seam = (rig.seam_indices or {}).get(f"arm_{side}_belowel")
            if seam is not None:
                out = _smooth_seam(out, seam, rig.adj,
                                       in_mask=(w > 1e-3))
            _propagate_pivots(pivots, anchor, R, (f"wrist_{side}",))

        prefix = f"{side.lower()}_wrist"
        yaw, pitch, roll = _clamp_joint(
            "wrist",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"wrist_{side}")
        w = rig.weights.get(f"arm_{side}_hand")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)

    # ── Leg chain per side ─────────────────────────────────
    for side in ("L", "R"):
        prefix = f"{side.lower()}_hip"
        yaw, pitch, roll = _clamp_joint(
            "hip",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"hip_{side}")
        w = rig.weights.get(f"leg_{side}_full")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)
            seam = (rig.seam_indices or {}).get(f"leg_{side}_full")
            if seam is not None:
                out = _smooth_seam(out, seam, rig.adj,
                                       in_mask=(w > 1e-3))
            _propagate_pivots(pivots, anchor, R,
                                (f"knee_{side}", f"ankle_{side}"))

        prefix = f"{side.lower()}_knee"
        yaw, pitch, roll = _clamp_joint(
            "knee",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"knee_{side}")
        w = rig.weights.get(f"leg_{side}_belowkn")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)
            seam = (rig.seam_indices or {}).get(f"leg_{side}_belowkn")
            if seam is not None:
                out = _smooth_seam(out, seam, rig.adj,
                                       in_mask=(w > 1e-3))
            _propagate_pivots(pivots, anchor, R, (f"ankle_{side}",))

        prefix = f"{side.lower()}_ankle"
        yaw, pitch, roll = _clamp_joint(
            "ankle",
            float(getattr(params, f"{prefix}_yaw",   0.0)),
            float(getattr(params, f"{prefix}_pitch", 0.0)),
            float(getattr(params, f"{prefix}_roll",  0.0)))
        anchor = pivots.get(f"ankle_{side}")
        w = rig.weights.get(f"leg_{side}_foot")
        if anchor is not None and w is not None and (
                abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3):
            R = _rmat(yaw, pitch, roll)
            out = _rotate_weighted(out, anchor, R, w)

    return out
