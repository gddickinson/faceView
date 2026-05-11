"""Full-body avatar meshes — male / female with blend slider.

The body OBJs (CC0 from MakeHuman base) ship with their own heads,
necks and shoulders already in the right anatomical relationship.
Rather than guessing percentages we use that pre-existing geometry
as the anchor:

1. **Anatomical canon for the body** — the Vitruvian/da Vinci ratio
   that adult humans are ≈ **7.5 head-heights** tall. We use this to
   locate the body's chin from the mesh's overall height; that line
   is where the head meets the neck.
2. **Real landmark for the ICT head** — vertex #964 (the peak-
   displacement vertex of the ``jawOpen`` blendshape) is the chin
   tip of the ICT-FaceKit model. Its Y coordinate (~−6.47) is the
   ICT chin line.
3. **1:1 head-size match** — scale the body so its head_height
   (z_span / 7.5) equals the ICT head_height (crown − chin), then
   translate so the body's chin lands exactly on the ICT chin. The
   ICT head replaces the body head one-for-one in size and position
   while sitting on the body's preserved neck and clavicles.

Anatomical reference values (kept as constants below)
-----------------------------------------------------
* ``HEAD_HEIGHTS_PER_BODY = 7.5``  — Vitruvian ratio for adult humans.
* Adult male ≈ 175 cm height, ≈ 22.5 cm head, 7.78 heads.
* Adult female ≈ 162 cm height, ≈ 21.0 cm head, 7.71 heads.
* Female total height ≈ 92.6 % of male — already encoded in the
  source OBJs (1.64 m / 1.68 m = 0.976 in this asset).
* ICT-FaceKit ``jawOpen`` peak vertex index = ``964`` (chin tip).

Axis conventions
----------------
* body OBJ: x=lateral, y=depth (-Y forward), z=height
* ICT head: x=lateral, y=height, z=depth (+Z forward)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir


# ── Anatomical constants ──────────────────────────────────────────────

# Vitruvian canon — adult human total height in head-heights.
HEAD_HEIGHTS_PER_BODY: float = 7.5

# ICT-FaceKit landmark indices (validated empirically against the
# bundled blendshape definitions).
ICT_CHIN_VERT_IDX: int = 964   # peak-displacement vert of `jawOpen`


@dataclass
class BodyMesh:
    verts: np.ndarray         # (N, 3) float32 — already in ICT frame
    tris: np.ndarray          # (M, 3) int32
    colors: np.ndarray        # (N, 3) float32 in [0, 1]
    specular: np.ndarray      # (N,) float32
    emissive: np.ndarray      # (N,) float32
    parts: np.ndarray | None = None  # (N,) int32, body-part labels


# Body-part classification labels.
BP_TORSO     = 0
BP_NECK      = 1
BP_LEFT_ARM  = 2
BP_RIGHT_ARM = 3
BP_LEFT_LEG  = 4
BP_RIGHT_LEG = 5
BP_HEAD      = 6
BP_NAMES = {
    0: "torso", 1: "neck", 2: "left_arm", 3: "right_arm",
    4: "left_leg", 5: "right_leg", 6: "head",
}

# Fine-grained body-part labels. Returned by classify_body_parts_fine.
# Pure Vitruvian threshold logic, keyed off chin_y + head_h, no skeleton
# dependency. Sufficient for per-region colouring and approximate
# bone-driven masks (e.g. only-deform-this-finger animations).
BPF_NECK         = 0
BPF_CHEST        = 1
BPF_ABDOMEN      = 2
BPF_PELVIS_SKIN  = 3
BPF_UPPER_ARM_L  = 4
BPF_UPPER_ARM_R  = 5
BPF_FOREARM_L    = 6
BPF_FOREARM_R    = 7
BPF_HAND_L       = 8
BPF_HAND_R       = 9
BPF_THIGH_L      = 10
BPF_THIGH_R      = 11
BPF_SHIN_L       = 12
BPF_SHIN_R       = 13
BPF_FOOT_L       = 14
BPF_FOOT_R       = 15
BPF_NAMES = {
    0: "neck", 1: "chest", 2: "abdomen", 3: "pelvis_skin",
    4: "upper_arm_L", 5: "upper_arm_R",
    6: "forearm_L",   7: "forearm_R",
    8: "hand_L",      9: "hand_R",
    10: "thigh_L",   11: "thigh_R",
    12: "shin_L",    13: "shin_R",
    14: "foot_L",    15: "foot_R",
}


def classify_body_parts(verts: np.ndarray, chin_y: float,
                          head_h: float) -> np.ndarray:
    """Assign each body vertex to a body-part label based on its
    Y/X position. Returns ``(N,)`` int32 array of BP_* values.

    Used for debug visualization + as a HARD classification mask
    in joint rotations (so e.g. a shoulder rotation can be limited
    to ARM verts only, never picking up TORSO pixels).
    """
    if len(verts) == 0:
        return np.zeros(0, dtype=np.int32)
    HH = float(head_h)
    shoulder_y = chin_y - HH * 0.50
    hip_y      = chin_y - HH * 3.00
    wrist_y    = chin_y - HH * 3.20
    # X thresholds — arms hang at ~0.55 head-heights from spine,
    # legs at ~0.10 from spine.
    arm_inner  = HH * 0.55
    leg_inner  = HH * 0.10
    x = verts[:, 0]
    y = verts[:, 1]

    labels = np.full(len(verts), BP_TORSO, dtype=np.int32)
    # Above shoulder line — neck (head proper is the ICT mesh, not body)
    labels[y > shoulder_y] = BP_NECK
    # ARM region: |X| > arm_inner AND between shoulder and wrist.
    # Arms hang lateral — use X to disambiguate from legs/torso.
    arm_y_band = (y < shoulder_y) & (y > wrist_y)
    labels[arm_y_band & (x > arm_inner)]  = BP_LEFT_ARM
    labels[arm_y_band & (x < -arm_inner)] = BP_RIGHT_ARM
    # LEG region: below hip AND not too central (groin stays torso).
    # Arms can extend below hip Y in bbox, but their X is larger
    # than the leg X range so they keep ARM label from above.
    leg_y_band = y < hip_y
    leg_x_band_l = (x > leg_inner) & (x < arm_inner)
    leg_x_band_r = (x < -leg_inner) & (x > -arm_inner)
    labels[leg_y_band & leg_x_band_l] = BP_LEFT_LEG
    labels[leg_y_band & leg_x_band_r] = BP_RIGHT_LEG
    return labels


def _try_load_painted_labels(n_verts: int) -> np.ndarray | None:
    """Load painted body-part labels written by
    `tools.import_part_painting`. Tries the per-morph files first
    (``body_part_labels_male.npz`` / ``..._female.npz``) and falls
    back to the legacy single-file ``body_part_labels.npz``. Returns
    the saved labels only if the vert count matches the requested
    body mesh; otherwise None and the threshold classifier runs."""
    base = assets_dir()
    candidates = [
        base / "body_part_labels_male.npz",
        base / "body_part_labels_female.npz",
        base / "body_part_labels.npz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            d = np.load(path)
            if int(d.get("n_verts", -1)) != n_verts:
                continue
            labels = np.asarray(d["labels"], dtype=np.int32)
            if len(labels) != n_verts:
                continue
            return labels
        except Exception:
            continue
    return None


def classify_body_parts_fine(verts: np.ndarray, chin_y: float,
                              head_h: float) -> np.ndarray:
    """Per-vertex finer body-part labels (16 regions).

    If `assets/body_part_labels.npz` exists with the matching vert
    count, those manually-painted labels are returned (the override
    produced by `tools.import_part_painting`). Otherwise the
    threshold classifier runs.

    Y boundaries are body-relative (fractions of the verts' actual
    Y span) so the thresholds line up with the avatar's real
    proportions instead of a Vitruvian estimate. ``chin_y`` and
    ``head_h`` are still used for X-band thresholds (arm/leg widths
    are governed by head size, not torso height).
    """
    if len(verts) == 0:
        return np.zeros(0, dtype=np.int32)
    saved = _try_load_painted_labels(len(verts))
    if saved is not None:
        return saved
    HH = float(head_h)
    body_top = float(verts[:, 1].max())
    body_bot = float(verts[:, 1].min())
    bh = max(1.0, body_top - body_bot)

    # Body-relative Y bands (matches `avatar_landmarks` fractions).
    shoulder_y = body_top - 0.04 * bh
    xiphoid_y  = body_top - 0.28 * bh   # bottom of rib cage
    hip_y      = body_top - 0.42 * bh
    crotch_y   = body_top - 0.50 * bh
    elbow_y    = body_top - 0.20 * bh
    wrist_y    = body_top - 0.46 * bh
    knee_y     = body_top - 0.65 * bh
    ankle_y    = body_top - 0.93 * bh

    # X thresholds. ``hand_x`` separates the actual hand (far lateral
    # at wrist Y, ~HH×1.0) from the upper outer thigh (which crosses
    # HH×0.55). The leg group then fills *every* non-hand X below
    # the hip — a fixed leg_outer would always miss either the wide
    # upper thigh or the wide outer foot at different Y levels.
    arm_inner = HH * 0.55
    hand_x    = HH * 0.95
    leg_inner = HH * 0.05

    x = verts[:, 0]
    y = verts[:, 1]
    abs_x = np.abs(x)

    labels = np.full(len(verts), BPF_CHEST, dtype=np.int32)

    # Torso column (|X| < arm_inner) split by Y.
    torso_x = abs_x < arm_inner
    labels[torso_x & (y > shoulder_y)] = BPF_NECK
    labels[torso_x & (y <= shoulder_y) & (y > xiphoid_y)] = BPF_CHEST
    labels[torso_x & (y <= xiphoid_y) & (y > hip_y)] = BPF_ABDOMEN
    labels[torso_x & (y <= hip_y) & (y > crotch_y)] = BPF_PELVIS_SKIN

    # Arms — between shoulder and wrist Y, X outside torso column.
    arm_band = (y <= shoulder_y) & (y > wrist_y) & (abs_x >= arm_inner)
    upper = arm_band & (y > elbow_y)
    fore  = arm_band & (y <= elbow_y)
    labels[upper & (x > 0)] = BPF_UPPER_ARM_L
    labels[upper & (x < 0)] = BPF_UPPER_ARM_R
    labels[fore  & (x > 0)] = BPF_FOREARM_L
    labels[fore  & (x < 0)] = BPF_FOREARM_R

    # Hand — below wrist, FAR lateral, AND within ~0.6 head-heights
    # of the wrist (any farther down is foot territory regardless of
    # |x|). Without this Y cap the wide outer foot verts get mis-
    # tagged as hand and rotate with shoulder/elbow movements.
    hand_top_y = wrist_y
    hand_bot_y = wrist_y - HH * 0.65
    hand_band = ((y <= hand_top_y) & (y >= hand_bot_y)
                  & (abs_x >= hand_x))
    labels[hand_band & (x > 0)] = BPF_HAND_L
    labels[hand_band & (x < 0)] = BPF_HAND_R

    # Legs — everything below hip that isn't hand. Y bands separate
    # thigh / shin / foot, sign of X separates L vs R. Since hand
    # band is now Y-bounded, feet at large |x| are no longer
    # competing with hands — drop the abs_x cap so wide outer-foot
    # verts are properly captured.
    leg_y = (y <= hip_y) & ~hand_band
    thigh_b = leg_y & (y > knee_y)
    shin_b  = leg_y & (y <= knee_y) & (y > ankle_y)
    foot_b  = leg_y & (y <= ankle_y)
    side_l = x >= 0
    side_r = x < 0
    labels[thigh_b & side_l] = BPF_THIGH_L
    labels[thigh_b & side_r] = BPF_THIGH_R
    labels[shin_b  & side_l] = BPF_SHIN_L
    labels[shin_b  & side_r] = BPF_SHIN_R
    labels[foot_b  & side_l] = BPF_FOOT_L
    labels[foot_b  & side_r] = BPF_FOOT_R
    # Restore pelvis_skin in the central column above crotch_y; the
    # leg sweep above overwrote it because the inner thigh shares
    # X with the pelvis groin region.
    pelvis_band = (y <= hip_y) & (y > crotch_y) & (abs_x < leg_inner)
    labels[pelvis_band] = BPF_PELVIS_SKIN
    return labels


def part_movement_summary(verts_before: np.ndarray,
                           verts_after: np.ndarray,
                           parts: np.ndarray) -> dict:
    """Return per-body-part mean displacement (ICT units). Useful
    for debugging which regions are unintentionally moving during
    a particular effect.
    """
    if verts_before is None or verts_after is None:
        return {}
    diff = np.linalg.norm(verts_after - verts_before, axis=1)
    out = {}
    for label, name in BP_NAMES.items():
        m = parts == label
        if m.any():
            out[name] = {
                "mean": float(diff[m].mean()),
                "max":  float(diff[m].max()),
                "count": int(m.sum()),
            }
    return out


def _body_dir() -> Path:
    return assets_dir() / "body_meshes"


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ parser — vertex positions + triangulated faces."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idxs = []
                for token in line.split()[1:]:
                    idx = int(token.split("/")[0]) - 1
                    idxs.append(idx)
                for k in range(1, len(idxs) - 1):
                    tris.append([idxs[0], idxs[k], idxs[k + 1]])
    return (np.array(verts, dtype=np.float32),
            np.array(tris, dtype=np.int32))


# ── Body OBJ loading + anatomy ────────────────────────────────────────


@lru_cache(maxsize=2)
def _load_body_obj(which: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached raw OBJ load (full body, head still attached)."""
    path = _body_dir() / f"body_{which}.obj"
    if not path.exists():
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))
    return _parse_obj(path)


def _body_anatomy(verts: np.ndarray) -> dict:
    """Anatomical landmarks for a body mesh, derived from the
    Vitruvian 7.5-heads ratio.

    Body axis: z is height. We use the OBJ's actual extents (so the
    male and female meshes stay in their own units) and apply the
    canonical head-height fraction. This matches what an anatomy
    textbook says about the body — and crucially gives identical
    treatment to both bodies even though they're different sizes.
    """
    if len(verts) == 0:
        return {}
    z = verts[:, 2]
    z_max, z_min = float(z.max()), float(z.min())
    z_span = z_max - z_min
    head_h = z_span / HEAD_HEIGHTS_PER_BODY
    chin_z = z_max - head_h
    # Neck is roughly half a head below the chin; shoulders sit
    # ~1.4 heads down from the crown (anatomical canon, used only
    # for diagnostic info).
    neck_z = z_max - head_h * 1.20
    shoulder_z = z_max - head_h * 1.40
    return {
        "crown": z_max,
        "chin": chin_z,
        "neck": neck_z,
        "shoulder": shoulder_z,
        "foot": z_min,
        "head_height": head_h,
        "total_height": z_span,
    }


def _strip_above(verts: np.ndarray, tris: np.ndarray, height_axis: int,
                 cut_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop verts whose height ≥ ``cut_value`` and reindex tris.

    Used to remove the body's head while preserving its neck stub
    and shoulders — the cut is placed at the body's chin line.
    """
    if len(verts) == 0:
        return verts, tris
    keep = verts[:, height_axis] < cut_value
    if not keep.any():
        return verts, tris
    keep_idx = np.where(keep)[0]
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
    tri_keep = keep[tris].all(axis=1)
    return verts[keep], remap[tris[tri_keep]].astype(np.int32)


def _strip_above_sloped(
    verts: np.ndarray, tris: np.ndarray,
    height_axis: int, depth_axis: int,
    cut_back: float, cut_front: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Strip the body's head with a Z-sloped cut: keep verts up to
    ``cut_back`` at the back of the head and only up to ``cut_front``
    at the front. A smoothstep along ``depth_axis`` (body's depth)
    blends the two limits — no kink where they meet.

    Why: with a flat ``chin``-level cut, the body's preserved neck
    sticks out at the front and crosses ICT's chin curve, distorting
    the chin. With a sloped cut the front is trimmed lower so ICT's
    chin curves down into a clean body-neck silhouette, while the
    back/nape stays at chin level for the seamless head-to-body join.
    """
    if len(verts) == 0:
        return verts, tris
    h = verts[:, height_axis]
    d = verts[:, depth_axis]
    # Body OBJ uses ``-Y forward``, so the front face is at MIN d and
    # the back/nape is at MAX d. Smoothstep from front (cut_front) to
    # back (cut_back).
    d_lo = float(np.percentile(d, 10))
    d_hi = float(np.percentile(d, 90))
    t = np.clip((d - d_lo) / max(1e-6, d_hi - d_lo), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    cut = cut_front + t * (cut_back - cut_front)
    keep = h < cut
    if not keep.any():
        return verts, tris
    keep_idx = np.where(keep)[0]
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
    tri_keep = keep[tris].all(axis=1)
    return verts[keep], remap[tris[tri_keep]].astype(np.int32)


def _boundary_ring_indices(tris: np.ndarray) -> np.ndarray:
    """Vertex indices that lie on the mesh's open boundary.

    A boundary edge appears in exactly one triangle (manifold mesh).
    Returns the unique vertex indices forming all such edges.
    """
    if len(tris) == 0:
        return np.array([], dtype=np.int32)
    edges: dict[tuple[int, int], int] = {}
    for t in tris:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    bv: set[int] = set()
    for (a, b), c in edges.items():
        if c == 1:
            bv.add(a)
            bv.add(b)
    return np.array(sorted(bv), dtype=np.int32)


def boundary_ring_at_height(verts: np.ndarray, tris: np.ndarray,
                              prefer: str = "top",
                              height_axis: int = 1,
                              band_frac: float = 0.05) -> np.ndarray:
    """Return vertex indices of the boundary ring near the top
    (``prefer='top'``) or bottom (``'bottom'``) of the mesh.

    ``band_frac`` is the height-window around the extreme used to
    select ring members — 5 % of the boundary's height span by
    default. Caller can use the returned indices to read ring
    positions, compute a centroid, build a bridge mesh, etc.
    """
    bv = _boundary_ring_indices(tris)
    if len(bv) == 0:
        return bv
    bv_h = verts[bv, height_axis]
    h_min = float(bv_h.min())
    h_max = float(bv_h.max())
    h_span = max(1e-6, h_max - h_min)
    if prefer == "top":
        thr = h_max - h_span * band_frac
        return bv[bv_h >= thr]
    thr = h_min + h_span * band_frac
    return bv[bv_h <= thr]


def _morph_top_to_ict(body_verts: np.ndarray,
                       ict_verts_ref: np.ndarray,
                       body_top_y: float,
                       band_h: float,
                       ) -> np.ndarray:
    """Per-Y lateral + depth morph of body's upper region toward
    ICT's outline at each Y level.

    For every Y slice within ``[body_top_y - band_h, body_top_y]``
    we measure the body's natural lateral X half-extent and depth
    Z bounds and ICT's at the same Y, then drive each body vertex
    toward the ICT extents using a smoothstep weight (1.0 at the
    top of the band → 0.0 at the bottom). Below the band the body
    keeps its natural anatomy.

    Result: the body's upper back / upper chest / shoulder caps
    smoothly taper toward ICT's neck so the join reads as one
    continuous shape rather than two stacked cylinders.
    """
    if len(body_verts) == 0 or band_h <= 0:
        return body_verts
    band_bottom = body_top_y - band_h
    n_bins = 16
    bin_h = band_h / max(n_bins - 1, 1)
    half_bin = bin_h * 0.6  # window slightly wider than bin spacing

    body_x_ext = np.zeros(n_bins, dtype=np.float32)
    body_z_lo = np.zeros(n_bins, dtype=np.float32)
    body_z_hi = np.zeros(n_bins, dtype=np.float32)
    ict_x_ext = np.zeros(n_bins, dtype=np.float32)
    ict_z_lo = np.zeros(n_bins, dtype=np.float32)
    ict_z_hi = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        y = band_bottom + i * bin_h
        bm = (body_verts[:, 1] > y - half_bin) & \
              (body_verts[:, 1] < y + half_bin)
        if bm.any():
            body_x_ext[i] = float(np.abs(body_verts[bm, 0]).max())
            body_z_lo[i] = float(body_verts[bm, 2].min())
            body_z_hi[i] = float(body_verts[bm, 2].max())
        im = (ict_verts_ref[:, 1] > y - half_bin) & \
              (ict_verts_ref[:, 1] < y + half_bin)
        if im.any():
            ict_x_ext[i] = float(np.abs(ict_verts_ref[im, 0]).max())
            ict_z_lo[i] = float(ict_verts_ref[im, 2].min())
            ict_z_hi[i] = float(ict_verts_ref[im, 2].max())

    in_band = (body_verts[:, 1] > band_bottom) & \
               (body_verts[:, 1] <= body_top_y + 1e-3)
    if not in_band.any():
        return body_verts

    out = body_verts.copy()
    idxs = np.where(in_band)[0]
    vy = body_verts[idxs, 1]
    bin_i = np.clip(np.round((vy - band_bottom) / max(bin_h, 1e-6)),
                       0, n_bins - 1).astype(np.int32)
    # Smoothstep weight per vert: 0 at band_bottom, 1 at body_top_y.
    t = np.clip((vy - band_bottom) / max(band_h, 1e-6), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)

    bx = body_x_ext[bin_i]
    ix = ict_x_ext[bin_i]
    safe_bx = np.where(bx > 1e-6, bx, 1.0)
    target_x = (1.0 - w) * bx + w * ix
    sx = np.where(bx > 1e-6, target_x / safe_bx, 1.0)
    # Cap aggressive shrinking — going below 0.30 collapses the mesh
    # and kinks triangles. Allow up to 0.30..1.5.
    sx = np.clip(sx, 0.30, 1.5)
    out[idxs, 0] = body_verts[idxs, 0] * sx

    bz_centre = (body_z_lo[bin_i] + body_z_hi[bin_i]) / 2.0
    bz_half = (body_z_hi[bin_i] - body_z_lo[bin_i]) / 2.0
    iz_centre = (ict_z_lo[bin_i] + ict_z_hi[bin_i]) / 2.0
    iz_half = (ict_z_hi[bin_i] - ict_z_lo[bin_i]) / 2.0
    safe_bz_half = np.where(bz_half > 1e-6, bz_half, 1.0)
    target_centre = (1.0 - w) * bz_centre + w * iz_centre
    target_half = (1.0 - w) * bz_half + w * iz_half
    sz = np.where(bz_half > 1e-6, target_half / safe_bz_half, 1.0)
    sz = np.clip(sz, 0.30, 1.5)
    out[idxs, 2] = target_centre + (body_verts[idxs, 2] - bz_centre) * sz
    return out


def _laplacian_smooth_band(verts: np.ndarray, tris: np.ndarray,
                              y_centre: float, y_half: float,
                              iterations: int = 3) -> np.ndarray:
    """In-place Laplacian smoothing of verts within a Y band around
    ``y_centre``. Each iteration replaces each band-vert's position
    with the average of itself and its neighbours via shared edges.
    Smooths kinks at the body's top boundary so the silhouette
    transitions from neck-rotation-tracking back to the static
    torso shape gradually.

    Smoothing strength is weighted by distance from ``y_centre`` —
    full at the centre line, zero at ±y_half.
    """
    if len(verts) == 0 or len(tris) == 0 or y_half <= 0:
        return verts
    # Build per-vertex neighbour list from triangle edges.
    n = len(verts)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for t in tris:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        nbrs[a].add(b); nbrs[a].add(c)
        nbrs[b].add(a); nbrs[b].add(c)
        nbrs[c].add(a); nbrs[c].add(b)
    nbrs_arr = [np.array(sorted(s), dtype=np.int32) for s in nbrs]
    # Smoothing weight per vertex — peaks at y_centre, fades over ±y_half.
    y = verts[:, 1]
    w = 1.0 - np.clip(np.abs(y - y_centre) / max(1e-6, y_half),
                          0.0, 1.0)
    w = (w * w * (3.0 - 2.0 * w)).astype(np.float32)
    out = verts.copy().astype(np.float32)
    for _ in range(iterations):
        new = out.copy()
        for i in range(n):
            if w[i] <= 1e-3:
                continue
            ns = nbrs_arr[i]
            if len(ns) == 0:
                continue
            avg = out[ns].mean(axis=0)
            new[i] = out[i] * (1.0 - w[i]) + avg * w[i]
        out = new
    return out


def _close_open_top(verts: np.ndarray, tris: np.ndarray,
                      height_axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Close the open top boundary left by ``_strip_above``.

    After stripping the body's head, the topmost cross-cut triangles
    are dropped — this leaves an open ring of edges around the neck
    that's visible to the camera as a jagged horizontal seam. This
    function detects the open boundary at the top, adds a centroid
    vertex, and fan-triangulates the boundary into a closed cap.

    The cap inherits no special colour — callers append matching
    rows to ``colors`` / ``specular`` / ``emissive``.
    """
    if len(verts) == 0 or len(tris) == 0:
        return verts, tris
    # Build undirected edge -> triangle-count map.
    edges: dict[tuple[int, int], int] = {}
    for t in tris:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    # Boundary edges appear in exactly one triangle (manifold mesh).
    boundary = [e for e, c in edges.items() if c == 1]
    if len(boundary) < 3:
        return verts, tris
    # Restrict to the TOP boundary loop — boundary verts within
    # 5 % of head_height of the maximum height. (A stripped-at-the-
    # top body has no other open boundaries, but be defensive in
    # case the source OBJ had non-manifold edges elsewhere.)
    boundary_verts = set()
    for a, b in boundary:
        boundary_verts.add(a)
        boundary_verts.add(b)
    boundary_idx = np.array(sorted(boundary_verts), dtype=np.int32)
    bv_h = verts[boundary_idx, height_axis]
    h_max = float(bv_h.max())
    h_min = float(verts[:, height_axis].min())
    threshold = h_max - (h_max - h_min) * 0.05
    top_mask = bv_h > threshold
    top_set = set(boundary_idx[top_mask].tolist())
    if len(top_set) < 3:
        return verts, tris
    # Centroid vertex closes the cap.
    top_verts = verts[np.array(list(top_set), dtype=np.int32)]
    centroid = top_verts.mean(axis=0).astype(verts.dtype)
    new_idx = len(verts)
    verts_out = np.vstack([verts, centroid[None, :]])
    # Fan-triangulate the top-boundary edges from the centroid.
    new_tris = []
    for a, b in boundary:
        if a in top_set and b in top_set:
            # Wind so the cap normal points up (height_axis +).
            new_tris.append([new_idx, b, a])
    if not new_tris:
        return verts, tris
    tris_out = np.vstack([tris, np.array(new_tris, dtype=tris.dtype)])
    return verts_out, tris_out


# ── ICT head landmarks ────────────────────────────────────────────────


def _ict_head_anatomy(ict_verts: np.ndarray) -> dict:
    """Anatomical landmarks for the ICT head mesh.

    Uses real mesh landmarks rather than heuristics:
    * ``crown`` — highest vertex of the mesh (top of cranium).
    * ``chin``  — Y coordinate of vertex 964 (jawOpen peak), which
      is the chin tip of ICT-FaceKit's neutral pose.

    head_height = crown − chin, in ICT units (≈ 21).
    """
    if len(ict_verts) <= ICT_CHIN_VERT_IDX:
        return {}
    crown_y = float(ict_verts[:, 1].max())
    chin_y = float(ict_verts[ICT_CHIN_VERT_IDX, 1])
    return {
        "crown": crown_y,
        "chin": chin_y,
        "head_height": crown_y - chin_y,
    }


# ── Body → ICT frame transform ────────────────────────────────────────


def _to_ict_frame(verts: np.ndarray,
                  tris: np.ndarray,
                  ict_verts_ref: np.ndarray,
                  body_anat: dict,
                  ict_anchor: np.ndarray | None = None) -> np.ndarray:
    """Place a (head-stripped) body into the ICT frame so its chin
    lines up with the ICT head's chin AND its neck axis matches
    ICT's neck axis (lateral + depth).

    Steps:
    1. Axis-swap (body is z-up, ICT is y-up).
    2. Centre laterally + depth (initial pass).
    3. Scale so body's measured ``head_height`` equals ICT's measured
       ``head_height``.
    4. Translate Y so body's chin lands on ICT's chin Y.
    5. **Z-align body's neck axis with ICT's neck axis**. Without
       this, body's spine-centred mesh sits behind ICT's face-
       forward chin (~4 ICT units), creating a visible step in
       profile view. We compute both meshes' neck-band Z centroids
       at the chin level and shift the body so they coincide.
       Both centroids are derived from the LIVE deformed ICT mesh,
       so the alignment tracks identity-blend morphing.
    """
    if len(verts) == 0:
        return verts

    # 1. Axis swap to ICT frame.
    swapped = np.column_stack([
        verts[:, 0],
        verts[:, 2],
        verts[:, 1],
    ]).astype(np.float32)
    # 2. Centre laterally + depth (rough first pass; depth gets a
    # second alignment in step 5).
    swapped[:, 0] -= (swapped[:, 0].min() + swapped[:, 0].max()) / 2
    swapped[:, 2] -= (swapped[:, 2].min() + swapped[:, 2].max()) / 2

    ict_anat = _ict_head_anatomy(ict_verts_ref)
    body_head_h = float(body_anat.get("head_height", 0.0))
    if not ict_anat or body_head_h <= 0:
        return swapped

    # 3. Scale: body's "1 head" must equal ICT's "1 head".
    scale = ict_anat["head_height"] / body_head_h
    swapped *= scale

    # 4. Y-translate — body's anatomical chin → ICT's chin. We use
    # body_anat["chin"] (the canonical 7.5-heads landmark) rather
    # than swapped[:, 1].max(), because the caller will strip ABOVE
    # the body's clavicle line — i.e. above the head AND the neck —
    # so swapped's eventual max won't be at chin level. Anchoring
    # via the conceptual chin keeps the body's clavicles at the
    # right Y relative to ICT's chin regardless of where we strip.
    body_chin_y_scaled = float(body_anat["chin"]) * scale
    swapped[:, 1] += (ict_anat["chin"] - body_chin_y_scaled)

    # 5. Z-align — match the body's chin-band 90th-percentile Z (the
    # forward-most chin / face-edge) to ICT's chin-band 90p Z. The
    # caller (gen_body_mesh) lowers ``morph_band_h`` to reduce the
    # body-top warp because this strong shift already puts the body's
    # natural chin/clavicle line at ICT's chin without needing
    # heavy morph-blending — keeping the body's torso shape intact.
    chin_y = ict_anat["chin"]
    head_h = ict_anat["head_height"]
    band = head_h * 0.10
    body_band = (swapped[:, 1] > chin_y - band) & \
                (swapped[:, 1] < chin_y + band)
    ict_band  = (ict_verts_ref[:, 1] > chin_y - band) & \
                (ict_verts_ref[:, 1] < chin_y + band)
    if body_band.any() and ict_band.any():
        body_z_chin = float(np.percentile(swapped[body_band, 2], 90))
        ict_z_chin  = float(np.percentile(ict_verts_ref[ict_band, 2], 90))
    else:
        body_z_chin = float(swapped[:, 2].max())
        ict_z_chin  = float(ict_verts_ref[:, 2].max())
    swapped[:, 2] += (ict_z_chin - body_z_chin)
    return swapped


def gen_body_mesh(ict_verts_ref: np.ndarray,
                  morph: float = 1.0,
                  color_hex: str = "#3a7088") -> BodyMesh | None:
    """Generate a body mesh in ICT coordinates.

    ``morph`` ∈ [-1, 1] selects female (-1) or male (+1). Body-part
    labels (BPF) are baked only at the two extremes (see
    ``body_part_labels_{male,female}.npz``), so intermediate values
    are snapped to the nearest baked extreme; a blended mesh would
    have a vert count that matches neither NPZ and would fall back
    to the threshold classifier, producing flyaway voxels during
    rigging.
    """
    male_v, tris_m = _load_body_obj("male")
    female_v, tris_f = _load_body_obj("female")
    if len(male_v) == 0 or len(female_v) == 0:
        return None

    # Snap morph to nearest baked extreme so labels match.
    morph = 1.0 if float(morph) >= 0.0 else -1.0
    if male_v.shape == female_v.shape:
        v_raw = male_v if morph >= 0 else female_v
        tris = tris_m
    else:
        v_raw = male_v if morph >= 0 else female_v
        tris = tris_m if morph >= 0 else tris_f

    anat = _body_anatomy(v_raw)
    if not anat:
        return None

    # Sloped strip: preserve body's full neck at the back (cut at
    # chin Z) but lower the cut at the front so ICT's chin curve
    # isn't disrupted by the body's preserved under-chin flesh.
    # FACEVIEW_KINK_FIX env var still allows the older approaches
    # for A/B comparison.
    import os as _os
    _approach = _os.environ.get("FACEVIEW_KINK_FIX", "below_chin")
    chin_cut = float(anat["chin"])
    neck_cut = float(anat["neck"])
    if _approach == "below_chin":
        # Sloped cut: chin level at back (preserves the seamless
        # back-of-head join), trimmed close to the neck level at
        # the front so ICT's full chin/throat curve has room to
        # descend without colliding with the body's preserved
        # front-neck flesh.
        front_cut = chin_cut + (neck_cut - chin_cut) * 0.85
        v_stripped, tris_stripped = _strip_above_sloped(
            v_raw, tris,
            height_axis=2, depth_axis=1,
            cut_back=chin_cut, cut_front=front_cut)
    else:
        cut = neck_cut
        v_stripped, tris_stripped = _strip_above(
            v_raw, tris, height_axis=2, cut_value=cut)
    if len(v_stripped) == 0:
        return None

    verts = _to_ict_frame(v_stripped, tris_stripped, ict_verts_ref,
                            body_anat=anat)
    if len(verts) == 0:
        return None

    # Per-Y morph of body's upper region (top ~25% of head height —
    # roughly the shoulders + upper chest / upper back / clavicle
    # area) toward ICT's outline so the silhouette transitions
    # smoothly across the join.
    ict_anat = _ict_head_anatomy(ict_verts_ref)
    if ict_anat:
        body_top_y = float(verts[:, 1].max())
        # Approach can use a wider morph band to blend body's top
        # into ICT outline more gradually.
        if _approach == "wide_morph":
            morph_band_h = ict_anat["head_height"] * 0.70
        elif _approach == "no_morph":
            morph_band_h = 0.0
        else:
            # Wide enough to cover both the back-of-neck and the
            # lower-front-cut throat region, so ICT's throat curve
            # transitions smoothly into the body's neck/clavicle
            # outline. Sloped strip leaves the front trimmed lower,
            # so the band needs ≥ 0.40 head_h to reach those verts.
            morph_band_h = ict_anat["head_height"] * 0.40
        if morph_band_h > 0:
            verts = _morph_top_to_ict(verts, ict_verts_ref,
                                         body_top_y=body_top_y,
                                         band_h=morph_band_h)

    # Close body's open top so the camera can't see through. After
    # the morph, the cap edge matches ICT's neck outline — the cap
    # is hidden under ICT's bust / lower-jaw mesh from any angle.
    verts, tris_capped = _close_open_top(verts, tris_stripped, height_axis=1)

    # Laplacian smoothing pass at the body's top boundary — softens
    # any kink between body's morphed top and the natural body shape
    # below, and prepares a smoother surface for the head-rotation
    # tracking to deform across.
    if ict_anat:
        body_top_y = float(verts[:, 1].max())
        smooth_centre = body_top_y - ict_anat["head_height"] * 0.10
        smooth_half   = ict_anat["head_height"] * 0.20
        verts = _laplacian_smooth_band(verts, tris_capped,
                                          y_centre=smooth_centre,
                                          y_half=smooth_half,
                                          iterations=2)

    n = len(verts)
    s = color_hex.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        rgb = (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
               int(s[4:6], 16) / 255.0)
    except (ValueError, IndexError):
        rgb = (0.35, 0.55, 0.65)
    colors = np.tile(np.array(rgb, dtype=np.float32), (n, 1))
    specular = np.full(n, 0.15, dtype=np.float32)
    emissive = np.zeros(n, dtype=np.float32)
    # Per-vertex body-part labels — used as a hard classification
    # mask in joint rotations so e.g. an arm rotation can never
    # pick up torso/leg verts even if their X happens to fall on
    # the smoothstep boundary.
    ict_anat_for_parts = _ict_head_anatomy(ict_verts_ref)
    parts = classify_body_parts(
        verts,
        chin_y=ict_anat_for_parts.get("chin", 0.0),
        head_h=ict_anat_for_parts.get("head_height", 21.0),
    )
    return BodyMesh(verts=verts.astype(np.float32),
                    tris=tris_capped.astype(np.int32),
                    colors=colors,
                    specular=specular,
                    emissive=emissive,
                    parts=parts)
