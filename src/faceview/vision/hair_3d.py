"""3D hair-shell meshes — procedural geometry for the avatar.

Each style is a generator function that returns
``(vertices, triangles, colors, normals, specular)`` arrays in
*ICT model coordinates* (same space as the deformed ICT mesh).
The renderer appends them to the ICT vertex stream before the GL
pass so the hair receives the same neck-pivot rotation and camera-
orbit transforms as the head.

Coordinates: +Y up (crown is highest Y), +Z forward (face → +Z),
ICT mesh extents roughly (-9..9, -25..12, 7..12) — the hair gen
functions auto-fit using the ICT mesh's bbox so identity weights
that change head shape still get fitting hair.

Styles:
  none / short_cap / fringe / side_part / long_straight /
  curly_afro / mohawk / ponytail / wild_spikes / buzz
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HairMesh:
    verts: np.ndarray         # (N, 3) float32
    tris: np.ndarray          # (M, 3) int32
    colors: np.ndarray        # (N, 3) float32 in [0, 1]
    specular: np.ndarray      # (N,) float32
    emissive: np.ndarray      # (N,) float32 — always 0 for hair


def _hex_rgb_f(c: str, fb=(0.18, 0.10, 0.04)) -> tuple[float, float, float]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
                int(s[4:6], 16) / 255.0)
    except (ValueError, IndexError):
        return fb


def _compute_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per-vertex normals via incident-face averaging."""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    out = np.zeros_like(verts)
    np.add.at(out, tris[:, 0], fn)
    np.add.at(out, tris[:, 1], fn)
    np.add.at(out, tris[:, 2], fn)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


def _ict_head_anchor(ict_verts: np.ndarray) -> tuple[
    np.ndarray, float, float, float
]:
    """Crown anchor + size hints from the ICT mesh.

    Returns:
        crown: (3,) — point at top of head, on midline
        head_w: half-width along X at the upper-skull region
        head_d: half-depth along Z at the upper-skull region
        head_h: full height of the head region (crown→chin)
    """
    y = ict_verts[:, 1]
    y_max = float(y.max())
    y_min = float(y.min())
    span = max(1e-6, y_max - y_min)
    # Top 4 % of vertices form the crown (~upper 1.4 units in our
    # ~34-unit-tall ICT mesh).
    thr = y_max - span * 0.04
    crown_v = ict_verts[y > thr]
    crown = crown_v.mean(axis=0)
    # Skull-cap dimensions: take a slab near the crown (top ~25 %)
    # rather than the full upper half so we don't include the jaw.
    cap_band = ict_verts[y > y_max - span * 0.25]
    head_w = float((cap_band[:, 0].max() - cap_band[:, 0].min()) / 2.0)
    head_d = float((cap_band[:, 2].max() - cap_band[:, 2].min()) / 2.0)
    # Full head height ≈ crown to chin (~half the mesh span).
    head_h = float(span * 0.55)
    return crown, head_w, head_d, head_h


# ── primitive builders ────────────────────────────────────────────


def _build_dome(centre: np.ndarray, rx: float, ry: float, rz: float,
                  rings: int = 12, segs: int = 20,
                  y_squash: float = 1.0,
                  noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Half-ellipsoid dome above ``centre``.

    ``rings`` = vertical resolution, ``segs`` = horizontal.
    ``y_squash`` < 1 → flatter cap; > 1 → taller dome.
    ``noise`` adds per-vertex radial jitter for hair-like irregularity.
    """
    rng = np.random.default_rng(7)
    verts = []
    for i in range(rings + 1):
        # Phi from 0 (top) to π/2 (equator).
        phi = (i / rings) * (math.pi / 2)
        sp, cp = math.sin(phi), math.cos(phi)
        for j in range(segs):
            theta = (j / segs) * 2 * math.pi
            st, ct = math.sin(theta), math.cos(theta)
            jx = (rng.random() - 0.5) * 2 * noise
            jy = (rng.random() - 0.5) * 2 * noise
            jz = (rng.random() - 0.5) * 2 * noise
            x = centre[0] + (rx + jx) * sp * ct
            y = centre[1] + (ry + jy) * cp * y_squash
            z = centre[2] + (rz + jz) * sp * st
            verts.append([x, y, z])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for i in range(rings):
        for j in range(segs):
            j2 = (j + 1) % segs
            a = i * segs + j
            b = i * segs + j2
            c = (i + 1) * segs + j
            d = (i + 1) * segs + j2
            tris.append([a, b, c])
            tris.append([b, d, c])
    return verts, np.array(tris, dtype=np.int32)


def _build_strand(start: np.ndarray, end: np.ndarray,
                    width: float, segments: int = 5
                    ) -> tuple[np.ndarray, np.ndarray]:
    """A thin ribbon strand from ``start`` to ``end``.

    Two parallel edges separated by ``width`` along X; segments
    interpolate between start and end. Renders as a quad strip.
    """
    verts = []
    for i in range(segments + 1):
        t = i / segments
        c = start * (1 - t) + end * t
        verts.append([c[0] - width / 2, c[1], c[2]])
        verts.append([c[0] + width / 2, c[1], c[2]])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for i in range(segments):
        a = i * 2
        b = i * 2 + 1
        c = (i + 1) * 2
        d = (i + 1) * 2 + 1
        tris.append([a, c, b])
        tris.append([b, c, d])
    return verts, np.array(tris, dtype=np.int32)


def _build_spike(base: np.ndarray, tip: np.ndarray,
                   base_radius: float, sides: int = 5
                   ) -> tuple[np.ndarray, np.ndarray]:
    """A cone-spike from ``base`` (with radius) to ``tip``."""
    verts = [tip.tolist()]
    direction = tip - base
    if np.linalg.norm(direction) < 1e-6:
        return np.array([base, tip], dtype=np.float32), \
            np.array([[0, 1, 1]], dtype=np.int32)
    # Find any perpendicular vector for the base ring.
    up = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(direction / np.linalg.norm(direction), up)) > 0.95:
        up = np.array([1, 0, 0], dtype=np.float32)
    side1 = np.cross(direction, up)
    side1 /= max(1e-6, np.linalg.norm(side1))
    side2 = np.cross(direction, side1)
    side2 /= max(1e-6, np.linalg.norm(side2))
    for i in range(sides):
        a = i / sides * 2 * math.pi
        ring_pt = base + (math.cos(a) * side1 + math.sin(a) * side2) * base_radius
        verts.append(ring_pt.tolist())
    verts = np.array(verts, dtype=np.float32)
    tris = []
    # Tip = vert 0; ring = verts 1..sides.
    for i in range(sides):
        a = 1 + i
        b = 1 + (i + 1) % sides
        tris.append([0, a, b])
    return verts, np.array(tris, dtype=np.int32)


def _combine(*pieces: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not pieces:
        return np.zeros((0, 3), dtype=np.float32), \
            np.zeros((0, 3), dtype=np.int32)
    all_verts = []
    all_tris = []
    offset = 0
    for v, t in pieces:
        all_verts.append(v)
        all_tris.append(t + offset)
        offset += len(v)
    return (np.vstack(all_verts).astype(np.float32),
            np.vstack(all_tris).astype(np.int32))


# ── per-style generators ──────────────────────────────────────────


def _style_short_cap(ict_verts, color):
    crown, hw, hd, _ = _ict_head_anchor(ict_verts)
    # Equator at forehead-line (~ hw below crown); top above crown
    # by ~hw*0.25. Result is a substantial 3D cap that wraps from
    # forehead all the way up + over.
    centre = crown.copy()
    centre[1] -= hw * 1.0
    return _build_dome(centre, rx=hw * 1.08, ry=hw * 1.25,
                          rz=hd * 1.05, rings=14, segs=28,
                          y_squash=1.0, noise=hw * 0.05)


def _style_fringe(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    cap = _style_short_cap(ict_verts, color)
    # Fringe panel: a forward-tilted curved triangle below the cap.
    fringe_verts = []
    fringe_tris = []
    n_strands = 9
    fringe_top_y = crown[1] - hw * 0.40
    fringe_bot_y = crown[1] - hw * 0.95
    for i in range(n_strands):
        t = i / (n_strands - 1)
        x = crown[0] + (t - 0.5) * hw * 1.6
        # Slight bow forward (positive Z).
        z_top = crown[2] + hd * 0.35
        z_bot = z_top + hd * 0.12
        # Thin strand.
        fringe_verts.append([x - hw * 0.04, fringe_top_y, z_top])
        fringe_verts.append([x + hw * 0.04, fringe_top_y, z_top])
        fringe_verts.append([x - hw * 0.06, fringe_bot_y, z_bot])
        fringe_verts.append([x + hw * 0.06, fringe_bot_y, z_bot])
    fringe_verts = np.array(fringe_verts, dtype=np.float32)
    for i in range(n_strands):
        a, b, c, d = i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3
        fringe_tris.append([a, c, b])
        fringe_tris.append([b, c, d])
    return _combine(cap, (fringe_verts, np.array(fringe_tris, dtype=np.int32)))


def _style_side_part(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    cap = _style_short_cap(ict_verts, color)
    # Sweep panel — extra hair on screen-right (positive X side).
    sweep_verts = []
    sweep_tris = []
    n = 10
    for i in range(n):
        t = i / (n - 1)
        x_top = crown[0] - hw * 0.15
        x_bot = crown[0] + hw * 0.95
        x = x_top * (1 - t) + x_bot * t
        y = crown[1] + math.sin(t * math.pi) * hw * 0.18 - hw * 0.20
        z = crown[2] + hd * 0.30
        sweep_verts.append([x, y, z])
        sweep_verts.append([x, y - hw * 0.10, z + hd * 0.05])
    sweep_verts = np.array(sweep_verts, dtype=np.float32)
    for i in range(n - 1):
        a, b = i * 2, i * 2 + 1
        c, d = (i + 1) * 2, (i + 1) * 2 + 1
        sweep_tris.append([a, c, b])
        sweep_tris.append([b, c, d])
    return _combine(cap, (sweep_verts, np.array(sweep_tris, dtype=np.int32)))


def _style_long_straight(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    cap = _style_short_cap(ict_verts, color)
    # Two side curtains hanging down to chest.
    pieces = [cap]
    for sign in (-1, +1):
        for layer, dz in enumerate((-0.0, hd * 0.15)):
            top = np.array([sign * hw * 0.95,
                              crown[1] - hw * 0.30,
                              crown[2] - hd * 0.10 + dz], dtype=np.float32)
            bottom = np.array([sign * hw * 1.05,
                                  crown[1] - hh * 1.6,
                                  crown[2] - hd * 0.40 + dz],
                                 dtype=np.float32)
            v, t = _build_strand(top, bottom, width=hw * 0.9, segments=8)
            pieces.append((v, t))
    return _combine(*pieces)


def _style_curly_afro(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    # Big puffy halo: equator at forehead, dome tall + wide.
    centre = crown.copy()
    centre[1] -= hw * 1.05
    base_v, base_t = _build_dome(centre, rx=hw * 1.45, ry=hw * 1.55,
                                     rz=hd * 1.40, rings=12, segs=24,
                                     y_squash=1.0, noise=hw * 0.18)
    # Random curl bumps on the dome surface for that wooly look.
    rng = np.random.default_rng(11)
    pieces = [(base_v, base_t)]
    for _ in range(45):
        a = rng.uniform(0, math.pi / 2)
        b = rng.uniform(0, 2 * math.pi)
        bx = centre[0] + math.sin(a) * math.cos(b) * hw * 1.45
        by = centre[1] + math.cos(a) * hw * 1.55
        bz = centre[2] + math.sin(a) * math.sin(b) * hd * 1.40
        v, t = _build_dome(np.array([bx, by, bz], dtype=np.float32),
                              rx=hw * 0.16, ry=hw * 0.16, rz=hw * 0.16,
                              rings=4, segs=8, y_squash=1.0)
        pieces.append((v, t))
    return _combine(*pieces)


def _style_mohawk(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    pieces = []
    n_spikes = 9
    for i in range(n_spikes):
        t = i / (n_spikes - 1)
        z = crown[2] + (t - 0.5) * hd * 1.6
        # Tall central spikes, lower at front + back.
        height = hw * 1.4 - abs(t - 0.5) * hw * 0.7
        base = np.array([crown[0], crown[1] - hw * 0.10, z],
                          dtype=np.float32)
        tip = np.array([crown[0], crown[1] + height, z],
                          dtype=np.float32)
        v, tt = _build_spike(base, tip,
                                base_radius=hw * 0.25, sides=6)
        pieces.append((v, tt))
    # Base strip running front-to-back along the head crown.
    base_centre = crown.copy()
    base_centre[1] -= hw * 0.15
    v_strip, t_strip = _build_dome(base_centre, rx=hw * 0.30,
                                       ry=hw * 0.30, rz=hd * 0.95,
                                       rings=5, segs=12, y_squash=0.85)
    pieces.append((v_strip, t_strip))
    return _combine(*pieces)


def _style_ponytail(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    cap = _style_short_cap(ict_verts, color)
    # Tail hanging behind the head, slightly swung to one side.
    top = np.array([hw * 0.10, crown[1] - hw * 0.25,
                       crown[2] - hd * 1.05], dtype=np.float32)
    bottom = np.array([hw * 0.20, crown[1] - hh * 1.4,
                          crown[2] - hd * 1.40], dtype=np.float32)
    v, t = _build_strand(top, bottom, width=hw * 0.5, segments=6)
    return _combine(cap, (v, t))


def _style_wild_spikes(ict_verts, color):
    crown, hw, hd, hh = _ict_head_anchor(ict_verts)
    cap = _style_short_cap(ict_verts, color)
    pieces = [cap]
    rng = np.random.default_rng(29)
    for _ in range(28):
        # Anchor on the upper hemisphere of the cap (above crown).
        a = rng.uniform(0, math.pi / 2)
        b = rng.uniform(0, 2 * math.pi)
        sx = math.sin(a) * math.cos(b)
        sy = math.cos(a)
        sz = math.sin(a) * math.sin(b)
        bx = crown[0] + sx * hw * 1.05
        by = crown[1] - hw * 0.10 + sy * hw * 1.05
        bz = crown[2] + sz * hd * 1.0
        base = np.array([bx, by, bz], dtype=np.float32)
        # Spikes extend radially outward from the crown.
        out_len = hw * (0.6 + rng.random() * 0.6)
        tip = base + np.array([sx, sy + 0.6, sz],
                                dtype=np.float32) * out_len
        v, t = _build_spike(base, tip,
                              base_radius=hw * 0.18, sides=4)
        pieces.append((v, t))
    return _combine(*pieces)


def _style_buzz(ict_verts, color):
    """Very-thin cap, tight to the skull, low height."""
    crown, hw, hd, _ = _ict_head_anchor(ict_verts)
    centre = crown.copy()
    centre[1] -= hw * 0.95
    return _build_dome(centre, rx=hw * 1.02, ry=hw * 1.05,
                          rz=hd * 1.02, rings=8, segs=20,
                          y_squash=0.95, noise=hw * 0.01)


_STYLE_FUNCS = {
    "short_cap":     _style_short_cap,
    "fringe":        _style_fringe,
    "side_part":     _style_side_part,
    "long_straight": _style_long_straight,
    "curly_afro":    _style_curly_afro,
    "mohawk":        _style_mohawk,
    "ponytail":      _style_ponytail,
    "wild_spikes":   _style_wild_spikes,
    "buzz":          _style_buzz,
}


def list_styles() -> list[str]:
    return ["none"] + list(_STYLE_FUNCS.keys())


def _mouth_anchor(ict_verts: np.ndarray, model
                    ) -> tuple[np.ndarray, float] | None:
    """Centroid of the lip ring on the deformed ICT mesh + a size
    hint (lip-ring radius). Used to anchor the tongue mesh.
    """
    idx = model.name_to_idx.get("mouthClose")
    if idx is None:
        return None
    mags = np.linalg.norm(model.deltas[idx], axis=1)
    top = np.argsort(-mags)[:80]
    pts = ict_verts[top]
    centre = pts.mean(axis=0)
    radius = float(np.linalg.norm(pts - centre, axis=1).mean())
    return centre, max(0.5, radius)


def gen_tongue_mesh(ict_verts: np.ndarray, model,
                      color_hex: str = "#5a1820",
                      *,
                      extend: float = 0.5,
                      lateral: float = 0.0,
                      vertical: float = 0.0,
                      curl: float = 0.0,
                      taper: float = 0.4,
                      jaw_open: float = 0.0,
                      ) -> HairMesh | None:
    """Anatomically-rooted dynamic 3D tongue.

    Travels along the inside of the mouth: rooted at the back of
    the oral cavity (attached to the mandible — drops with
    ``jaw_open``), passes through the lip exit on the midline,
    then the visible tip is steered by extend / lateral / vertical
    / curl / taper. The internal segment never bows outward, so
    the body stays inside the face.

    Parameters (all in [-1, 1] unless noted):
      extend   : -1 fully retracted (returns None) → +1 stuck way out;
                 0 = tip at the lip exit.
      lateral  : tip side-shift after exiting the mouth.
      vertical : tip up/down after exiting; +1 over upper lip,
                 -1 over lower lip.
      curl     : longitudinal flex of the EXTERNAL segment only.
      taper    : 0 blunt, 1 sharply pointed.
      jaw_open : 0..1, drops the tongue root + lip exit with the
                 mandible.
    """
    if extend <= -0.95:
        return None

    anchor = _mouth_anchor(ict_verts, model)
    if anchor is None:
        return None
    centre, lip_r = anchor

    head_w = float(ict_verts[:, 0].max() - ict_verts[:, 0].min())
    head_h = float(ict_verts[:, 1].max() - ict_verts[:, 1].min())

    # ── anatomical anchors (in ICT model coordinates) ──
    # Root: back of the oral cavity. Deep behind the lips, slightly
    # below midline (sits on the mandibular floor).
    root = np.array([0.0,
                       centre[1] - head_h * 0.005,
                       centre[2] - head_w * 0.07], dtype=np.float32)
    # Lip exit: where the tongue passes between the lips. Always on
    # the midline at the lip-ring centroid Z (slightly forward to
    # clear the inside of the lips).
    lip_exit = np.array([0.0,
                            centre[1] - head_h * 0.002,
                            centre[2] + lip_r * 0.20], dtype=np.float32)

    # Jaw open drops the mandible — tongue root drops fully, lip
    # exit drops a bit (lower lip moves more than upper).
    if jaw_open > 0.01:
        drop = head_h * 0.10 * jaw_open
        root[1] -= drop
        lip_exit[1] -= drop * 0.4

    # ── tip (depends on extension sign) ──
    if extend < 0.0:
        # Retracted: tip stays INSIDE the mouth, between root and
        # lip exit. No lateral / vertical / curl applied (the
        # tongue is curled up inside the closed mouth).
        retract_t = -extend  # 0..1
        tip = lip_exit + (root - lip_exit) * (retract_t * 0.55)
        external_present = False
    else:
        # Tip exits the lips and moves under user control. Distance
        # forward scales linearly with extension.
        forward = head_w * 0.22 * extend
        tip = lip_exit.copy()
        tip[2] += forward
        # Lateral + vertical scaled with extension so retracted-ish
        # tip doesn't slosh wildly.
        tip[0] += lateral * head_w * 0.10 * extend
        tip[1] += vertical * head_h * 0.08 * extend
        external_present = True

    # ── centerline build ──
    # Internal segment: root → lip_exit. Always a smooth curve along
    # the inside of the mouth — no user displacement here, so the
    # body never bows out through the cheek.
    int_rings = 6
    # External segment: lip_exit → tip. Bezier with curl displacement
    # applied to the control point (only applies if extended).
    ext_rings = 8 if external_present else 0
    if external_present:
        ext_mid = (lip_exit + tip) * 0.5
        ext_ctrl = ext_mid.copy()
        ext_ctrl[1] += curl * head_h * 0.12 * extend
    else:
        ext_ctrl = lip_exit
    int_mid = (root + lip_exit) * 0.5
    # Slight droop on the internal segment (tongue rests on mouth
    # floor when relaxed).
    int_ctrl = int_mid.copy()
    int_ctrl[1] -= head_h * 0.01

    rings = int_rings + ext_rings
    segs = 14
    base_width = head_w * 0.06
    base_height = head_w * 0.03

    def bezier(p0, p1, p2, t):
        return ((1 - t) ** 2 * p0
                + 2 * (1 - t) * t * p1
                + t ** 2 * p2)

    def bezier_tan(p0, p1, p2, t):
        return 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)

    verts = []
    for i in range(rings + 1):
        if i <= int_rings:
            # Internal segment: t_int ∈ [0, 1]
            t_local = i / max(1, int_rings)
            bt = bezier(root, int_ctrl, lip_exit, t_local)
            tang = bezier_tan(root, int_ctrl, lip_exit, t_local)
            t_global = i / rings
        else:
            # External segment.
            t_local = (i - int_rings) / max(1, ext_rings)
            bt = bezier(lip_exit, ext_ctrl, tip, t_local)
            tang = bezier_tan(lip_exit, ext_ctrl, tip, t_local)
            t_global = i / rings

        tang_len = float(np.linalg.norm(tang))
        if tang_len < 1e-6:
            tang = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            tang = tang / tang_len
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(tang, world_up)
        slen = float(np.linalg.norm(side))
        side = (side / slen) if slen > 1e-6 \
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.cross(side, tang)
        up /= max(1e-6, np.linalg.norm(up))
        scale = (1.0 - taper * t_global)
        end_envelope = math.sin(math.pi * t_global) ** 0.5 \
            if t_global < 0.95 \
            else math.sin(math.pi * 0.95) ** 0.5 * \
                (1 - (t_global - 0.95) / 0.05)
        end_envelope = max(0.05, end_envelope)
        w = base_width * scale * end_envelope
        h = base_height * scale * end_envelope
        for j in range(segs):
            theta = (j / segs) * 2 * math.pi
            sw = math.cos(theta) * w
            sh = math.sin(theta) * h
            v = bt + side * sw + up * sh
            verts.append(v)
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for i in range(rings):
        for j in range(segs):
            j2 = (j + 1) % segs
            a = i * segs + j
            b = i * segs + j2
            c = (i + 1) * segs + j
            d = (i + 1) * segs + j2
            tris.append([a, c, b])
            tris.append([b, c, d])
    tris = np.array(tris, dtype=np.int32)

    n = len(verts)
    color = _hex_rgb_f(color_hex)
    colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
    rng = np.random.default_rng(43)
    noise = (rng.random((n, 1)) - 0.5).astype(np.float32) * 0.10
    colors = np.clip(colors * (1.0 + noise), 0.0, 1.0)
    specular = np.full(n, 0.10, dtype=np.float32)
    emissive = np.zeros(n, dtype=np.float32)
    return HairMesh(verts=verts, tris=tris, colors=colors,
                       specular=specular, emissive=emissive)


def gen_hair_mesh(style: str, ict_verts: np.ndarray,
                    color_hex: str = "#3a2418") -> HairMesh | None:
    """Generate a 3D hair mesh in ICT model coordinates.

    Returns ``None`` for ``style == "none"``.
    """
    if style == "none" or style not in _STYLE_FUNCS:
        return None
    color = _hex_rgb_f(color_hex)
    verts, tris = _STYLE_FUNCS[style](ict_verts, color)
    if len(verts) == 0:
        return None
    n = len(verts)
    colors = np.tile(np.array(color, dtype=np.float32), (n, 1))
    # Subtle per-vertex jitter to break up the flat-color look.
    rng = np.random.default_rng(31)
    noise = (rng.random((n, 1)) - 0.5).astype(np.float32) * 0.18
    colors = np.clip(colors * (1.0 + noise), 0.0, 1.0)
    specular = np.full(n, 0.20, dtype=np.float32)
    emissive = np.zeros(n, dtype=np.float32)
    return HairMesh(verts=verts.astype(np.float32),
                       tris=tris.astype(np.int32),
                       colors=colors,
                       specular=specular,
                       emissive=emissive)
