"""Full-body avatar meshes — male / female with blend slider.

Loads body_male.obj and body_female.obj (CC0 from faceforge's
human-base-meshes-bundle export). Each is a single-mesh closed
body with no internal anatomy.

The body OBJ uses a different axis convention than the ICT face:
  body: x=lateral (0.92..1.76), y=depth (-0.12..0.17), z=height (0..1.64)
  ICT : x=lateral, y=height (-19..14), z=depth (-9..13)

We pre-bake an axis swap so the body lives in ICT's coordinate
frame, scale + translate it so the body's neck top sits at the
ICT bust bottom, then expose a male↔female morph by linearly
interpolating between the two pre-loaded vertex arrays (faces are
identical).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir


@dataclass
class BodyMesh:
    verts: np.ndarray         # (N, 3) float32 — already in ICT frame
    tris: np.ndarray          # (M, 3) int32
    colors: np.ndarray        # (N, 3) float32 in [0, 1]
    specular: np.ndarray      # (N,) float32
    emissive: np.ndarray      # (N,) float32


def _body_dir() -> Path:
    return assets_dir() / "body_meshes"


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ parser: vertex positions + triangle face indices."""
    verts: list[list[float]] = []
    tris: list[list[int]] = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                # Face indices may be a/b/c or a//c or just a — take
                # the first slot (vertex index, 1-based).
                idxs = []
                for token in line.split()[1:]:
                    idx = int(token.split("/")[0]) - 1
                    idxs.append(idx)
                # Triangulate fans for n-gons.
                for k in range(1, len(idxs) - 1):
                    tris.append([idxs[0], idxs[k], idxs[k + 1]])
    return (np.array(verts, dtype=np.float32),
            np.array(tris, dtype=np.int32))


@lru_cache(maxsize=2)
def _load_body_raw(which: str) -> tuple[np.ndarray, np.ndarray]:
    """Raw OBJ load (verts, tris) for ``"male"`` or ``"female"``,
    with the body's own head + neck stripped — leaving torso /
    arms / legs only. The ICT face replaces the head.
    """
    path = _body_dir() / f"body_{which}.obj"
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32), \
            np.zeros((0, 3), dtype=np.int32)
    verts, tris = _parse_obj(path)
    return _strip_head(verts, tris)


def _strip_head(verts: np.ndarray, tris: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Drop verts above the neck line and any triangle that
    references them. Reindex remaining triangles.

    Body OBJ axes pre-swap: z = height (0..1.64). The head + neck
    occupy the top ~13 % of the height. We keep verts with z below
    `head_floor` so the resulting mesh stops at the shoulder-line
    where the ICT head's bust will overlap.
    """
    if len(verts) == 0:
        return verts, tris
    z_max = float(verts[:, 2].max())
    z_min = float(verts[:, 2].min())
    span = z_max - z_min
    # Head + neck = top 13 % of body. Cut at the shoulder line so
    # ICT bust overlaps neatly with the body's clavicle area.
    head_floor = z_max - span * 0.13
    keep = verts[:, 2] < head_floor
    if not keep.any():
        return verts, tris
    # Reindex triangles: keep only those whose three verts are all
    # below the head floor.
    keep_idx = np.where(keep)[0]
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[keep_idx] = np.arange(len(keep_idx), dtype=np.int32)
    tri_keep_mask = keep[tris].all(axis=1)
    new_tris = remap[tris[tri_keep_mask]]
    return verts[keep], new_tris.astype(np.int32)


def _to_ict_frame(verts: np.ndarray,
                    ict_verts_ref: np.ndarray) -> np.ndarray:
    """Transform body verts into ICT coordinate space.

    Body axes: x=lateral, y=depth (-Y forward), z=height.
    ICT  axes: x=lateral, y=height,             z=depth (+Z forward).

    Steps:
    1. Swap Y↔Z (height up to Y axis).
    2. Centre laterally (body is in [+X..+X], not centred).
    3. Scale so the body's top (head crown level) lines up with
       a position above the ICT bust bottom — body head is replaced
       by the ICT head, but body's neck top should reach the ICT
       collar / shoulders.
    4. Translate vertically so body's NECK top meets ICT bust
       bottom and laterally so x=0 is the spine midline.
    """
    if len(verts) == 0:
        return verts
    # 1. Axis swap to ICT frame:
    #    body x → ICT x  (lateral, identical convention)
    #    body z → ICT y  (height up)
    #    body y → ICT z  (depth, flipped to make face forward)
    # The body OBJ was authored with -Y as the "front" direction;
    # flipping to +Z (ICT's forward) means we negate twice and
    # actually take +Y → ICT z. Net effect: body faces +Z (camera).
    swapped = np.column_stack([
        verts[:, 0],
        verts[:, 2],
        verts[:, 1],
    ]).astype(np.float32)

    # 2. Centre laterally (subtract midline).
    x_mid = (swapped[:, 0].min() + swapped[:, 0].max()) / 2
    swapped[:, 0] -= x_mid
    # And depth midline.
    z_mid = (swapped[:, 2].min() + swapped[:, 2].max()) / 2
    swapped[:, 2] -= z_mid

    # 3. Scale: realistic human is ~7.5 head-heights tall; headless
    # body fills ~6.5 of those. ICT head-only span ≈ ict_y_span × 0.55.
    body_h = swapped[:, 1].max() - swapped[:, 1].min()
    ict_head_h = (ict_verts_ref[:, 1].max() - ict_verts_ref[:, 1].min()) * 0.55
    scale = (ict_head_h * 6.5) / max(body_h, 1e-6)
    swapped *= scale

    # 4. Translate: line up the body's neck top (now max-Y after
    # head removal) with the ICT chin so the head sits directly
    # on the body's shoulders. The ICT bust mesh that extends below
    # the chin is hidden inside the body's upper torso.
    body_top_y = swapped[:, 1].max()
    y_min = float(ict_verts_ref[:, 1].min())
    y_max = float(ict_verts_ref[:, 1].max())
    # ICT chin sits ~50 % up from y_min. Body top should reach that
    # so head appears to grow out of the body.
    chin_y = y_min + (y_max - y_min) * 0.50
    swapped[:, 1] += (chin_y - body_top_y)
    return swapped


def gen_body_mesh(ict_verts_ref: np.ndarray,
                    morph: float = 0.0,
                    color_hex: str = "#3a7088") -> BodyMesh | None:
    """Generate a body mesh in ICT coordinates.

    ``morph`` ∈ [-1, 1] interpolates between fully-female (-1) and
    fully-male (+1). The two OBJ files share vertex count + face
    indices so a linear lerp is straightforward.
    """
    male_v, tris_m = _load_body_raw("male")
    female_v, tris_f = _load_body_raw("female")
    if len(male_v) == 0 or len(female_v) == 0:
        return None
    if male_v.shape != female_v.shape:
        # Different topologies — fall back to whichever is closer to
        # the slider.
        v_raw = male_v if morph > 0 else female_v
        tris = tris_m if morph > 0 else tris_f
    else:
        # Linear blend: morph -1 → female, +1 → male, 0 → 50/50.
        a = (morph + 1.0) * 0.5
        v_raw = female_v * (1.0 - a) + male_v * a
        tris = tris_m

    verts = _to_ict_frame(v_raw, ict_verts_ref)
    if len(verts) == 0:
        return None

    n = len(verts)
    # Skin colour — slightly cyan-tinted so it reads with the xray
    # head palette without clashing.
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
    return BodyMesh(verts=verts.astype(np.float32),
                       tris=tris.astype(np.int32),
                       colors=colors,
                       specular=specular,
                       emissive=emissive)
