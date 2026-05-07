"""Rank ICT personas by how well the BP3D anatomy fits their head.

For each ICT identity preset we measure two metrics on the jelly
composite:

  - residual: mean Euclidean distance (in pixels) between ICT and
    BP3D anchor pairs *after* the rigid similarity transform that
    aligns BP3D to ICT. Lower = the rigid warp could line the
    anchor sets up more closely → BP3D's natural proportions are
    closer to that ICT head shape.

  - silhouette mismatch (px): number of pixels where the ICT
    silhouette and warped-BP3D silhouette disagree. Lower = the
    BP3D anatomy fits inside the ICT head shape more tightly.

Output:
  - docs/images/ict_jelly_ranking.png — comparison grid sorted by
    fit (best on the left).
  - stdout: ranking table.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from faceview.vision.anatomy_catalog import specs_for_layer_set
from faceview.vision.anatomy_meshes import list_available_meshes
from faceview.vision.ict_face import (
    _NECK_MUSCLE_TOKENS, _bp3d_feature_points_3d, _hex_to_rgb,
    _ict_feature_points_3d_for, _is_neck_muscle,
    _project_bp3d_to_pixel, _project_ict_to_pixel,
    _shared_anatomy_renderer, render_face_ict,
)
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


SIZE = (360, 360)
BG = "#000810"

# All ICT identity presets shipped in personas.json (the ones with
# real identity_weights — skip the sci-fi style flips).
PERSONAS = [
    "ict_face_3d",
    "ict_male",
    "ict_female",
    "ict_claude",
    "ict_alt1",
    "ict_alt2",
    "ict_male_young",
    "ict_male_middle",
    "ict_male_elder",
    "ict_female_young",
    "ict_female_middle",
    "ict_female_elder",
]


def _persona_iw(name: str) -> dict[str, float]:
    raw = load_persona(name).identity_weights or {}
    return {k: float(v) for k, v in raw.items()
             if isinstance(v, (int, float))}


def _features_present_specs() -> list:
    avail = set(list_available_meshes())
    raw = [s for s in specs_for_layer_set("features") if s.fma in avail]
    return [s for s in raw if s.category != "vertebra"
             and not _is_neck_muscle(s.name)]


def _silhouette_mask(img: np.ndarray, bg_rgb) -> np.ndarray:
    bg = np.array(bg_rgb, dtype=np.float32)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    return (diff > 25.0).astype(np.uint8)


def measure(persona_name: str) -> dict:
    """Run the full alignment pipeline for one persona; return metrics."""
    iw = _persona_iw(persona_name)
    specs = _features_present_specs()
    bg_rgb = _hex_to_rgb(BG)

    # Project all anchors through both renderers' MVPs.
    ict_3d = _ict_feature_points_3d_for(iw)
    bp3_3d = _bp3d_feature_points_3d() or {}
    ict_pix = _project_ict_to_pixel(ict_3d, 0.0, 0.0, SIZE,
                                       identity_weights=iw)
    bp3_pix = _project_bp3d_to_pixel(bp3_3d, specs, 0.0, 0.0, SIZE)
    used = sorted(set(ict_pix) & set(bp3_pix))
    if len(used) < 5:
        return {"persona": persona_name, "residual": float("inf"),
                "silhouette_mismatch": -1, "n_anchors": len(used)}

    src = np.array([bp3_pix[k] for k in used], dtype=np.float32)
    dst = np.array([ict_pix[k] for k in used], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return {"persona": persona_name, "residual": float("inf"),
                "silhouette_mismatch": -1, "n_anchors": len(used)}

    # Apply the same warp to the src points and measure residual.
    src_h = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
    src_warped = (M @ src_h.T).T
    diff = src_warped - dst
    residual = float(np.linalg.norm(diff, axis=1).mean())

    # Silhouette mismatch via the actual jelly composite path.
    p = FaceParams()
    apply_persona(p, load_persona(persona_name))
    # Force jelly mode regardless of persona's style.
    p._persona_style = "jelly"
    composite = render_face_ict(p, size=SIZE)

    # ICT-only silhouette for the same params.
    p._persona_style = "xray"
    ict_only = render_face_ict(p, size=SIZE)

    # BP3D-only (warped) — render anatomy then apply same M.
    ana = _shared_anatomy_renderer().render(
        specs, SIZE, yaw=0.0, pitch=0.0, bg=bg_rgb,
    )
    ana_warped = cv2.warpAffine(
        ana, M, SIZE,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in bg_rgb),
    )
    ict_mask = _silhouette_mask(ict_only, bg_rgb)
    ana_mask = _silhouette_mask(ana_warped, bg_rgb)
    # Restrict to the upper 75 % of frame so the bust mismatch
    # (which the silhouette clip handles in the visible composite)
    # doesn't dominate the score — we want to compare HEAD fit.
    h = SIZE[1]
    cutoff = int(h * 0.75)
    ict_mask_head = ict_mask.copy()
    ana_mask_head = ana_mask.copy()
    ict_mask_head[cutoff:] = 0
    ana_mask_head[cutoff:] = 0
    xor = np.logical_xor(ict_mask_head, ana_mask_head)
    silhouette_mismatch = int(xor.sum())

    return {
        "persona": persona_name,
        "residual": residual,
        "silhouette_mismatch": silhouette_mismatch,
        "n_anchors": len(used),
        "scale": float(np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)),
        "_composite": composite,
    }


def main() -> None:
    results = []
    for name in PERSONAS:
        try:
            r = measure(name)
            results.append(r)
            print(f"{name:22s}  residual={r['residual']:6.2f}px  "
                  f"silh_mismatch={r['silhouette_mismatch']:6d}px  "
                  f"scale={r['scale']:.3f}  n={r['n_anchors']}")
        except Exception as e:
            print(f"{name:22s}  ERROR: {e}")

    # Combined fit score: normalised residual + normalised silhouette.
    res_arr = np.array([r["residual"] for r in results], dtype=np.float64)
    sil_arr = np.array([r["silhouette_mismatch"] for r in results],
                          dtype=np.float64)

    def norm(a):
        a = np.where(np.isfinite(a), a, np.nan)
        lo = np.nanmin(a)
        hi = np.nanmax(a)
        if hi <= lo:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    score = norm(res_arr) + norm(sil_arr)
    for r, s in zip(results, score):
        r["score"] = float(s)
    results.sort(key=lambda r: r["score"])

    print()
    print("=== Ranking (best fit → worst) ===")
    for i, r in enumerate(results):
        print(f"{i+1:2d}. {r['persona']:22s}  score={r['score']:.3f}  "
              f"resid={r['residual']:6.2f}  silh={r['silhouette_mismatch']:6d}")

    # Comparison grid: best at top, worst at bottom (3 cols × 4 rows).
    cells = []
    for r in results:
        img = r["_composite"].copy()
        label = (f"{r['persona']}\n"
                  f"score={r['score']:.2f}  "
                  f"resid={r['residual']:.1f}px  "
                  f"silh={r['silhouette_mismatch']}px")
        for i, line in enumerate(label.split("\n")):
            cv2.putText(img, line, (8, 22 + i * 18),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                          (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(img)

    cols = 3
    rows = (len(cells) + cols - 1) // cols
    while len(cells) < rows * cols:
        cells.append(np.zeros_like(cells[0]))
    grid = np.vstack([
        np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)
    ])
    out = Path("docs/images/ict_jelly_ranking.png")
    cv2.imwrite(str(out), grid)
    print(f"\nwrote {out} {grid.shape}")

    # Also dump a json record so the result is queryable.
    json_path = Path("docs/images/ict_jelly_ranking.json")
    json_path.write_text(json.dumps([
        {k: v for k, v in r.items() if k != "_composite"}
        for r in results
    ], indent=2))
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
