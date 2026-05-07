"""Jelly-mode alignment assessment.

Generates several diagnostic images for the BP3D-anatomy /
ICT-jelly composite:

1. ``ict_jelly_assessment_grid.png`` — per-emotion (rows) ×
   per-yaw-angle (cols) grid showing the jelly composite with
   ICT and BP3D anchor markers overlaid in two colours so you can
   see how closely the warp pulls them onto each other.

2. ``ict_jelly_assessment_heatmap.png`` — per-emotion silhouette
   diff: where the BP3D anatomy fails to fall inside the ICT skin
   silhouette, brighter = more mismatch.

3. ``ict_jelly_assessment_rotation.gif`` — yaw -25°…25° rotating
   loop on a single neutral face (animated, ~12 frames).
"""
from __future__ import annotations

import math
import sys

import cv2
import numpy as np

from faceview.vision.ict_face import (
    _align_anatomy_to_ict, _bp3d_feature_points_3d, _hex_to_rgb,
    _ict_feature_points_3d_for, _project_bp3d_to_pixel,
    _project_ict_to_pixel, _shared_anatomy_renderer,
    render_face_ict,
)
from faceview.vision.anatomy_catalog import specs_for_layer_set
from faceview.vision.anatomy_meshes import list_available_meshes
from faceview.vision.ict_face import _NECK_MUSCLE_TOKENS, _is_neck_muscle
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


PERSONA = "ict_jelly_young"
SIZE = (360, 360)
BG = "#000810"


def emotions() -> dict[str, FaceParams]:
    return {
        "neutral": FaceParams(),
        "happy": FaceParams(smile=0.85, cheek_raise=0.7),
        "angry": FaceParams(brow_lower=0.9, upper_lid_raise=0.4),
        "sad": FaceParams(smile=-0.6, lip_corner_drop=0.8,
                            inner_brow_raise=0.7),
        "jaw_open": FaceParams(jaw_open=0.85, upper_lid_raise=0.5),
    }


def _persona_iw(name: str) -> dict[str, float]:
    raw = load_persona(name).identity_weights or {}
    return {k: float(v) for k, v in raw.items()
             if isinstance(v, (int, float))}


def _features_present_specs() -> list:
    avail = set(list_available_meshes())
    raw = [s for s in specs_for_layer_set("features") if s.fma in avail]
    return [s for s in raw if s.category != "vertebra"
             and not _is_neck_muscle(s.name)]


def _draw_anchors(img: np.ndarray, ict_pix: dict, bp3_pix: dict,
                   used: list[str]) -> np.ndarray:
    """ICT anchors as green crosses; BP3D anchors as red dots; lines
    between corresponding pairs in yellow. Larger image annotation."""
    out = img.copy()
    for k in used:
        ix, iy = ict_pix[k]
        bx, by = bp3_pix[k]
        cv2.line(out, (int(bx), int(by)), (int(ix), int(iy)),
                  (0, 220, 220), 1, cv2.LINE_AA)
        cv2.circle(out, (int(bx), int(by)), 4, (0, 0, 220), -1, cv2.LINE_AA)
        cv2.drawMarker(out, (int(ix), int(iy)), (0, 220, 0),
                        cv2.MARKER_CROSS, 8, 2, cv2.LINE_AA)
    return out


def _silhouette_mask(img: np.ndarray, bg_rgb: tuple[int, int, int]) -> np.ndarray:
    bg = np.array(bg_rgb, dtype=np.float32)
    diff = np.linalg.norm(img.astype(np.float32) - bg[None, None, :], axis=2)
    return (diff > 25.0).astype(np.uint8)


def assessment_grid() -> np.ndarray:
    """Per-emotion × per-yaw grid with anchor markers."""
    yaws = [-0.45, -0.20, 0.0, 0.20, 0.45]
    emos = emotions()
    iw = _persona_iw(PERSONA)
    specs = _features_present_specs()
    bg_rgb = _hex_to_rgb(BG)

    rows = []
    for emo_name, emo_params in emos.items():
        cells = []
        for yaw in yaws:
            p = FaceParams(**{**emo_params.__dict__, "yaw": yaw})
            apply_persona(p, load_persona(PERSONA))

            ict_3d = _ict_feature_points_3d_for(iw)
            bp3_3d = _bp3d_feature_points_3d() or {}
            ict_pix = _project_ict_to_pixel(ict_3d, yaw * 0.6, 0, SIZE,
                                                identity_weights=iw)
            bp3_pix = _project_bp3d_to_pixel(bp3_3d, specs, yaw * 0.6, 0, SIZE)
            used = sorted(set(ict_pix) & set(bp3_pix))

            img = render_face_ict(p, size=SIZE)
            img = _draw_anchors(img, ict_pix, bp3_pix, used)
            cv2.putText(img, f"{emo_name} {int(math.degrees(yaw*0.6))}°",
                          (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                          (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def assessment_heatmap() -> np.ndarray:
    """Per-emotion silhouette mismatch heatmap — bright pixels mean
    the BP3D anatomy projects outside the ICT skin silhouette."""
    iw = _persona_iw(PERSONA)
    specs = _features_present_specs()
    bg_rgb = _hex_to_rgb(BG)
    cells = []

    for emo_name, emo_params in emotions().items():
        p = FaceParams(**emo_params.__dict__)
        apply_persona(p, load_persona(PERSONA))

        # Render ICT-only silhouette (force xray, no jelly).
        p._persona_style = "xray"
        ict_only = render_face_ict(p, size=SIZE)
        p._persona_style = "jelly"

        # Render warped BP3D silhouette without ICT overlay.
        ana = _shared_anatomy_renderer().render(
            specs, SIZE, yaw=0.0, pitch=0.0, bg=bg_rgb,
        )
        ana_warped = _align_anatomy_to_ict(
            ana, bg_rgb, specs, 0.0, 0.0, SIZE,
            identity_weights=iw,
        )
        ict_mask = _silhouette_mask(ict_only, bg_rgb)
        ana_mask = _silhouette_mask(ana_warped, bg_rgb)

        # XOR — pixels where exactly one mask is set.
        xor = np.logical_xor(ict_mask, ana_mask).astype(np.float32)
        # Distance from the ICT mask: how far inside / outside is the
        # mismatch? Larger = worse.
        dist_inside = cv2.distanceTransform(ict_mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(1 - ict_mask, cv2.DIST_L2, 3)

        only_ana_outside = ana_mask * (1 - ict_mask)
        only_ict_no_ana = ict_mask * (1 - ana_mask)
        weighted = (only_ana_outside * dist_outside
                    + only_ict_no_ana * dist_inside)
        norm = np.clip(weighted / 30.0, 0.0, 1.0)
        heat = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        # Composite: faint ICT silhouette under heatmap.
        ict_dim = (ict_only * 0.35).astype(np.uint8)
        out = np.where(heat[:, :, None] > 8, heat_color, ict_dim)
        out = out.astype(np.uint8)

        n_pixels = int(xor.sum())
        cv2.putText(out, f"{emo_name}: {n_pixels}px", (8, 22),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                      (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(out)

    return np.hstack(cells)


def rotation_gif(out_path: str) -> None:
    """yaw -25°…25° loop on the neutral face."""
    try:
        import imageio.v2 as imageio
    except Exception:
        print("imageio not available; skipping gif", file=sys.stderr)
        return
    frames = []
    for yaw in np.linspace(-0.45, 0.45, 12):
        p = FaceParams(yaw=float(yaw))
        apply_persona(p, load_persona(PERSONA))
        img = render_face_ict(p, size=SIZE)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Loop back.
    frames = frames + frames[::-1]
    imageio.mimsave(out_path, frames, duration=80, loop=0)


def main() -> None:
    grid = assessment_grid()
    cv2.imwrite("docs/images/ict_jelly_assessment_grid.png", grid)
    print(f"wrote ict_jelly_assessment_grid.png {grid.shape}")

    heat = assessment_heatmap()
    cv2.imwrite("docs/images/ict_jelly_assessment_heatmap.png", heat)
    print(f"wrote ict_jelly_assessment_heatmap.png {heat.shape}")

    rotation_gif("docs/images/ict_jelly_assessment_rotation.gif")
    print("wrote ict_jelly_assessment_rotation.gif")


if __name__ == "__main__":
    main()
