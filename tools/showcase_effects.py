"""Render a thumbnail per effect at peak intensity (u≈0.5).

Output: docs/images/effects_<category>.png — one per category, a
grid of labelled thumbnails. Plus docs/images/effects_all.png — a
master grid of every effect.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from faceview.vision.effects import (
    POST_HANDLERS, PRE_HANDLERS, REGISTRY, Stage, specs_by_category,
)
from faceview.vision.ict_face import render_face_ict
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


SIZE = (320, 320)
PERSONA = "ict_xray_young"


def _base() -> np.ndarray:
    p = FaceParams()
    apply_persona(p, load_persona(PERSONA))
    return render_face_ict(p, size=SIZE)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    bar_h = 28
    cv2.rectangle(out, (0, 0), (img.shape[1], bar_h), (24, 24, 24), -1)
    cv2.putText(out, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                  (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_post(spec) -> np.ndarray:
    base = _base()
    handler = POST_HANDLERS.get(spec.name)
    if handler is None:
        return base
    # Peak intensity at u=0.5.
    return handler(base, 0.5, 1.0)


def _render_pre(spec) -> np.ndarray:
    p = FaceParams()
    apply_persona(p, load_persona(PERSONA))
    handler = PRE_HANDLERS.get(spec.name)
    if handler is not None:
        handler(p, 0.5, 1.0)
    return render_face_ict(p, size=SIZE)


def _grid(thumbnails: list[np.ndarray], cols: int = 4) -> np.ndarray:
    if not thumbnails:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    rows = (len(thumbnails) + cols - 1) // cols
    while len(thumbnails) < rows * cols:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    grid = np.vstack([
        np.hstack(thumbnails[r * cols:(r + 1) * cols]) for r in range(rows)
    ])
    return grid


def main() -> None:
    by_cat = specs_by_category()
    all_cells: list[np.ndarray] = []
    for cat, specs in by_cat.items():
        cells: list[np.ndarray] = []
        for spec in specs:
            if spec.stage == Stage.POST:
                img = _render_post(spec)
            else:
                img = _render_pre(spec)
            cells.append(_label(img, spec.label))
        out = _grid(cells, cols=4)
        path = f"docs/images/effects_{cat}.png"
        cv2.imwrite(path, out)
        print(f"wrote {path} ({len(specs)} effects)")
        all_cells.extend(cells)

    # Combined grid.
    big = _grid(all_cells, cols=5)
    cv2.imwrite("docs/images/effects_all.png", big)
    print(f"wrote docs/images/effects_all.png ({len(all_cells)} total)")


if __name__ == "__main__":
    main()
