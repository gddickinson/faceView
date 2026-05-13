"""Demo grid for the dynamic-xray effects.

Shows the hairless xray head shifting skin tone with mood (anger →
red, sad → blue, fear → pale, happy → cyan, jaw open → magenta)
with a glowing pulsed iris on every panel. Output saved to
``docs/images/ict_xray_moods.png``.
"""
from __future__ import annotations

import time
import numpy as np
import cv2

from faceview.vision.ict_face import render_face_ict
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


def _frame(label: str, **kwargs) -> np.ndarray:
    p = FaceParams(**kwargs)
    apply_persona(p, load_persona("ict_xray"))
    img = render_face_ict(p, size=(320, 320))
    bgr = img.copy()
    cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def build_grid() -> np.ndarray:
    panels = [
        _frame("neutral"),
        _frame("happy", smile=0.85, cheek_raise=0.7),
        _frame("angry", brow_lower=0.9, upper_lid_raise=0.4),
        _frame("sad", smile=-0.6, lip_corner_drop=0.8, inner_brow_raise=0.7),
        _frame("fear (pale)", upper_lid_raise=0.9, mouth_stretch=0.6,
               inner_brow_raise=0.6),
        _frame("jaw open (glow)", jaw_open=0.85, upper_lid_raise=0.3),
    ]
    rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
    return np.vstack(rows)


def main() -> None:
    grid = build_grid()
    out = "docs/images/ict_xray_moods.png"
    cv2.imwrite(out, grid)
    print(f"wrote {out} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
