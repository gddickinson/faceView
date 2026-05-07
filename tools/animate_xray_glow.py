"""Glow-eye demonstration: 8-frame strip across one pulse cycle.

Shows the iris pulse over ~2 seconds. Output to
``docs/images/ict_xray_glow.png``.
"""
from __future__ import annotations

import time
import math
import numpy as np
import cv2

from faceview.vision.ict_face import render_face_ict, _STYLE_PULSE
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


def main() -> None:
    panels: list[np.ndarray] = []
    p = FaceParams(eye_open=1.0)
    apply_persona(p, load_persona("ict_xray"))

    # _emit_pulse_for reads time.monotonic — sleep between frames so
    # the pulse advances visibly. xray pulse is 0.5 Hz → 2 s period.
    period = 1.0 / _STYLE_PULSE["xray"][2]
    n = 8
    for i in range(n):
        img = render_face_ict(p, size=(320, 320))
        # Crop to eye region (centre-y, top-third).
        h, w, _ = img.shape
        crop = img[int(h * 0.30):int(h * 0.55), int(w * 0.20):int(w * 0.80)]
        panels.append(cv2.resize(crop, (240, 100)))
        time.sleep(period / n)

    # Stack as a horizontal strip.
    strip = np.hstack(panels)
    out = "docs/images/ict_xray_glow.png"
    cv2.imwrite(out, strip)
    print(f"wrote {out} ({strip.shape[1]}x{strip.shape[0]})")


if __name__ == "__main__":
    main()
