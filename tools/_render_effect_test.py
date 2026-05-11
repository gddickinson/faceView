"""Render a specific body effect at peak intensity for both genders
and save side-by-side PNGs — for verifying override changes without
restarting the GUI (which caches the rig)."""
import os
import sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image

from tools.extreme_pose_relabel import _make_neutral_params


def _render(gender: str, effect: str, size=(360, 640)):
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS
    handler = HANDLERS.get(f"pre_{effect}") or HANDLERS.get(effect)
    p = _make_neutral_params(gender)
    if handler:
        handler(p, 0.5, 1.0)
    return render_face_ict(p, size=tuple(size))


def main():
    effect = sys.argv[1] if len(sys.argv) > 1 else "stretch_up"
    panels = []
    for gender in ("male", "female"):
        rgb = _render(gender, effect)
        panels.append(rgb)
    panel = np.hstack(panels)
    out = f"/tmp/effect_{effect}_post_skel.png"
    Image.fromarray(panel).save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
