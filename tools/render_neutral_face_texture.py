"""Generate the neutral face texture for the face_warp_2d render mode.

Renders the BP3D photo-anatomical head once at neutral pose via the
GPU pipeline, crops to the face-box convention used by the 86-landmark
template, and saves to ``assets/data/neutral_face.png``. Run this once
after copying the BP3D STL meshes into place.

The texture is intentionally cropped so that landmark template
coordinates ``(x, y) in [0, 1]^2`` map roughly to the face features
in the rendered image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PIL import Image  # noqa: E402

from faceview.assets import assets_dir  # noqa: E402
from faceview.core.logger import configure as configure_logging, get_logger  # noqa: E402


log = get_logger("render_face_texture")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--layer-set", default="lifelike")
    parser.add_argument(
        "--out", type=Path,
        default=assets_dir() / "data" / "neutral_face.png",
    )
    args = parser.parse_args()

    configure_logging()

    from faceview.vision.gpu_renderer import render_face_faceforge_gpu
    from faceview.vision.sim_face import FaceParams

    p = FaceParams.neutral()
    bgr = render_face_faceforge_gpu(p, (args.size, args.size),
                                      layer_set=args.layer_set)
    raw = Image.fromarray(bgr[:, :, ::-1])

    # Hand-tuned crop. The face_oval landmark template has chin at
    # y=0.96 and hairline_top at y=0.08 — so the head occupies
    # 0.08..0.96 = 88% of the face-box height. We extend the bottom
    # to y=1.0 so a few pixels of neck are visible (chin won't sit
    # on the bottom edge during jaw-open).
    head_top = 20
    head_h = 338
    unit = head_h / (0.96 - 0.08)            # ~384 px per face-box unit
    # Extend bottom by 4% of unit for neck.
    extended_h = int(unit + 0.04 * unit)
    top = int(head_top - 0.08 * unit)
    left = int((args.size - unit) / 2)
    cropped = raw.crop((left, top, left + int(unit), top + extended_h))
    out = cropped.resize((args.size, args.size), Image.LANCZOS)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    log.info("texture.saved", path=str(args.out), size=out.size)
    print(f"saved neutral face texture: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
