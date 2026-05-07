"""Demo movies for the layered anatomy renderer + photo-anatomical bridge.

Outputs in ``docs/images/``:

- ``anatomy_layers_grid.png`` — 6-panel grid: skin, x-ray, muscles, eyeballs,
  brain, skull. Same expression in each panel.
- ``anatomy_peel.gif`` — peel-away animation: opaque skin fades to muscles
  to skull to brain. ~3 s loop.
- ``anatomy_meshes_rotate.gif`` — photo-anatomical mode (BodyParts3D STLs)
  rotating. Only emitted when meshes are present.

Usage::

    python -m tools.animate_anatomy_layers
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from faceview.core.logger import configure as configure_logging, get_logger  # noqa: E402
from faceview.utils.paths import docs_image_dir  # noqa: E402
from faceview.vision.anatomy_meshes import meshes_available  # noqa: E402
from faceview.vision.faceforge_bridge import render_face_faceforge  # noqa: E402
from faceview.vision.sim_face import FaceParams  # noqa: E402
from faceview.vision.sim_face_layered import (  # noqa: E402
    LAYER_PRESETS,
    render_face_layered,
)


log = get_logger("anim_layers")


def _grid(images: dict, size: tuple[int, int], cols: int = 3, label_h: int = 28) -> Image.Image:
    cell_w, cell_h = size
    n = len(images)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows), (10, 12, 16))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(images.items()):
        r, c = divmod(i, cols)
        x = c * cell_w
        y = r * (cell_h + label_h)
        sheet.paste(im, (x, y))
        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + label_h],
                        fill=(20, 22, 28))
        draw.text((x + cell_w // 2, y + cell_h + label_h // 2),
                   label, fill=(220, 220, 220), anchor="mm", font=font)
    return sheet


def render_grid(out: Path, size: tuple[int, int]) -> Path:
    p = FaceParams.happy()
    panels = {}
    for label, mode in [
        ("skin (anatomical)", "anatomy_layers"),
        ("x-ray", "anatomy_xray"),
        ("muscle masses", "anatomy_muscles"),
        ("eyeballs", "anatomy_eyeballs"),
        ("brain", "anatomy_brain"),
        ("skull only", "anatomy_skull"),
    ]:
        bgr = render_face_layered(p, size, layers=mode)
        panels[label] = Image.fromarray(bgr[:, :, ::-1])
    sheet = _grid(panels, size, cols=3)
    sheet.save(out)
    log.info("anim.grid_saved", path=str(out))
    return out


def render_peel(out: Path, size: tuple[int, int], fps: int = 18,
                 loop_seconds: float = 4.0) -> Path:
    """Peel-away: cycle skin → muscles → skull → brain → skull → muscles → skin."""
    p = FaceParams.happy()
    n = int(loop_seconds * fps)
    frames: list[Image.Image] = []
    for i in range(n):
        f = i / n  # 0..1 over the cycle
        # Cosine-eased fade across 4 phases.
        phase = f * 4.0
        if phase < 1.0:
            # full skin → faded skin
            t = phase
            spec = [("skull", 1.0), ("brain", 1.0), ("eyeballs", 1.0),
                    ("muscle_masses", 1.0), ("skin", 1.0 - t)]
        elif phase < 2.0:
            t = phase - 1.0
            spec = [("skull", 1.0), ("brain", 1.0), ("eyeballs", 1.0),
                    ("muscle_masses", 1.0 - t)]
        elif phase < 3.0:
            t = phase - 2.0
            spec = [("skull", 1.0 - 0.6 * t), ("brain", 1.0)]
        else:
            t = phase - 3.0
            # back-fade: bring skin back through layers
            spec = [
                ("skull", 0.4 + 0.6 * t),
                ("brain", 1.0 - 0.5 * t),
                ("eyeballs", t),
                ("muscle_masses", t),
                ("skin", t),
            ]
        bgr = render_face_layered(p, size, layers=spec, show_hair=False)
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    delay = max(40, int(1000 / fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=delay, loop=0, optimize=False, disposal=2,
    )
    log.info("anim.peel_saved", path=str(out), frames=len(frames))
    return out


def render_meshes_rotate(out: Path, size: tuple[int, int], fps: int = 18,
                          loop_seconds: float = 4.0) -> Path | None:
    if not meshes_available():
        log.warning("anim.meshes_skip", reason="no STLs in assets/anatomy_meshes")
        return None
    n = int(loop_seconds * fps)
    frames: list[Image.Image] = []
    p = FaceParams.neutral()
    for i in range(n):
        f = i / n
        p.yaw = math.sin(f * math.tau) * 0.7
        p.pitch = math.cos(f * math.tau) * 0.15
        bgr = render_face_faceforge(p, size, layer_set="all_head_neck")
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    delay = max(40, int(1000 / fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:],
        duration=delay, loop=0, optimize=False, disposal=2,
    )
    log.info("anim.meshes_saved", path=str(out), frames=len(frames))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs=2, default=[280, 280])
    parser.add_argument("--peel-size", type=int, nargs=2, default=[400, 400])
    args = parser.parse_args()

    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    images = docs_image_dir()
    grid = render_grid(images / "anatomy_layers_grid.png", tuple(args.size))
    peel = render_peel(images / "anatomy_peel.gif", tuple(args.peel_size))
    meshes = render_meshes_rotate(images / "anatomy_meshes_rotate.gif",
                                    tuple(args.peel_size))

    print("anatomy demos saved:")
    print(f"  grid:  {grid}")
    print(f"  peel:  {peel}")
    print(f"  meshes:{meshes if meshes else '(skipped — no STLs)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
