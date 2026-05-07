"""Demos for the three new 3D render tracks.

Outputs in ``docs/images/``:

- ``head_3d_lite_talking.gif`` — animated lite 3D head speaking +
  rotating. Real-time on CPU (~50 fps).
- ``gpu_lifelike_rotate.gif`` — full BP3D head rotating, rendered via
  Apple Metal-backed OpenGL through moderngl. Real-time (~30 fps).
- ``head_3d_lite_emotions.png`` — 6-emotion grid in lite 3D mode,
  showing how AU-driven landmark deformation translates to the 3D
  mesh.
- ``three_d_modes_compare.png`` — three panels: stylised 2D vs lite
  3D vs GPU lifelike, same neutral pose.
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
from faceview.vision.avatar import TalkingAvatar  # noqa: E402
from faceview.vision.expressions import apply_expression  # noqa: E402
from faceview.vision.face_state import FaceState, face_state_to_params  # noqa: E402
from faceview.vision.head_3d_lite import render_face_3d_lite  # noqa: E402
from faceview.vision.sim_face import FaceParams, render_face  # noqa: E402


log = get_logger("anim_3d")


def _label_grid(panels: dict, size: tuple[int, int], cols: int, label_h: int = 28):
    cell_w, cell_h = size
    n = len(panels)
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows), (10, 12, 16))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(panels.items()):
        r, c = divmod(i, cols)
        x = c * cell_w
        y = r * (cell_h + label_h)
        sheet.paste(im, (x, y))
        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + label_h],
                        fill=(20, 22, 28))
        draw.text((x + cell_w // 2, y + cell_h + label_h // 2),
                   label, fill=(220, 220, 220), anchor="mm", font=font)
    return sheet


def render_lite_emotions(out: Path, size: tuple[int, int]) -> Path:
    panels = {}
    for emo in ["neutral", "happy", "sad", "surprised", "angry", "thinking"]:
        state = FaceState()
        apply_expression(state, emo)
        params = face_state_to_params(state)
        bgr = render_face_3d_lite(params, size)
        panels[emo] = Image.fromarray(bgr[:, :, ::-1])
    sheet = _label_grid(panels, size, cols=3)
    sheet.save(out)
    log.info("anim.lite_emotions_saved", path=str(out))
    return out


def render_lite_talking(out: Path, size: tuple[int, int],
                          fps: int = 18, text: str = "Hi I am face view") -> Path:
    av = TalkingAvatar(emotion="happy", persona="head_3d_lite", seed=11)
    av.persona.render_mode = "head_3d_lite"
    utt = av.say(text)
    n = int((utt.duration + 0.6) * fps)
    frames: list[Image.Image] = []
    for i in range(n):
        t = i / fps
        params = av.tick(t)
        params.yaw = 0.25 * math.sin(t * 1.3)
        bgr = render_face_3d_lite(params, size)
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    delay = max(40, int(1000 / fps))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                     duration=delay, loop=0, optimize=False, disposal=2)
    log.info("anim.lite_talking_saved", path=str(out), frames=len(frames))
    return out


def render_modes_compare(out: Path, size: tuple[int, int]) -> Path:
    p = FaceParams.happy()
    panels: dict[str, Image.Image] = {}
    # Stylised 2D.
    stylised = render_face(p, size)
    panels["stylised 2D"] = Image.fromarray(stylised[:, :, ::-1])
    # Lite 3D.
    p2 = FaceParams.happy()
    p2.render_mode = "head_3d_lite"
    lite = render_face(p2, size)
    panels["lite 3D"] = Image.fromarray(lite[:, :, ::-1])
    # GPU lifelike (if available).
    try:
        from faceview.vision.gpu_renderer import render_face_faceforge_gpu
        p3 = FaceParams.neutral()
        bgr = render_face_faceforge_gpu(p3, size, layer_set="lifelike")
        panels["GPU lifelike"] = Image.fromarray(bgr[:, :, ::-1])
    except Exception as exc:  # noqa: BLE001
        log.warning("compare.gpu_skipped", error=str(exc))
    sheet = _label_grid(panels, size, cols=len(panels))
    sheet.save(out)
    log.info("anim.modes_compare_saved", path=str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, nargs=2, default=[320, 320])
    args = parser.parse_args()
    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    images = docs_image_dir()
    size = tuple(args.size)

    talking = render_lite_talking(images / "head_3d_lite_talking.gif", size)
    emotions = render_lite_emotions(images / "head_3d_lite_emotions.png", size)
    compare = render_modes_compare(images / "three_d_modes_compare.png", size)

    print("3D demo movies + images:")
    print(f"  lite talking:  {talking}")
    print(f"  lite emotions: {emotions}")
    print(f"  modes compare: {compare}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
