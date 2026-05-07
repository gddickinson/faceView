"""Demo movies for the anatomical renderer.

Outputs in ``docs/images/``:

- ``anatomical_talking.gif`` — TalkingAvatar speaking a phrase rendered
  with the anatomical pipeline (FACS-driven landmarks + 43-muscle
  catalog).
- ``anatomical_overlay.gif`` — same utterance with the muscle activation
  layer drawn on top, so you can see *which* muscles are firing for
  each viseme + idle behaviour.
- ``anatomical_compare.gif`` — side-by-side stylised vs anatomical for
  the same animation.
- ``anatomical_emotions.png`` — labelled grid of 8 emotion presets
  rendered anatomically.

Usage::

    python -m tools.animate_anatomical
    python -m tools.animate_anatomical --text "Tell me about anatomy."
"""

from __future__ import annotations

import argparse
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
from faceview.vision.face_state import face_state_to_params, FaceState  # noqa: E402
from faceview.vision.personas import apply_persona, load_persona  # noqa: E402
from faceview.vision.sim_face import render_face  # noqa: E402


log = get_logger("animate_anat")


def _avatar_frames(text: str, mode: str, fps: int, size: tuple[int, int]) -> list[Image.Image]:
    av = TalkingAvatar(emotion="happy", persona="anatomical", seed=11)
    av.persona.render_mode = mode
    utt = av.say(text)
    total = max(1.0, utt.duration + 0.6)
    n = int(total * fps)
    frames: list[Image.Image] = []
    for i in range(n):
        t = i / fps
        params = av.tick(t)
        bgr = render_face(params, size)
        frames.append(Image.fromarray(bgr[:, :, ::-1]))
    log.info("anim.frames", mode=mode, n=n, dur=total)
    return frames


def _save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    delay = max(40, int(1000 / fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:],
        duration=delay, loop=0, optimize=False, disposal=2,
    )
    log.info("anim.saved", path=str(path), frames=len(frames), delay_ms=delay)


def _side_by_side(left: list[Image.Image], right: list[Image.Image]) -> list[Image.Image]:
    n = min(len(left), len(right))
    out: list[Image.Image] = []
    for i in range(n):
        a = left[i]
        b = right[i]
        canvas = Image.new("RGB", (a.width + b.width + 8, max(a.height, b.height)),
                            (8, 10, 14))
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width + 8, 0))
        out.append(canvas)
    return out


def _emotion_grid(out: Path, size: tuple[int, int], cols: int = 4) -> Path:
    emos = ["neutral", "happy", "sad", "angry",
            "surprised", "fear", "disgust", "thinking"]
    rows = (len(emos) + cols - 1) // cols
    cell_w, cell_h = size
    label_h = 28
    sheet = Image.new("RGB", (cell_w * cols, (cell_h + label_h) * rows), (10, 12, 16))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    persona = load_persona("anatomical")
    for i, emo in enumerate(emos):
        # Build a baseline FaceState for this emotion and tick a few times so
        # the rendered preset settles.
        state = FaceState()
        apply_expression(state, emo)
        params = face_state_to_params(state)
        apply_persona(params, persona)
        bgr = render_face(params, size)
        portrait = Image.fromarray(bgr[:, :, ::-1])
        r, c = divmod(i, cols)
        x = c * cell_w
        y = r * (cell_h + label_h)
        sheet.paste(portrait, (x, y))
        draw = ImageDraw.Draw(sheet)
        draw.rectangle([x, y + cell_h, x + cell_w, y + cell_h + label_h],
                        fill=(20, 22, 28))
        draw.text((x + cell_w // 2, y + cell_h + label_h // 2),
                   emo, fill=(220, 220, 220), anchor="mm", font=font)

    sheet.save(out)
    log.info("anim.grid_saved", path=str(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Hi! I'm faceView. I can see, hear, and talk.")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", type=int, nargs=2, default=[400, 400])
    args = parser.parse_args()

    configure_logging()
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    images = docs_image_dir()
    size = tuple(args.size)

    # 1. Anatomical talking GIF.
    anat = _avatar_frames(args.text, "anatomical", args.fps, size)
    _save_gif(anat, images / "anatomical_talking.gif", args.fps)

    # 2. Anatomy-overlay variant (muscles glowing).
    overlay = _avatar_frames(args.text, "anatomy_overlay", args.fps, size)
    _save_gif(overlay, images / "anatomical_overlay.gif", args.fps)

    # 3. Side-by-side stylised vs anatomical.
    av_styl = TalkingAvatar(emotion="happy", persona="default", seed=11)
    av_styl.say(args.text)
    n = min(len(anat), int((av_styl._utterance.duration + 0.6) * args.fps))
    styl_frames: list[Image.Image] = []
    for i in range(n):
        t = i / args.fps
        params = av_styl.tick(t)
        styl_frames.append(Image.fromarray(render_face(params, size)[:, :, ::-1]))
    compare = _side_by_side(styl_frames, anat[:n])
    _save_gif(compare, images / "anatomical_compare.gif", args.fps)

    # 4. Emotion grid.
    _emotion_grid(images / "anatomical_emotions.png", (260, 260), cols=4)

    print("anatomical demos saved:")
    for name in ["anatomical_talking.gif", "anatomical_overlay.gif",
                 "anatomical_compare.gif", "anatomical_emotions.png"]:
        print(f"  {images / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
