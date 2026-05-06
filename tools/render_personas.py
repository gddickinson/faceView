"""Render every bundled persona at a fixed emotion into a contact sheet.

Outputs ``docs/images/personas.png`` — a row of N portraits, one per persona,
labelled with the persona name. Useful as a README block for the persona
system and a visual regression target.

Usage::

    python -m tools.render_personas
    python -m tools.render_personas --emotion happy --size 320 320
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before any PySide6 import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("FACEVIEW_HEADLESS", "1")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from faceview.utils.paths import docs_image_dir  # noqa: E402
from faceview.vision.avatar import TalkingAvatar  # noqa: E402
from faceview.vision.personas import list_personas  # noqa: E402
from faceview.vision.sim_face import render_face  # noqa: E402


_LABEL_HEIGHT = 28


def _render_one(name: str, emotion: str, size: tuple[int, int]) -> Image.Image:
    """Render a single persona portrait at the requested emotion."""
    av = TalkingAvatar(emotion=emotion, persona=name, seed=11)
    # Tick a few frames so the AU smoothing has reached the baseline.
    for i in range(8):
        params = av.tick(i * 0.05)
    bgr = render_face(params, size)
    return Image.fromarray(bgr[:, :, ::-1])


def render_sheet(
    *,
    emotion: str,
    size: tuple[int, int],
    out: Path,
    cols: int = 4,
) -> Path:
    names = list_personas()
    rows = (len(names) + cols - 1) // cols
    cell_w, cell_h = size
    sheet_w = cell_w * cols
    sheet_h = (cell_h + _LABEL_HEIGHT) * rows
    sheet = Image.new("RGB", (sheet_w, sheet_h), (12, 14, 18))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except OSError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(sheet)

    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        x = c * cell_w
        y = r * (cell_h + _LABEL_HEIGHT)
        portrait = _render_one(name, emotion, size)
        sheet.paste(portrait, (x, y))
        # Label band underneath.
        band_y = y + cell_h
        draw.rectangle([x, band_y, x + cell_w, band_y + _LABEL_HEIGHT],
                       fill=(20, 22, 28))
        draw.text(
            (x + cell_w // 2, band_y + _LABEL_HEIGHT // 2),
            name, fill=(220, 220, 220), anchor="mm", font=font,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", default="happy")
    parser.add_argument("--size", type=int, nargs=2, default=[300, 300])
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    # QApplication is required for QPainter inside render_face.
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841

    out = docs_image_dir() / "personas.png"
    saved = render_sheet(
        emotion=args.emotion,
        size=tuple(args.size),
        out=out,
        cols=args.cols,
    )
    print(f"persona sheet saved: {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
