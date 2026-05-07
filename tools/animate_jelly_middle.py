"""Sample views for the ict_jelly_middle persona.

Outputs:
- ict_jelly_middle_emotions.png — neutral / happy / angry / sad /
  fear / jaw_open emotion grid.
- ict_jelly_middle_rotation.png — yaw -30°…30° in 7 frames.
- ict_jelly_middle_talking.gif — animated talking head with
  subtle sway, ~60 frames.
- ict_xray_young_default.png — preview of the new default Claude
  persona (no anatomy, dynamic sci-fi).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from faceview.vision.ict_face import render_face_ict
from faceview.vision.personas import apply_persona, load_persona
from faceview.vision.sim_face import FaceParams


SIZE = (360, 360)
JELLY = "ict_jelly_middle"
DEFAULT = "ict_xray_young"


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                  (255, 255, 255), 1, cv2.LINE_AA)
    return img


def emotion_grid(persona: str, out_path: str) -> None:
    cases = [
        ("neutral", FaceParams()),
        ("happy", FaceParams(smile=0.85, cheek_raise=0.7)),
        ("angry", FaceParams(brow_lower=0.9, upper_lid_raise=0.4)),
        ("sad", FaceParams(smile=-0.6, lip_corner_drop=0.8,
                            inner_brow_raise=0.7)),
        ("fear", FaceParams(upper_lid_raise=0.9, mouth_stretch=0.6,
                              inner_brow_raise=0.6)),
        ("jaw_open", FaceParams(jaw_open=0.85, upper_lid_raise=0.5)),
    ]
    cells = []
    for label, p in cases:
        apply_persona(p, load_persona(persona))
        img = render_face_ict(p, size=SIZE)
        cells.append(_label(img, label))
    rows = [np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)]
    grid = np.vstack(rows)
    cv2.imwrite(out_path, grid)
    print(f"wrote {out_path}")


def rotation_strip(persona: str, out_path: str) -> None:
    yaws = np.linspace(-0.55, 0.55, 7)
    cells = []
    for yaw in yaws:
        p = FaceParams(yaw=float(yaw))
        apply_persona(p, load_persona(persona))
        img = render_face_ict(p, size=SIZE)
        cells.append(_label(img, f"{int(math.degrees(yaw * 0.6))}°"))
    cv2.imwrite(out_path, np.hstack(cells))
    print(f"wrote {out_path}")


def talking_gif(persona: str, out_path: str) -> None:
    """Animated talking head — driven by TalkingAvatar.tick so the
    real talking-sway code path runs."""
    try:
        import imageio.v2 as imageio
    except Exception:
        print("imageio unavailable; skipping gif")
        return
    from faceview.vision.avatar import TalkingAvatar

    av = TalkingAvatar(persona=persona, emotion="happy")
    av.say("Hello — let me show you the talking head sway in this jelly mode "
            "while I speak naturally.")

    frames = []
    n = 60
    fps = 20
    for i in range(n):
        t = av._t0 + i / fps
        params = av.tick(t)
        img = render_face_ict(params, size=SIZE)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(out_path, frames, duration=1000 / fps, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames)")


def main() -> None:
    emotion_grid(JELLY, "docs/images/ict_jelly_middle_emotions.png")
    rotation_strip(JELLY, "docs/images/ict_jelly_middle_rotation.png")
    talking_gif(JELLY, "docs/images/ict_jelly_middle_talking.gif")
    emotion_grid(DEFAULT, "docs/images/ict_xray_young_emotions.png")
    talking_gif(DEFAULT, "docs/images/ict_xray_young_talking.gif")


if __name__ == "__main__":
    main()
