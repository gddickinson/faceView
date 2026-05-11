"""Capture side-view renders of the avatar at varying head pitch so
we can see how far the neck base shifts. Overlay horizontal reference
lines through the rest-pose shoulder/neck-base/chin Y so any drift
is obvious.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _params(pitch_slider: float, gender: str = "male"):
    """pitch_slider ∈ [-1, +1]; internally scaled to ±0.4 rad ≈ ±23°."""
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = float(np.deg2rad(90.0))  # side view
    p._camera_pitch = 0.0
    p._camera_zoom = 0.65
    p.pitch = float(pitch_slider)
    return p


def _capture_rest_landmarks(gender):
    """Render the rest pose and snapshot the shoulder + chin Y."""
    from faceview.vision.ict_face import render_face_ict
    p = _params(0.0, gender)
    img = render_face_ict(p, size=(360, 640))
    return img


def render_label(img, text, font_size=14):
    pil = Image.fromarray(img)
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    drw.rectangle((0, 0, 220, font_size + 8), fill=(0, 0, 0))
    drw.text((5, 3), text, fill=(255, 235, 180), font=font)
    return np.asarray(pil)


def overlay_lines(img, ref_lines):
    """Draw horizontal reference lines at the given pixel Y values."""
    pil = Image.fromarray(img).convert("RGBA")
    drw = ImageDraw.Draw(pil)
    for y, color, label in ref_lines:
        drw.line([(0, y), (pil.width, y)], fill=color, width=1)
        drw.text((pil.width - 60, y - 14), label, fill=color)
    return np.asarray(pil.convert("RGB"))


def main():
    from faceview.vision.ict_face import render_face_ict
    pitches = [-1.0, -0.5, 0.0, 0.5, 1.0]  # slider units
    rows = []
    for gender in ("male", "female"):
        # Find rest-pose reference lines by inspecting the rest image.
        # Pick lines visually based on image proportions of the body
        # in the camera frame.
        # The body fills roughly y=120..620; shoulder ~y=240; neck-base ~y=210; chin ~y=180.
        ref_lines = [
            (160, (220, 220, 80), "chin"),
            (200, (80, 220, 80), "neck-base"),
            (250, (220, 80, 80), "shoulder-line"),
        ]
        cells = []
        for d in pitches:
            p = _params(d, gender)
            img = render_face_ict(p, size=(360, 640))
            img = overlay_lines(img, ref_lines)
            deg = d * 0.4 * 180.0 / np.pi
            img = render_label(img, f"{gender} pitch={d:+.2f} ({deg:+.1f}°)")
            cells.append(img)
        rows.append(np.hstack(cells))
    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/nod_baseline.png")
    print("wrote /tmp/nod_baseline.png shape:", out.shape)


if __name__ == "__main__":
    main()
