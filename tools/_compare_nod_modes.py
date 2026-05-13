"""Render side-view comparison of all FACEVIEW_NOD_MODE values at
±1.0 pitch. Cropped tight on the head+neck+upper-torso region with
horizontal reference lines through chin / neck-base / shoulder so
the bend point and any base-of-neck drift is obvious.

Layout (3 rows × 5 cols):
    Row 1: pitch=-1.0 (head down)  modes: current / sharper /
              spine_ripple / anchored / sharp_anchored
    Row 2: pitch=0.0 (rest, identical for all modes)
    Row 3: pitch=+1.0 (head up)    modes: same five
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MODES = ["current", "sharper", "spine_ripple", "anchored", "sharp_anchored"]
LABELS = {
    "current": "CURRENT (legacy)",
    "sharper": "SHARPER (C4-T1 → 0)",
    "spine_ripple": "SPINE RIPPLE (tiny T1-T4)",
    "anchored": "ANCHORED (snap below -0.3)",
    "sharp_anchored": "SHARP + ANCHORED",
}


def _params(pitch_slider: float, gender: str):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = float(np.deg2rad(90.0))
    p._camera_pitch = 0.0
    p._camera_zoom = 1.4
    p.pitch = pitch_slider
    return p


def render(pitch, gender, mode):
    os.environ["FACEVIEW_NOD_MODE"] = mode
    # Bust any module-level caches.
    import importlib, faceview.vision.ict_face as ict
    importlib.reload(ict)
    p = _params(pitch, gender)
    return ict.render_face_ict(p, size=(900, 1200))


def crop_neck(img: np.ndarray) -> np.ndarray:
    """Crop the head + neck + upper-torso band."""
    h, w = img.shape[:2]
    return img[int(h*0.08):int(h*0.50), int(w*0.30):int(w*0.70)]


def label_with_lines(img, title):
    pil = Image.fromarray(img).convert("RGBA")
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    # Reference lines (visually picked for the cropped frame).
    h = pil.height
    refs = [
        (int(h * 0.40), (255, 235, 80), "chin"),
        (int(h * 0.58), (80, 255, 80), "neck-base"),
        (int(h * 0.70), (255, 100, 100), "shoulder"),
    ]
    for y, c, lab in refs:
        drw.line([(0, y), (pil.width, y)], fill=c, width=1)
        drw.text((pil.width - 110, y - 22), lab, fill=c, font=font)
    drw.rectangle((0, 0, pil.width, 36), fill=(0, 0, 0))
    drw.text((6, 6), title, fill=(255, 235, 180), font=font)
    return np.asarray(pil.convert("RGB"))


def main():
    gender = "male"
    rows = []
    for row_label, pitch in (("DOWN -22.9°", -1.0),
                              ("REST 0°", 0.0),
                              ("UP +22.9°", +1.0)):
        cells = []
        for mode in MODES:
            img = render(pitch, gender, mode)
            cropped = crop_neck(img)
            cells.append(label_with_lines(cropped,
                f"{LABELS[mode]} | {row_label}"))
        rows.append(np.hstack(cells))
    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/nod_modes_compare.png")
    print(f"wrote /tmp/nod_modes_compare.png shape: {out.shape}")


if __name__ == "__main__":
    main()
