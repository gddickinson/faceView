"""Render rest pose + pitched pose as an OVERLAY for each nod mode.
Rest is drawn in red, pitched in blue, alpha-blended. Any drift at
the neck base shows up as red+blue ghosting; perfect anchoring shows
red (rest) and blue (head) only.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MODES = ["current", "sharper", "spine_ripple", "anchored", "sharp_anchored"]
LABELS = {
    "current": "CURRENT (legacy)",
    "sharper": "SHARPER",
    "spine_ripple": "SPINE_RIPPLE",
    "anchored": "ANCHORED",
    "sharp_anchored": "SHARP+ANCHORED",
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
    import importlib, faceview.vision.ict_face as ict
    importlib.reload(ict)
    p = _params(pitch, gender)
    return ict.render_face_ict(p, size=(900, 1200))


def crop_neck(img):
    h, w = img.shape[:2]
    return img[int(h*0.08):int(h*0.50), int(w*0.30):int(w*0.70)]


def overlay(rest_img, pitched_img):
    """Tint rest red, pitched cyan, blend. Drift = magenta/yellow
    fringes; aligned regions = grey/white."""
    rest = rest_img.astype(np.float32)
    pose = pitched_img.astype(np.float32)
    # Convert to luma
    rest_l = (rest[..., 0]*0.3 + rest[..., 1]*0.59 + rest[..., 2]*0.11)
    pose_l = (pose[..., 0]*0.3 + pose[..., 1]*0.59 + pose[..., 2]*0.11)
    out = np.zeros_like(rest)
    out[..., 0] = rest_l                # rest in red channel
    out[..., 1] = pose_l * 0.5          # pitched in green half
    out[..., 2] = pose_l                # pitched in blue
    return np.clip(out, 0, 255).astype(np.uint8)


def label_with_lines(img, title):
    pil = Image.fromarray(img).convert("RGBA")
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    h = pil.height
    refs = [
        (int(h * 0.40), (255, 235, 80), "chin"),
        (int(h * 0.58), (80, 255, 80), "neck-base"),
        (int(h * 0.70), (255, 100, 100), "shoulder"),
    ]
    for y, c, lab in refs:
        drw.line([(0, y), (pil.width, y)], fill=c, width=1)
        drw.text((pil.width - 130, y - 26), lab, fill=c, font=font)
    drw.rectangle((0, 0, pil.width, 40), fill=(0, 0, 0))
    drw.text((6, 7), title, fill=(255, 235, 180), font=font)
    return np.asarray(pil.convert("RGB"))


def main():
    gender = "male"
    # Rest is identical across modes, so we just use any. Use "current".
    rest = crop_neck(render(0.0, gender, "current"))
    rows = []
    for pitch, row_lbl in ((-1.0, "DOWN -22.9°"), (+1.0, "UP +22.9°")):
        cells = []
        for mode in MODES:
            pitched = crop_neck(render(pitch, gender, mode))
            blended = overlay(rest, pitched)
            cells.append(label_with_lines(blended,
                f"{LABELS[mode]} | {row_lbl}"))
        rows.append(np.hstack(cells))
    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/nod_overlay_compare.png")
    print(f"wrote /tmp/nod_overlay_compare.png shape: {out.shape}")


if __name__ == "__main__":
    main()
