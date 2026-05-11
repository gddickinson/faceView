"""Zoom comparison of shoulder seam: hard vs graded weights at peak
arm raise — the place where graded weights should make the biggest
visible difference."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _make_neutral_params(gender):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_zoom = 0.55
    return p


def render(gender, effect, mode, size=(720, 1280)):
    os.environ["FACEVIEW_RIG_WEIGHT_MODE"] = mode
    import faceview.vision.body_rig as br
    br._cached_rig_state.cache_clear()
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS
    p = _make_neutral_params(gender)
    if effect != "neutral":
        HANDLERS[effect](p, 0.5, 1.0)
    return render_face_ict(p, size=size)


def label(img, text):
    pil = Image.fromarray(img)
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    drw.rectangle((0, 0, 360, 28), fill=(0, 0, 0))
    drw.text((5, 4), text, fill=(255, 235, 180), font=font)
    return np.asarray(pil)


def crop_shoulders(img):
    """Zoom to shoulder/upper-torso area where seam artifacts show."""
    h, w = img.shape[:2]
    # Body sits around centre, head around top quarter. Crop shoulders.
    y0 = int(h * 0.18)
    y1 = int(h * 0.50)
    x0 = int(w * 0.20)
    x1 = int(w * 0.80)
    return img[y0:y1, x0:x1]


def main():
    rows = []
    for g in ("male", "female"):
        for effect in ("arms_up", "arms_crossed", "stretch_up", "clap"):
            hard  = render(g, effect, "hard")
            graded = render(g, effect, "graded_3ring")
            zh = crop_shoulders(hard)
            zg = crop_shoulders(graded)
            row = np.hstack([label(zh, f"{g} {effect} HARD"),
                             label(zg, f"{g} {effect} GRADED")])
            rows.append(row)
    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/seam_compare.png")
    print("wrote /tmp/seam_compare.png  shape:", out.shape)


if __name__ == "__main__":
    main()
