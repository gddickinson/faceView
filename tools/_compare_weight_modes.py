"""Render side-by-side hard vs graded_3ring weight modes."""
import os, sys, importlib
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


def render(gender, effect, mode, size=(360, 640)):
    # Force the chosen weight mode then INVALIDATE the rig cache
    os.environ["FACEVIEW_RIG_WEIGHT_MODE"] = mode
    import faceview.vision.body_rig as br
    br._cached_rig_state.cache_clear()
    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS
    p = _make_neutral_params(gender)
    if effect != "neutral":
        HANDLERS[effect](p, 0.5, 1.0)
    return render_face_ict(p, size=tuple(size))


def label(img, text):
    pil = Image.fromarray(img)
    drw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    drw.rectangle((0, 0, 200, 22), fill=(0, 0, 0))
    drw.text((5, 3), text, fill=(255, 235, 180), font=font)
    return np.asarray(pil)


def main():
    effects = ["neutral", "arms_up", "arms_out", "salute",
                "wave_left", "clap", "arms_crossed", "stretch_up"]
    rows = []
    for g in ("male", "female"):
        for mode in ("hard", "graded_3ring"):
            cells = []
            for e in effects:
                img = render(g, e, mode)
                cells.append(label(img, f"{e}"))
            row = np.hstack(cells)
            # Side label
            side_pil = Image.fromarray(row)
            drw = ImageDraw.Draw(side_pil)
            drw.rectangle((0, row.shape[0]-22, 280, row.shape[0]),
                          fill=(0, 0, 0))
            drw.text((5, row.shape[0]-19), f"{g} / {mode}",
                       fill=(255, 220, 120))
            rows.append(np.asarray(side_pil))
    out = np.vstack(rows)
    Image.fromarray(out).save("/tmp/weight_mode_compare.png")
    print("wrote /tmp/weight_mode_compare.png")


if __name__ == "__main__":
    main()
