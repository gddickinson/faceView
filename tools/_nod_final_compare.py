"""Final before/after comparison: legacy CURRENT vs new SPINE_RIPPLE
default. Large side-by-side side-view of the head + neck + upper
torso at ±22.9° head pitch, with overlay rest-pose reference.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _params(pitch_slider, gender):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_yaw = float(np.deg2rad(90.0))
    p._camera_zoom = 0.9
    p.pitch = pitch_slider
    return p


def render(pitch, gender, mode):
    os.environ["FACEVIEW_NOD_MODE"] = mode
    import importlib, faceview.vision.ict_face as ict
    importlib.reload(ict)
    return ict.render_face_ict(_params(pitch, gender), size=(900, 1200))


def crop_neck(img):
    h, w = img.shape[:2]
    return img[int(h*0.05):int(h*0.55), int(w*0.20):int(w*0.80)]


def overlay_rest_outline(img, rest_img, color=(255, 80, 80)):
    """Draw the rest pose's outline edge over the pitched image so
    drift in the body region is visible as a deviation between
    silhouettes."""
    g_rest = (rest_img[..., 0].astype(np.float32) * 0.3
              + rest_img[..., 1] * 0.59 + rest_img[..., 2] * 0.11)
    edge_x = np.abs(np.diff(g_rest, axis=1, prepend=g_rest[:, :1]))
    edge_y = np.abs(np.diff(g_rest, axis=0, prepend=g_rest[:1, :]))
    edge = np.maximum(edge_x, edge_y) > 25.0
    out = img.copy()
    for c in range(3):
        out[..., c] = np.where(edge, color[c], out[..., c])
    return out


def label_with_lines(img, title):
    pil = Image.fromarray(img).convert("RGBA")
    drw = ImageDraw.Draw(pil)
    try:
        font_t = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
        font_r = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
    except Exception:
        font_t = font_r = ImageFont.load_default()
    h = pil.height
    refs = [
        (int(h * 0.38), (255, 235, 80), "chin"),
        (int(h * 0.54), (80, 255, 80), "neck-base"),
        (int(h * 0.66), (255, 100, 200), "shoulder"),
    ]
    for y, c, lab in refs:
        drw.line([(0, y), (pil.width, y)], fill=c, width=1)
        drw.text((pil.width - 130, y - 22), lab, fill=c, font=font_r)
    drw.rectangle((0, 0, pil.width, 44), fill=(0, 0, 0))
    drw.text((10, 10), title, fill=(255, 240, 180), font=font_t)
    return np.asarray(pil.convert("RGB"))


def main():
    gender = "male"
    rest_current = crop_neck(render(0.0, gender, "current"))
    cells = []
    for mode_label, mode in (("BEFORE (legacy)", "current"),
                              ("AFTER (spine_ripple)", "spine_ripple")):
        for pitch_label, pitch in (("DOWN -22.9°", -1.0),
                                    ("UP +22.9°", +1.0)):
            pitched = crop_neck(render(pitch, gender, mode))
            overlaid = overlay_rest_outline(pitched, rest_current)
            cells.append(label_with_lines(overlaid,
                f"{mode_label} | {pitch_label}"))
    # 2x2 grid
    top = np.hstack(cells[:2])
    bot = np.hstack(cells[2:])
    out = np.vstack([top, bot])
    Image.fromarray(out).save("/tmp/nod_final_compare.png")
    print(f"wrote /tmp/nod_final_compare.png shape: {out.shape}")


if __name__ == "__main__":
    main()
