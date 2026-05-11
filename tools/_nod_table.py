"""Render the per-Y-band displacement comparison as a clean table
image so the relative improvements across modes are easy to scan."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

MODES = ["current", "sharper", "spine_ripple", "anchored", "sharp_anchored"]


def _params(pitch_slider, gender):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_zoom = 0.55
    p.pitch = pitch_slider
    return p


def measure(mode, pitch=+1.0):
    os.environ["FACEVIEW_NOD_MODE"] = mode
    import importlib, faceview.vision.ict_face as ict
    importlib.reload(ict)
    calls = []
    orig = ict._apply_cervical_cascade

    def hook(verts, yaw, p_in, roll, chin_y, head_h, pivot_z=0.0):
        before = verts.copy()
        out = orig(verts, yaw, p_in, roll, chin_y, head_h, pivot_z=pivot_z)
        calls.append((before, out, chin_y, head_h))
        return out

    ict._apply_cervical_cascade = hook
    try:
        ict.render_face_ict(_params(pitch, "male"), size=(360, 640))
    finally:
        ict._apply_cervical_cascade = orig

    # Body mesh = call[1]
    v0, v1, chin_y, head_h = calls[1]
    disp = np.linalg.norm(v1 - v0, axis=1)
    y_norm = (v0[:, 1] - chin_y) / head_h
    bands = {
        "upper-neck (C1-C3)":   ((-0.20, +0.00), None),
        "mid-neck (C4-C6)":     ((-0.40, -0.20), None),
        "neck-base (C7-T1)":    ((-0.50, -0.40), None),
        "upper-torso/clav.":    ((-1.00, -0.50), None),
        "mid-torso":            ((-2.00, -1.00), None),
    }
    out = {}
    for name, ((lo, hi), _) in bands.items():
        m = (y_norm >= lo) & (y_norm < hi)
        out[name] = float(disp[m].mean()) if m.any() else 0.0
    return out


def main():
    rows = {m: measure(m, +1.0) for m in MODES}
    bands = list(next(iter(rows.values())).keys())

    # Render table
    cell_w = 220
    cell_h = 44
    h = cell_h * (len(bands) + 2)
    w = cell_w * (len(MODES) + 1)
    img = Image.new("RGB", (w, h), (16, 18, 22))
    drw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 18)
        font_t = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except Exception:
        font = font_t = ImageFont.load_default()

    drw.text((10, 6),
        "Body-mesh mean displacement (ICT units) at pitch=+1.0 (+22.9°)",
        fill=(255, 240, 180), font=font_t)

    # Header row
    y = cell_h
    drw.rectangle((0, y, w, y + cell_h), fill=(40, 44, 52))
    drw.text((10, y + 12), "Y band", fill=(220, 220, 220), font=font)
    for i, m in enumerate(MODES):
        x = (i + 1) * cell_w
        lab = m
        if m == "spine_ripple":
            lab = m + " ★"
        drw.text((x + 6, y + 12), lab, fill=(200, 235, 255), font=font)

    # Data rows
    for r, band in enumerate(bands):
        ry = (r + 2) * cell_h
        bg = (24, 26, 30) if r % 2 == 0 else (32, 34, 38)
        drw.rectangle((0, ry, w, ry + cell_h), fill=bg)
        drw.text((10, ry + 12), band, fill=(220, 220, 220), font=font)
        cur = rows["current"][band]
        for i, m in enumerate(MODES):
            v = rows[m][band]
            x = (i + 1) * cell_w
            if cur > 1e-5 and m != "current":
                pct = 100.0 * (1 - v / cur)
                txt = f"{v:.4f}  ({pct:+.0f}%)"
            else:
                txt = f"{v:.4f}"
            color = (200, 200, 200)
            if cur > 1e-5 and m != "current":
                if v < cur * 0.1:
                    color = (130, 240, 130)
                elif v < cur * 0.5:
                    color = (240, 240, 130)
            drw.text((x + 6, ry + 12), txt, fill=color, font=font)

    img.save("/tmp/nod_table.png")
    print("wrote /tmp/nod_table.png")


if __name__ == "__main__":
    main()
