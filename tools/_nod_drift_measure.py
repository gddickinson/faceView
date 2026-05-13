"""Measure per-Y-band displacement for BOTH the ICT head mesh and
the body mesh when a head pitch is applied. The user's "base of neck
moves" complaint shows up in both meshes — the ICT mesh's collar
region and the body mesh's upper neck."""
import os
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _params(pitch_slider: float, gender: str):
    from faceview.vision.sim_face import FaceParams
    p = FaceParams.neutral()
    p.identity_weights = {"genderHead": 1.0 if gender == "male" else -1.0}
    p._show_body = True
    p._body_morph = 1.0 if gender == "male" else -1.0
    p._camera_zoom = 0.55
    p.pitch = pitch_slider
    return p


def measure(pitch=+1.0, gender="male"):
    import faceview.vision.ict_face as ict_face
    orig = ict_face._apply_cervical_cascade
    calls = []

    def hook(verts, yaw, pitch_in, roll, chin_y, head_h, pivot_z=0.0):
        before = verts.copy()
        out = orig(verts, yaw, pitch_in, roll, chin_y, head_h, pivot_z=pivot_z)
        calls.append({"before": before, "after": out, "chin_y": chin_y,
                      "head_h": head_h})
        return out

    ict_face._apply_cervical_cascade = hook
    try:
        ict_face.render_face_ict(_params(pitch, gender), size=(360, 640))
    finally:
        ict_face._apply_cervical_cascade = orig

    # call[0] = ICT head mesh; call[1] = body mesh
    for i, name in enumerate(("ICT_HEAD_MESH", "BODY_MESH")):
        if i >= len(calls):
            continue
        c = calls[i]
        v0, v1 = c["before"], c["after"]
        chin_y, head_h = c["chin_y"], c["head_h"]
        disp = np.linalg.norm(v1 - v0, axis=1)
        y_norm = (v0[:, 1] - chin_y) / head_h
        print(f"\n=== {name} ({len(v0)} verts) ===")
        bands = [
            (+0.7, +1.5, "crown / forehead"),
            (+0.3, +0.7, "skull / face"),
            (+0.0, +0.3, "above C1 / chin-region"),
            (-0.2, +0.0, "upper neck (C1-C3)"),
            (-0.4, -0.2, "mid neck (C4-C6)"),
            (-0.5, -0.4, "neck-base (C7-T1)"),
            (-1.0, -0.5, "upper torso / clavicle"),
            (-2.0, -1.0, "mid torso"),
            (-7.0, -2.0, "lower body"),
        ]
        print(f"{'band':<28} {'count':>6} {'mean':>8} {'max':>8} {'p95':>8}")
        for lo, hi, lab in bands:
            m = (y_norm >= lo) & (y_norm < hi)
            if not m.any():
                continue
            d = disp[m]
            print(f"{lab:<28} {m.sum():>6} {d.mean():>8.4f} {d.max():>8.4f}"
                  f" {np.percentile(d,95):>8.4f}")


def main():
    print("\n*** pitch=+1.0 (head up +22.9°) ***")
    measure(+1.0, "male")
    print("\n\n*** pitch=-1.0 (head down -22.9°) ***")
    measure(-1.0, "male")


if __name__ == "__main__":
    main()
