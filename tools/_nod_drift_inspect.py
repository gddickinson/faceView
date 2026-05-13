"""Diagnostic: print the cervical cascade's actual parameters
(chin_y, head_h, body vert Y range) to understand why the body is
moving so much."""
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


def main():
    import faceview.vision.ict_face as ict_face
    orig = ict_face._apply_cervical_cascade
    call_count = [0]

    def hook(verts, yaw, pitch, roll, chin_y, head_h, pivot_z=0.0):
        call_count[0] += 1
        n = call_count[0]
        print(f"\n--- cascade call #{n} ---")
        print(f"  verts: {len(verts)}  yaw={yaw:.4f}  pitch={pitch:.4f}  roll={roll:.4f}")
        print(f"  chin_y={chin_y:.4f}  head_h={head_h:.4f}  pivot_z={pivot_z:.4f}")
        print(f"  vert Y range: [{verts[:,1].min():.3f} .. {verts[:,1].max():.3f}]")
        print(f"  vert Y - chin_y range: [{(verts[:,1].min() - chin_y):.3f} .. {(verts[:,1].max() - chin_y):.3f}]")
        # Show what fraction of head_h each Y band is at
        print(f"  body Y range in head_h units: [{(verts[:,1].min() - chin_y)/head_h:.3f} .. {(verts[:,1].max() - chin_y)/head_h:.3f}]")
        # Vertebra Y positions
        from faceview.vision.ict_face import (VERTEBRA_Y_FRACS,
            VERTEBRA_FRACTIONS_PITCH)
        print("  vertebra absolute Ys + cumul pitch:")
        for fy, fp in zip(VERTEBRA_Y_FRACS, VERTEBRA_FRACTIONS_PITCH):
            print(f"    y={chin_y + fy*head_h:.3f}  frac={fy:+.3f}  cumul_pitch={fp:.4f}")
        result = orig(verts, yaw, pitch, roll, chin_y, head_h, pivot_z=pivot_z)
        disp = np.linalg.norm(result - verts, axis=1)
        print(f"  result disp: mean={disp.mean():.4f}  max={disp.max():.4f}  p95={np.percentile(disp,95):.4f}")
        return result

    ict_face._apply_cervical_cascade = hook
    try:
        ict_face.render_face_ict(_params(+1.0, "male"), size=(360, 640))
    finally:
        ict_face._apply_cervical_cascade = orig


if __name__ == "__main__":
    main()
