"""Inspect what's actually flying away during a body effect.
Reports the top N verts by displacement from rest, with their labels
and rest positions, so we can target them directly."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from tools.extreme_pose_relabel import _make_neutral_params, _capture_rig_io
from tools.isolated_voxel_relabel import BPF_NAMES

def main():
    gender = sys.argv[1] if len(sys.argv) > 1 else "male"
    effect = sys.argv[2] if len(sys.argv) > 2 else "arms_up"
    topn = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    from faceview.vision.ict_face import render_face_ict
    from faceview.vision.effects_pre import HANDLERS

    p = _make_neutral_params(gender)
    handler = HANDLERS.get(f"pre_{effect}") or HANDLERS.get(effect)
    if handler:
        handler(p, 0.5, 1.0)
    cap = _capture_rig_io(lambda: render_face_ict(p, size=(360, 640)))
    rest = cap["rest"]; posed = cap["posed"]
    fine = cap["fine_eff"]
    disp = np.linalg.norm(posed - rest, axis=1)
    med_per_label = {}
    for lid in range(16):
        m = fine == lid
        if m.sum() == 0:
            continue
        med_per_label[lid] = float(np.median(disp[m]))

    # Find verts that moved 3x more than the median for their label
    is_outlier = np.zeros(len(rest), bool)
    for lid in range(16):
        m = fine == lid
        if m.sum() < 5:
            continue
        med = med_per_label[lid]
        # Flag verts >> their label's median
        is_outlier |= m & (disp > max(med * 3 + 5, 10.0))

    out_idx = np.argsort(-disp)[:topn]
    print(f"Effect: {effect}  gender: {gender}")
    print(f"Median disp per label: " +
            ", ".join(f"{BPF_NAMES[k]}={v:.1f}" for k, v in
                      sorted(med_per_label.items())))
    print(f"\nTop {topn} fliers (disp, vi, label, rest_xyz, posed_xyz):")
    print(f"  outliers flagged: {int(is_outlier.sum())}")
    for vi in out_idx:
        x, y, z = rest[vi]
        px, py, pz = posed[vi]
        flag = "*" if is_outlier[vi] else " "
        print(f" {flag} disp={disp[vi]:7.2f}  vi={vi:5d}  "
                f"lab={BPF_NAMES[int(fine[vi])]:9s}  "
                f"rest=({x:+6.1f},{y:+6.1f},{z:+6.1f})  "
                f"posed=({px:+6.1f},{py:+6.1f},{pz:+6.1f})")

if __name__ == "__main__":
    main()
