"""Pre-compute compact ICT-FaceKit blendshape deltas.

ICT-FaceKit ships a 386 MB tree of OBJ files (one per blendshape).
This tool reads the neutral mesh + every blendshape OBJ from a local
ICT-FaceKit clone, computes per-vertex deltas (blendshape_verts -
neutral_verts), and writes a single compressed ``.npz`` we can ship.

Usage::

    git clone https://github.com/USC-ICT/ICT-FaceKit /tmp/ICT-FaceKit
    python -m tools.build_ict_blendshapes /tmp/ICT-FaceKit

Writes to ``src/faceview/assets/data/ict/face_kit.npz``:
- ``vertices``     (N, 3) float32 neutral positions
- ``triangles``    (M, 3) int32  index buffer
- ``names``        list[str]      blendshape names (ARKit-aligned)
- ``deltas``       (B, N, 3) float32   per-blendshape vertex deltas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _parse_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with path.open() as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                p = line.split()[1:]
                idxs = [int(t.split("/")[0]) - 1 for t in p]
                for i in range(1, len(idxs) - 1):
                    tris.append((idxs[0], idxs[i], idxs[i + 1]))
    return (np.asarray(verts, dtype=np.float32),
            np.asarray(tris, dtype=np.int32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ict_dir", type=Path,
                         help="Local clone of USC-ICT/ICT-FaceKit")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent.parent / "src" / "faceview"
                / "assets" / "data" / "ict" / "face_kit.npz",
    )
    args = parser.parse_args()

    fxm = args.ict_dir / "FaceXModel"
    if not fxm.is_dir():
        print(f"ERROR: {fxm} not found", file=sys.stderr)
        return 2

    neutral_path = fxm / "generic_neutral_mesh.obj"
    if not neutral_path.exists():
        print(f"ERROR: {neutral_path} missing", file=sys.stderr)
        return 2

    print(f"reading neutral from {neutral_path}")
    base_verts, tris = _parse_obj(neutral_path)
    print(f"  {len(base_verts)} verts, {len(tris)} tris")

    deltas: list[np.ndarray] = []
    names: list[str] = []
    for obj in sorted(fxm.glob("*.obj")):
        if obj.name == "generic_neutral_mesh.obj":
            continue
        bs_verts, _ = _parse_obj(obj)
        if bs_verts.shape != base_verts.shape:
            print(f"  SKIP {obj.name} — shape mismatch", file=sys.stderr)
            continue
        delta = bs_verts - base_verts
        deltas.append(delta.astype(np.float32))
        names.append(obj.stem)

    delta_arr = np.stack(deltas, axis=0)
    print(f"computed {len(deltas)} blendshapes; deltas shape: {delta_arr.shape}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        vertices=base_verts,
        triangles=tris,
        deltas=delta_arr,
        names=np.array(names),
    )
    print(f"saved {args.out}  (size: {args.out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
