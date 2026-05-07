"""FaceScape / FaceVerse non-commercial scan loader.

[FaceScape](https://nju-3dv.github.io/projects/FaceScape/) provides
18 760 textured 3D faces (938 subjects × 20 expressions) at pore-
level detail. [FaceVerse](https://github.com/LizhenWangT/FaceVerse-Dataset)
adds 2 688 high-quality scans from a DLSR rig.

LICENCE
-------
Both are **non-commercial research only**. Sign up at the project
sites, agree to the licence, then download the OBJ + texture
subsets. Do NOT bundle the data here — users opt in deliberately.

DATA LAYOUT
-----------
After download, place an extracted scan at::

    assets/data/facescape/<id>/<expression>.obj
    assets/data/facescape/<id>/<expression>.jpg

Or symlink from your local FaceScape clone.

Render modes: ``facescape_3d`` / ``faceverse_3d``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _scan_dir() -> Path:
    return assets_dir() / "data" / "facescape"


def _list_scan_dirs() -> list[Path]:
    d = _scan_dir()
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_dir()]


def _parse_obj_with_uvs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ parser — verts + tris only (UVs ignored)."""
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
            np.asarray(tris, dtype=np.uint32))


@lru_cache(maxsize=8)
def load_facescape_scan(subject_id: str, expression: str = "1_neutral"):
    obj_path = _scan_dir() / subject_id / f"{expression}.obj"
    if not obj_path.exists():
        raise MissingDependency(
            "FaceScape scan", "vision",
            hint=(
                f"Download FaceScape from "
                "https://nju-3dv.github.io/projects/FaceScape/ "
                f"(non-commercial research licence) and extract a "
                f"scan to {obj_path}."
            ),
        )
    return _parse_obj_with_uvs(obj_path)


def render_face_facescape(
    params,
    size: tuple[int, int] = (480, 480),
    *,
    subject_id: str = "1",
    expression: str = "1_neutral",
) -> np.ndarray:
    iw = getattr(params, "identity_weights", {}) or {}
    if "facescape_subject" in iw and isinstance(iw["facescape_subject"], str):
        subject_id = iw["facescape_subject"]
    if "facescape_expression" in iw and isinstance(iw["facescape_expression"], str):
        expression = iw["facescape_expression"]

    verts, tris = load_facescape_scan(subject_id, expression)

    v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
    tn = np.cross(v1 - v0, v2 - v0)
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9)
    normals = np.zeros_like(verts)
    np.add.at(normals, tris[:, 0], tn)
    np.add.at(normals, tris[:, 1], tn)
    np.add.at(normals, tris[:, 2], tn)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)

    from faceview.vision.ict_face import _ensure_renderer
    rend = _ensure_renderer()
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    scale = 1.6 / max(span, 1e-6)
    skin = np.tile(np.array([0.92, 0.78, 0.69], dtype=np.float32),
                    (len(verts), 1))
    skin_spec = np.full(len(verts), 0.30, dtype=np.float32)
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    return rend.render(
        verts=verts.astype(np.float32),
        normals=normals.astype(np.float32),
        triangles=tris.astype(np.uint32),
        vert_colors=skin, vert_spec=skin_spec,
        centre=centre.astype(np.float32), scale=float(scale),
        yaw=yaw, pitch=pitch, size=size, bg=(10, 13, 18),
    )
