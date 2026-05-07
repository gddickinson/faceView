"""MetaHuman head FBX bridge (Gumroad CC distribution).

[Dragonboots' MetaHuman head FBX](https://dragonboots.gumroad.com/l/metahumanhead)
is a free Gumroad distribution that extracts the head + teeth + eye
models with all 52 ARKit-aligned blendshapes from Epic Games'
MetaHuman library. It's the highest-fidelity per-vertex animatable
head we know about; the catch is the FBX format.

DEPS
----
FBX is a closed Autodesk format. Open-source readers exist:

- ``aspose-3d`` (commercial Aspose, free trial)
- ``PyAssimp`` (the assimp library binding)
- Blender's Python API (``bpy``) inside Blender's Python

Each is heavy (10s of MB) and fails silently in obscure ways. We
lazy-import any of them; users opt in by installing one.

DATA
----
Download ``MetaHuman_52_blendshapes.fbx`` from Gumroad and place at
``assets/data/metahuman/head.fbx``.

Render mode: ``metahuman_3d``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _fbx_path() -> Path:
    return assets_dir() / "data" / "metahuman" / "head.fbx"


@lru_cache(maxsize=1)
def _load_fbx_with_assimp(path: Path):
    try:
        import pyassimp
    except ImportError as exc:
        raise MissingDependency(
            "pyassimp", "vision",
            hint=(
                "Install with `pip install pyassimp` "
                "(needs the assimp shared library — "
                "`brew install assimp` on macOS)."
            ),
        ) from exc
    return pyassimp.load(str(path))


def render_face_metahuman(
    params,
    size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    """Render a MetaHuman head — requires FBX + reader installed."""
    path = _fbx_path()
    if not path.exists():
        raise MissingDependency(
            "MetaHuman head FBX", "vision",
            hint=(
                "Download from https://dragonboots.gumroad.com/l/metahumanhead "
                f"and place at {path}."
            ),
        )

    scene = _load_fbx_with_assimp(path)
    if not scene.meshes:
        raise RuntimeError("FBX contained no meshes")

    # First mesh is typically the head. MetaHuman head FBX ships
    # several meshes (head / teeth / eyes); we pick the largest.
    largest = max(scene.meshes, key=lambda m: len(m.vertices))
    verts = np.asarray(largest.vertices, dtype=np.float32)
    tris = np.asarray(largest.faces, dtype=np.uint32)

    # Blendshape application: assimp exposes `mAnimMeshes` per mesh
    # with morph targets named after the corresponding shape. We
    # accumulate ARKit-driven deltas the same way as ICT.
    from faceview.vision.anatomy import face_params_to_au_values
    from faceview.vision.arkit_blendshapes import au_to_arkit_values
    arkit = au_to_arkit_values(face_params_to_au_values(params))
    for anim_mesh in getattr(largest, "anim_meshes", []) or []:
        name = getattr(anim_mesh, "mName", "")
        weight = float(arkit.get(name, 0.0))
        if weight == 0:
            continue
        delta = np.asarray(anim_mesh.vertices, dtype=np.float32) - verts
        verts = verts + weight * delta

    # Normals.
    if largest.normals is not None and len(largest.normals) == len(verts):
        normals = np.asarray(largest.normals, dtype=np.float32)
    else:
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
    skin_spec = np.full(len(verts), 0.40, dtype=np.float32)
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    return rend.render(
        verts=verts, normals=normals, triangles=tris,
        vert_colors=skin, vert_spec=skin_spec,
        centre=centre.astype(np.float32), scale=float(scale),
        yaw=yaw, pitch=pitch, size=size, bg=(10, 13, 18),
    )
