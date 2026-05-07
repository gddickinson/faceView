"""Ready Player Me GLB avatar loader.

[Ready Player Me](https://readyplayer.me/) provides cross-platform
3D avatars with ARKit-aligned blendshapes, distributed as glTF/GLB
binary files at ``https://models.readyplayer.me/<id>.glb``. This
module fetches/loads the binary, extracts the head mesh + 52
ARKit-named morph targets, and renders through our existing
moderngl Phong / SSS pipeline.

End-user customisation: a faceView user can plug in any RPM avatar
URL and our FACS pipeline drives it. The avatar comes with its
own skin texture + hair mesh, so it bypasses our procedural-hair
overlay.

Render mode: ``rpm_3d``.
Cache: avatars are kept in ``assets/data/rpm/<id>.glb``.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _rpm_dir() -> Path:
    p = assets_dir() / "data" / "rpm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _avatar_path(avatar_id: str) -> Path:
    return _rpm_dir() / f"{avatar_id}.glb"


def fetch_avatar(avatar_id: str) -> Path:
    """Download a Ready Player Me GLB if not already cached.

    ``avatar_id`` is the UUID-like string in the avatar URL.
    """
    target = _avatar_path(avatar_id)
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"https://models.readyplayer.me/{avatar_id}.glb"
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    return target


@dataclass
class RPMAvatar:
    vertices: np.ndarray      # (N, 3)
    triangles: np.ndarray     # (M, 3) uint32
    normals: np.ndarray       # (N, 3)
    morph_targets: dict[str, np.ndarray]   # name → (N, 3) deltas
    base_color: tuple[float, float, float]


@lru_cache(maxsize=4)
def load_rpm_avatar(avatar_id: str) -> RPMAvatar:
    """Parse a Ready Player Me GLB → RPMAvatar."""
    try:
        import pygltflib
    except ImportError as exc:
        raise MissingDependency(
            "pygltflib", "vision",
            hint="Install with `pip install pygltflib`.",
        ) from exc

    path = _avatar_path(avatar_id)
    if not path.exists():
        path = fetch_avatar(avatar_id)
    glb = pygltflib.GLTF2.load(str(path))

    # Find the head mesh: RPM ships separate meshes per body part;
    # we want "Wolf3D_Head" or any mesh whose name contains "Head".
    head_idx = None
    for i, mesh in enumerate(glb.meshes):
        if mesh.name and "head" in mesh.name.lower():
            head_idx = i
            break
    if head_idx is None and glb.meshes:
        head_idx = 0
    if head_idx is None:
        raise RuntimeError("no meshes in RPM avatar")

    mesh = glb.meshes[head_idx]
    primitive = mesh.primitives[0]

    # Extract vertex positions, normals, indices via the binary buffer.
    binary = glb.binary_blob()

    def _accessor(idx: int) -> np.ndarray:
        acc = glb.accessors[idx]
        bv = glb.bufferViews[acc.bufferView]
        offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
        count = acc.count
        # Component types: 5126=FLOAT, 5125=UINT32, 5123=UINT16
        ctype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc.componentType]
        per = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
        nbytes = count * per * np.dtype(ctype).itemsize
        arr = np.frombuffer(binary[offset:offset + nbytes], dtype=ctype)
        return arr.reshape(count, per) if per > 1 else arr

    verts = _accessor(primitive.attributes.POSITION).astype(np.float32)
    norms = (_accessor(primitive.attributes.NORMAL).astype(np.float32)
             if primitive.attributes.NORMAL is not None else None)
    tris = _accessor(primitive.indices).astype(np.uint32).reshape(-1, 3)

    # Morph targets — RPM names them after ARKit shapes.
    morphs: dict[str, np.ndarray] = {}
    target_names = (mesh.extras or {}).get("targetNames", []) if mesh.extras else []
    if not target_names and primitive.targets:
        target_names = [f"target_{i}" for i in range(len(primitive.targets))]
    for ti, target in enumerate(primitive.targets or []):
        if "POSITION" in target:
            delta = _accessor(target["POSITION"]).astype(np.float32)
            name = target_names[ti] if ti < len(target_names) else f"target_{ti}"
            morphs[name] = delta

    if norms is None:
        # Compute per-vertex normals from triangles.
        v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
        tn = np.cross(v1 - v0, v2 - v0)
        tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9)
        norms = np.zeros_like(verts)
        np.add.at(norms, tris[:, 0], tn)
        np.add.at(norms, tris[:, 1], tn)
        np.add.at(norms, tris[:, 2], tn)
        norms /= np.maximum(np.linalg.norm(norms, axis=1, keepdims=True), 1e-9)

    return RPMAvatar(
        vertices=verts, triangles=tris, normals=norms,
        morph_targets=morphs,
        base_color=(0.92, 0.78, 0.69),
    )


def render_face_rpm(
    params,
    size: tuple[int, int] = (480, 480),
    *,
    avatar_id: str = "default",
) -> np.ndarray:
    """Render a Ready Player Me head with FACS-driven blendshapes."""
    iw = getattr(params, "identity_weights", {}) or {}
    requested_id = iw.get("rpm_id", avatar_id)
    if not isinstance(requested_id, str):
        requested_id = avatar_id
    avatar = load_rpm_avatar(requested_id)

    from faceview.vision.anatomy import face_params_to_au_values
    from faceview.vision.arkit_blendshapes import au_to_arkit_values

    arkit_coefs = au_to_arkit_values(face_params_to_au_values(params))
    verts = avatar.vertices.copy()
    for name, value in arkit_coefs.items():
        if value == 0 or name not in avatar.morph_targets:
            continue
        verts += float(value) * avatar.morph_targets[name]

    # Reuse the ICT moderngl renderer infrastructure.
    from faceview.vision.ict_face import _ensure_renderer
    rend = _ensure_renderer()
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    scale = 1.6 / max(span, 1e-6)

    skin_color = np.tile(np.array(avatar.base_color, dtype=np.float32),
                          (len(verts), 1))
    skin_spec = np.full(len(verts), 0.35, dtype=np.float32)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    bg = _hex_to_rgb(getattr(params, "background", "#0a0d12"))
    return rend.render(
        verts=verts.astype(np.float32),
        normals=avatar.normals.astype(np.float32),
        triangles=avatar.triangles.astype(np.uint32),
        vert_colors=skin_color,
        vert_spec=skin_spec,
        centre=centre.astype(np.float32),
        scale=float(scale),
        yaw=yaw, pitch=pitch,
        size=size, bg=bg,
    )


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
