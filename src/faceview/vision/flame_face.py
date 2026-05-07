"""FLAME 3D head model bridge (PyTorch).

[FLAME](https://flame.is.tue.mpg.de/) (Faces Learned with an
Articulated Model and Expressions) is a statistical 3D head model
from MPI-IS, learned from 33 000+ aligned scans. It has:

- 100 PCA shape modes (identity)
- 100 PCA expression modes
- Articulated jaw + neck + eyeballs as separate joints
- Pose-dependent corrective blendshapes

Differentiable PyTorch implementation at
[soubhiksanyal/FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
(also on PyPI as ``FLAME-PyTorch``).

DEPS + DATA
-----------
Heavy: requires ``torch`` (~2 GB), ``FLAME-PyTorch`` PyPI package,
plus the FLAME model file (~100 MB) downloaded from
https://flame.is.tue.mpg.de/download.php (CC-BY academic licence —
sign up first, agree to non-commercial terms).

Place the model at ``assets/data/flame/generic_model.pkl`` after
download.

LICENCE NOTE
------------
FLAME is released under CC-BY (academic). Commercial use needs a
separate licence from MPI-IS. We bundle this bridge module but not
the model file; users opt in deliberately by installing the deps
and downloading the model.

Render mode: ``flame_3d``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _flame_model_path() -> Path:
    return assets_dir() / "data" / "flame" / "generic_model.pkl"


def _ensure_flame():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise MissingDependency(
            "torch", "vision",
            hint="Install with `pip install torch` (~2 GB).",
        ) from exc
    try:
        from FLAME_PyTorch.flame_model import FLAME  # noqa: F401
        return FLAME
    except ImportError:
        try:
            from flame_pytorch.flame import FLAME  # noqa: F401
            return FLAME
        except ImportError as exc:
            raise MissingDependency(
                "FLAME-PyTorch", "vision",
                hint=(
                    "Install with `pip install FLAME-PyTorch` or "
                    "`git clone https://github.com/soubhiksanyal/FLAME_PyTorch`. "
                    "Then download the model from "
                    "https://flame.is.tue.mpg.de/download.php (CC-BY) "
                    "and place at "
                    f"{_flame_model_path()}."
                ),
            ) from exc


@lru_cache(maxsize=1)
def load_flame_model():
    FLAME = _ensure_flame()
    path = _flame_model_path()
    if not path.exists():
        raise MissingDependency(
            "FLAME model file", "vision",
            hint=(
                f"Download generic_model.pkl from "
                "https://flame.is.tue.mpg.de/download.php and "
                f"place at {path}."
            ),
        )
    # FLAME's constructor signature varies by upstream version; we
    # use the most common form.
    config = type("FLAMEConfig", (), {
        "flame_model_path": str(path),
        "static_landmark_embedding_path": "",
        "dynamic_landmark_embedding_path": "",
        "shape_params": 100,
        "expression_params": 100,
        "pose_params": 6,
        "use_face_contour": False,
        "use_3D_translation": False,
        "optimize_eyeballpose": False,
        "optimize_neckpose": False,
        "num_worker": 0,
        "batch_size": 1,
    })()
    import torch
    return FLAME(config).cpu().eval(), torch


def render_face_flame(
    params,
    size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    """Render a FLAME face. Lazy-imports torch + FLAME-PyTorch."""
    flame, torch = load_flame_model()

    iw = getattr(params, "identity_weights", {}) or {}
    shape = torch.zeros(1, 100)
    expr = torch.zeros(1, 100)
    pose = torch.zeros(1, 6)
    for k, v in iw.items():
        if not isinstance(v, (int, float)) or not isinstance(k, str):
            continue
        if k.startswith("flame_shape_"):
            try:
                idx = int(k.split("_", 2)[2])
                if 0 <= idx < 100:
                    shape[0, idx] = float(v)
            except ValueError:
                continue
        elif k.startswith("flame_expr_"):
            try:
                idx = int(k.split("_", 2)[2])
                if 0 <= idx < 100:
                    expr[0, idx] = float(v)
            except ValueError:
                continue

    # FLAME forward pass — exact API varies; this is the common form.
    with torch.no_grad():
        verts, _ = flame(shape, expr, pose)
    verts_np = verts[0].cpu().numpy().astype(np.float32)

    # Render through ICT's moderngl pipeline.
    faces = flame.faces.astype(np.uint32) if hasattr(flame, "faces") else None
    if faces is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    from faceview.vision.ict_face import _ensure_renderer
    rend = _ensure_renderer()
    centre = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2
    span = float(np.linalg.norm(verts_np.max(axis=0) - verts_np.min(axis=0)))
    scale = 1.6 / max(span, 1e-6)

    v0 = verts_np[faces[:, 0]]; v1 = verts_np[faces[:, 1]]
    v2 = verts_np[faces[:, 2]]
    tn = np.cross(v1 - v0, v2 - v0)
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-9)
    vn = np.zeros_like(verts_np)
    np.add.at(vn, faces[:, 0], tn)
    np.add.at(vn, faces[:, 1], tn)
    np.add.at(vn, faces[:, 2], tn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)

    skin = np.tile(np.array([0.92, 0.78, 0.69], dtype=np.float32),
                    (len(verts_np), 1))
    skin_spec = np.full(len(verts_np), 0.30, dtype=np.float32)
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    return rend.render(
        verts=verts_np, normals=vn.astype(np.float32),
        triangles=faces, vert_colors=skin, vert_spec=skin_spec,
        centre=centre.astype(np.float32), scale=float(scale),
        yaw=yaw, pitch=pitch, size=size, bg=(10, 13, 18),
    )
