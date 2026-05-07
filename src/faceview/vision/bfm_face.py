"""Basel Face Model 2017 bridge via ``eos-py``.

[The Basel Face Model 2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html)
is a classic statistical 3D morphable face model. The
[eos](https://github.com/patrikhuber/eos) Python bindings (``pip
install eos-py``) provide a lightweight loader that takes shape +
expression coefficients and produces an OBJ-ready mesh.

This module wraps that bridge for our pipeline. It lazy-imports
``eos`` and a downloaded BFM 2017 H5 file, then renders the
resulting mesh through our existing GPU path.

KNOWN LIMITATION
----------------
The PyPI ``eos-py`` wheels at the time of writing are **x86_64 only**.
On Apple Silicon (M1/M2/M3) the import fails with an architecture
mismatch. Workarounds:

- Run faceview under Rosetta 2 (``arch -x86_64 python ...``)
- Build eos from source for arm64
  (https://github.com/patrikhuber/eos)
- Wait for upstream arm64 wheels

For now the module raises :class:`MissingDependency` on Apple
Silicon, so the rest of the project keeps working.

ALSO REQUIRED — the BFM 2017 H5 file. Download from
https://faces.dmi.unibas.ch/bfm/bfm2017.html and place at
``assets/data/bfm/model2017-1_bfm_nomouth.h5``.

Render mode: ``bfm_3d``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _bfm_path() -> Path:
    return assets_dir() / "data" / "bfm" / "model2017-1_bfm_nomouth.h5"


def _ensure_eos():
    try:
        import eos  # noqa: F401
        return eos
    except ImportError as exc:
        raise MissingDependency(
            "eos-py", "vision",
            hint=(
                "Install with `pip install eos-py`. On Apple Silicon "
                "the wheel is x86_64 only — run under Rosetta 2 "
                "(`arch -x86_64 python -m faceview`) or build eos "
                "from source for arm64."
            ),
        ) from exc


@lru_cache(maxsize=1)
def load_bfm():
    """Load BFM 2017 model + 100 random shape coefficients."""
    eos = _ensure_eos()
    path = _bfm_path()
    if not path.exists():
        raise MissingDependency(
            "BFM 2017 model file", "vision",
            hint=(
                f"Download model2017-1_bfm_nomouth.h5 from "
                "https://faces.dmi.unibas.ch/bfm/bfm2017.html and "
                f"place at {path}."
            ),
        )
    morphable = eos.morphablemodel.load_model(str(path))
    return morphable


def render_face_bfm(
    params,
    size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    """Render a BFM face. Raises MissingDependency on Apple Silicon."""
    eos = _ensure_eos()
    morphable = load_bfm()

    # Sample identity from persona.identity_weights with bfm_* keys.
    iw = getattr(params, "identity_weights", {}) or {}
    n_shape = morphable.get_shape_model().get_num_principal_components()
    shape_coeffs = [0.0] * n_shape
    for k, v in iw.items():
        if not isinstance(k, str) or not k.startswith("bfm_"):
            continue
        try:
            idx = int(k.split("_", 1)[1])
            if 0 <= idx < n_shape:
                shape_coeffs[idx] = float(v)
        except ValueError:
            continue

    sample = morphable.draw_sample(shape_coeffs, [], [])
    verts = np.asarray(sample.vertices, dtype=np.float32)
    tris = np.asarray(sample.tvi, dtype=np.uint32)

    # Reuse the ICT moderngl renderer infrastructure for skin shading.
    from faceview.vision.ict_face import _ensure_renderer
    rend = _ensure_renderer()
    centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
    span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    scale = 1.6 / max(span, 1e-6)

    # Use vertex normals computed from triangles.
    v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
    tri_n = np.cross(v1 - v0, v2 - v0)
    tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-9)
    vn = np.zeros_like(verts)
    np.add.at(vn, tris[:, 0], tri_n)
    np.add.at(vn, tris[:, 1], tri_n)
    np.add.at(vn, tris[:, 2], tri_n)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)

    skin_color = np.tile(np.array([0.92, 0.78, 0.69], dtype=np.float32),
                          (len(verts), 1))
    skin_spec = np.full(len(verts), 0.30, dtype=np.float32)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    return rend.render(
        verts=verts.astype(np.float32),
        normals=vn.astype(np.float32),
        triangles=tris,
        vert_colors=skin_color,
        vert_spec=skin_spec,
        centre=centre.astype(np.float32),
        scale=float(scale),
        yaw=yaw, pitch=pitch,
        size=size,
        bg=(10, 13, 18),
    )
