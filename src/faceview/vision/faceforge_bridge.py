"""Bridge to faceforge-style 3D rendering.

faceforge (the sibling anatomy app) uses BodyParts3D STL meshes for a
true medical-quality head/body. faceView doesn't depend on faceforge at
runtime, but if its **head + neck STL subset** has been copied into
``assets/anatomy_meshes/`` (via ``tools.copy_anatomy_meshes``), this
bridge renders them through faceView's lightweight CPU rasteriser in
:mod:`faceview.vision.anatomy_meshes`.

This is the **photo-anatomical** end of the rendering spectrum. Like
the layered illustration mode, it composes selectable layers — but
each layer is a real STL mesh, not a stylised QPainterPath. It will
not look like a textbook diagram; it will look like a CT atlas slice.

Falls back gracefully when meshes aren't present:
:class:`MissingDependency` with the path to populate.
"""

from __future__ import annotations

import math
import numpy as np

from faceview.core.errors import MissingDependency
from faceview.vision.anatomy_catalog import (
    head_neck_fmas,
    specs_for_layer_set,
)
from faceview.vision.anatomy_meshes import (
    list_available_meshes,
    load_mesh,
    mesh_dir,
    meshes_available,
    render_meshes,
)


# Cache the FMA list (read-only after first call).
HEAD_NECK_FMAS = head_neck_fmas()


def _resolve_present(names: list[str]) -> list[str]:
    """Filter ``names`` to those whose .stl files are actually on disk."""
    avail = set(list_available_meshes())
    return [n for n in names if n in avail]


def _params_to_pose(params) -> tuple[float, float]:
    """Convert FaceParams.yaw/pitch in [-1, 1] to radians (±0.6 rad)."""
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.6
    return yaw, pitch


# Mapping from human-friendly layer set names to catalog presets.
_LAYER_ALIASES = {
    "skull_bones": "skull_only",
    "skull_only": "skull_only",
    "vertebrae": "vertebrae",
    "muscles": "muscles",
    "features": "features",
    "lifelike": "lifelike",
    "xray": "xray",
    "all_head_neck": "all_head_neck",
}


def render_face_faceforge(
    params,
    size: tuple[int, int] = (640, 480),
    *,
    layer_set: str = "lifelike",
) -> np.ndarray:
    """Render the BodyParts3D head + neck meshes at the given pose.

    ``layer_set`` selects which structures and which materials to use:

    - ``lifelike`` — full opaque skin over muscles + bones (default).
      Looks like a 3D portrait when meshes are present.
    - ``xray`` — same content, skin translucent (~0.35 opacity).
    - ``muscles`` — bones + muscles, no skin.
    - ``skull_only`` — bones + vertebrae only.
    - ``features`` — bones + muscles + eyes + ears + nose cartilages.
    - ``vertebrae`` — cervical spine only.
    - ``all_head_neck`` — every catalog mesh (synonym for ``xray``).

    Raises :class:`MissingDependency` when the mesh directory is empty.
    """
    if not meshes_available():
        raise MissingDependency(
            "BodyParts3D STL meshes",
            install_hint=(
                "Copy the head + neck STL subset into "
                f"{mesh_dir()} via:\n"
                "  python -m tools.copy_anatomy_meshes "
                "/path/to/bodyparts3D/stl"
            ),
        )

    catalog_name = _LAYER_ALIASES.get(layer_set, layer_set)
    specs = specs_for_layer_set(catalog_name)
    avail = set(list_available_meshes())
    specs = [s for s in specs if s.fma in avail]

    if not specs:
        raise MissingDependency(
            "BodyParts3D STL meshes",
            install_hint=(
                f"No STLs matched layer set '{layer_set}' in {mesh_dir()}. "
                "Run `python -m tools.copy_anatomy_meshes "
                "/path/to/bodyparts3D/stl` to populate."
            ),
        )

    meshes = [load_mesh(s.fma) for s in specs]
    yaw, pitch = _params_to_pose(params)
    bg = getattr(params, "background", "#0a0d12")
    bg_rgb = _hex_to_rgb(bg)
    return render_meshes(
        meshes, size,
        yaw=yaw, pitch=pitch, bg_color=bg_rgb,
        materials=specs,
    )


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16


# Convenience helper for tools / tests.
def faceforge_status() -> dict:
    """Report what's available — used by `/state` and CLI diagnostics."""
    avail = list_available_meshes()
    return {
        "meshes_available": bool(avail),
        "mesh_count": len(avail),
        "mesh_dir": str(mesh_dir()),
        "expected_head_neck": len(HEAD_NECK_FMAS),
    }
