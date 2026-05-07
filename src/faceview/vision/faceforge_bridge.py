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
from faceview.vision.anatomy_meshes import (
    list_available_meshes,
    load_mesh,
    mesh_dir,
    meshes_available,
    render_meshes,
    HEAD_NECK_FMAS,
)


# ── default layer presets ───────────────────────────────────────────


_PRESET_LAYERS: dict[str, list[str]] = {
    # FMA codes; loaded from disk if present.
    "skull_bones": [
        "FMA46565",  # skull
        "FMA52748",  # mandible
        "FMA52747",  # zygomatic bone
    ],
    "vertebrae": [
        "FMA12519", "FMA12520", "FMA12521", "FMA12522",
        "FMA12523", "FMA12524", "FMA12525",
    ],
    "expression_muscles": [],   # filled at load time from the catalogue
    "all_head_neck": list(HEAD_NECK_FMAS.keys()),
}


def _resolve_present(names: list[str]) -> list[str]:
    """Filter ``names`` to those whose .stl files are actually on disk."""
    avail = set(list_available_meshes())
    return [n for n in names if n in avail]


def _params_to_pose(params) -> tuple[float, float]:
    """Convert FaceParams.yaw/pitch in [-1, 1] to radians (±0.6 rad)."""
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.6
    return yaw, pitch


def render_face_faceforge(
    params,
    size: tuple[int, int] = (640, 480),
    *,
    layer_set: str = "skull_bones",
) -> np.ndarray:
    """Render the BP3D head + neck subset at the given pose.

    ``layer_set`` selects which structures to include
    (``skull_bones`` | ``vertebrae`` | ``expression_muscles`` |
    ``all_head_neck``). When the mesh dir is empty this raises
    :class:`MissingDependency` with the populate hint.
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

    names = _PRESET_LAYERS.get(layer_set, ["skull_bones"])
    if layer_set == "all_head_neck":
        names = _resolve_present(list(HEAD_NECK_FMAS.keys()))
    elif not names:  # e.g. expression_muscles — derive from catalogue
        names = _resolve_present(list(HEAD_NECK_FMAS.keys()))
    else:
        names = _resolve_present(names)

    if not names:
        raise MissingDependency(
            "BodyParts3D STL meshes",
            install_hint=(
                "Mesh directory is present but empty. Copy STLs via "
                "`python -m tools.copy_anatomy_meshes "
                "/path/to/bodyparts3D/stl`."
            ),
        )

    meshes = [load_mesh(n) for n in names]
    yaw, pitch = _params_to_pose(params)
    bg = getattr(params, "background", "#0a0d12")
    bg_rgb = _hex_to_rgb(bg)
    return render_meshes(
        meshes, size,
        yaw=yaw, pitch=pitch, bg_color=bg_rgb,
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
