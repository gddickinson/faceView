"""Expression presets — load FACS AU activations from JSON.

The bundled ``expressions.json`` (adapted from the FaceForge anatomy app)
defines 12 emotion presets (neutral, happy, sad, angry, surprised, fear,
disgust, contempt, pout, kiss, pain, thinking) as named AU dictionaries.
Each preset can also include head pose targets (``headYaw``, ``headPitch``,
``headRoll``) — see ``thinking`` for an example.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from faceview.assets import assets_dir
from faceview.vision.face_state import AU_IDS, FaceState


@lru_cache(maxsize=1)
def _load() -> dict[str, dict[str, float]]:
    path = assets_dir() / "config" / "expressions.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text())


def expression_names() -> list[str]:
    """Return the names of all bundled expression presets."""
    return list(_load().keys())


def get_expression(name: str) -> dict[str, float]:
    """Return the raw AU dictionary for ``name`` (empty dict if missing)."""
    return dict(_load().get(name, {}))


def apply_expression(state: FaceState, name: str, weight: float = 1.0) -> FaceState:
    """Set AUs and head-pose fields on ``state`` from preset ``name``.

    Existing AU values are *replaced* (not added) to match the preset, scaled
    by ``weight``. Returns the same state for chaining.
    """
    preset = get_expression(name)
    if not preset:
        return state

    state.reset_aus()
    for k, v in preset.items():
        if k in AU_IDS:
            state.set(k, float(v) * weight)
        elif k == "headYaw":
            state.head_yaw = float(v) * weight
        elif k == "headPitch":
            state.head_pitch = float(v) * weight
        elif k == "headRoll":
            state.head_roll = float(v) * weight
    return state
