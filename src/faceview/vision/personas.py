"""Persona presets — appearance overrides applied on top of an animated FaceState.

A :class:`Persona` carries the *static* identity bits of the rendered face
(skin tone, hair colour, lip colour, background) so they can be swapped at
runtime without touching the dynamic FACS animation. The :class:`TalkingAvatar`
holds one persona and applies it to every :class:`FaceParams` returned by
``tick()``.

Bundled presets live in ``assets/config/personas.json``. Additional presets
(or fully custom ones) can be passed straight to :func:`apply_persona`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from faceview.assets import assets_dir
from faceview.vision.sim_face import FaceParams


@dataclass
class Persona:
    """Static appearance overlay for the procedural face.

    ``render_mode`` selects the renderer family (``stylised`` /
    ``anatomical`` / ``ict_face_3d`` / etc).

    ``identity_weights`` is an optional dict of ICT-FaceKit identity
    PCA coefficients (e.g. ``{"identity001": 2.0, "identity002": -1.5}``)
    used by the ``ict_face_3d`` renderer to vary the base face shape
    away from the neutral mean — different combinations give
    different individuals.
    """
    name: str
    skin_hue: float = 28.0
    hair_color: str = "#2c1810"
    lip_color: str = "#a44a4a"
    background: str = "#0c0f14"
    render_mode: str = "stylised"
    identity_weights: dict = field(default_factory=dict)
    # Iris colour for realistic eyes (most common in humans is
    # brown ≈ #5a3818; blue ~#4f6e85; green ~#5a7035; hazel ~#7a5530).
    eye_color: str = "#5a3818"
    # Skin saturation/value multipliers — tone down or tan up.
    skin_saturation: float = 0.32
    skin_value: float = 0.86
    # Render style — flips the ICT palette + shader uniforms for
    # sci-fi looks. Default "natural" is the realistic skin path.
    # "neon" / "transparent" / "cyberpunk" / "xray" select preset
    # palettes designed for stylised animation.
    style: str = "natural"


@lru_cache(maxsize=1)
def _load_personas_json() -> dict[str, dict]:
    path = assets_dir() / "config" / "personas.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text())


def list_personas() -> list[str]:
    return sorted(_load_personas_json().keys())


def load_persona(name: str) -> Persona:
    """Load a bundled persona by name. Falls back to ``default`` if missing."""
    data = _load_personas_json()
    raw = data.get(name) or data.get("default") or {}
    iw = raw.get("identity_weights", {}) or {}
    # Coerce numeric values to float; pass strings through (some
    # entries — e.g. ``mh_target`` — are mode-specific tag names
    # rather than coefficients).
    norm_iw: dict[str, float | str] = {}
    for k, v in iw.items():
        if isinstance(v, (int, float)):
            norm_iw[k] = float(v)
        else:
            norm_iw[k] = v
    return Persona(
        name=name,
        skin_hue=float(raw.get("skin_hue", 28.0)),
        hair_color=str(raw.get("hair_color", "#2c1810")),
        lip_color=str(raw.get("lip_color", "#a44a4a")),
        background=str(raw.get("background", "#0c0f14")),
        render_mode=str(raw.get("render_mode", "stylised")),
        identity_weights=norm_iw,
        eye_color=str(raw.get("eye_color", "#5a3818")),
        skin_saturation=float(raw.get("skin_saturation", 0.32)),
        skin_value=float(raw.get("skin_value", 0.86)),
        style=str(raw.get("style", "natural")),
    )


def apply_persona(params: FaceParams, persona: Persona) -> FaceParams:
    """Mutate ``params`` to carry the persona's static appearance fields."""
    params.skin_hue = persona.skin_hue
    params.hair_color = persona.hair_color
    params.lip_color = persona.lip_color
    params.background = persona.background
    params.render_mode = persona.render_mode
    # Identity coefficients for the ICT face renderer (no-op for
    # other modes).
    params.identity_weights = dict(persona.identity_weights)
    # Forward extended persona fields onto FaceParams as attributes.
    # ICT renderer reads these; other modes ignore them.
    params._persona_eye_color = persona.eye_color
    params._persona_skin_sat = persona.skin_saturation
    params._persona_skin_val = persona.skin_value
    params._persona_style = persona.style
    return params


def persona_summary(personas: Iterable[Persona]) -> list[dict]:
    return [asdict(p) for p in personas]
