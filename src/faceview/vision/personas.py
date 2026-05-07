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
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from faceview.assets import assets_dir
from faceview.vision.sim_face import FaceParams


@dataclass
class Persona:
    """Static appearance overlay for the procedural face.

    ``render_mode`` selects the renderer family (``stylised`` /
    ``anatomical`` / ``anatomy_overlay`` / ``wireframe``). The default
    stays on ``stylised`` so existing tests and personas keep their
    look.
    """
    name: str
    skin_hue: float = 28.0
    hair_color: str = "#2c1810"
    lip_color: str = "#a44a4a"
    background: str = "#0c0f14"
    render_mode: str = "stylised"


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
    return Persona(
        name=name,
        skin_hue=float(raw.get("skin_hue", 28.0)),
        hair_color=str(raw.get("hair_color", "#2c1810")),
        lip_color=str(raw.get("lip_color", "#a44a4a")),
        background=str(raw.get("background", "#0c0f14")),
        render_mode=str(raw.get("render_mode", "stylised")),
    )


def apply_persona(params: FaceParams, persona: Persona) -> FaceParams:
    """Mutate ``params`` to carry the persona's static appearance fields."""
    params.skin_hue = persona.skin_hue
    params.hair_color = persona.hair_color
    params.lip_color = persona.lip_color
    params.background = persona.background
    params.render_mode = persona.render_mode
    return params


def persona_summary(personas: Iterable[Persona]) -> list[dict]:
    return [asdict(p) for p in personas]
