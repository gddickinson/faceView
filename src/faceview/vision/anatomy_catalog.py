"""Unified anatomical mesh catalogue lifted from faceforge.

Loads the bundled head + neck JSON configs (skull bones, face features,
expression muscles, jaw muscles, neck muscles, skin, cervical
vertebrae) and exposes them as a flat list of :class:`MeshSpec`
records the renderer can iterate. Each spec carries:

- ``fma`` — BodyParts3D FMA identifier (also the STL filename).
- ``name`` — human-readable label.
- ``category`` — bone / muscle / feature / skin.
- ``color`` — per-mesh RGB (lifted directly from faceforge's catalogue).
- ``opacity`` / ``shininess`` — material parameters; only ``skin`` and
  some ``feature`` meshes set these.
- ``draw_order`` — back-to-front render order. Skin draws last so that
  with reduced opacity the underlying anatomy shows through.

The catalogue is loaded once and cached. ``head_neck_fmas()`` returns
the full FMA list — used by ``copy_anatomy_meshes.py`` so the script
copies exactly the STLs the renderer can render.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from faceview.assets import assets_dir


@dataclass
class MeshSpec:
    fma: str
    name: str
    category: str            # bone / muscle / feature / skin / vertebra
    color: tuple[int, int, int]
    opacity: float = 1.0
    shininess: float = 6.0
    draw_order: int = 100    # lower = earlier (further back)
    metadata: dict = field(default_factory=dict)


def _decode_color(c: int | None, default=(220, 210, 195)) -> tuple[int, int, int]:
    if c is None:
        return default
    return ((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF)


def _config_dir() -> Path:
    return assets_dir() / "config" / "anatomy"


def _load(name: str) -> list | dict:
    path = _config_dir() / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_catalog() -> list[MeshSpec]:
    out: list[MeshSpec] = []

    # 1. Skull bones (front-most bone layer behind muscles)
    for entry in _load("skull_bones"):
        out.append(MeshSpec(
            fma=entry["stl"],
            name=entry["name"],
            category="bone",
            color=_decode_color(entry.get("color"), default=(220, 210, 195)),
            shininess=4.0,
            draw_order=20,
            metadata={"group": entry.get("group", "")},
        ))

    # 2. Cervical vertebrae
    cerv = _load("cervical_vertebrae")
    if isinstance(cerv, dict):
        for k, v in cerv.items():
            stl = v.get("stl") if isinstance(v, dict) else None
            if not stl:
                continue
            out.append(MeshSpec(
                fma=stl, name=k, category="vertebra",
                color=_decode_color(v.get("color"), default=(214, 205, 185)),
                shininess=4.0,
                draw_order=15,
            ))
    elif isinstance(cerv, list):
        for v in cerv:
            stl = v.get("stl") if isinstance(v, dict) else None
            if not stl:
                continue
            out.append(MeshSpec(
                fma=stl, name=v.get("name", stl), category="vertebra",
                color=_decode_color(v.get("color"), default=(214, 205, 185)),
                shininess=4.0,
                draw_order=15,
            ))

    # 3. Expression muscles + jaw muscles (~65 total)
    for cat_file, draw in (("expression_muscles", 40),
                            ("jaw_muscles", 35),
                            ("neck_muscles", 30)):
        for entry in _load(cat_file):
            out.append(MeshSpec(
                fma=entry["stl"],
                name=entry["name"],
                category="muscle",
                color=_decode_color(entry.get("color"), default=(190, 90, 95)),
                shininess=8.0,
                draw_order=draw,
                metadata={"auMap": entry.get("auMap", {})},
            ))

    # 4. Face features (eyeballs, ears, eyebrow muscles, nose cartilage)
    for entry in _load("face_features"):
        cat = entry.get("category", "feature")
        out.append(MeshSpec(
            fma=entry["stl"],
            name=entry["name"],
            category=cat,
            color=_decode_color(entry.get("color"), default=(245, 240, 230)),
            opacity=float(entry.get("opacity", 1.0)),
            shininess=float(entry.get("shininess", 8.0)),
            draw_order=50,
            metadata={"type": entry.get("type"), "animated": entry.get("animated", False)},
        ))

    # 5. Skin — translucent outermost layer.
    for entry in _load("skin"):
        out.append(MeshSpec(
            fma=entry["stl"],
            name=entry["name"],
            category="skin",
            color=_decode_color(entry.get("color"), default=(220, 192, 170)),
            opacity=float(entry.get("opacity", 0.35)),
            shininess=float(entry.get("shininess", 5.0)),
            draw_order=90,
        ))

    return out


@lru_cache(maxsize=1)
def head_neck_fmas() -> dict[str, str]:
    """Return ``{fma: name}`` for all meshes the catalogue references."""
    return {spec.fma: spec.name for spec in load_catalog()}


def specs_by_category() -> dict[str, list[MeshSpec]]:
    out: dict[str, list[MeshSpec]] = {}
    for s in load_catalog():
        out.setdefault(s.category, []).append(s)
    return out


def specs_for_layer_set(layer_set: str) -> list[MeshSpec]:
    """Pre-built layer compositions for the renderer."""
    cats = specs_by_category()
    if layer_set == "skull_only":
        return cats.get("bone", []) + cats.get("vertebra", [])
    if layer_set == "muscles":
        return (cats.get("bone", []) + cats.get("vertebra", []) +
                cats.get("muscle", []))
    if layer_set == "features":
        return (cats.get("bone", []) + cats.get("vertebra", []) +
                cats.get("muscle", []) + cats.get("eyes", []) +
                cats.get("ears", []) + cats.get("nose", []) +
                cats.get("eyebrows", []) + cats.get("feature", []))
    if layer_set == "lifelike":
        # Photo-anatomical face: skin opaque enough to read as a face but
        # not so dense that the eyeballs underneath get fully obscured.
        all_specs = []
        for s in load_catalog():
            # Make a fresh copy so we don't mutate the cached catalog.
            from dataclasses import replace
            opacity = 0.92 if s.category == "skin" else s.opacity
            shininess = 8.0 if s.category == "skin" else s.shininess
            all_specs.append(replace(s, opacity=opacity, shininess=shininess))
        return all_specs
    if layer_set == "xray":
        # Same content as lifelike but with skin translucent.
        return list(load_catalog())
    if layer_set == "vertebrae":
        return cats.get("vertebra", [])
    if layer_set == "all_head_neck":
        return list(load_catalog())
    raise ValueError(f"unknown layer_set: {layer_set}")
