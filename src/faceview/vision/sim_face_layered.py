"""Layered anatomy compositor.

Stacks the anatomical layers (skull → brain → eyeballs → muscle masses
→ skin) back-to-front so you can see what's underneath when an upper
layer is hidden or faded. Layer composition is driven by a list of
``(layer_name, alpha)`` tuples — a Persona's ``layers`` config.

Layer order (back to front):
    0. background vignette
    1. skull (bone)
    2. brain (inside cranium)
    3. eyeballs (full sclera spheres in orbits)
    4. muscle_masses (solid expression muscles)
    5. skin (the regular anatomical face)
    6. muscle_overlay (translucent activation gradient)

Each layer takes the deformed landmark dict produced by the anatomy
module — they all share the same coordinate frame, so the result reads
as one head with peelable layers, not a bunch of separate diagrams.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainter,
    QRadialGradient,
)

from faceview.vision.anatomy import (
    deform_landmarks,
    face_params_to_au_values,
    landmark_template,
    load_muscles,
)
from faceview.vision.anatomy_brain import draw_brain
from faceview.vision.anatomy_eyeballs import draw_eyeballs
from faceview.vision.anatomy_muscle_masses import draw_muscle_masses
from faceview.vision.anatomy_skull import draw_skull
from faceview.vision.sim_face import FaceParams
from faceview.vision.sim_face_anatomical import _scale_pts, _skin_palette
from faceview.vision.sim_face_anatomical_parts import (
    draw_brows,
    draw_cheeks,
    draw_eyes,
    draw_hair_back,
    draw_hair_front,
    draw_mouth,
    draw_nose,
    draw_skin,
    draw_skin_shading,
    smooth_path,
)
from faceview.vision.sim_face_anatomy_overlay import draw_muscle_overlay


LAYER_NAMES = ("skull", "brain", "eyeballs", "muscle_masses", "skin",
                "muscle_overlay")
LayerSpec = tuple[str, float]   # (name, alpha)


# Common preset layer stacks the renderer can pick by mode name.
LAYER_PRESETS: dict[str, list[LayerSpec]] = {
    "anatomy_layers": [
        ("skull", 1.0),
        ("brain", 0.95),
        ("eyeballs", 1.0),
        ("muscle_masses", 0.95),
        ("skin", 0.85),
    ],
    "anatomy_skull": [
        ("skull", 1.0),
    ],
    "anatomy_brain": [
        ("skull", 0.30),
        ("brain", 1.0),
    ],
    "anatomy_muscles": [
        ("skull", 0.30),
        ("muscle_masses", 1.0),
        ("muscle_overlay", 0.55),
    ],
    "anatomy_xray": [
        ("skull", 0.45),
        ("brain", 0.40),
        ("eyeballs", 0.55),
        ("muscle_masses", 0.40),
        ("skin", 0.45),
    ],
    "anatomy_eyeballs": [
        ("skull", 0.30),
        ("eyeballs", 1.0),
    ],
}


def _draw_skin_full(p: QPainter, params: FaceParams, template, pts,
                     au_values, box) -> None:
    """The full anatomical skin pipeline (skin + features), no hair."""
    skin = _skin_palette(params.skin_hue)
    face_pts = [pts[lm.name] for lm in template if lm.group == "face_oval"]
    face_path = smooth_path(face_pts, close=True)
    draw_skin(p, face_path, skin, box)
    draw_skin_shading(p, pts, skin, params, au_values, box)
    draw_cheeks(p, pts, skin, au_values, box)
    draw_brows(p, pts, params, au_values, box)
    draw_eyes(p, pts, params, au_values, box)
    draw_nose(p, pts, skin, params, au_values, box)
    draw_mouth(p, pts, params, skin, au_values, box)


def _resolve_layers(mode_or_layers: str | list[LayerSpec]) -> list[LayerSpec]:
    if isinstance(mode_or_layers, str):
        if mode_or_layers in LAYER_PRESETS:
            return list(LAYER_PRESETS[mode_or_layers])
        raise ValueError(f"unknown layer preset: {mode_or_layers}")
    return list(mode_or_layers)


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return arr[:, :, ::-1].copy()


def render_face_layered(
    params: FaceParams,
    size: tuple[int, int] = (640, 480),
    *,
    layers: str | Iterable[LayerSpec] = "anatomy_layers",
    show_hair: bool | None = None,
) -> np.ndarray:
    """Compose a layered anatomical view of the head.

    ``layers`` may be a preset name (``anatomy_layers``,
    ``anatomy_skull``, ``anatomy_brain``, ``anatomy_xray``,
    ``anatomy_muscles``, ``anatomy_eyeballs``) or an explicit list of
    ``(layer_name, alpha)`` pairs.

    Hair is shown only when the skin layer is fully opaque, unless
    ``show_hair`` is overridden.
    """
    spec = _resolve_layers(layers if not isinstance(layers, str) else layers)
    w, h = size

    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(params.background))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    # Background vignette.
    bg = QColor(params.background)
    grad = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
    grad.setColorAt(0.0, bg.lighter(120))
    grad.setColorAt(1.0, bg.darker(125))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRect(0, 0, w, h)

    # Compute deformed landmarks once — every layer shares them.
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    au_values = face_params_to_au_values(params)
    muscles = load_muscles()
    deformed = deform_landmarks(base, au_values, muscles=muscles)

    fw = min(w, h) * 0.78
    fx = (w - fw) / 2.0 + params.yaw * fw * 0.04
    fy = (h - fw) / 2.0 + params.pitch * fw * 0.04
    box = (fx, fy, fw, fw)
    pts = _scale_pts(deformed, template, box)

    # Hair only really makes sense when skin is fully opaque (otherwise
    # you'd see hair sticking out of a transparent head).
    skin_alpha = next((a for n, a in spec if n == "skin"), 0.0)
    show_hair_eff = show_hair if show_hair is not None else (skin_alpha >= 0.99)

    if show_hair_eff:
        draw_hair_back(p, pts, params, box)

    for name, alpha in spec:
        if alpha <= 0.001:
            continue
        if name == "skull":
            draw_skull(p, pts, box, alpha=alpha)
        elif name == "brain":
            draw_brain(p, pts, box, alpha=alpha)
        elif name == "eyeballs":
            draw_eyeballs(
                p, pts, box, alpha=alpha,
                pupil_x=float(getattr(params, "pupil_x", 0.0)),
                pupil_y=float(getattr(params, "pupil_y", 0.0)),
            )
        elif name == "muscle_masses":
            draw_muscle_masses(p, muscles, au_values, box, alpha=alpha)
        elif name == "skin":
            p.save()
            if alpha < 1.0:
                p.setOpacity(alpha)
            _draw_skin_full(p, params, template, pts, au_values, box)
            p.restore()
        elif name == "muscle_overlay":
            p.save()
            p.setOpacity(alpha)
            draw_muscle_overlay(p, muscles, au_values, box)
            p.restore()
        else:
            raise ValueError(f"unknown layer: {name}")

    if show_hair_eff:
        draw_hair_front(p, pts, params, box)

    p.end()
    return _qimage_to_bgr(img)
