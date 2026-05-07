"""Anatomically-grounded 2D face renderer.

Compared to the stylised renderer in :mod:`faceview.vision.sim_face`, this
one is built on the 86-point landmark template and 43 expression muscles
in :mod:`faceview.vision.anatomy` (catalogue lifted from the faceforge
anatomy project). AU activations resolve through real anatomical pulls
— zygomaticus major lifts the mouth corner up *and* outward, levator
labii alaeque nasi pulls the upper lip and nasal wing together,
mentalis pushes the lower lip up via the chin pad — before any drawing
happens. Emotions read as expressions rather than smile/jaw knob
movement.

Three render modes are exposed via :func:`render_face_anatomical`:

- ``"anatomical"`` — fully shaded face with anatomical proportions and
  AU-driven landmark deformation. The default for the avatar in
  ``FACEVIEW_RENDER_MODE=anatomical``.
- ``"anatomy_overlay"`` — same face plus a translucent muscle layer
  glowing in proportion to each muscle's current activation. Useful
  for inspection / teaching / debugging an emotion preset.
- ``"wireframe"`` — landmark dots + group polylines on a dark
  background. Cheap, deterministic, useful for tests and for showing
  the underlying landmark template.

The renderer takes :class:`~faceview.vision.sim_face.FaceParams` so it
is a drop-in alternative to the stylised path. ``Persona`` overlays
still apply (skin tone, hair colour, lip colour, background).
"""

from __future__ import annotations

from typing import Literal

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
    Landmark,
    deform_landmarks,
    face_params_to_au_values,
    landmark_template,
    load_muscles,
)
from faceview.vision.sim_face import FaceParams
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
from faceview.vision.sim_face_anatomy_overlay import (
    draw_muscle_overlay,
    draw_wireframe,
)


RenderMode = Literal["anatomical", "anatomy_overlay", "wireframe"]


# ── colour helpers ────────────────────────────────────────────────────


def _skin_palette(hue: float) -> dict[str, QColor]:
    """Build a coherent skin palette from a single hue."""
    base = QColor.fromHsvF((hue % 360) / 360.0, 0.46, 0.92)
    shadow = QColor.fromHsvF((hue % 360) / 360.0, 0.55, 0.74)
    deep = QColor.fromHsvF((hue % 360) / 360.0, 0.62, 0.55)
    rim = QColor.fromHsvF((hue % 360) / 360.0, 0.30, 1.00)
    blush = QColor.fromHsvF(((hue + 4) % 360) / 360.0, 0.55, 0.94)
    return {"base": base, "shadow": shadow, "deep": deep, "rim": rim, "blush": blush}


# ── coordinate transform ──────────────────────────────────────────────


def _scale_pts(
    deformed: list[tuple[float, float]],
    landmarks: list[Landmark],
    box: tuple[float, float, float, float],
) -> dict[str, QPointF]:
    """Map deformed normalised coords to image-space QPointF dict by name."""
    x0, y0, w, _ = box
    return {
        lm.name: QPointF(x0 + dx * w, y0 + dy * w)
        for lm, (dx, dy) in zip(landmarks, deformed)
    }


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return arr[:, :, ::-1].copy()


# ── primary entry ─────────────────────────────────────────────────────


def render_face_anatomical(
    params: FaceParams,
    size: tuple[int, int] = (640, 480),
    *,
    mode: RenderMode = "anatomical",
) -> np.ndarray:
    """Render an anatomically-grounded face. Returns a BGR uint8 array."""
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

    # Compute deformed landmark positions for this FaceParams.
    template = landmark_template()
    base = [(lm.x, lm.y) for lm in template]
    au_values = face_params_to_au_values(params)
    muscles = load_muscles()
    deformed = deform_landmarks(base, au_values, muscles=muscles)

    # Centre the face in the image with a slight pose offset.
    fw = min(w, h) * 0.78
    fx = (w - fw) / 2.0 + params.yaw * fw * 0.04
    fy = (h - fw) / 2.0 + params.pitch * fw * 0.04
    box = (fx, fy, fw, fw)

    pts = _scale_pts(deformed, template, box)

    if mode == "wireframe":
        draw_wireframe(p, template, pts, w, h)
    else:
        _draw_face_layers(p, params, template, pts, au_values, box)
        if mode == "anatomy_overlay":
            draw_muscle_overlay(p, muscles, au_values, box)

    p.end()
    return _qimage_to_bgr(img)


# ── composition ───────────────────────────────────────────────────────


def _draw_face_layers(p, params, template, pts, au_values, box) -> None:
    """Paint the layered anatomical face."""
    skin = _skin_palette(params.skin_hue)

    # 1. Hair behind head — covers most of the silhouette plus a bit more.
    draw_hair_back(p, pts, params, box)

    # 2. Face oval (skin) — closed face_oval landmark group.
    face_pts = [pts[lm.name] for lm in template if lm.group == "face_oval"]
    face_path = smooth_path(face_pts, close=True)
    draw_skin(p, face_path, skin, box)

    # 3. Anatomical shading: temple/jaw shadow, brow shadow, nasolabial
    # fold, mentolabial sulcus.
    draw_skin_shading(p, pts, skin, params, au_values, box)

    # 4. Cheek apples (AU6 lift + AU12 smile blush).
    draw_cheeks(p, pts, skin, au_values, box)

    # 5. Hair fringe over forehead.
    draw_hair_front(p, pts, params, box)

    # 6. Brows.
    draw_brows(p, pts, params, au_values, box)

    # 7. Eyes (sclera, iris, limbus, lashes).
    draw_eyes(p, pts, params, au_values, box)

    # 8. Nose.
    draw_nose(p, pts, skin, params, au_values, box)

    # 9. Mouth / lips with optional inner cavity + teeth.
    draw_mouth(p, pts, params, skin, au_values, box)
