"""Stylised brain layer.

Sits inside the cranium (visible when skull layer is removed/faded).
Renders the four cerebral lobes (frontal, parietal, temporal,
occipital), the cerebellum, and a brainstem stub. Gyri/sulci are drawn
as wavy lines for surface texture — illustrative, not anatomically
indexed.

Like :mod:`faceview.vision.anatomy_skull`, this is a 2D illustration:
positioned correctly relative to the cranium but stylised, not derived
from medical imaging.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)


# Anatomical / textbook colour scheme. Frontal warmer; occipital cooler.
_FRONTAL = QColor(220, 165, 165)
_PARIETAL = QColor(225, 175, 175)
_TEMPORAL = QColor(210, 158, 158)
_OCCIPITAL = QColor(195, 145, 152)
_CEREBELLUM = QColor(195, 165, 175)
_BRAINSTEM = QColor(225, 200, 195)
_GYRUS = QColor(165, 100, 105)


def _with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


def _shaded_fill(p: QPainter, base: QColor, focus: QPointF,
                  radius: float, alpha: float) -> QBrush:
    light = QColor(base.lighter(120))
    light.setAlphaF(alpha)
    mid = QColor(base)
    mid.setAlphaF(alpha)
    dark = QColor(base.darker(135))
    dark.setAlphaF(alpha)
    g = QRadialGradient(focus, radius)
    g.setColorAt(0.0, light)
    g.setColorAt(0.55, mid)
    g.setColorAt(1.0, dark)
    return QBrush(g)


def _draw_gyri(p: QPainter, region: QPainterPath, intensity: float = 0.9,
                spacing: float = 6.0, scale: float = 1.0) -> None:
    """Wavy lines inside ``region`` to suggest cortical convolutions."""
    p.save()
    p.setClipPath(region)
    pen = QPen(_with_alpha(_GYRUS, 0.40 * intensity),
                max(1.0, 1.4 * scale))
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    rect = region.boundingRect()
    y0 = rect.top()
    y1 = rect.bottom()
    x0 = rect.left()
    x1 = rect.right()
    span = max(1.0, x1 - x0)
    rows = max(3, int((y1 - y0) / spacing))
    for i in range(rows):
        y = y0 + (i + 0.5) * (y1 - y0) / rows
        path = QPainterPath()
        path.moveTo(QPointF(x0, y))
        for k in range(8):
            f = (k + 1) / 8
            x = x0 + f * span
            yy = y + 2.5 * scale * math.sin(f * math.pi * 4 + i * 0.6)
            path.lineTo(QPointF(x, yy))
        p.drawPath(path)
    p.restore()


def draw_brain(p: QPainter, pts: dict[str, QPointF], box,
                *, alpha: float = 1.0) -> None:
    """Paint the cerebral lobes + cerebellum inside the cranium.

    Anatomical positioning relative to the deformed landmarks. ``alpha``
    fades the whole brain layer for the layered compositor.
    """
    bx, by, bw, _ = box
    s = bw

    glabella = pts.get("glabella", QPointF(bx + bw * 0.5, by + bw * 0.34))
    forehead_l = pts["forehead_l"]
    forehead_r = pts["forehead_r"]
    temple_l = pts["temple_l"]
    temple_r = pts["temple_r"]
    hairline_top = pts["hairline_top"]

    # Vertical reference lines.
    cx = (forehead_l.x() + forehead_r.x()) / 2
    top_y = hairline_top.y() - s * 0.05
    front_y = forehead_l.y() - s * 0.005
    waist_y = (top_y + front_y) / 2

    # 1. Brainstem (thin, behind nasal aperture — comes out at bottom-back).
    stem_path = QPainterPath()
    stem_path.moveTo(QPointF(cx - s * 0.025, waist_y + s * 0.18))
    stem_path.cubicTo(
        QPointF(cx - s * 0.030, waist_y + s * 0.26),
        QPointF(cx - s * 0.025, waist_y + s * 0.34),
        QPointF(cx - s * 0.020, waist_y + s * 0.40),
    )
    stem_path.lineTo(QPointF(cx + s * 0.020, waist_y + s * 0.40))
    stem_path.cubicTo(
        QPointF(cx + s * 0.025, waist_y + s * 0.34),
        QPointF(cx + s * 0.030, waist_y + s * 0.26),
        QPointF(cx + s * 0.025, waist_y + s * 0.18),
    )
    stem_path.closeSubpath()
    p.setBrush(_with_alpha(_BRAINSTEM, alpha))
    p.setPen(QPen(_with_alpha(_BRAINSTEM.darker(135), alpha), s * 0.002))
    p.drawPath(stem_path)

    # 2. Cerebellum — sits behind / below the cerebrum.
    cere = QPainterPath()
    cere_centre = QPointF(cx, waist_y + s * 0.20)
    cere.addEllipse(cere_centre, s * 0.10, s * 0.060)
    p.setBrush(_shaded_fill(p, _CEREBELLUM, cere_centre, s * 0.10, alpha))
    p.setPen(QPen(_with_alpha(_CEREBELLUM.darker(140), alpha), s * 0.0025))
    p.drawPath(cere)
    # Cerebellar foliations (parallel curved lines).
    p.save()
    p.setClipPath(cere)
    pen = QPen(_with_alpha(_GYRUS.darker(115), 0.6 * alpha), 1.2)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(8):
        f = (i + 0.5) / 8
        y = (cere_centre.y() - s * 0.060) + f * s * 0.120
        path = QPainterPath()
        path.moveTo(QPointF(cere_centre.x() - s * 0.10, y))
        path.quadTo(QPointF(cere_centre.x(), y + 1.0),
                     QPointF(cere_centre.x() + s * 0.10, y))
        p.drawPath(path)
    p.restore()

    # 3. Temporal lobes (left + right, lateral, lower).
    for side, x_anchor in (("l", temple_l.x() + s * 0.03),
                            ("r", temple_r.x() - s * 0.03)):
        c = QPointF(x_anchor, waist_y + s * 0.13)
        path = QPainterPath()
        path.addEllipse(c, s * 0.075, s * 0.055)
        p.setBrush(_shaded_fill(p, _TEMPORAL, c, s * 0.10, alpha))
        p.setPen(QPen(_with_alpha(_TEMPORAL.darker(140), alpha), s * 0.0025))
        p.drawPath(path)
        _draw_gyri(p, path, intensity=alpha, spacing=s * 0.018, scale=1.0)

    # 4. Occipital lobe — back of cerebrum, behind the parietal.
    occ_centre = QPointF(cx, waist_y + s * 0.10)
    occ = QPainterPath()
    occ.addEllipse(occ_centre, s * 0.10, s * 0.07)
    p.setBrush(_shaded_fill(p, _OCCIPITAL, occ_centre, s * 0.12, alpha))
    p.setPen(QPen(_with_alpha(_OCCIPITAL.darker(140), alpha), s * 0.0025))
    p.drawPath(occ)
    _draw_gyri(p, occ, intensity=alpha, spacing=s * 0.018, scale=1.0)

    # 5. Parietal lobe — top of cerebrum, behind frontal.
    par_centre = QPointF(cx, top_y + s * 0.10)
    par = QPainterPath()
    par.moveTo(QPointF(cx - s * 0.18, top_y + s * 0.02))
    par.quadTo(QPointF(cx, top_y - s * 0.04),
                QPointF(cx + s * 0.18, top_y + s * 0.02))
    par.cubicTo(
        QPointF(cx + s * 0.16, top_y + s * 0.14),
        QPointF(cx + s * 0.05, top_y + s * 0.16),
        QPointF(cx, top_y + s * 0.16),
    )
    par.cubicTo(
        QPointF(cx - s * 0.05, top_y + s * 0.16),
        QPointF(cx - s * 0.16, top_y + s * 0.14),
        QPointF(cx - s * 0.18, top_y + s * 0.02),
    )
    par.closeSubpath()
    p.setBrush(_shaded_fill(p, _PARIETAL, par_centre, s * 0.20, alpha))
    p.setPen(QPen(_with_alpha(_PARIETAL.darker(140), alpha), s * 0.0025))
    p.drawPath(par)
    _draw_gyri(p, par, intensity=alpha, spacing=s * 0.020, scale=1.0)

    # 6. Frontal lobe — biggest, at the front.
    fr_centre = QPointF(cx, (top_y + front_y) / 2 + s * 0.02)
    fr = QPainterPath()
    fr.moveTo(QPointF(cx - s * 0.20, front_y))
    fr.quadTo(QPointF(cx - s * 0.21, top_y + s * 0.06),
                QPointF(cx - s * 0.16, top_y + s * 0.02))
    fr.quadTo(QPointF(cx, top_y - s * 0.02),
                QPointF(cx + s * 0.16, top_y + s * 0.02))
    fr.quadTo(QPointF(cx + s * 0.21, top_y + s * 0.06),
                QPointF(cx + s * 0.20, front_y))
    fr.cubicTo(
        QPointF(cx + s * 0.15, front_y + s * 0.02),
        QPointF(cx + s * 0.06, front_y - s * 0.005),
        QPointF(cx, front_y),
    )
    fr.cubicTo(
        QPointF(cx - s * 0.06, front_y - s * 0.005),
        QPointF(cx - s * 0.15, front_y + s * 0.02),
        QPointF(cx - s * 0.20, front_y),
    )
    fr.closeSubpath()
    p.setBrush(_shaded_fill(p, _FRONTAL, fr_centre, s * 0.25, alpha))
    p.setPen(QPen(_with_alpha(_FRONTAL.darker(140), alpha), s * 0.0025))
    p.drawPath(fr)
    _draw_gyri(p, fr, intensity=alpha, spacing=s * 0.020, scale=1.1)

    # 7. Central sulcus — divides frontal from parietal.
    sulcus = QPainterPath()
    sulcus.moveTo(QPointF(cx - s * 0.17, top_y + s * 0.04))
    sulcus.cubicTo(
        QPointF(cx - s * 0.08, top_y + s * 0.02),
        QPointF(cx + s * 0.08, top_y + s * 0.02),
        QPointF(cx + s * 0.17, top_y + s * 0.04),
    )
    pen = QPen(_with_alpha(_GYRUS.darker(120), alpha), s * 0.004)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(sulcus)

    # 8. Inter-hemispheric fissure — vertical line down midline.
    fissure = QPainterPath()
    fissure.moveTo(QPointF(cx, top_y - s * 0.02))
    fissure.lineTo(QPointF(cx, front_y))
    p.drawPath(fissure)
