"""Stylised skull layer.

Drawable beneath the skin / muscle layers in the layered anatomy renderer.
Shapes are anatomically positioned (cranium aligns with the face oval,
orbital cavities surround the eyes, mandible follows the jawline) but the
*style* is illustrative rather than medical-imaging accurate — closer to
an anatomy textbook diagram than a CT scan.

The single public entry point is :func:`draw_skull` — call from a
QPainter context with the deformed landmark dict produced by the
anatomical renderer's compositor.
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


_BONE = QColor(220, 210, 195)         # Pale ivory bone
_BONE_SHADOW = QColor(168, 155, 140)
_BONE_DEEP = QColor(105, 92, 78)
_TOOTH = QColor(238, 230, 210)
_TOOTH_SHADOW = QColor(180, 168, 145)
_CAVITY = QColor(28, 22, 18)


def _bone_brush(bx: float, by: float, w: float, h: float, alpha: float) -> QBrush:
    grad = QRadialGradient(QPointF(bx + w * 0.42, by + h * 0.30), max(w, h) * 0.7)
    light = QColor(_BONE)
    mid = QColor(_BONE_SHADOW)
    dark = QColor(_BONE_DEEP)
    light.setAlphaF(alpha)
    mid.setAlphaF(alpha)
    dark.setAlphaF(alpha)
    grad.setColorAt(0.0, light)
    grad.setColorAt(0.55, mid)
    grad.setColorAt(1.0, dark)
    return QBrush(grad)


def draw_skull(p: QPainter, pts: dict[str, QPointF], box,
               *, alpha: float = 1.0) -> None:
    """Paint a stylised skull at the face's anatomical position.

    ``pts`` is the deformed landmark dict from the anatomical renderer.
    ``box`` is the (x, y, w, h) face-box rectangle. ``alpha`` lets the
    compositor fade the skull through translucent skin / muscle.
    """
    bx, by, bw, _ = box
    s = bw

    # Reference points from the deformed landmark template.
    chin = pts["chin"]
    temple_l = pts["temple_l"]
    temple_r = pts["temple_r"]
    forehead_l = pts["forehead_l"]
    forehead_r = pts["forehead_r"]
    hairline_top = pts["hairline_top"]
    eye_l_centre = QPointF(
        (pts["eye_l_upper_2"].x() + pts["eye_l_lower_2"].x()) / 2,
        (pts["eye_l_upper_2"].y() + pts["eye_l_lower_2"].y()) / 2,
    )
    eye_r_centre = QPointF(
        (pts["eye_r_upper_2"].x() + pts["eye_r_lower_2"].x()) / 2,
        (pts["eye_r_upper_2"].y() + pts["eye_r_lower_2"].y()) / 2,
    )
    nose_root = pts["nose_root"]
    nose_tip = pts["nose_tip"]
    nose_alar_l = pts["nose_alar_l"]
    nose_alar_r = pts["nose_alar_r"]
    lip_corner_l = pts["lip_corner_l"]
    lip_corner_r = pts["lip_corner_r"]
    jaw_l4 = pts["jaw_l4"]
    jaw_r4 = pts["jaw_r4"]

    # 1. Cranium — slightly bigger than the face_oval at the top, narrows
    # at the temples toward the mandible.
    cranium = QPainterPath()
    cranium.moveTo(QPointF(jaw_l4.x() - s * 0.02, jaw_l4.y()))
    cranium.quadTo(
        QPointF(temple_l.x() - s * 0.02, (temple_l.y() + jaw_l4.y()) / 2),
        QPointF(temple_l.x() - s * 0.01, temple_l.y()),
    )
    cranium.quadTo(
        QPointF(forehead_l.x() - s * 0.02, hairline_top.y() - s * 0.08),
        QPointF(hairline_top.x(), hairline_top.y() - s * 0.10),
    )
    cranium.quadTo(
        QPointF(forehead_r.x() + s * 0.02, hairline_top.y() - s * 0.08),
        QPointF(temple_r.x() + s * 0.01, temple_r.y()),
    )
    cranium.quadTo(
        QPointF(temple_r.x() + s * 0.02, (temple_r.y() + jaw_r4.y()) / 2),
        QPointF(jaw_r4.x() + s * 0.02, jaw_r4.y()),
    )
    # Mandible curve — follow jawline + chin.
    cranium.lineTo(pts["jaw_r3"])
    cranium.lineTo(pts["jaw_r2"])
    cranium.lineTo(pts["jaw_r1"])
    cranium.lineTo(QPointF(chin.x() + s * 0.01, chin.y() + s * 0.01))
    cranium.lineTo(QPointF(chin.x() - s * 0.01, chin.y() + s * 0.01))
    cranium.lineTo(pts["jaw_l1"])
    cranium.lineTo(pts["jaw_l2"])
    cranium.lineTo(pts["jaw_l3"])
    cranium.closeSubpath()

    p.setBrush(_bone_brush(bx, by, bw, bw, alpha))
    edge = QColor(_BONE_DEEP)
    edge.setAlphaF(min(1.0, alpha * 1.1))
    p.setPen(QPen(edge, s * 0.0035))
    p.drawPath(cranium)

    # 2. Coronal suture line — from temple to temple over forehead.
    suture_pen = QPen(QColor(120, 105, 90, int(180 * alpha)), s * 0.003,
                       Qt.PenStyle.DashLine)
    p.setPen(suture_pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    suture = QPainterPath()
    suture.moveTo(QPointF(temple_l.x() + s * 0.01, temple_l.y() - s * 0.04))
    suture.cubicTo(
        QPointF(forehead_l.x(), forehead_l.y() - s * 0.04),
        QPointF(forehead_r.x(), forehead_r.y() - s * 0.04),
        QPointF(temple_r.x() - s * 0.01, temple_r.y() - s * 0.04),
    )
    p.drawPath(suture)

    # Sagittal suture (top of skull).
    sag = QPainterPath()
    sag.moveTo(QPointF(hairline_top.x(), hairline_top.y() - s * 0.10))
    sag.lineTo(QPointF(hairline_top.x(), forehead_l.y() - s * 0.04))
    p.drawPath(sag)

    # 3. Orbital cavities — bigger than the eyes.
    orbit_pen = QPen(QColor(_BONE_DEEP), s * 0.0035)
    orbit_pen.setColor(QColor(_BONE_DEEP.red(), _BONE_DEEP.green(), _BONE_DEEP.blue(),
                                int(255 * min(1.0, alpha * 1.1))))
    p.setPen(orbit_pen)
    cav = QColor(_CAVITY)
    cav.setAlphaF(alpha)
    p.setBrush(cav)
    for c in (eye_l_centre, eye_r_centre):
        rect_w = s * 0.085
        rect_h = s * 0.080
        p.drawEllipse(QPointF(c.x(), c.y() + s * 0.005), rect_w, rect_h)

    # 4. Pyriform aperture — pear-shaped nasal opening.
    p.setBrush(cav)
    aperture = QPainterPath()
    top = QPointF(nose_root.x(), nose_root.y() + s * 0.025)
    btm_l = QPointF(nose_alar_l.x() + s * 0.005, nose_alar_l.y() + s * 0.005)
    btm_r = QPointF(nose_alar_r.x() - s * 0.005, nose_alar_r.y() + s * 0.005)
    spine = QPointF(nose_tip.x(), nose_tip.y() + s * 0.005)
    aperture.moveTo(top)
    aperture.cubicTo(QPointF(top.x() - s * 0.04, top.y() + s * 0.03),
                      btm_l,
                      QPointF(btm_l.x() + s * 0.01, btm_l.y() + s * 0.005))
    aperture.lineTo(spine)
    aperture.lineTo(QPointF(btm_r.x() - s * 0.01, btm_r.y() + s * 0.005))
    aperture.cubicTo(btm_r,
                      QPointF(top.x() + s * 0.04, top.y() + s * 0.03),
                      top)
    aperture.closeSubpath()
    p.drawPath(aperture)

    # 5. Zygomatic arches — outline curves from temple to maxilla.
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.setPen(QPen(_BONE_DEEP, s * 0.0035))
    for t, e_centre in ((temple_l, eye_l_centre), (temple_r, eye_r_centre)):
        arch = QPainterPath()
        arch.moveTo(QPointF(t.x() + (s * 0.02 if t.x() < bx + bw / 2 else -s * 0.02),
                              t.y() + s * 0.02))
        arch.cubicTo(
            QPointF((t.x() + e_centre.x()) / 2, t.y() + s * 0.04),
            QPointF(e_centre.x() + (s * 0.04 if e_centre.x() > t.x() else -s * 0.04),
                     e_centre.y() + s * 0.04),
            QPointF(e_centre.x(), e_centre.y() + s * 0.075),
        )
        p.drawPath(arch)

    # 6. Maxilla + upper teeth row.
    teeth_top_y = (lip_corner_l.y() + lip_corner_r.y()) / 2 + s * 0.015
    teeth_bot_y = teeth_top_y + s * 0.035
    span_l = lip_corner_l.x() - s * 0.005
    span_r = lip_corner_r.x() + s * 0.005
    n_teeth = 8
    p.setBrush(_TOOTH)
    p.setPen(QPen(_TOOTH_SHADOW, s * 0.002))
    for i in range(n_teeth):
        tw = (span_r - span_l) / n_teeth
        x = span_l + i * tw
        rect = QPainterPath()
        rect.addRoundedRect(x, teeth_top_y, tw - s * 0.003, s * 0.022,
                              s * 0.005, s * 0.005)
        p.drawPath(rect)

    # 7. Mandible / lower teeth row.
    teeth2_top_y = teeth_bot_y + s * 0.005
    p.setBrush(_TOOTH)
    for i in range(n_teeth):
        tw = (span_r - span_l) / n_teeth
        x = span_l + i * tw
        rect = QPainterPath()
        rect.addRoundedRect(x, teeth2_top_y, tw - s * 0.003, s * 0.022,
                              s * 0.005, s * 0.005)
        p.drawPath(rect)

    # 8. Mental protuberance (chin point) — bone surface highlight.
    p.setBrush(QColor(_BONE.red(), _BONE.green(), _BONE.blue(), int(220 * alpha)))
    p.setPen(QPen(_BONE_SHADOW, s * 0.0025))
    p.drawEllipse(
        QPointF(chin.x(), chin.y() - s * 0.010),
        s * 0.025, s * 0.018,
    )

    # 9. Mastoid process hint — a small bump behind the ear.
    for ear in ("ear_lower_l", "ear_lower_r"):
        e = pts[ear]
        p.setBrush(_BONE_SHADOW)
        p.drawEllipse(QPointF(e.x(), e.y()), s * 0.020, s * 0.018)
