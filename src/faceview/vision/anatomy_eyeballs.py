"""Stylised eyeball layer.

Visible when the skin layer is removed — full sclera spheres sitting in
the orbital cavities, with iris, pupil, optic-nerve stub, and a small
extraocular-muscle hint. The skin renderer's eye drawing only paints
what's visible through the palpebral fissure (the eye opening); this
layer paints the *whole* globe the way an anatomy textbook does.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)


_SCLERA = QColor(245, 240, 230)
_SCLERA_SHADOW = QColor(208, 200, 188)
_IRIS = QColor(80, 60, 38)
_PUPIL = QColor(8, 6, 4)
_NERVE = QColor(220, 200, 180)
_VESSEL = QColor(180, 60, 60)


def _with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


def draw_eyeballs(p: QPainter, pts: dict[str, QPointF], box,
                   *, alpha: float = 1.0,
                   pupil_x: float = 0.0, pupil_y: float = 0.0) -> None:
    """Paint full eye globes at the orbital landmark positions."""
    bx, by, bw, _ = box
    s = bw

    eye_l_centre = QPointF(
        (pts["eye_l_upper_2"].x() + pts["eye_l_lower_2"].x()) / 2,
        (pts["eye_l_upper_2"].y() + pts["eye_l_lower_2"].y()) / 2,
    )
    eye_r_centre = QPointF(
        (pts["eye_r_upper_2"].x() + pts["eye_r_lower_2"].x()) / 2,
        (pts["eye_r_upper_2"].y() + pts["eye_r_lower_2"].y()) / 2,
    )

    globe_r = s * 0.058

    for c in (eye_l_centre, eye_r_centre):
        # Optic nerve stub (behind / lateral to the globe).
        nerve = QPainterPath()
        nerve_origin = QPointF(c.x() + (s * 0.030 if c.x() > bx + bw / 2 else -s * 0.030),
                                c.y() + s * 0.005)
        nerve.moveTo(QPointF(c.x(), c.y()))
        nerve.quadTo(
            QPointF((c.x() + nerve_origin.x()) / 2, c.y() + s * 0.020),
            nerve_origin,
        )
        pen = QPen(_with_alpha(_NERVE, alpha), s * 0.018)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(nerve)

        # Sclera globe with shading.
        sg = QRadialGradient(QPointF(c.x() - s * 0.012, c.y() - s * 0.012), globe_r * 1.6)
        light = QColor(_SCLERA)
        light.setAlphaF(alpha)
        mid = QColor(_SCLERA_SHADOW)
        mid.setAlphaF(alpha)
        dark = QColor(_SCLERA_SHADOW.darker(135))
        dark.setAlphaF(alpha)
        sg.setColorAt(0.0, light)
        sg.setColorAt(0.6, mid)
        sg.setColorAt(1.0, dark)
        p.setBrush(QBrush(sg))
        p.setPen(QPen(_with_alpha(_SCLERA_SHADOW.darker(150), alpha), s * 0.003))
        p.drawEllipse(c, globe_r, globe_r)

        # Sclera vessels — a few thin red curves.
        vessel_pen = QPen(_with_alpha(_VESSEL, 0.45 * alpha), s * 0.0015)
        p.setPen(vessel_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for k, (a0x, a0y, b0x, b0y) in enumerate([
            (-0.04,  0.01,  -0.025,  0.02),
            ( 0.045,  0.005,  0.030,  0.018),
            (-0.04, -0.012, -0.025, -0.005),
        ]):
            v = QPainterPath()
            v.moveTo(QPointF(c.x() + s * a0x, c.y() + s * a0y))
            v.quadTo(QPointF(c.x() + s * (a0x + b0x) / 2, c.y() + s * (a0y + b0y) / 2 + s * 0.005),
                      QPointF(c.x() + s * b0x, c.y() + s * b0y))
            p.drawPath(v)

        # Iris.
        ix = c.x() + pupil_x * s * 0.020
        iy = c.y() + pupil_y * s * 0.012
        iris_r = s * 0.024
        ig = QRadialGradient(QPointF(ix - s * 0.005, iy - s * 0.005), iris_r * 1.3)
        l = QColor(_IRIS.lighter(160))
        l.setAlphaF(alpha)
        m = QColor(_IRIS)
        m.setAlphaF(alpha)
        d = QColor(_IRIS.darker(140))
        d.setAlphaF(alpha)
        ig.setColorAt(0.0, l)
        ig.setColorAt(0.55, m)
        ig.setColorAt(1.0, d)
        p.setBrush(QBrush(ig))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(ix, iy), iris_r, iris_r)
        # Limbal ring.
        p.setPen(QPen(_with_alpha(QColor(20, 12, 8), alpha), s * 0.003))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(ix, iy), iris_r, iris_r)
        # Pupil.
        p.setBrush(_with_alpha(_PUPIL, alpha))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(ix, iy), iris_r * 0.40, iris_r * 0.40)
        # Specular highlight.
        p.setBrush(QColor(255, 255, 255, int(220 * alpha)))
        p.drawEllipse(
            QPointF(ix - iris_r * 0.45, iy - iris_r * 0.45),
            iris_r * 0.20, iris_r * 0.20,
        )
