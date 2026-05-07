"""Solid muscle-mass layer.

The :func:`draw_muscle_overlay` in
:mod:`faceview.vision.sim_face_anatomy_overlay` shows muscle activation
as a translucent gradient on top of the rendered face — useful for
debugging emotion presets but not anatomically opaque. This module
draws each of the 43 expression muscles as a *solid* fleshy shape at
its anatomical centroid, oriented along the fiber direction. With the
skin layer off, the layered renderer shows the muscle layer as a
textbook-style myology view.

Activation also brightens the colour and slightly enlarges the muscle
along the fiber axis (a 2D analogue of contraction). This means the
same animation pipeline that drives skin deformation also drives the
muscle layer's thickness — they stay coherent.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)

from faceview.vision.anatomy import Muscle, muscle_activation


_MUSCLE_BASE = QColor(165, 75, 78)        # Resting muscle red.
_MUSCLE_LIGHT = QColor(205, 110, 110)
_MUSCLE_DEEP = QColor(115, 38, 42)
_FASCIA = QColor(220, 200, 195)


def _with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


def _activated_colour(activation: float, alpha: float) -> tuple[QColor, QColor, QColor]:
    """Bright redder when contracting, dusky red at rest."""
    if activation > 0.05:
        light = QColor(225, 110, 110)
        mid = QColor(195, 70, 75)
        dark = QColor(145, 38, 42)
    else:
        light = QColor(_MUSCLE_LIGHT)
        mid = QColor(_MUSCLE_BASE)
        dark = QColor(_MUSCLE_DEEP)
    return _with_alpha(light, alpha), _with_alpha(mid, alpha), _with_alpha(dark, alpha)


def _muscle_path(cx: float, cy: float, fx: float, fy: float,
                  rx: float, ry: float) -> QPainterPath:
    """Oriented ellipse — long axis along the fiber direction."""
    path = QPainterPath()
    fn = math.hypot(fx, fy)
    if fn < 1e-6:
        path.addEllipse(QPointF(cx, cy), rx, ry)
        return path
    # Build a unit-orthonormal rotation aligned with the fiber.
    ux, uy = fx / fn, fy / fn
    px, py = -uy, ux
    # Sample 32 points along an ellipse in fiber-space, transform to scene.
    for i in range(33):
        ang = i / 32 * math.tau
        a = math.cos(ang) * rx  # along fiber
        b = math.sin(ang) * ry  # perpendicular
        x = cx + a * ux + b * px
        y = cy + a * uy + b * py
        if i == 0:
            path.moveTo(QPointF(x, y))
        else:
            path.lineTo(QPointF(x, y))
    path.closeSubpath()
    return path


def draw_muscle_masses(p: QPainter, muscles: list[Muscle],
                        au_values: dict[str, float],
                        box,
                        *, alpha: float = 1.0,
                        show_fibers: bool = True) -> None:
    """Paint solid muscle masses at their anatomical positions."""
    bx, by, bw, _ = box
    s = bw

    # 1. Soft fascia / connective-tissue layer beneath the muscles, so the
    # face still reads as a head when only the muscle layer is on.
    fascia_brush = QRadialGradient(QPointF(bx + bw * 0.5, by + bw * 0.5), bw * 0.7)
    fas_light = QColor(_FASCIA)
    fas_light.setAlphaF(0.45 * alpha)
    fas_dark = QColor(_FASCIA.darker(120))
    fas_dark.setAlphaF(0.25 * alpha)
    fascia_brush.setColorAt(0.0, fas_light)
    fascia_brush.setColorAt(1.0, fas_dark)
    p.setBrush(QBrush(fascia_brush))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(bx + bw * 0.5, by + bw * 0.5), bw * 0.45, bw * 0.55)

    # 2. Each muscle as an oriented ellipse, colour-shifted by activation.
    for m in muscles:
        a = muscle_activation(m, au_values)
        # Contraction = shorter along fiber, slightly fatter perpendicular.
        contraction = a * 0.18
        cx = bx + m.cx * bw
        cy = by + m.cy * bw
        # Radial sphincters (orbicularis): both axes equal.
        if m.fx == 0.0 and m.fy == 0.0:
            rx = m.radius * bw * 0.55
            ry = m.radius * bw * 0.55 * (1.0 - 0.10 * a)
            fx, fy = 1.0, 0.0
        else:
            rx = m.radius * bw * (1.05 - contraction)
            ry = m.radius * bw * 0.45 * (1.0 + contraction * 0.6)
            fx, fy = m.fx, m.fy

        path = _muscle_path(cx, cy, fx, fy, rx, ry)
        light, mid, dark = _activated_colour(a, alpha)
        focus = QPointF(cx - 0.4 * rx * fx, cy - 0.4 * rx * fy)
        g = QRadialGradient(focus, max(rx, ry) * 1.5)
        g.setColorAt(0.0, light)
        g.setColorAt(0.55, mid)
        g.setColorAt(1.0, dark)
        p.setBrush(QBrush(g))
        p.setPen(QPen(_with_alpha(_MUSCLE_DEEP, alpha), s * 0.0015))
        p.drawPath(path)

        # 3. Fiber striations along the long axis.
        if show_fibers and rx > s * 0.012:
            p.save()
            p.setClipPath(path)
            stripe_pen = QPen(_with_alpha(_MUSCLE_DEEP.darker(115), 0.5 * alpha),
                                max(0.8, s * 0.0018))
            p.setPen(stripe_pen)
            n_stripes = 6
            for i in range(n_stripes):
                f = (i + 0.5) / n_stripes
                # Perpendicular offset.
                fn = math.hypot(fx, fy)
                ux, uy = fx / fn, fy / fn
                px_, py_ = -uy, ux
                offset = (f - 0.5) * 2 * ry
                start = QPointF(cx - rx * ux + offset * px_, cy - rx * uy + offset * py_)
                end = QPointF(cx + rx * ux + offset * px_, cy + rx * uy + offset * py_)
                p.drawLine(start, end)
            p.restore()
