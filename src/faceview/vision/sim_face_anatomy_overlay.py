"""Overlay + wireframe drawers for the anatomical renderer.

Split out of :mod:`faceview.vision.sim_face_anatomical_parts` purely to
keep each module under the project file-size budget. Both functions are
self-contained — they take the already-deformed landmark dict from the
main renderer and add a layer on top.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient

from faceview.vision.anatomy import muscle_activation
from faceview.vision.sim_face_anatomical_parts import smooth_path


# Hue-per-AU palette so the overlay is self-explanatory: AU12 (smile)
# glows green, AU4 (brow lower) red, AU22 (lip funneler) yellow, etc.
_AU_HUES = {
    "AU1": 200, "AU2": 200, "AU4": 0,   "AU5": 30,
    "AU6": 320, "AU9": 0,   "AU12": 120, "AU15": 240,
    "AU20": 270, "AU22": 60, "AU25": 180, "AU26": 180,
}


def draw_muscle_overlay(p: QPainter, muscles, au_values, box) -> None:
    """Paint translucent muscle activation gradients with fiber ticks."""
    bx, by, bw, _ = box
    p.setPen(Qt.PenStyle.NoPen)
    for m in muscles:
        a = muscle_activation(m, au_values)
        if a < 0.05:
            continue
        dom_au = max(
            m.au_map.items(),
            key=lambda kv: kv[1] * float(au_values.get(kv[0], 0.0)),
        )[0]
        hue = _AU_HUES.get(dom_au, 0)
        col = QColor.fromHsvF(hue / 360.0, 0.85, 1.0)
        col.setAlphaF(0.55 * a)
        cx = bx + m.cx * bw
        cy = by + m.cy * bw
        rad = m.radius * bw
        rg = QRadialGradient(QPointF(cx, cy), rad)
        rg.setColorAt(0.0, col)
        rg.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QPointF(cx, cy), rad, rad)
        if a > 0.3 and not (m.fx == 0.0 and m.fy == 0.0):
            pen = QPen(col.darker(140), 2.0)
            p.setPen(pen)
            p.drawLine(
                QPointF(cx, cy),
                QPointF(cx + m.fx * rad * 0.7, cy + m.fy * rad * 0.7),
            )
            p.setPen(Qt.PenStyle.NoPen)


def draw_wireframe(p: QPainter, template, pts, w: int, h: int) -> None:
    """Landmark dots + group polylines on a dark background."""
    p.setBrush(QColor(15, 18, 22))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRect(0, 0, w, h)
    pen = QPen(QColor(120, 200, 255, 170), 1.4)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    groups: dict[str, list[QPointF]] = {}
    for lm in template:
        groups.setdefault(lm.group, []).append(pts[lm.name])
    for name, qpts in groups.items():
        if name in {
            "face_oval", "lip_outer_upper", "lip_outer_lower",
            "eye_l_upper", "eye_l_lower", "eye_r_upper", "eye_r_lower",
            "brow_l", "brow_r", "lip_inner_upper", "lip_inner_lower",
            "nose",
        }:
            close = name == "face_oval"
            p.drawPath(smooth_path(qpts, close=close))
    p.setBrush(QColor(255, 220, 80))
    p.setPen(Qt.PenStyle.NoPen)
    for lm in template:
        q = pts[lm.name]
        p.drawEllipse(q, 2.2, 2.2)
