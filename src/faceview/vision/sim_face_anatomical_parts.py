"""Feature drawers for the anatomical renderer.

Split out of :mod:`faceview.vision.sim_face_anatomical` purely to keep
the file size budget. Each drawer takes the QPainter, the dict of
deformed landmark QPointFs, the FaceParams + AU values, and the
face-box rectangle. The dispatcher in the main module decides the
order in which they're called.
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

from faceview.vision.anatomy import landmark_template, muscle_activation


# ── path helpers ──────────────────────────────────────────────────────


def smooth_path(points: list[QPointF], close: bool = True) -> QPainterPath:
    """Catmull-Rom-style smooth path through ``points``."""
    n = len(points)
    if n < 3:
        path = QPainterPath()
        if points:
            path.moveTo(points[0])
            for q in points[1:]:
                path.lineTo(q)
            if close:
                path.closeSubpath()
        return path

    if close:
        ext = [points[-1]] + points + [points[0], points[1]]
    else:
        ext = [points[0]] + points + [points[-1]]

    path = QPainterPath()
    path.moveTo(points[0])
    for i in range(1, n + (1 if close else 0)):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0,
                     p1.y() + (p2.y() - p0.y()) / 6.0)
        c2 = QPointF(p2.x() - (p3.x() - p1.x()) / 6.0,
                     p2.y() - (p3.y() - p1.y()) / 6.0)
        path.cubicTo(c1, c2, p2)
    if close:
        path.closeSubpath()
    return path


def box_scale(box) -> float:
    return float(box[2])


def hex_color(c: str) -> QColor:
    return QColor(c)


# ── feature drawers ───────────────────────────────────────────────────


def draw_skin(p: QPainter, face_path: QPainterPath, skin, box) -> None:
    bx, by, bw, _ = box
    grad = QRadialGradient(
        QPointF(bx + bw * 0.42, by + bw * 0.30), bw * 0.85,
    )
    grad.setColorAt(0.0, skin["rim"])
    grad.setColorAt(0.45, skin["base"])
    grad.setColorAt(1.0, skin["deep"])
    p.setBrush(QBrush(grad))
    p.setPen(QPen(skin["deep"].darker(115), 1.3))
    p.drawPath(face_path)


def draw_skin_shading(p, pts, skin, params, au_values, box) -> None:
    s = box_scale(box)

    # Side shadow (right side).
    shade = QLinearGradient(QPointF(pts["temple_l"].x(), 0),
                             QPointF(pts["temple_r"].x(), 0))
    shade.setColorAt(0.0, QColor(0, 0, 0, 35))
    shade.setColorAt(0.4, QColor(0, 0, 0, 0))
    shade.setColorAt(0.6, QColor(0, 0, 0, 0))
    shade.setColorAt(1.0, QColor(0, 0, 0, 35))
    p.setBrush(QBrush(shade))
    p.setPen(Qt.PenStyle.NoPen)
    face_pts = [pts[lm.name] for lm in landmark_template() if lm.group == "face_oval"]
    face_path = smooth_path(face_pts, close=True)
    p.drawPath(face_path)

    # Brow shadow.
    for side in ("l", "r"):
        b_outer = pts[f"brow_{side}_0" if side == "l" else f"brow_{side}_4"]
        b_inner = pts[f"brow_{side}_4" if side == "l" else f"brow_{side}_0"]
        shadow_path = QPainterPath()
        shadow_path.moveTo(b_outer)
        shadow_path.quadTo(
            QPointF((b_outer.x() + b_inner.x()) / 2, b_outer.y() + s * 0.020),
            b_inner,
        )
        shadow_path.lineTo(QPointF(b_inner.x(), b_inner.y() + s * 0.025))
        shadow_path.quadTo(
            QPointF((b_outer.x() + b_inner.x()) / 2, b_outer.y() + s * 0.045),
            QPointF(b_outer.x(), b_outer.y() + s * 0.025),
        )
        shadow_path.closeSubpath()
        p.setBrush(QColor(0, 0, 0, 24))
        p.drawPath(shadow_path)

    # Nasolabial fold.
    fold_strength = max(0.0, au_values["AU12"] * 0.6 + au_values["AU6"] * 0.4
                        + au_values["AU9"] * 0.5)
    if fold_strength > 0.15:
        for side, alar, corner in (("l", "nose_alar_l", "lip_corner_l"),
                                    ("r", "nose_alar_r", "lip_corner_r")):
            a = pts[alar]
            c = pts[corner]
            mid = QPointF((a.x() + c.x()) / 2 + (s * 0.012 if side == "r" else -s * 0.012),
                          (a.y() + c.y()) / 2)
            pen = QPen(QColor(0, 0, 0, int(60 * fold_strength)))
            pen.setWidthF(s * 0.005)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            path.moveTo(a)
            path.quadTo(mid, c)
            p.drawPath(path)

    # Mentolabial sulcus.
    chin_path = QPainterPath()
    chin_path.moveTo(QPointF(pts["lip_lower_l2"].x(), pts["lip_lower_mid"].y() + s * 0.025))
    chin_path.quadTo(
        QPointF(pts["lip_lower_mid"].x(), pts["lip_lower_mid"].y() + s * 0.040),
        QPointF(pts["lip_lower_r2"].x(), pts["lip_lower_mid"].y() + s * 0.025),
    )
    pen = QPen(QColor(0, 0, 0, 30))
    pen.setWidthF(s * 0.004)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(chin_path)


def draw_cheeks(p, pts, skin, au_values, box) -> None:
    s = box_scale(box)
    a6 = au_values["AU6"]
    a12 = au_values["AU12"]
    intensity = 0.55 + 0.45 * max(a6, a12)
    blush = QColor(skin["blush"])
    blush.setAlphaF(0.30 * intensity)
    p.setPen(Qt.PenStyle.NoPen)
    for c in ("cheek_l", "cheek_r"):
        cp = pts[c]
        cy = cp.y() - s * 0.012 * a6
        rg = QRadialGradient(QPointF(cp.x(), cy), s * 0.07)
        rg.setColorAt(0.0, blush)
        rg.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(rg))
        p.drawEllipse(QPointF(cp.x(), cy), s * 0.07, s * 0.05)


def draw_brows(p, pts, params, au_values, box) -> None:
    s = box_scale(box)
    hair = hex_color(params.hair_color)
    body = QPen(hair.darker(110), s * 0.012)
    body.setCapStyle(Qt.PenCapStyle.RoundCap)
    hairs = QPen(hair.darker(125), s * 0.0042)
    hairs.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setBrush(Qt.BrushStyle.NoBrush)

    for side in ("l", "r"):
        bp = [pts[f"brow_{side}_{i}"] for i in range(5)]
        path = smooth_path(bp, close=False)
        p.setPen(body)
        p.drawPath(path)
        p.setPen(hairs)
        for i in range(len(bp) - 1):
            a, b = bp[i], bp[i + 1]
            for k in range(3):
                f = (k + 0.5) / 3
                x = a.x() + (b.x() - a.x()) * f
                y = a.y() + (b.y() - a.y()) * f
                tx, ty = (b.x() - a.x()), (b.y() - a.y())
                tn = (tx * tx + ty * ty) ** 0.5 + 1e-6
                tx /= tn; ty /= tn
                length = s * 0.022
                p.drawLine(QPointF(x - tx * length / 2, y - ty * length / 2 - s * 0.006),
                            QPointF(x + tx * length / 2, y + ty * length / 2))


def draw_eyes(p, pts, params, au_values, box) -> None:
    s = box_scale(box)
    blink = 1.0 - float(getattr(params, "eye_open", 1.0))
    blink = max(0.0, min(1.0, blink))
    a5 = au_values["AU5"]
    a6 = au_values["AU6"]

    for side in ("l", "r"):
        upper = [pts[f"eye_{side}_upper_{i}"] for i in range(5)]
        lower = [pts[f"eye_{side}_lower_{i}"] for i in range(5)]
        if blink > 0.0 or a5 > 0.0:
            for i, q in enumerate(upper):
                target_y = lower[i].y() if blink > 0 else q.y() - s * 0.010 * a5
                ny = q.y() + (target_y - q.y()) * blink
                upper[i] = QPointF(q.x(), ny)
        if a6 > 0.0:
            for i, q in enumerate(lower):
                lower[i] = QPointF(q.x(), q.y() - s * 0.006 * a6)

        eye_outline = upper + list(reversed(lower[1:-1]))
        eye_path = smooth_path(eye_outline, close=True)

        p.setBrush(QColor(245, 240, 230))
        p.setPen(QPen(QColor(60, 40, 30, 120), s * 0.003))
        p.drawPath(eye_path)

        if blink < 0.95:
            p.save()
            p.setClipPath(eye_path)
            iris_c = pts[f"iris_{side}"]
            ix = iris_c.x() + float(getattr(params, "pupil_x", 0.0)) * s * 0.010
            iy = iris_c.y() + float(getattr(params, "pupil_y", 0.0)) * s * 0.006
            iris_r = s * 0.016
            iris_grad = QRadialGradient(QPointF(ix - s * 0.006, iy - s * 0.006), iris_r)
            iris_grad.setColorAt(0.0, QColor(160, 130, 90))
            iris_grad.setColorAt(0.5, QColor(80, 60, 38))
            iris_grad.setColorAt(1.0, QColor(40, 28, 18))
            p.setBrush(QBrush(iris_grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(ix, iy), iris_r, iris_r)
            pen = QPen(QColor(20, 12, 8, 200), s * 0.0035)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(ix, iy), iris_r, iris_r)
            p.setBrush(QColor(8, 6, 4))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(ix, iy), iris_r * 0.36, iris_r * 0.36)
            p.setBrush(QColor(255, 255, 255, 220))
            p.drawEllipse(QPointF(ix - iris_r * 0.45, iy - iris_r * 0.45),
                           iris_r * 0.18, iris_r * 0.18)
            p.restore()

        lid_pen = QPen(QColor(40, 25, 15), s * 0.0055)
        lid_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(lid_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(smooth_path(upper, close=False))

        lash_pen = QPen(QColor(30, 20, 10, 180), s * 0.003)
        p.setPen(lash_pen)
        for q in lower[1:-1]:
            p.drawLine(QPointF(q.x(), q.y()),
                        QPointF(q.x(), q.y() + s * 0.008))


def draw_nose(p, pts, skin, params, au_values, box) -> None:
    s = box_scale(box)
    a9 = au_values["AU9"]

    bridge_path = QPainterPath()
    bridge_path.moveTo(pts["nose_root"])
    bridge_path.quadTo(pts["nose_dorsum_l"], pts["nose_alar_l"])
    bridge_path.lineTo(pts["nose_tip"])
    bridge_path.lineTo(pts["nose_alar_r"])
    bridge_path.quadTo(pts["nose_dorsum_r"], pts["nose_root"])
    bridge_path.closeSubpath()

    grad = QLinearGradient(
        QPointF(pts["nose_root"].x() - s * 0.05, pts["nose_root"].y()),
        QPointF(pts["nose_root"].x() + s * 0.05, pts["nose_tip"].y()),
    )
    grad.setColorAt(0.0, QColor(0, 0, 0, 0))
    grad.setColorAt(0.5, QColor(0, 0, 0, 14))
    grad.setColorAt(1.0, QColor(0, 0, 0, 36))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPath(bridge_path)

    p.setBrush(QColor(255, 255, 255, 22))
    p.drawEllipse(QPointF(pts["nose_tip"].x() - s * 0.005,
                           pts["nose_tip"].y() - s * 0.010),
                   s * 0.020, s * 0.012)

    p.setBrush(skin["shadow"])
    p.setPen(QPen(skin["deep"].darker(120), s * 0.002))
    for k in ("nose_alar_l", "nose_alar_r"):
        a = pts[k]
        p.drawEllipse(QPointF(a.x(), a.y()), s * 0.022, s * 0.013)

    p.setBrush(QColor(20, 14, 12))
    p.setPen(Qt.PenStyle.NoPen)
    for k in ("nostril_l", "nostril_r"):
        n = pts[k]
        ny = n.y() - s * 0.005 * a9
        p.drawEllipse(QPointF(n.x(), ny), s * 0.011, s * 0.006)

    if a9 > 0.25:
        pen = QPen(QColor(0, 0, 0, int(80 * a9)), s * 0.003)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        gp = QPainterPath()
        gp.moveTo(QPointF(pts["nose_root"].x() - s * 0.025,
                           pts["nose_root"].y() - s * 0.005))
        gp.quadTo(pts["nose_root"], QPointF(pts["nose_root"].x() + s * 0.025,
                                              pts["nose_root"].y() - s * 0.005))
        p.drawPath(gp)


def draw_mouth(p, pts, params, skin, au_values, box) -> None:
    s = box_scale(box)
    lip = hex_color(params.lip_color)
    lip_dark = lip.darker(140)
    lip_light = lip.lighter(120)

    upper_outer = [pts[k] for k in
                   ["lip_corner_l", "lip_upper_l2", "cupid_l",
                    "cupid_top", "cupid_r", "lip_upper_r2", "lip_corner_r"]]
    lower_outer = [pts[k] for k in
                   ["lip_corner_r", "lip_lower_r2", "lip_lower_mid",
                    "lip_lower_l2", "lip_corner_l"]]
    inner_upper = [pts[k] for k in ["inner_u_l", "inner_u_m", "inner_u_r"]]
    inner_lower = [pts[k] for k in ["inner_l_r", "inner_l_m", "inner_l_l"]]

    a25 = au_values["AU25"]
    a26 = au_values["AU26"]
    open_mouth = max(a25, a26) > 0.20

    grad = QLinearGradient(
        QPointF(0, upper_outer[3].y()),
        QPointF(0, lower_outer[2].y() + s * 0.005),
    )
    grad.setColorAt(0.0, lip_dark)
    grad.setColorAt(0.4, lip)
    grad.setColorAt(1.0, lip_light)
    p.setBrush(QBrush(grad))
    p.setPen(QPen(lip_dark.darker(115), s * 0.0035))
    p.drawPath(smooth_path(upper_outer + list(reversed(lower_outer[1:-1])),
                              close=True))

    if open_mouth:
        cavity = QPainterPath()
        cavity.moveTo(inner_upper[0])
        for q in inner_upper[1:]:
            cavity.lineTo(q)
        for q in inner_lower:
            cavity.lineTo(q)
        cavity.closeSubpath()
        p.setBrush(QColor(40, 22, 22))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(cavity)

        if a26 > 0.10:
            top = inner_upper
            bot_y = (inner_upper[1].y() + inner_lower[1].y()) / 2
            bot = [QPointF(q.x(), bot_y) for q in top]
            teeth_path = QPainterPath()
            teeth_path.moveTo(top[0])
            for q in top[1:]:
                teeth_path.lineTo(q)
            for q in reversed(bot):
                teeth_path.lineTo(q)
            teeth_path.closeSubpath()
            tg = QLinearGradient(QPointF(0, top[0].y()), QPointF(0, bot_y))
            tg.setColorAt(0.0, QColor(245, 235, 220))
            tg.setColorAt(1.0, QColor(210, 200, 180))
            p.setBrush(QBrush(tg))
            p.setPen(QPen(QColor(150, 130, 110), s * 0.002))
            p.drawPath(teeth_path)
            pen = QPen(QColor(170, 150, 130, 200), s * 0.002)
            p.setPen(pen)
            for i in range(1, len(top)):
                f = i / len(top)
                x = top[0].x() + (top[-1].x() - top[0].x()) * f
                p.drawLine(QPointF(x, top[0].y()), QPointF(x, bot_y))

    pen = QPen(lip_dark.darker(140), s * 0.0025)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    bow = QPainterPath()
    bow.moveTo(upper_outer[2])
    bow.quadTo(upper_outer[3],
                QPointF(upper_outer[4].x(), upper_outer[4].y()))
    p.drawPath(bow)


def draw_hair_back(p, pts, params, box) -> None:
    s = box_scale(box)
    hair = hex_color(params.hair_color)
    p.setBrush(QBrush(hair))
    p.setPen(Qt.PenStyle.NoPen)
    cap = QPainterPath()
    cap.moveTo(QPointF(pts["temple_l"].x() - s * 0.03,
                        pts["temple_l"].y() + s * 0.06))
    cap.quadTo(
        QPointF(pts["temple_l"].x() - s * 0.04, pts["hairline_top"].y() - s * 0.10),
        QPointF(pts["hairline_top"].x(), pts["hairline_top"].y() - s * 0.10),
    )
    cap.quadTo(
        QPointF(pts["temple_r"].x() + s * 0.04, pts["hairline_top"].y() - s * 0.10),
        QPointF(pts["temple_r"].x() + s * 0.03, pts["temple_r"].y() + s * 0.06),
    )
    cap.lineTo(pts["temple_r"])
    cap.quadTo(
        QPointF(pts["hairline_r"].x() + s * 0.01, pts["hairline_r"].y()),
        pts["hairline_top"],
    )
    cap.quadTo(
        QPointF(pts["hairline_l"].x() - s * 0.01, pts["hairline_l"].y()),
        pts["temple_l"],
    )
    cap.closeSubpath()
    p.drawPath(cap)


def draw_hair_front(p, pts, params, box) -> None:
    s = box_scale(box)
    hair = hex_color(params.hair_color)
    hair_light = hair.lighter(140)

    fringe = QPainterPath()
    fringe.moveTo(pts["temple_l"])
    fringe.cubicTo(
        QPointF(pts["temple_l"].x() + s * 0.04, pts["hairline_l"].y() + s * 0.02),
        QPointF(pts["hairline_top"].x() + s * 0.10, pts["hairline_top"].y() + s * 0.04),
        QPointF(pts["temple_r"].x() - s * 0.02, pts["hairline_r"].y() + s * 0.04),
    )
    fringe.cubicTo(
        QPointF(pts["hairline_top"].x() + s * 0.05, pts["hairline_top"].y() - s * 0.01),
        QPointF(pts["hairline_top"].x() - s * 0.05, pts["hairline_top"].y() - s * 0.01),
        pts["temple_l"],
    )
    fringe.closeSubpath()
    p.setBrush(QBrush(hair))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPath(fringe)

    pen = QPen(hair_light, s * 0.0035)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(8):
        f = i / 7.0
        x0 = pts["temple_l"].x() + (pts["temple_r"].x() - pts["temple_l"].x()) * f
        y0 = pts["hairline_top"].y() + s * 0.02 + s * 0.012 * math.sin(math.pi * f * 1.4)
        x1 = x0 + s * 0.025
        y1 = y0 - s * 0.018
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))


# Overlay + wireframe drawers live in
# ``sim_face_anatomy_overlay`` to keep this module under budget.
