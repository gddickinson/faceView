"""QPainter helpers for individual face parts.

Kept separate from :mod:`faceview.vision.sim_face` so each renderer stays
under the project's 500-line file budget. Every helper takes a ``QPainter``
plus a small geometry context dict and the relevant :class:`FaceParams`.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient


# ── Color helpers ───────────────────────────────────────────────────────


def skin_color(hue: float, lightness: float = 0.78, saturation: float = 0.45) -> QColor:
    c = QColor()
    c.setHsvF((hue / 360.0) % 1.0, saturation, lightness, 1.0)
    return c


def darken(c: QColor, factor: int = 130) -> QColor:
    return c.darker(factor)


def lighten(c: QColor, factor: int = 110) -> QColor:
    return c.lighter(factor)


# ── Brows ───────────────────────────────────────────────────────────────


def draw_brows(p: QPainter, ctx: dict, params) -> None:
    """Stroke-based eyebrows that respond to AU1/AU2/AU4.

    Each brow is six short, slightly-overlapping bristles drawn at an angle
    influenced by the inner/outer brow raise and the brow lowerer (AU4).
    """
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    hair_color = QColor(params.hair_color)

    base_y = cy - fh * 0.36
    # AU1 lifts the inner end, AU2 lifts the outer end. AU4 (brow_lower) drops both.
    inner_lift = params.inner_brow_raise * fh * 0.05
    outer_lift = params.outer_brow_raise * fh * 0.04
    overall = params.brow_raise * fh * 0.04
    drop = params.brow_lower * fh * 0.05

    n_strokes = 12

    for sign in (-1, 1):
        # Inner & outer endpoint positions; sign = -1 left brow, +1 right.
        inner_x = cx + sign * fw * 0.18
        outer_x = cx + sign * fw * 0.46
        inner_y = base_y - inner_lift - overall + drop
        outer_y = base_y - outer_lift - overall + drop * 0.6
        # Slight arch — the highest point sits about 35% from the inner end.
        arch_x = inner_x + (outer_x - inner_x) * 0.40
        arch_y = (inner_y + outer_y) / 2 - fh * 0.025

        # AU4 also pulls the brows toward each other (frown).
        inner_x += -sign * params.brow_lower * fw * 0.025

        # First, a single solid stroke giving the brow its body. Strokes for
        # texture sit on top.
        body_pen = QPen(hair_color, fh * 0.018)
        body_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(body_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        body = QPainterPath()
        body.moveTo(QPointF(inner_x, inner_y + 2))
        body.quadTo(QPointF(arch_x, arch_y), QPointF(outer_x, outer_y))
        p.drawPath(body)

        # Hair strokes that lie nearly flat along the brow direction.
        stroke_pen = QPen(hair_color.lighter(110), 1.4)
        stroke_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(stroke_pen)
        for i in range(n_strokes):
            f = i / max(1, n_strokes - 1)
            # Quadratic interpolation along the brow curve.
            x = (1 - f) ** 2 * inner_x + 2 * (1 - f) * f * arch_x + f ** 2 * outer_x
            y = (1 - f) ** 2 * inner_y + 2 * (1 - f) * f * arch_y + f ** 2 * outer_y
            # Tangent direction (derivative of the quadratic curve).
            tx = 2 * (1 - f) * (arch_x - inner_x) + 2 * f * (outer_x - arch_x)
            ty = 2 * (1 - f) * (arch_y - inner_y) + 2 * f * (outer_y - arch_y)
            length = math.hypot(tx, ty)
            if length < 1e-3:
                continue
            tx /= length
            ty /= length
            stroke_len = fh * 0.020
            p.drawLine(
                QPointF(x - tx * stroke_len * 0.4, y - ty * stroke_len * 0.4),
                QPointF(x + tx * stroke_len * 0.6, y + ty * stroke_len * 0.6),
            )

    # Glabellar furrow (between brows) when AU4 is strong.
    if params.brow_lower > 0.4:
        pen = QPen(QColor(60, 40, 40, int(120 * params.brow_lower)))
        pen.setWidthF(1.5)
        p.setPen(pen)
        gy = base_y - overall + drop * 0.5
        p.drawLine(QPointF(cx - 2, gy + 4), QPointF(cx - 2, gy + fh * 0.05))
        p.drawLine(QPointF(cx + 4, gy + 4), QPointF(cx + 4, gy + fh * 0.05))


# ── Eyes ────────────────────────────────────────────────────────────────


def draw_eyes(p: QPainter, ctx: dict, params) -> None:
    """Almond-shaped eyes with separate upper/lower lids responding to AU5/AU6.

    Drawing order: white sclera → iris → pupil → specular → lashes → upper lid.
    The lids are filled with skin tone so they overdraw the iris/sclera and
    produce the right shape when ``eye_open`` is small (blink) or
    ``cheek_raise`` is high (smiling squint).
    """
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    skin = ctx["skin"]
    skin_dark = ctx["skin_dark"]

    eye_y = cy - fh * 0.18
    eye_w = fw * 0.21
    base_h = fh * 0.10
    # AU5 widens the eye (upper lid raise); AU6 narrows from below (cheek raise).
    open_factor = max(0.05, params.eye_open) + 0.20 * params.upper_lid_raise
    eye_h = base_h * min(1.4, open_factor)
    bottom_lift = params.cheek_raise * fh * 0.020

    for sign in (-1, 1):
        ex = cx + sign * fw * 0.30

        # Almond shape: two arcs joined at the corners.
        almond = _almond_path(ex, eye_y, eye_w, eye_h, bottom_lift)

        # 1. Sclera (eye-white) — fill the almond.
        sclera = QColor("#fafafa")
        p.setBrush(QBrush(sclera))
        p.setPen(QPen(skin_dark.darker(110), 1.2))
        p.drawPath(almond)

        if params.eye_open > 0.18:
            # 2. Iris with radial gradient.
            iris_r = min(eye_w * 0.40, eye_h * 0.55)
            iris_x = ex + params.pupil_x * (eye_w * 0.18)
            iris_y = eye_y + params.pupil_y * (eye_h * 0.10)

            iris_grad = QRadialGradient(QPointF(iris_x - iris_r * 0.2, iris_y - iris_r * 0.2), iris_r * 1.4)
            iris_grad.setColorAt(0.0, QColor("#5d7aae"))
            iris_grad.setColorAt(0.6, QColor("#3a4a6c"))
            iris_grad.setColorAt(1.0, QColor("#1f2a45"))

            # Clip iris/pupil to the almond so they don't peek out at the corners.
            p.save()
            p.setClipPath(almond)
            p.setBrush(QBrush(iris_grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(iris_x, iris_y), iris_r, iris_r)

            # 3. Pupil.
            p.setBrush(QBrush(QColor("#0a0a0a")))
            p.drawEllipse(QPointF(iris_x, iris_y), iris_r * 0.40, iris_r * 0.40)

            # 4. Specular highlight.
            p.setBrush(QBrush(QColor(255, 255, 255, 220)))
            p.drawEllipse(
                QPointF(iris_x - iris_r * 0.30, iris_y - iris_r * 0.32),
                iris_r * 0.18,
                iris_r * 0.18,
            )
            p.restore()

        # 5. Cheek-raise lid line on the bottom — subtle skin-fold above cheek.
        if params.cheek_raise > 0.2:
            pen = QPen(skin_dark.darker(115), 1.2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            crease = QPainterPath()
            crease.moveTo(QPointF(ex - eye_w / 2, eye_y + eye_h / 2 - bottom_lift + 1))
            crease.quadTo(
                QPointF(ex, eye_y + eye_h / 2 - bottom_lift - fh * 0.005 * params.cheek_raise),
                QPointF(ex + eye_w / 2, eye_y + eye_h / 2 - bottom_lift + 1),
            )
            p.drawPath(crease)

        # 6. Eyelashes — thin strokes along the upper edge, only when open.
        if params.eye_open > 0.4:
            lash_pen = QPen(QColor("#1a0d0d"))
            lash_pen.setWidthF(1.4)
            lash_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            p.setPen(lash_pen)
            n = 6
            for i in range(n):
                f = i / (n - 1)
                lx = ex - eye_w / 2 + f * eye_w
                ly = eye_y - eye_h / 2 + math.sin(math.pi * f) * (-fh * 0.005)
                # Lash angles: outer lashes flick outward; inner stay vertical.
                ang = -0.55 + 1.1 * f
                p.drawLine(
                    QPointF(lx, ly),
                    QPointF(lx + math.sin(ang) * sign * 4.5, ly - math.cos(ang) * fh * 0.012),
                )


def _almond_path(cx: float, cy: float, w: float, h: float, bottom_lift: float) -> QPainterPath:
    """Build an almond-shaped path centred at (cx, cy)."""
    half_w = w / 2
    half_h = h / 2
    p = QPainterPath()
    left = QPointF(cx - half_w, cy)
    right = QPointF(cx + half_w, cy)
    top_ctl = QPointF(cx, cy - half_h * 1.2)
    bot_ctl = QPointF(cx, cy + half_h - bottom_lift)
    p.moveTo(left)
    p.quadTo(top_ctl, right)
    p.quadTo(bot_ctl, left)
    p.closeSubpath()
    return p


# ── Cheeks (apples + blush) ─────────────────────────────────────────────


def draw_cheeks(p: QPainter, ctx: dict, params) -> None:
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]

    # AU6 cheek raise — apples that lift toward the eyes.
    apple_y = cy + fh * 0.10 - params.cheek_raise * fh * 0.06
    apple_alpha = int(40 + 80 * max(0.0, max(params.cheek_raise, params.smile)))
    apple_r = fw * 0.16

    for sign in (-1, 1):
        x = cx + sign * fw * 0.42
        rad = QRadialGradient(QPointF(x, apple_y), apple_r)
        rad.setColorAt(0.0, QColor(220, 130, 130, apple_alpha))
        rad.setColorAt(1.0, QColor(220, 130, 130, 0))
        p.setBrush(QBrush(rad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(x, apple_y), apple_r, apple_r * 0.65)


# ── Nose ────────────────────────────────────────────────────────────────


def draw_nose(p: QPainter, ctx: dict, params) -> None:
    cx = ctx["cx"]
    cy = ctx["cy"]
    fh = ctx["face_h"]
    skin_dark = ctx["skin_dark"]

    # Bridge line (subtle).
    pen = QPen(skin_dark.darker(125), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    bridge_top = QPointF(cx - 5, cy - fh * 0.08)
    bridge_bot = QPointF(cx + 4, cy + fh * 0.05)
    path = QPainterPath()
    path.moveTo(bridge_top)
    path.cubicTo(QPointF(cx - 12, cy), QPointF(cx + 8, cy + fh * 0.04), bridge_bot)
    p.drawPath(path)

    # Tip roundness — small dark crescent under nose for a tip shadow.
    p.setBrush(QBrush(skin_dark.darker(115)))
    p.setPen(Qt.PenStyle.NoPen)
    tip_r = fh * 0.02
    p.drawEllipse(QPointF(cx + 3, cy + fh * 0.06), tip_r * 1.4, tip_r * 0.7)

    # Nostrils — small dark ovals; AU9 (nose wrinkle) flares them slightly.
    nostril_r = fh * 0.012 * (1 + 0.5 * params.nose_wrinkle)
    p.setBrush(QBrush(QColor(40, 25, 25)))
    for sign in (-1, 1):
        nx = cx + sign * fh * 0.025
        ny = cy + fh * 0.075
        p.drawEllipse(QPointF(nx, ny), nostril_r, nostril_r * 0.65)

    # Nose-wrinkle bunch — a small dark crease at the bridge top.
    if params.nose_wrinkle > 0.3:
        pen = QPen(QColor(60, 40, 40, int(120 * params.nose_wrinkle)))
        pen.setWidthF(1.2)
        p.setPen(pen)
        p.drawLine(QPointF(cx - 6, cy - fh * 0.10), QPointF(cx + 6, cy - fh * 0.10))


# ── Mouth ───────────────────────────────────────────────────────────────


def draw_mouth(p: QPainter, ctx: dict, params) -> None:
    """Lip-aware mouth that distinguishes visemes.

    The mouth is built from a smoothly-curved upper-lip top (Cupid's bow)
    and a lower-lip bottom. The interior gap between them is the open mouth.
    Width and height are driven by ``mouth_pucker`` (AU22), ``mouth_stretch``
    (AU20), ``smile``, ``lip_corner_drop`` (AU15), and ``jaw_open``.
    """
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    skin_dark = ctx["skin_dark"]
    lip = QColor(params.lip_color)
    lip_dark = lip.darker(120)
    lip_light = lip.lighter(115)

    mouth_y = cy + fh * 0.32

    # Width: base, widened by stretch+smile, narrowed by pucker.
    w_factor = (1.0 + 0.45 * params.mouth_stretch + 0.25 * max(0.0, params.smile)) \
               * (1.0 - 0.55 * params.mouth_pucker)
    mouth_w = fw * 0.30 * max(0.45, w_factor)

    # Vertical opening from jaw drop.
    open_h = fh * 0.18 * params.jaw_open

    # Corner offset: smile lifts corners up; lip_corner_drop pulls them down.
    # The two contributions overlap on a sad face — combine but cap the total
    # so the lip line stays gracefully curved rather than over-extended.
    smile_pos = max(0.0, params.smile)
    smile_neg = max(0.0, -params.smile)
    corner_dy = -params.smile * fh * 0.075 + params.lip_corner_drop * fh * 0.040
    corner_dy = max(-fh * 0.090, min(fh * 0.060, corner_dy))
    # Middle-of-mouth offset — for a smile the *centre* sags below the
    # corners (∪). For a frown a slight inversion (∩) is enough.
    mid_dy = (
        smile_pos * fh * 0.050
        - smile_neg * fh * 0.014
        - params.lip_corner_drop * fh * 0.012
    )
    # Reduce the cupid's bow apex height for frowns so the upper lip stays
    # rounded rather than pointing up like a triangle.
    upper_h_scale = 1.0 - 0.75 * smile_neg - 0.55 * params.lip_corner_drop
    upper_h_scale = max(0.30, upper_h_scale)

    half_w = mouth_w / 2
    cl = QPointF(cx - half_w, mouth_y + corner_dy)   # left corner
    cr = QPointF(cx + half_w, mouth_y + corner_dy)   # right corner
    mid_y = mouth_y + mid_dy                          # centre of the lip line

    # Upper lip thickness — fatter when puckered, flatter when frowning.
    upper_h = fh * 0.030 * (1 + 0.6 * params.mouth_pucker) * upper_h_scale
    # Cupid's bow dip — small dip in the middle of the upper lip's top edge.
    bow_dip = fh * 0.012 * (1 - 0.4 * params.mouth_stretch)

    # Lower lip thickness — also fatter when puckered.
    lower_h = fh * 0.040 * (1 + 0.5 * params.mouth_pucker)

    if open_h > 4:
        # Open mouth: upper lip top curve, interior, teeth, lower lip.
        # The upper-lip top runs from cl, up over (mid_y - upper_h) with a
        # cupid's bow, then to cr.
        upper_top_y = mid_y - upper_h
        upper_top = QPainterPath()
        upper_top.moveTo(cl)
        upper_top.cubicTo(
            QPointF(cx - half_w * 0.55, mouth_y + corner_dy - upper_h * 0.85),
            QPointF(cx - half_w * 0.18, upper_top_y),
            QPointF(cx, upper_top_y + bow_dip),
        )
        upper_top.cubicTo(
            QPointF(cx + half_w * 0.18, upper_top_y),
            QPointF(cx + half_w * 0.55, mouth_y + corner_dy - upper_h * 0.85),
            cr,
        )
        # Underside of the upper lip — follows the lip line (cl → mid → cr).
        upper_inner = QPainterPath()
        upper_inner.moveTo(cl)
        upper_inner.quadTo(QPointF(cx, mid_y - 1), cr)
        upper = QPainterPath(upper_top)
        rev = QPainterPath(upper_inner)
        upper.connectPath(rev.toReversed())
        upper.closeSubpath()

        p.setBrush(QBrush(lip))
        p.setPen(QPen(lip_dark, 1))
        p.drawPath(upper)

        # Mouth interior — between upper-lip-bottom and lower-lip-top.
        bot_top_mid = mid_y + open_h
        interior = QPainterPath()
        interior.moveTo(cl)
        interior.quadTo(QPointF(cx, mid_y - 1), cr)
        interior.quadTo(QPointF(cx, bot_top_mid - 1), cl)
        interior.closeSubpath()
        p.setBrush(QBrush(QColor("#3a1a1a")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(interior)

        # Upper teeth strip.
        if open_h > 14 and params.mouth_pucker < 0.6:
            teeth_w = mouth_w * 0.78
            teeth_h = max(4.0, min(open_h * 0.35, fh * 0.025))
            tx = cx - teeth_w / 2
            ty = mid_y + 1
            grad = QLinearGradient(0, ty, 0, ty + teeth_h)
            grad.setColorAt(0.0, QColor("#fbf6e8"))
            grad.setColorAt(1.0, QColor("#d8cfb3"))
            p.setBrush(QBrush(grad))
            p.setPen(QPen(QColor("#a89a76"), 1))
            p.drawRoundedRect(QRectF(tx, ty, teeth_w, teeth_h), 3, 3)
            p.setPen(QPen(QColor("#b6a988"), 1))
            for i in range(1, 6):
                xx = tx + teeth_w * i / 6
                p.drawLine(QPointF(xx, ty + 1), QPointF(xx, ty + teeth_h - 1))

        # Lower lip — upper edge follows the bottom of the interior, lower
        # edge bulges down by lower_h.
        lower = QPainterPath()
        lower.moveTo(cl)
        lower.quadTo(QPointF(cx, bot_top_mid - 1), cr)
        lower.cubicTo(
            QPointF(cx + half_w * 0.55, bot_top_mid + lower_h * 1.0),
            QPointF(cx - half_w * 0.55, bot_top_mid + lower_h * 1.0),
            cl,
        )
        lower.closeSubpath()
        grad = QLinearGradient(0, bot_top_mid, 0, bot_top_mid + lower_h)
        grad.setColorAt(0.0, lip_light)
        grad.setColorAt(0.7, lip)
        grad.setColorAt(1.0, lip_dark)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(lip_dark, 1))
        p.drawPath(lower)

    else:
        # Closed mouth — upper lip (cupid's bow) + lower lip + lip line.
        upper = QPainterPath()
        upper.moveTo(cl)
        upper.cubicTo(
            QPointF(cx - half_w * 0.55, mouth_y + corner_dy - upper_h * 0.85),
            QPointF(cx - half_w * 0.18, mid_y - upper_h),
            QPointF(cx, mid_y - upper_h + bow_dip),
        )
        upper.cubicTo(
            QPointF(cx + half_w * 0.18, mid_y - upper_h),
            QPointF(cx + half_w * 0.55, mouth_y + corner_dy - upper_h * 0.85),
            cr,
        )
        # Close along the lip line through (cx, mid_y).
        upper.quadTo(QPointF(cx, mid_y), cl)
        upper.closeSubpath()
        p.setBrush(QBrush(lip))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(upper)

        # Lower lip — top edge along the lip line, bottom edge bulges down.
        lower = QPainterPath()
        lower.moveTo(cl)
        lower.quadTo(QPointF(cx, mid_y), cr)
        lower.cubicTo(
            QPointF(cx + half_w * 0.55, mid_y + lower_h * 1.0),
            QPointF(cx - half_w * 0.55, mid_y + lower_h * 1.0),
            cl,
        )
        lower.closeSubpath()
        grad = QLinearGradient(0, mid_y, 0, mid_y + lower_h)
        grad.setColorAt(0.0, lip_light)
        grad.setColorAt(0.8, lip)
        grad.setColorAt(1.0, lip_dark)
        p.setBrush(QBrush(grad))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPath(lower)

        # Lip line between upper and lower lips — bends with the smile.
        pen = QPen(lip_dark.darker(140), 1.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        line = QPainterPath()
        line.moveTo(cl)
        line.quadTo(QPointF(cx, mid_y), cr)
        p.drawPath(line)

    # Subtle chin/lip shadow — small, low-alpha; just under the lower lip.
    shadow = QRadialGradient(QPointF(cx, mouth_y + fh * 0.10), fw * 0.16)
    shadow.setColorAt(0.0, QColor(0, 0, 0, 22))
    shadow.setColorAt(1.0, QColor(0, 0, 0, 0))
    p.setBrush(QBrush(shadow))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(cx, mouth_y + fh * 0.10), fw * 0.14, fh * 0.04)
