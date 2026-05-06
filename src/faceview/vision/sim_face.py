"""Parametric simulated face for testing without a webcam.

Renders a stylised face into a numpy BGR array using QPainter. The face is
expressive enough to drive the full vision pipeline end-to-end: presence
(it's a face-shaped blob), mouth-activity (jaw_open varies), and emotion
(brow + smile coefficients). Identity won't classify it as the *owner* but
it will produce stable embeddings, which is the right behaviour.

The face is fully deterministic for a given :class:`FaceParams` — useful
when generating reproducible README screenshots and in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen


@dataclass
class FaceParams:
    # Pose
    yaw: float = 0.0          # -1 (left) .. 1 (right)
    pitch: float = 0.0        # -1 (up) .. 1 (down)
    # Expression knobs (0..1 unless noted)
    eye_open: float = 1.0     # 0 = closed (blink)
    jaw_open: float = 0.0
    smile: float = 0.0        # -1 (frown) .. 1 (smile)
    brow_raise: float = 0.0   # -1 (frown) .. 1 (raised)
    pupil_x: float = 0.0      # -1 .. 1
    pupil_y: float = 0.0      # -1 .. 1
    # Skin tone (just for variety)
    skin_hue: float = 28.0    # HSV-style hue 0..360
    background: str = "#0c0f14"

    # Convenience constructors
    @classmethod
    def neutral(cls) -> "FaceParams":
        return cls()

    @classmethod
    def happy(cls) -> "FaceParams":
        return cls(smile=0.85, eye_open=0.9, brow_raise=0.15)

    @classmethod
    def surprised(cls) -> "FaceParams":
        return cls(brow_raise=0.9, eye_open=1.0, jaw_open=0.55)

    @classmethod
    def sad(cls) -> "FaceParams":
        return cls(smile=-0.6, brow_raise=-0.3, eye_open=0.7)

    @classmethod
    def speaking(cls, t: float) -> "FaceParams":
        """Animated talking-mouth based on a time scalar (seconds)."""
        return cls(
            jaw_open=0.18 + 0.18 * (1 + math.sin(t * 8)) / 2,
            smile=0.12 + 0.06 * math.sin(t * 1.7),
            brow_raise=0.05 * math.sin(t * 0.5),
            pupil_x=0.15 * math.sin(t * 0.7),
        )


# ── helpers ─────────────────────────────────────────────────────────────


def _skin_color(hue: float, lightness: float = 0.78) -> QColor:
    c = QColor()
    c.setHsvF((hue / 360.0) % 1.0, 0.45, lightness, 1.0)
    return c


def _qimage_to_bgr(img: QImage) -> np.ndarray:
    """Convert a QImage to a BGR uint8 numpy array (cv2-compatible)."""
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    # QImage stores RGB; cv2 expects BGR.
    return arr[:, :, ::-1].copy()


# ── renderer ─────────────────────────────────────────────────────────────


def render_face(params: FaceParams, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Render a face with the given parameters; return a BGR uint8 array."""
    w, h = size
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(params.background))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx, cy = w / 2.0, h / 2.0 + 8 * params.pitch * 30  # tiny vertical shift
    face_w = min(w, h) * 0.38
    face_h = face_w * 1.18
    yaw_offset = params.yaw * face_w * 0.10  # parallax fudge
    cx += yaw_offset

    # ── Head silhouette ─────────────────────────────────────────────
    skin = _skin_color(params.skin_hue)
    skin_dark = _skin_color(params.skin_hue, lightness=0.62)

    grad = QLinearGradient(cx - face_w, cy - face_h, cx + face_w, cy + face_h)
    grad.setColorAt(0.0, skin.lighter(110))
    grad.setColorAt(1.0, skin_dark)
    p.setBrush(QBrush(grad))
    p.setPen(QPen(skin_dark.darker(120), 2))
    p.drawEllipse(QPointF(cx, cy), face_w, face_h)

    # ── Hair (decorative arc on top) ─────────────────────────────────
    hair = QColor("#2c1810")
    p.setBrush(QBrush(hair))
    p.setPen(Qt.PenStyle.NoPen)
    hair_rect = QRectF(cx - face_w, cy - face_h - face_h * 0.05,
                       face_w * 2, face_h * 0.65)
    p.drawChord(hair_rect, 30 * 16, 120 * 16)

    # ── Eyebrows ─────────────────────────────────────────────────────
    brow_y = cy - face_h * 0.34 - params.brow_raise * face_h * 0.06
    brow_w = face_w * 0.34
    brow_h = face_h * 0.05
    p.setBrush(QBrush(hair))
    for sign in (-1, 1):
        path = QPainterPath()
        x0 = cx + sign * face_w * 0.36 - brow_w / 2
        y0 = brow_y - sign * params.brow_raise * 4 + (1 - params.brow_raise) * 2
        path.moveTo(x0, y0)
        path.cubicTo(
            QPointF(x0 + brow_w * 0.3, y0 - brow_h * 1.1),
            QPointF(x0 + brow_w * 0.7, y0 - brow_h * 1.1),
            QPointF(x0 + brow_w, y0),
        )
        path.lineTo(x0 + brow_w, y0 + brow_h * 0.6)
        path.lineTo(x0, y0 + brow_h * 0.6)
        path.closeSubpath()
        p.drawPath(path)

    # ── Eyes ─────────────────────────────────────────────────────────
    eye_y = cy - face_h * 0.19
    eye_w = face_w * 0.20
    eye_h = face_h * 0.10 * max(0.05, params.eye_open)
    for sign in (-1, 1):
        ex = cx + sign * face_w * 0.34
        # Eye white
        p.setBrush(QBrush(QColor("#fafafa")))
        p.setPen(QPen(skin_dark.darker(110), 1.5))
        p.drawEllipse(QPointF(ex, eye_y), eye_w / 2, eye_h / 2)
        # Iris + pupil
        if params.eye_open > 0.1:
            iris_r = min(eye_w, eye_h) * 0.45
            iris_x = ex + params.pupil_x * (eye_w * 0.18)
            iris_y = eye_y + params.pupil_y * (eye_h * 0.12)
            p.setBrush(QBrush(QColor("#3a4a6c")))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(iris_x, iris_y), iris_r, iris_r)
            p.setBrush(QBrush(QColor("#0a0a0a")))
            p.drawEllipse(QPointF(iris_x, iris_y), iris_r * 0.45, iris_r * 0.45)
            # Specular highlight
            p.setBrush(QBrush(QColor("#ffffff")))
            p.drawEllipse(QPointF(iris_x - iris_r * 0.25, iris_y - iris_r * 0.25),
                          iris_r * 0.18, iris_r * 0.18)

    # ── Nose (a soft shadow stroke) ──────────────────────────────────
    p.setPen(QPen(skin_dark.darker(130), 2))
    p.setBrush(Qt.BrushStyle.NoBrush)
    nose_top = QPointF(cx - 4, cy - face_h * 0.08)
    nose_bot = QPointF(cx + 4, cy + face_h * 0.05)
    path = QPainterPath()
    path.moveTo(nose_top)
    path.cubicTo(QPointF(cx - 12, cy), QPointF(cx + 8, cy + face_h * 0.04), nose_bot)
    p.drawPath(path)

    # ── Mouth ────────────────────────────────────────────────────────
    mouth_y = cy + face_h * 0.30
    mouth_w = face_w * (0.34 + 0.06 * params.smile)
    open_h = face_h * 0.18 * params.jaw_open
    smile_offset = params.smile * face_h * 0.06

    if params.jaw_open > 0.04:
        # Open mouth — dark interior with teeth strip
        p.setBrush(QBrush(QColor("#3a1e1e")))
        p.setPen(QPen(skin_dark.darker(160), 2))
        rect = QRectF(cx - mouth_w / 2, mouth_y - open_h / 2,
                      mouth_w, open_h)
        p.drawRoundedRect(rect, open_h / 2, open_h / 2)
        # Teeth strip (only when wide enough)
        if open_h > 16:
            p.setBrush(QBrush(QColor("#f1ede1")))
            p.setPen(Qt.PenStyle.NoPen)
            t_h = max(4.0, open_h * 0.22)
            t_rect = QRectF(rect.x() + 6, rect.y() + 4,
                            rect.width() - 12, t_h)
            p.drawRoundedRect(t_rect, 2, 2)
    else:
        # Closed-mouth curve. In Y-down coords, a smile has corners ABOVE
        # the midpoint (smaller Y) and a frown has corners BELOW it.
        p.setPen(QPen(QColor("#7a2c2c"), 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        corners_y = mouth_y - smile_offset * 0.4   # smile pulls corners up
        ctrl_y = mouth_y + smile_offset            # smile pushes middle down
        left = QPointF(cx - mouth_w / 2, corners_y)
        right = QPointF(cx + mouth_w / 2, corners_y)
        ctrl = QPointF(cx, ctrl_y)
        path.moveTo(left)
        path.quadTo(ctrl, right)
        p.drawPath(path)

    # ── Cheek blush (subtle) ─────────────────────────────────────────
    p.setBrush(QBrush(QColor(220, 130, 130, 50)))
    p.setPen(Qt.PenStyle.NoPen)
    for sign in (-1, 1):
        p.drawEllipse(
            QPointF(cx + sign * face_w * 0.42, cy + face_h * 0.18),
            face_w * 0.10,
            face_h * 0.06,
        )

    p.end()
    return _qimage_to_bgr(img)
