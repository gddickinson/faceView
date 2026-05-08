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
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)


@dataclass
class FaceParams:
    """Renderer input. Most fields are 0..1 unless noted.

    Coarse fields (``smile``, ``brow_raise``, ``jaw_open``) preserve backward
    compatibility with hand-built scenes. The "AU-grade" fields below let the
    renderer distinguish visemes (``mouth_pucker`` vs ``mouth_stretch``) and
    show subtle expression cues (``cheek_raise``, ``upper_lid_raise``, the
    individual brow components). When in doubt :func:`face_state_to_params`
    populates them all from FACS Action Units.
    """

    # Pose
    yaw: float = 0.0          # -1 (left) .. 1 (right)
    pitch: float = 0.0        # -1 (up) .. 1 (down)

    # Coarse expression knobs
    eye_open: float = 1.0     # 0 = closed (blink)
    jaw_open: float = 0.0     # 0..1
    smile: float = 0.0        # -1 (frown) .. 1 (smile)
    brow_raise: float = 0.0   # -1 (frown) .. 1 (raised) — master fallback
    pupil_x: float = 0.0      # -1 .. 1
    pupil_y: float = 0.0      # -1 .. 1

    # Appearance
    skin_hue: float = 28.0    # HSV-style hue 0..360
    background: str = "#0c0f14"
    hair_color: str = "#2c1810"
    lip_color: str = "#a44a4a"

    # Renderer selection. "stylised" = the layered cartoony pipeline below.
    # "anatomical" / "anatomy_overlay" / "wireframe" route to
    # ``vision.sim_face_anatomical.render_face_anatomical`` instead.
    render_mode: str = "stylised"

    # ICT-FaceKit identity coefficients (PCA modes that vary base
    # face shape). Empty dict = use the neutral mean face.
    identity_weights: dict = None

    # AU-grade detail (all 0..1, defaults 0 → ignored by old call-sites)
    mouth_pucker: float = 0.0       # AU22 lip funneler — pulls lips inward and forward
    mouth_stretch: float = 0.0      # AU20 lip stretch — widens the mouth horizontally
    cheek_raise: float = 0.0        # AU6 cheek raise — lifts cheeks, narrows lower-eye
    nose_wrinkle: float = 0.0       # AU9 nose wrinkle — bunches skin around nose
    upper_lid_raise: float = 0.0    # AU5 upper lid raise — eye-widening
    inner_brow_raise: float = 0.0   # AU1 — inner brow tips lift (sad/surprised)
    outer_brow_raise: float = 0.0   # AU2 — outer brow tips lift (surprise)
    brow_lower: float = 0.0         # AU4 — both brows down + together (anger/concentration)
    lip_corner_drop: float = 0.0    # AU15 — lip corners drop (sadness)
    chin_raise: float = 0.0         # AU17 — chin raises, lower lip up (pout)
    upper_lip_raise: float = 0.0    # AU10 — upper lip pulled up (snarl / disgust)
    dimpler: float = 0.0            # AU14 — corner pulled inward (smirk)
    lip_tighten: float = 0.0        # AU23 — lips drawn tight, narrowed
    lip_press: float = 0.0          # AU24 — lips pressed together

    # Convenience constructors
    @classmethod
    def neutral(cls) -> "FaceParams":
        return cls()

    @classmethod
    def happy(cls) -> "FaceParams":
        return cls(smile=0.85, eye_open=0.9, brow_raise=0.15, cheek_raise=0.7)

    @classmethod
    def surprised(cls) -> "FaceParams":
        return cls(
            brow_raise=0.9, eye_open=1.0, jaw_open=0.55,
            inner_brow_raise=0.9, outer_brow_raise=0.9, upper_lid_raise=0.8,
        )

    @classmethod
    def sad(cls) -> "FaceParams":
        return cls(
            smile=-0.6, brow_raise=-0.3, eye_open=0.7,
            inner_brow_raise=0.7, lip_corner_drop=0.8,
        )

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


def _draw_background(p: QPainter, w: int, h: int, params: FaceParams) -> None:
    """Soft radial vignette so the head reads as foreground."""
    bg = QColor(params.background)
    grad = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) * 0.7)
    grad.setColorAt(0.0, bg.lighter(125))
    grad.setColorAt(1.0, bg.darker(120))
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRect(0, 0, w, h)


def _draw_ears(p: QPainter, ctx: dict, params: FaceParams) -> None:
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    skin = ctx["skin"]
    skin_dark = ctx["skin_dark"]
    for sign in (-1, 1):
        ex = cx + sign * fw * 0.99
        ey = cy + fh * 0.02
        p.setBrush(QBrush(skin))
        p.setPen(QPen(skin_dark.darker(115), 1.2))
        p.drawEllipse(QPointF(ex, ey), fw * 0.10, fh * 0.14)
        # Inner ear shadow.
        p.setBrush(QBrush(skin_dark.darker(135)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(ex, ey + fh * 0.01), fw * 0.04, fh * 0.07)


def _draw_head_skin(p: QPainter, ctx: dict, params: FaceParams) -> None:
    """Skin layer — head silhouette with rim light + side shading."""
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    skin = ctx["skin"]
    skin_dark = ctx["skin_dark"]

    # 1. Forward-lit skin gradient (light from upper-left).
    grad = QRadialGradient(
        QPointF(cx - fw * 0.20, cy - fh * 0.30),
        max(fw, fh) * 1.6,
    )
    grad.setColorAt(0.0, skin.lighter(115))
    grad.setColorAt(0.55, skin)
    grad.setColorAt(1.0, skin_dark.darker(110))
    p.setBrush(QBrush(grad))
    p.setPen(QPen(skin_dark.darker(125), 1.5))
    p.drawEllipse(QPointF(cx, cy), fw, fh)

    # 2. Side-shadow on the right of the face (gives 3D fall-off).
    side = QLinearGradient(QPointF(cx, cy), QPointF(cx + fw, cy))
    side.setColorAt(0.0, QColor(0, 0, 0, 0))
    side.setColorAt(0.7, QColor(0, 0, 0, 0))
    side.setColorAt(1.0, QColor(0, 0, 0, 50))
    p.setBrush(QBrush(side))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(QPointF(cx, cy), fw, fh)



def _draw_hair(p: QPainter, ctx: dict, params: FaceParams) -> None:
    """A swept fringe and a back-of-head cap, with a few stroke highlights."""
    cx = ctx["cx"]
    cy = ctx["cy"]
    fw = ctx["face_w"]
    fh = ctx["face_h"]
    hair = QColor(params.hair_color)
    hair_light = hair.lighter(135)

    # 1. Cap (back of head — slightly larger than the skull).
    p.setBrush(QBrush(hair))
    p.setPen(Qt.PenStyle.NoPen)
    cap_path = QPainterPath()
    cap_path.moveTo(QPointF(cx - fw * 1.05, cy + fh * 0.10))
    cap_path.quadTo(
        QPointF(cx - fw * 1.10, cy - fh * 0.95),
        QPointF(cx, cy - fh * 1.05),
    )
    cap_path.quadTo(
        QPointF(cx + fw * 1.10, cy - fh * 0.95),
        QPointF(cx + fw * 1.05, cy + fh * 0.10),
    )
    cap_path.quadTo(
        QPointF(cx + fw * 0.85, cy - fh * 0.40),
        QPointF(cx + fw * 0.55, cy - fh * 0.55),
    )
    cap_path.quadTo(
        QPointF(cx, cy - fh * 0.95),
        QPointF(cx - fw * 0.55, cy - fh * 0.55),
    )
    cap_path.quadTo(
        QPointF(cx - fw * 0.85, cy - fh * 0.40),
        QPointF(cx - fw * 1.05, cy + fh * 0.10),
    )
    cap_path.closeSubpath()
    p.drawPath(cap_path)

    # 2. Swept fringe over the forehead — a thin curving wedge.
    fringe = QPainterPath()
    fringe.moveTo(QPointF(cx - fw * 0.85, cy - fh * 0.55))
    fringe.cubicTo(
        QPointF(cx - fw * 0.45, cy - fh * 0.78),
        QPointF(cx + fw * 0.30, cy - fh * 0.72),
        QPointF(cx + fw * 0.55, cy - fh * 0.50),
    )
    fringe.cubicTo(
        QPointF(cx + fw * 0.20, cy - fh * 0.60),
        QPointF(cx - fw * 0.25, cy - fh * 0.55),
        QPointF(cx - fw * 0.85, cy - fh * 0.55),
    )
    fringe.closeSubpath()
    p.setBrush(QBrush(hair))
    p.drawPath(fringe)

    # 3. A few hair-strand highlights for texture.
    pen = QPen(hair_light, 1.5)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(7):
        f = i / 6.0
        x0 = cx - fw * 0.85 + f * fw * 1.40
        y0 = cy - fh * 0.55 - 4 + 6 * math.sin(math.pi * f * 1.2)
        x1 = x0 + 18
        y1 = y0 - 14 + 4 * math.sin(math.pi * f)
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))


# ── renderer ─────────────────────────────────────────────────────────────


def render_face(params: FaceParams, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    """Render a face with the given parameters; return a BGR uint8 array.

    Dispatches to the anatomical renderer when ``params.render_mode`` is
    one of ``anatomical`` / ``anatomy_overlay`` / ``wireframe``. Defaults
    to the layered stylised pipeline (the cartoony look used since v0.1).
    """
    mode = getattr(params, "render_mode", "stylised") or "stylised"
    if mode in ("anatomical", "anatomy_overlay", "wireframe"):
        from faceview.vision.sim_face_anatomical import render_face_anatomical
        return render_face_anatomical(params, size, mode=mode)  # type: ignore[arg-type]
    if mode.startswith("anatomy_") and mode in (
        "anatomy_layers", "anatomy_skull", "anatomy_brain",
        "anatomy_muscles", "anatomy_xray", "anatomy_eyeballs",
    ):
        from faceview.vision.sim_face_layered import render_face_layered
        return render_face_layered(params, size, layers=mode)
    if mode == "faceforge_3d":
        from faceview.vision.faceforge_bridge import render_face_faceforge
        return render_face_faceforge(params, size)
    if mode == "faceforge_3d_gpu":
        from faceview.vision.gpu_renderer import render_face_faceforge_gpu
        return render_face_faceforge_gpu(params, size)
    if mode == "face_warp_2d":
        from faceview.vision.face_warp import render_face_warp
        return render_face_warp(params, size)
    if mode == "head_decimated_3d":
        from faceview.vision.head_decimated import render_face_decimated
        return render_face_decimated(params, size)
    if mode == "head_decimated_3d_gpu":
        from faceview.vision.head_decimated import render_face_decimated_gpu
        return render_face_decimated_gpu(params, size)
    if mode == "face_warp_3d":
        from faceview.vision.face_warp_atlas import render_face_warp_atlas
        return render_face_warp_atlas(params, size)
    if mode == "makehuman_3d":
        from faceview.vision.makehuman_mesh import render_face_makehuman
        return render_face_makehuman(params, size)
    if mode == "ict_face_3d":
        from faceview.vision.ict_face import render_face_ict
        return render_face_ict(params, size)
    if mode == "bfm_3d":
        from faceview.vision.bfm_face import render_face_bfm
        return render_face_bfm(params, size)
    if mode == "rpm_3d":
        from faceview.vision.rpm_avatar import render_face_rpm
        return render_face_rpm(params, size)
    if mode == "flame_3d":
        from faceview.vision.flame_face import render_face_flame
        return render_face_flame(params, size)
    if mode == "metahuman_3d":
        from faceview.vision.metahuman_face import render_face_metahuman
        return render_face_metahuman(params, size)
    if mode == "facescape_3d":
        from faceview.vision.facescape_face import render_face_facescape
        return render_face_facescape(params, size)

    from faceview.vision.sim_face_parts import (
        draw_brows,
        draw_cheeks,
        draw_eyes,
        draw_mouth,
        draw_nose,
        skin_color,
    )

    w, h = size
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(QColor(params.background))

    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    _draw_background(p, w, h, params)

    # Geometry context shared by helpers.
    cx, cy = w / 2.0, h / 2.0 + params.pitch * h * 0.04
    face_w = min(w, h) * 0.38
    face_h = face_w * 1.18
    cx += params.yaw * face_w * 0.10  # parallax fudge

    skin = skin_color(params.skin_hue)
    skin_dark = skin_color(params.skin_hue, lightness=0.62)

    ctx = {
        "cx": cx, "cy": cy,
        "face_w": face_w, "face_h": face_h,
        "skin": skin, "skin_dark": skin_dark,
    }

    # Layered rendering: ears → head skin → cheeks → hair → brows → eyes →
    # nose → mouth. Cheek apples sit under the hair so the fringe can crop them.
    _draw_ears(p, ctx, params)
    _draw_head_skin(p, ctx, params)
    draw_cheeks(p, ctx, params)
    _draw_hair(p, ctx, params)
    draw_brows(p, ctx, params)
    draw_eyes(p, ctx, params)
    draw_nose(p, ctx, params)
    draw_mouth(p, ctx, params)

    p.end()
    return _qimage_to_bgr(img)
