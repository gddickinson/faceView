"""Room-map window — top-down plan view of the camera scene.

Renders a 2-D plan of objects detected in the camera's field of
view, anchored at the camera position (bottom-centre of the
canvas) with the FOV cone fanning out forward. Dots are coloured
by class, sized by recency. Tracked objects (under active
``track_object`` calls) get trails.

Opened via View → Room map… (Ctrl+Shift+Z). While visible, the
backing :class:`RoomMapWorker` runs at ~1 Hz; while hidden it
stops ticking to save CPU.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, HeadPose, RoomMap


_BG = QColor("#181b22")
_GRID = QColor("#272b35")
_FOV_FILL = QColor(80, 110, 230, 35)
_FOV_EDGE = QColor("#3a6fe0")
_CAMERA = QColor("#5e72e4")
_HEAD_ARROW = QColor("#22a36b")
_TRAIL = QColor(200, 200, 200, 80)


# Palette for object dots — same labels get the same colour across
# frames (hash-based).
_PALETTE = [
    QColor("#1a73e8"), QColor("#22a36b"), QColor("#fd7e14"),
    QColor("#c0392b"), QColor("#9b51e0"), QColor("#e8a23a"),
    QColor("#8e44ad"), QColor("#16a085"), QColor("#d63384"),
    QColor("#0d6efd"),
]


def _colour_for(label: str) -> QColor:
    return _PALETTE[abs(hash(label)) % len(_PALETTE)]


class RoomMapCanvas(QWidget):
    """Custom-painted plan view. Subscribed to ROOM_MAP + HEAD_POSE."""

    # World-units → pixels at scale 1.0.
    BASE_SCALE = 140.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self._items: list = []
        self._units = "relative"
        self._hfov_deg = 65.0
        self._head_yaw = 0.0
        # Per-label rolling trail of (x, z) tuples.
        self._trails: dict[str, list[tuple[float, float]]] = {}
        bus = get_bus()
        bus.subscribe(EventType.ROOM_MAP, self._on_room_map)
        bus.subscribe(EventType.HEAD_POSE, self._on_head_pose)

    # ── bus ─────────────────────────────────────────────────

    def _on_room_map(self, m: RoomMap) -> None:
        if m is None:
            return
        self._items = list(m.items)
        self._units = m.units
        self._hfov_deg = m.hfov_deg
        for it in self._items:
            trail = self._trails.setdefault(it.label, [])
            trail.append((it.x, it.z))
            if len(trail) > 12:
                trail.pop(0)
        self.update()

    def _on_head_pose(self, hp: HeadPose) -> None:
        if hp is None:
            return
        self._head_yaw = float(hp.yaw)
        self.update()

    # ── paint ───────────────────────────────────────────────

    def paintEvent(self, _ev) -> None:  # noqa: N802 — Qt
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        # Camera at the bottom-centre of the canvas; +z goes UP the
        # screen, +x goes right. Scale by canvas size.
        cam_x = w / 2.0
        cam_y = h - 32.0
        scale = self.BASE_SCALE * min(w / 420.0, h / 380.0)

        p.fillRect(QRectF(0, 0, w, h), _BG)
        self._paint_grid(p, cam_x, cam_y, w, h, scale)
        self._paint_fov(p, cam_x, cam_y, scale)
        self._paint_trails(p, cam_x, cam_y, scale)
        self._paint_items(p, cam_x, cam_y, scale)
        self._paint_camera(p, cam_x, cam_y)
        self._paint_legend(p, w, h)

    def _paint_grid(self, p, cx, cy, w, h, scale) -> None:
        pen = QPen(_GRID, 1)
        p.setPen(pen)
        step = scale * 0.5  # half-unit grid
        x = cx % step
        while x < w:
            p.drawLine(QPointF(x, 0), QPointF(x, h))
            x += step
        y = cy % step
        while y < h:
            p.drawLine(QPointF(0, y), QPointF(w, y))
            y += step

    def _paint_fov(self, p, cx, cy, scale) -> None:
        half = math.radians(self._hfov_deg / 2.0)
        reach = scale * 3.5
        left_x = cx - reach * math.sin(half)
        right_x = cx + reach * math.sin(half)
        far_y = cy - reach * math.cos(half)
        cone = QPolygonF([
            QPointF(cx, cy),
            QPointF(left_x, far_y),
            QPointF(right_x, far_y),
        ])
        p.setBrush(QBrush(_FOV_FILL))
        p.setPen(QPen(_FOV_EDGE, 1, Qt.PenStyle.DashLine))
        p.drawPolygon(cone)

    def _paint_trails(self, p, cx, cy, scale) -> None:
        p.setPen(QPen(_TRAIL, 1.5))
        for trail in self._trails.values():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                x0, z0 = trail[i - 1]
                x1, z1 = trail[i]
                p.drawLine(
                    QPointF(cx + x0 * scale, cy - z0 * scale),
                    QPointF(cx + x1 * scale, cy - z1 * scale),
                )

    def _paint_items(self, p, cx, cy, scale) -> None:
        font = QFont()
        font.setPointSize(9)
        p.setFont(font)
        for item in self._items:
            sx = cx + item.x * scale
            sy = cy - item.z * scale
            colour = _colour_for(item.label)
            # Fade by staleness.
            age = max(0.0, time.time() - item.last_seen_ts)
            alpha = max(80, 255 - int(age * 40))
            fill = QColor(colour)
            fill.setAlpha(alpha)
            p.setBrush(QBrush(fill))
            p.setPen(QPen(colour.darker(150), 1))
            r = 7.0
            p.drawEllipse(QPointF(sx, sy), r, r)
            # Label + distance.
            distance = math.sqrt(item.x ** 2 + item.z ** 2)
            label = f"{item.label} · {distance:.1f}"
            p.setPen(QPen(QColor("#dde2ec"), 1))
            p.drawText(QPointF(sx + 10, sy - 8), label)

    def _paint_camera(self, p, cx, cy) -> None:
        # Camera triangle at origin.
        p.setBrush(QBrush(_CAMERA))
        p.setPen(QPen(_CAMERA.lighter(130), 1))
        tri = QPolygonF([
            QPointF(cx - 7, cy + 5),
            QPointF(cx + 7, cy + 5),
            QPointF(cx, cy - 9),
        ])
        p.drawPolygon(tri)
        # Head-pose heading arrow (user looking direction).
        if self._head_yaw != 0.0:
            length = 28.0
            angle = self._head_yaw * math.pi / 2.0  # yaw in [-1,1] → ±90°
            ex = cx + length * math.sin(angle)
            ey = cy - length * math.cos(angle)
            p.setPen(QPen(_HEAD_ARROW, 2))
            p.drawLine(QPointF(cx, cy), QPointF(ex, ey))

    def _paint_legend(self, p, w, h) -> None:
        p.setPen(QPen(QColor("#7a8290"), 1))
        font = QFont()
        font.setPointSize(8)
        p.setFont(font)
        p.drawText(QPointF(8, 16), f"units: {self._units}")
        p.drawText(QPointF(8, 30), f"hfov: {self._hfov_deg:.0f}°")
        if self._items:
            p.drawText(QPointF(8, 44),
                       f"{len(self._items)} item(s) on map")
        else:
            p.drawText(QPointF(8, 44), "no items yet — waiting for OBJECTS")


class RoomMapWindow(QWidget):
    """Standalone top-level window holding the canvas + a small toolbar."""

    def __init__(self, main_window) -> None:
        super().__init__()
        self._mw = main_window
        self.setWindowTitle("faceView — Room map")
        self.resize(560, 480)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header_row = QHBoxLayout()
        header = QLabel("Room map (top-down)")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        header_row.addWidget(header, 1)
        clear_btn = QPushButton("Clear trails", self)
        clear_btn.clicked.connect(self._clear_trails)
        header_row.addWidget(clear_btn)
        root.addLayout(header_row)

        self.canvas = RoomMapCanvas(self)
        root.addWidget(self.canvas, 1)

        hint = QLabel(
            "Camera at bottom; FOV cone fans forward. Distances are "
            "in relative units until a metric calibration is set "
            "(roadmap P16)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2;")
        root.addWidget(hint)

    def _clear_trails(self) -> None:
        self.canvas._trails.clear()
        self.canvas.update()

    # ── lifecycle: gate the worker on show/hide ──────────────────

    def showEvent(self, ev) -> None:  # noqa: N802 — Qt
        super().showEvent(ev)
        worker = getattr(self._mw, "room_map_worker", None)
        if worker is not None:
            worker.set_active(True)

    def closeEvent(self, ev) -> None:  # noqa: N802 — Qt
        worker = getattr(self._mw, "room_map_worker", None)
        if worker is not None:
            worker.set_active(False)
        super().closeEvent(ev)
