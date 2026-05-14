"""Cheap whole-frame scene descriptors: brightness + motion.

Neither metric needs a model. Both run on a downscaled copy of the FRAME
(~128 px on the long edge) so each call is < 1 ms. Published at ~5 Hz
on :data:`EventType.SCENE`.

Subscribed by :class:`PerceptionStore` (and the debug panel) so the LLM
gets a tiny extra context line on every turn: *"scene: well-lit, user
is fairly still"* or *"dim room, user is moving a lot"*.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, SceneInfo
from faceview.core.logger import get_logger


log = get_logger("scene")


class SceneAnalyzer:
    def __init__(self, throttle_hz: float = 5.0) -> None:
        self._period = 1.0 / max(1.0, throttle_hz)
        self._last_emit = 0.0
        self._prev_gray = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        try:
            import cv2  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise MissingDependency("opencv-python", "vision") from exc
        if self._started:
            return
        self._started = True
        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("scene.started")

    def _on_frame(self, frame) -> None:
        if frame is None:
            return
        now = time.time()
        if now - self._last_emit < self._period:
            return
        self._last_emit = now

        try:
            import cv2  # type: ignore
        except ImportError:
            return
        # Downscale for speed — 128 px on the long edge is plenty for
        # mean luminance and motion magnitude.
        h, w = frame.shape[:2]
        scale = 128.0 / max(h, w)
        if scale < 1.0:
            small = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean()) / 255.0
        motion = 0.0
        with self._lock:
            prev = self._prev_gray
            self._prev_gray = gray
        if prev is not None and prev.shape == gray.shape:
            diff = cv2.absdiff(prev, gray)
            motion = float(diff.mean()) / 255.0
            # Motion typically lives in [0, 0.05]. Rescale into 0..1.
            motion = min(1.0, motion * 20.0)

        get_bus().publish(
            EventType.SCENE,
            SceneInfo(
                brightness=brightness,
                brightness_label=_brightness_label(brightness),
                motion=motion,
                motion_label=_motion_label(motion),
            ),
        )


def _brightness_label(b: float) -> str:
    if b < 0.18:
        return "dark"
    if b < 0.35:
        return "dim"
    if b < 0.70:
        return "lit"
    return "bright"


def _motion_label(m: float) -> str:
    if m < 0.05:
        return "still"
    if m < 0.20:
        return "moving"
    return "active"
