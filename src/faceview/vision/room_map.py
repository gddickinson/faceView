"""Room-map worker — top-down plan of detected objects via MiDaS.

For each detected object (from :data:`EventType.OBJECTS`), we sample
the MiDaS depth at the bbox centre and project the pixel through
estimated camera intrinsics into a camera-relative 2-D point. A
short EMA on the per-label position keeps the dots from jittering.

Why monocular depth + projection instead of true SfM? SfM needs
camera motion which a static webcam doesn't have. Monocular depth
is rougher (relative units until calibrated) but works from any
single frame and the per-frame consistency is good enough for "the
cup is over there, the laptop is over there".

The worker runs on its own thread at a slow cadence (~1 Hz by
default) — MiDaS-small is ~80 MB on first load and ~150 ms per
inference on Apple Silicon. To save CPU, it only ticks when there
are subscribers on :data:`EventType.ROOM_MAP` (i.e. the room-map
panel is open). Subscribers register via :func:`set_active` from the
panel's show/hide events.

Distances are reported in **relative units** by default. A future
calibration step (P16) will multiply by a learned scale factor so
the panel can render them as metres.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Optional

import numpy as np

from faceview.core.event_bus import get_bus
from faceview.core.events import (
    EventType, ObjectsSeen, RoomMap, RoomMapItem,
)
from faceview.core.logger import get_logger


log = get_logger("room_map")


_DEFAULT_HFOV_DEG = 65.0
_DEFAULT_INTERVAL_S = 1.0
_EMA_ALPHA = 0.4
# Drop an item from the map this many seconds after we last saw it.
_STALE_AFTER_S = 5.0


def _hfov_from_env() -> float:
    raw = os.environ.get("FACEVIEW_CAMERA_HFOV_DEG")
    if not raw:
        return _DEFAULT_HFOV_DEG
    try:
        v = float(raw)
        return max(30.0, min(120.0, v))
    except ValueError:
        return _DEFAULT_HFOV_DEG


def _project(
    cx_px: float, cy_px: float, distance: float,
    frame_w: int, frame_h: int, hfov_rad: float,
) -> tuple[float, float, float]:
    """Project (image_x, image_y, distance) into camera-relative XYZ.

    ``+z`` is forward from the camera, ``+x`` is right, ``+y`` is
    up. For the top-down view we use (x, z); y is reported for
    completeness."""
    if frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0, 0.0
    # Pinhole model. focal_x is in pixels.
    focal_x = (frame_w / 2.0) / math.tan(hfov_rad / 2.0)
    focal_y = focal_x  # square pixels assumption
    theta_x = math.atan((cx_px - frame_w / 2.0) / max(1e-3, focal_x))
    theta_y = math.atan((cy_px - frame_h / 2.0) / max(1e-3, focal_y))
    x = distance * math.sin(theta_x)
    y = -distance * math.sin(theta_y)
    z = distance * math.cos(theta_x) * math.cos(theta_y)
    return x, y, z


class RoomMapWorker:
    """Singleton-ish worker; one per MainWindow.

    Lifecycle: ``start()`` spins up a background thread. ``stop()``
    joins it. ``set_active(True/False)`` toggles whether the inner
    tick actually does work — keep it False when no UI is consuming
    the map.
    """

    def __init__(
        self,
        hfov_deg: Optional[float] = None,
        interval_s: float = _DEFAULT_INTERVAL_S,
    ) -> None:
        self.hfov_deg = float(hfov_deg) if hfov_deg else _hfov_from_env()
        self.hfov_rad = math.radians(self.hfov_deg)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        # Latest signals.
        self._last_frame = None
        self._last_frame_size: tuple[int, int] = (0, 0)
        self._last_objects: list = []
        # Smoothed (x, z) per label.
        self._smoothed: dict[str, RoomMapItem] = {}
        # Subscribe immediately so we have state ready when activated.
        bus = get_bus()
        bus.subscribe(EventType.OBJECTS, self._on_objects)
        bus.subscribe(EventType.FRAME, self._on_frame)

    # ── lifecycle ────────────────────────────────────────────

    def start(self) -> bool:
        if self._thread is not None:
            return True
        self._thread = threading.Thread(
            target=self._loop, name="room-map", daemon=True,
        )
        self._thread.start()
        log.info("room_map.started", hfov_deg=self.hfov_deg,
                 interval_s=self.interval_s)
        return True

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    def set_active(self, on: bool) -> None:
        """Gate the inner work. Off → just buffer signals; no MiDaS."""
        self._active = bool(on)

    # ── bus handlers ─────────────────────────────────────────

    def _on_objects(self, payload) -> None:
        if not isinstance(payload, ObjectsSeen):
            return
        self._last_objects = list(payload.detections)

    def _on_frame(self, frame) -> None:
        if frame is None:
            return
        self._last_frame = frame
        h, w = frame.shape[:2]
        self._last_frame_size = (w, h)

    # ── inner loop ───────────────────────────────────────────

    def _loop(self) -> None:
        # Brief delay so vision workers can publish at least once.
        time.sleep(min(self.interval_s, 2.0))
        while not self._stop.is_set():
            try:
                if self._active:
                    self._tick()
            except Exception as exc:  # noqa: BLE001
                log.warning("room_map.tick_error", error=str(exc))
            # Poll the stop flag every second for responsive shutdown.
            for _ in range(int(self.interval_s)):
                if self._stop.is_set():
                    return
                time.sleep(1.0)

    def _tick(self) -> None:
        # Always trim stale items so a tracked object disappears even
        # when the frame goes blank (rather than the dot lingering
        # forever).
        now = time.time()
        for label, item in list(self._smoothed.items()):
            if now - item.last_seen_ts > _STALE_AFTER_S:
                del self._smoothed[label]

        frame = self._last_frame
        detections = self._last_objects
        if frame is None or not detections:
            # Still publish the (possibly thinned) map so the UI
            # reflects the trim.
            if self._smoothed:
                w, h = self._last_frame_size
                get_bus().publish(
                    EventType.ROOM_MAP,
                    RoomMap(
                        items=list(self._smoothed.values()),
                        frame_w=w, frame_h=h,
                        hfov_deg=self.hfov_deg, units="relative",
                    ),
                )
            return
        try:
            from faceview.vision.depth import DepthEstimator
            depth = DepthEstimator.shared().depth_map(frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("room_map.depth_failed", error=str(exc))
            return
        if depth is None or depth.size == 0:
            return
        # MiDaS: higher value = nearer. We invert to "distance proxy":
        # objects at the max-depth pixel become near 0; far objects
        # → near 1. Stays relative-units until P16 calibration.
        dmax = float(depth.max())
        dmin = float(depth.min())
        span = max(1e-3, dmax - dmin)
        # Normalize the centre depth of each detection.
        now = time.time()
        w, h = self._last_frame_size
        items_this_tick: dict[str, RoomMapItem] = {}
        for det in detections:
            bx, by, bw, bh = det.bbox
            cx = int(bx + bw / 2.0)
            cy = int(by + bh / 2.0)
            cx = max(0, min(depth.shape[1] - 1, cx))
            cy = max(0, min(depth.shape[0] - 1, cy))
            d_at = float(depth[cy, cx])
            # Distance proxy in [0, 1].
            distance = (dmax - d_at) / span
            # Scale up a bit so the dots aren't all squashed near
            # the camera. 3.0 is empirical — feels right at 720p.
            distance *= 3.0
            x, y, z = _project(
                cx, cy, distance, w, h, self.hfov_rad,
            )
            items_this_tick[det.label] = RoomMapItem(
                label=det.label, x=x, y=y, z=z,
                confidence=float(det.score), last_seen_ts=now,
            )
        # EMA-smooth per label.
        for label, fresh in items_this_tick.items():
            old = self._smoothed.get(label)
            if old is None:
                self._smoothed[label] = fresh
            else:
                old.x = _ema(old.x, fresh.x)
                old.y = _ema(old.y, fresh.y)
                old.z = _ema(old.z, fresh.z)
                old.confidence = fresh.confidence
                old.last_seen_ts = now
        # Publish.
        get_bus().publish(
            EventType.ROOM_MAP,
            RoomMap(
                items=list(self._smoothed.values()),
                frame_w=w, frame_h=h, hfov_deg=self.hfov_deg,
                units="relative",
            ),
        )


def _ema(prev: float, fresh: float) -> float:
    return _EMA_ALPHA * fresh + (1.0 - _EMA_ALPHA) * prev
