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

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from faceview.config import settings
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
        # Apply metric calibration if available — publish in metres
        # when a scale factor is set, otherwise keep "relative".
        scale = CalibrationStore.shared().scale
        if scale and scale > 0:
            items_out = [
                RoomMapItem(
                    label=it.label, x=it.x * scale, y=it.y * scale,
                    z=it.z * scale, confidence=it.confidence,
                    last_seen_ts=it.last_seen_ts,
                )
                for it in self._smoothed.values()
            ]
            units = "metres"
        else:
            items_out = list(self._smoothed.values())
            units = "relative"
        get_bus().publish(
            EventType.ROOM_MAP,
            RoomMap(
                items=items_out,
                frame_w=w, frame_h=h, hfov_deg=self.hfov_deg,
                units=units,
            ),
        )


def _ema(prev: float, fresh: float) -> float:
    return _EMA_ALPHA * fresh + (1.0 - _EMA_ALPHA) * prev


# ── Calibration store (P16 — metric scale factor) ────────────────


class CalibrationStore:
    """One-number calibration: ``scale`` converts relative units to
    metres. Loaded from ``~/.faceview/camera_calibration.json`` on
    boot; persisted on every change.

    The room-map worker multiplies every (x, y, z) by ``scale`` and
    sets ``RoomMap.units = "metres"`` when calibrated. When the
    file is absent we keep ``units="relative"`` — same behaviour as
    before P16.
    """

    _instance: "CalibrationStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "CalibrationStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = CalibrationStore()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.scale: Optional[float] = None
        self._load()

    def _path(self) -> Path:
        return settings.data_dir / "camera_calibration.json"

    def _load(self) -> None:
        p = self._path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text())
            s = float(data.get("scale") or 0.0)
            if s > 0:
                with self._lock:
                    self.scale = s
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("calibration.load_failed", error=str(exc))

    def set_scale(self, scale: float) -> bool:
        """Persist a new scale. ``scale <= 0`` is rejected."""
        if not scale or scale <= 0:
            return False
        with self._lock:
            self.scale = float(scale)
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({
                "scale": float(scale),
                "saved_at": time.time(),
            }, indent=2))
            tmp.replace(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("calibration.save_failed", error=str(exc))
            return False
        return True

    def clear(self) -> bool:
        """Drop calibration — room map reverts to relative units."""
        with self._lock:
            self.scale = None
        try:
            p = self._path()
            if p.exists():
                p.unlink()
        except OSError as exc:
            log.warning("calibration.clear_failed", error=str(exc))
            return False
        return True


# ── RoomMapStore (read-side singleton for the LLM tool) ───────────


class RoomMapStore:
    """Caches the latest :data:`EventType.ROOM_MAP` payload.

    The :class:`RoomMapWorker` is write-side and stops publishing
    when the panel is hidden. The store is the read-side: any caller
    (the ``describe_room_layout`` LLM tool, future MCP endpoints,
    the panel itself) reads from here. Survives the worker going
    idle — last-known positions linger until they age out."""

    _instance: "RoomMapStore | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "RoomMapStore":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = RoomMapStore()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[RoomMap] = None
        get_bus().subscribe(EventType.ROOM_MAP, self._on_map)

    def _on_map(self, m) -> None:
        if not isinstance(m, RoomMap):
            return
        with self._lock:
            self._latest = m

    def latest(self) -> Optional[RoomMap]:
        with self._lock:
            return self._latest


# ── describe_room_layout (used by the LLM tool) ───────────────────


def _zone_for(x: float, z: float) -> str:
    """Coarse direction label relative to the camera."""
    if abs(z) < 0.05 and abs(x) < 0.05:
        return "right at the camera"
    if z <= 0:
        return "behind the camera"
    angle = math.degrees(math.atan2(x, z))
    if abs(angle) < 15:
        return "directly ahead"
    if angle >= 15 and angle < 45:
        return "ahead and slightly to the right"
    if angle >= 45 and angle < 75:
        return "to the right"
    if angle >= 75:
        return "far right"
    if angle <= -15 and angle > -45:
        return "ahead and slightly to the left"
    if angle <= -45 and angle > -75:
        return "to the left"
    return "far left"


def describe_room_layout() -> str:
    """Build a one-paragraph natural-language description of the
    latest room map. Used by the LLM tool of the same name."""
    snap = RoomMapStore.shared().latest()
    if snap is None or not snap.items:
        return ("I don't have a room map yet — open the Room map "
                "window (View → Room map…) to start mapping objects.")
    # Sort items by distance so the description starts close-in.
    items = sorted(
        snap.items,
        key=lambda it: math.sqrt(it.x ** 2 + it.z ** 2),
    )
    unit = "m" if snap.units == "metres" else "units"
    bits: list[str] = []
    for it in items[:6]:
        dist = math.sqrt(it.x ** 2 + it.z ** 2)
        zone = _zone_for(it.x, it.z)
        bits.append(f"{it.label} is {dist:.1f} {unit} {zone}")
    summary = "; ".join(bits)
    if len(items) > 6:
        summary += f" — plus {len(items) - 6} other items"
    return f"Room layout (camera-relative): {summary}."
