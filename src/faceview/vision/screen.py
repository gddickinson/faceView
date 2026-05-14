"""Screen-region capture worker (P10).

Captures a chosen monitor at ~10 Hz and publishes
:data:`EventType.SCREEN_FRAME` events carrying the BGR
``np.ndarray``. The same vision tools that operate on the webcam
frame can be pointed at this stream via :class:`ScreenFrameGrabber`
+ the new ``look_at_screen`` tool.

Default OFF: capturing the screen is privacy-sensitive and on
macOS requires the user to grant **Screen Recording** permission
to the terminal / IDE that forked faceView. The menu item
"View → Screen capture" toggles it on/off; while active, the
status-bar Vision pill flashes orange (same indicator we use for
local VLM activity).

Dependencies: ``mss`` (~70 KB) for cross-platform screen grab.
Lazy-imported so faceView boots fine without it.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, PixelTransmission
from faceview.core.logger import get_logger


log = get_logger("screen")


DEFAULT_FPS = 10
DEFAULT_MONITOR = 1  # mss numbers from 1; 0 is "all monitors stitched"


class ScreenCaptureWorker:
    """Background thread that grabs the screen at a steady cadence.

    The worker has a write-only role on the bus (publishes
    SCREEN_FRAME); separate consumers (ScreenFrameGrabber, screen
    panel, future SCREEN-aware tools) read from there."""

    def __init__(
        self,
        monitor_index: int = DEFAULT_MONITOR,
        fps: int = DEFAULT_FPS,
    ) -> None:
        self.monitor_index = int(monitor_index)
        self.fps = max(1, min(30, int(fps)))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sct = None  # mss instance, lazy

    def start(self) -> bool:
        if self._thread is not None:
            return True
        try:
            import mss  # type: ignore  # noqa: F401
        except ImportError as exc:
            log.warning("screen.no_mss",
                        hint="pip install mss to enable screen capture")
            raise MissingDependency("mss", "vision") from exc
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="screen-capture", daemon=True,
        )
        self._thread.start()
        log.info("screen.started",
                 monitor=self.monitor_index, fps=self.fps)
        return True

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None
        # Last "active=False" so the recording indicator clears.
        try:
            get_bus().publish(
                EventType.PIXELS_LEAVING,
                PixelTransmission(active=False, destination="screen",
                                  tool="screen_capture"),
            )
        except Exception:  # noqa: BLE001
            pass

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── internals ────────────────────────────────────────────

    def _loop(self) -> None:
        import mss
        import numpy as np

        # Tell the privacy indicator we're sampling the screen.
        try:
            get_bus().publish(
                EventType.PIXELS_LEAVING,
                PixelTransmission(active=True, destination="screen",
                                  tool="screen_capture"),
            )
        except Exception:  # noqa: BLE001
            pass

        period = 1.0 / float(self.fps)
        with mss.mss() as sct:
            monitors = sct.monitors
            idx = max(1, min(len(monitors) - 1, self.monitor_index))
            mon = monitors[idx]
            bus = get_bus()
            while not self._stop.is_set():
                t0 = time.time()
                try:
                    raw = sct.grab(mon)
                    arr = np.array(raw, dtype="uint8")
                    # mss returns BGRA; drop alpha for our BGR contract.
                    bgr = arr[:, :, :3]
                    bus.publish(EventType.SCREEN_FRAME, bgr)
                except Exception as exc:  # noqa: BLE001
                    log.warning("screen.grab_failed", error=str(exc))
                    # Slow down on errors so we don't spam the log.
                    time.sleep(0.5)
                elapsed = time.time() - t0
                wait = max(0.0, period - elapsed)
                if wait > 0:
                    if self._stop.wait(timeout=wait):
                        return


# ── ScreenFrameGrabber — read-side singleton ──────────────────────


class ScreenFrameGrabber:
    """Mirror of :class:`faceview.llm.vision_tool.FrameGrabber` but
    listening on ``EventType.SCREEN_FRAME``. The new ``look_at_screen``
    tool reads from this; existing ``look_at_camera`` keeps using the
    webcam grabber so the two stay independent."""

    _instance: "ScreenFrameGrabber | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "ScreenFrameGrabber":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = ScreenFrameGrabber()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ts = 0.0
        get_bus().subscribe(EventType.SCREEN_FRAME, self._on_frame)

    def _on_frame(self, frame) -> None:
        if frame is None:
            return
        with self._lock:
            self._latest = frame
            self._latest_ts = time.time()

    def have_frame(self) -> bool:
        with self._lock:
            return self._latest is not None

    def latest_jpeg_b64(
        self,
        max_dim: int = 1280,
        quality: int = 75,
    ) -> Optional[tuple[str, str]]:
        """Same shape as FrameGrabber.latest_jpeg_b64 — return
        ``(b64_str, source)`` where source is always ``"screen"``."""
        import base64 as _b64
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        try:
            import cv2  # type: ignore
        except ImportError:
            return None
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality],
        )
        if not ok:
            return None
        return _b64.b64encode(buf.tobytes()).decode("ascii"), "screen"
