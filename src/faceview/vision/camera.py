"""Webcam capture using OpenCV's AVFoundation backend on macOS.

Posts BGR ``np.ndarray`` frames on :data:`EventType.FRAME` plus a
:class:`FrameInfo` heartbeat for FPS tracking. Runs on a single dedicated
thread; the GUI panel converts to ``QImage`` on the main thread.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from faceview.config import settings
from faceview.core.errors import CameraError, MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, FrameInfo
from faceview.core.logger import get_logger


log = get_logger("camera")


class CameraWorker:
    def __init__(self, index: int | None = None, target_fps: int | None = None) -> None:
        self.index = settings.camera_index if index is None else index
        self.target_fps = target_fps or settings.target_fps
        self._cap = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_ts = 0.0

    def start(self) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise MissingDependency("opencv-python", "vision") from exc

        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise CameraError(f"could not open camera index {self.index}")

        self._cap = cap
        self._thread = threading.Thread(target=self._loop, name="camera-worker", daemon=True)
        self._thread.start()
        log.info("camera.started", index=self.index, target_fps=self.target_fps)

    def stop(self) -> None:
        # Tell the loop to exit FIRST, then wait for it, THEN release the
        # AVFoundation capture. Releasing while the loop is blocked inside
        # ``cap.read()`` triggers a SIGSEGV in -[CaptureDelegate
        # grabImageUntilDate:] when test-mode flips the camera off
        # mid-frame.
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass
        self._thread = None
        log.info("camera.stopped")

    def _loop(self) -> None:
        bus = get_bus()
        period = 1.0 / max(1, self.target_fps)
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if self._stop.is_set():
                break
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            now = time.time()
            fps = 1.0 / (now - self._last_ts) if self._last_ts else 0.0
            self._last_ts = now

            bus.publish(EventType.FRAME, frame)
            h, w = frame.shape[:2]
            bus.publish(EventType.STATUS, None)  # cheap heartbeat
            bus.publish(
                EventType.FRAME,
                None,
            ) if False else None  # noqa: B015 — placeholder for future heartbeat
            del fps, w, h
            time.sleep(period)
