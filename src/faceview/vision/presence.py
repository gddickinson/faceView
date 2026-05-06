"""Face presence + count via MediaPipe Tasks FaceDetector (BlazeFace).

Subscribes to ``FRAME`` events and publishes ``PRESENCE`` with the number of
faces visible plus their bounding boxes.
"""

from __future__ import annotations

import threading
import time

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Presence
from faceview.core.logger import get_logger


log = get_logger("presence")


class PresenceDetector:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence
        self._detector = None
        self._lock = threading.Lock()
        self._last_emit = 0.0

    def start(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc

        self._mp = mp
        # Use the legacy FaceDetection solution — simpler than the Tasks API
        # and equally fast for presence-only use cases.
        self._detector = mp.solutions.face_detection.FaceDetection(  # type: ignore[attr-defined]
            model_selection=0,
            min_detection_confidence=self.min_confidence,
        )

        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("presence.started")

    def _on_frame(self, frame) -> None:
        if frame is None or self._detector is None:
            return
        # Throttle to ~10 Hz.
        now = time.time()
        if now - self._last_emit < 0.1:
            return
        self._last_emit = now

        try:
            res = self._detector.process(frame[:, :, ::-1])  # BGR → RGB
        except Exception as exc:  # noqa: BLE001
            log.warning("presence.error", error=str(exc))
            return

        bboxes: list[tuple[int, int, int, int]] = []
        if res and res.detections:
            h, w = frame.shape[:2]
            for det in res.detections:
                rb = det.location_data.relative_bounding_box
                x = int(rb.xmin * w)
                y = int(rb.ymin * h)
                ww = int(rb.width * w)
                hh = int(rb.height * h)
                bboxes.append((x, y, ww, hh))

        get_bus().publish(
            EventType.PRESENCE,
            Presence(face_count=len(bboxes), bboxes=bboxes),
        )
