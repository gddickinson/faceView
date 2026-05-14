"""Object detection via MediaPipe ObjectDetector (EfficientDet-Lite0).

Detects ~80 COCO classes (person, cup, laptop, book, …) at ~10 ms per
frame on Apple Silicon CPU. We throttle to ~3 Hz because object lists
don't change that fast and we want to leave CPU headroom for face mesh
+ gestures.

The .tflite model (~12 MB) is downloaded on first use into
``~/.faceview/models/efficientdet_lite0.tflite``.

Disable with ``FACEVIEW_OBJECTS=0``.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from faceview.config import settings
from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import (
    DetectedObject, EventType, ObjectsSeen,
)
from faceview.core.logger import get_logger


log = get_logger("objects")


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "object_detector/efficientdet_lite0/int8/latest/"
    "efficientdet_lite0.tflite"
)


def objects_enabled() -> bool:
    raw = os.environ.get("FACEVIEW_OBJECTS")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _model_path() -> Path:
    return settings.data_dir / "models" / "efficientdet_lite0.tflite"


def _ensure_model(path: Path, timeout: float = 30.0) -> bool:
    if path.exists() and path.stat().st_size > 1_000:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("objects.downloading", url=MODEL_URL)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as r:
            data = r.read()
        path.write_bytes(data)
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            OSError) as exc:
        log.warning("objects.model_fetch_failed", error=str(exc))
        return False


class ObjectDetector:
    def __init__(
        self,
        throttle_hz: float = 3.0,
        max_results: int = 6,
        score_threshold: float = 0.35,
    ) -> None:
        self._period = 1.0 / max(1.0, throttle_hz)
        self._last_emit = 0.0
        self._max_results = max_results
        self._score_threshold = score_threshold
        self._det = None
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> bool:
        if self._started or not objects_enabled():
            return self._started
        try:
            import mediapipe as mp  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc
        model = _model_path()
        if not _ensure_model(model):
            log.info("objects.disabled_no_model")
            return False
        try:
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc

        options = mp_vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            max_results=self._max_results,
            score_threshold=self._score_threshold,
        )
        self._det = mp_vision.ObjectDetector.create_from_options(options)

        get_bus().subscribe(EventType.FRAME, self._on_frame)
        self._started = True
        log.info("objects.started")
        return True

    def _on_frame(self, frame) -> None:
        if frame is None or self._det is None:
            return
        now = time.time()
        if now - self._last_emit < self._period:
            return
        self._last_emit = now
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(mp.ImageFormat.SRGB, rgb)
            res = self._det.detect(image)
        except Exception as exc:  # noqa: BLE001
            log.warning("objects.error", error=str(exc))
            return

        dets: list[DetectedObject] = []
        for d in res.detections:
            if not d.categories:
                continue
            cat = d.categories[0]
            bb = d.bounding_box
            dets.append(DetectedObject(
                label=cat.category_name,
                score=float(cat.score),
                bbox=(int(bb.origin_x), int(bb.origin_y),
                      int(bb.width), int(bb.height)),
            ))
        get_bus().publish(EventType.OBJECTS, ObjectsSeen(detections=dets))
