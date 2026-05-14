"""Hand gesture recognition via MediaPipe Tasks.

Wraps ``mediapipe.tasks.python.vision.GestureRecognizer`` with the small
prebuilt model ("gesture_recognizer.task", ~7 MB). The model recognises
seven base classes:

    None | Closed_Fist | Open_Palm | Pointing_Up |
    Thumb_Down | Thumb_Up | Victory | ILoveYou

We publish :data:`EventType.GESTURE` events at ~5 Hz with a normalised
label. The model file is downloaded on first use into
``~/.faceview/models/gesture_recognizer.task``.

Disable with ``FACEVIEW_GESTURES=0``. Skips cleanly if the model can't
be fetched (offline boot) — the rest of the perception stack continues
to work without gesture data.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from faceview.config import settings
from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Gesture
from faceview.core.logger import get_logger


log = get_logger("gestures")


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)


def gestures_enabled() -> bool:
    raw = os.environ.get("FACEVIEW_GESTURES")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _model_path() -> Path:
    return settings.data_dir / "models" / "gesture_recognizer.task"


def _ensure_model(path: Path, timeout: float = 30.0) -> bool:
    if path.exists() and path.stat().st_size > 1_000:
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("gestures.downloading", url=MODEL_URL)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=timeout) as r:
            data = r.read()
        path.write_bytes(data)
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError,
            OSError) as exc:
        log.warning("gestures.model_fetch_failed", error=str(exc))
        return False


_GESTURE_NAME_MAP = {
    "None": "none",
    "Closed_Fist": "closed_fist",
    "Open_Palm": "open_palm",
    "Pointing_Up": "pointing",
    "Thumb_Down": "thumbs_down",
    "Thumb_Up": "thumbs_up",
    "Victory": "victory",
    "ILoveYou": "i_love_you",
}


class GestureRecognizer:
    def __init__(self, throttle_hz: float = 5.0) -> None:
        self._period = 1.0 / max(1.0, throttle_hz)
        self._last_emit = 0.0
        self._lock = threading.Lock()
        self._rec = None
        self._mp_image = None
        self._started = False

    def start(self) -> bool:
        if self._started or not gestures_enabled():
            return self._started
        try:
            import mediapipe as mp  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc
        model = _model_path()
        if not _ensure_model(model):
            log.info("gestures.disabled_no_model")
            return False
        try:
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
        except ImportError as exc:
            raise MissingDependency("mediapipe", "vision") from exc

        options = mp_vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._rec = mp_vision.GestureRecognizer.create_from_options(options)
        self._mp_image = mp.Image
        self._mp = mp

        get_bus().subscribe(EventType.FRAME, self._on_frame)
        self._started = True
        log.info("gestures.started")
        return True

    def _on_frame(self, frame) -> None:
        if frame is None or self._rec is None:
            return
        now = time.time()
        if now - self._last_emit < self._period:
            return
        self._last_emit = now
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
            # MP rejects non-contiguous arrays — cvtColor returns a
            # fresh contiguous RGB buffer, unlike frame[:, :, ::-1].
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(mp.ImageFormat.SRGB, rgb)
            res = self._rec.recognize(image)
        except Exception as exc:  # noqa: BLE001
            log.warning("gestures.error", error=str(exc))
            return

        label = "none"
        hand = "none"
        confidence = 0.0
        if res.gestures:
            # Each entry is a list of category candidates for one hand.
            best_idx, best_cat = 0, None
            for i, cats in enumerate(res.gestures):
                if not cats:
                    continue
                top = cats[0]
                if (best_cat is None
                        or top.score > best_cat.score):
                    best_cat = top
                    best_idx = i
            if best_cat is not None:
                label = _GESTURE_NAME_MAP.get(
                    best_cat.category_name, best_cat.category_name.lower()
                )
                confidence = float(best_cat.score)
                if (res.handedness and best_idx < len(res.handedness)
                        and res.handedness[best_idx]):
                    h_top = res.handedness[best_idx][0]
                    hand = h_top.category_name.lower()  # "Left"/"Right"
                if len(res.gestures) >= 2:
                    hand = "both"

        get_bus().publish(
            EventType.GESTURE,
            Gesture(label=label, hand=hand, confidence=confidence),
        )
