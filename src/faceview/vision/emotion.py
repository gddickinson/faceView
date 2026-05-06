"""Emotion classification via DeepFace.

Heavy: DeepFace pulls TensorFlow on first use. The recogniser is throttled
to ~1 Hz to avoid saturating the M-series CPU.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import Emotion, EventType
from faceview.core.logger import get_logger


log = get_logger("emotion")


class EmotionAnalyzer:
    def __init__(self) -> None:
        self._df: Any = None
        self._lock = threading.Lock()
        self._last_emit = 0.0
        self._busy = False

    def start(self) -> None:
        try:
            from deepface import DeepFace  # type: ignore
        except ImportError as exc:
            raise MissingDependency("deepface", "emotion") from exc
        self._df = DeepFace
        get_bus().subscribe(EventType.FRAME, self._on_frame)
        log.info("emotion.started")

    def _on_frame(self, frame) -> None:
        if frame is None or self._df is None or self._busy:
            return
        now = time.time()
        if now - self._last_emit < 1.0:
            return
        self._last_emit = now

        # Run analyse in a small worker so we never block the camera thread.
        t = threading.Thread(target=self._analyse, args=(frame,), daemon=True)
        t.start()

    def _analyse(self, frame) -> None:
        with self._lock:
            self._busy = True
            try:
                result = self._df.analyze(
                    frame,
                    actions=["emotion"],
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                if isinstance(result, list):
                    result = result[0]
                emo = result.get("emotion") or {}
                if not emo:
                    return
                label = max(emo, key=emo.get)
                conf = float(emo[label]) / 100.0
                get_bus().publish(
                    EventType.EMOTION,
                    Emotion(label=label, confidence=conf, scores={k: float(v) / 100.0 for k, v in emo.items()}),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("emotion.error", error=str(exc))
            finally:
                self._busy = False
