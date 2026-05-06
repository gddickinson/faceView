"""Voice-Activity-Detection gate using silero-vad.

Subscribes to ``AUDIO_CHUNK`` and emits ``VAD_SPEECH_START`` /
``VAD_SPEECH_END`` along with a buffered utterance for STT.
"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("vad")


class VadGate:
    def __init__(self, threshold: float = 0.5, hangover_ms: int = 300) -> None:
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self._lock = threading.Lock()
        self._model = None
        self._is_speaking = False
        self._silence_run = 0
        self._utterance: list[np.ndarray] = []

    def start(self) -> None:
        try:
            from silero_vad import load_silero_vad  # type: ignore
        except ImportError as exc:
            raise MissingDependency("silero-vad", "speech") from exc

        self._model = load_silero_vad()
        bus = get_bus()
        bus.subscribe(EventType.AUDIO_CHUNK, self._on_chunk)
        log.info("vad.started", threshold=self.threshold)

    def _on_chunk(self, chunk: np.ndarray) -> None:
        # silero-vad expects float32 in [-1, 1] at 16 kHz.
        x = chunk.astype(np.float32) / 32768.0
        try:
            import torch  # type: ignore
            t = torch.from_numpy(x)
            prob = float(self._model(t, 16_000).item())  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            log.warning("vad.error", error=str(exc))
            return

        is_voiced = prob >= self.threshold
        bus = get_bus()
        with self._lock:
            if is_voiced:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._utterance.clear()
                    bus.publish(EventType.VAD_SPEECH_START)
                self._utterance.append(chunk)
                self._silence_run = 0
            elif self._is_speaking:
                self._utterance.append(chunk)
                self._silence_run += len(chunk)
                # convert hangover to samples
                if self._silence_run > self.hangover_ms * 16:
                    full = np.concatenate(self._utterance)
                    self._is_speaking = False
                    self._utterance.clear()
                    bus.publish(EventType.VAD_SPEECH_END, full)
