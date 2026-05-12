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
    # silero-vad v5+ requires exactly 512 samples per call at 16 kHz.
    SILERO_WINDOW = 512

    def __init__(self, threshold: float = 0.5, hangover_ms: int = 300) -> None:
        self.threshold = threshold
        self.hangover_ms = hangover_ms
        self._lock = threading.Lock()
        self._model = None
        self._is_speaking = False
        self._silence_run = 0
        self._utterance: list[np.ndarray] = []
        # Rolling buffer for chunks smaller than SILERO_WINDOW.
        self._pending: list[np.ndarray] = []
        self._pending_len = 0

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
        # Buffer incoming chunks until we have a full 512-sample window
        # for silero — anything smaller raises "Input audio chunk is too
        # short" inside the TorchScript model.
        self._pending.append(chunk)
        self._pending_len += len(chunk)
        while self._pending_len >= self.SILERO_WINDOW:
            combined = np.concatenate(self._pending)
            window = combined[: self.SILERO_WINDOW]
            leftover = combined[self.SILERO_WINDOW :]
            self._pending = [leftover] if leftover.size else []
            self._pending_len = leftover.size
            self._process_window(window)

    def _process_window(self, window: np.ndarray) -> None:
        x = window.astype(np.float32) / 32768.0
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
                self._utterance.append(window)
                self._silence_run = 0
            elif self._is_speaking:
                self._utterance.append(window)
                self._silence_run += len(window)
                if self._silence_run > self.hangover_ms * 16:
                    full = np.concatenate(self._utterance)
                    self._is_speaking = False
                    self._utterance.clear()
                    bus.publish(EventType.VAD_SPEECH_END, full)
