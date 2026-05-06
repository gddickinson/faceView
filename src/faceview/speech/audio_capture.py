"""Microphone capture using sounddevice.

Emits raw PCM chunks on the bus as :data:`EventType.AUDIO_CHUNK`. Downstream
VAD/STT workers consume the stream. The capture runs on its own thread —
sounddevice's callback is itself off-main but we additionally proxy chunks
through a Queue so consumers can be slow without dropping audio.

This module imports ``sounddevice`` lazily so the package is safe to import
without the ``[speech]`` extra installed.
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from faceview.config import settings
from faceview.core.errors import AudioError, MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("audio")


class AudioCapture:
    def __init__(
        self,
        sample_rate: int | None = None,
        chunk_ms: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate or settings.sample_rate
        self.chunk_ms = chunk_ms or settings.audio_chunk_ms
        self._stream = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def chunk_samples(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000

    def start(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise MissingDependency("sounddevice", "speech") from exc

        def cb(indata, frames, _time_info, status):  # noqa: N803 — sd API
            if status:
                log.warning("audio.status", status=str(status))
            try:
                self._q.put_nowait(indata.copy().reshape(-1))
            except queue.Full:
                pass  # drop on backpressure rather than block the audio thread

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_samples,
                callback=cb,
            )
            self._stream.start()
        except Exception as exc:
            raise AudioError(f"failed to open mic: {exc}") from exc

        self._thread = threading.Thread(target=self._fanout, name="audio-fanout", daemon=True)
        self._thread.start()
        log.info("audio.started", sr=self.sample_rate, chunk_ms=self.chunk_ms)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        log.info("audio.stopped")

    def _fanout(self) -> None:
        bus = get_bus()
        while not self._stop.is_set():
            try:
                chunk = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            bus.publish(EventType.AUDIO_CHUNK, chunk)
