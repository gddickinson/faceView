"""Speech-to-text using faster-whisper.

Listens for ``VAD_SPEECH_END`` events (final utterance buffer) and transcribes
synchronously on its own thread. We do not stream partials in this baseline —
real-time partials are an optimisation that can come later via RealtimeSTT.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from faceview.config import settings
from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Transcript
from faceview.core.logger import get_logger


log = get_logger("stt")


class SttWorker:
    def __init__(self, model_size: str = "small.en") -> None:
        self.model_size = model_size
        self._model = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise MissingDependency("faster-whisper", "speech") from exc

        self._model = WhisperModel(self.model_size, compute_type="int8")
        get_bus().subscribe(EventType.VAD_SPEECH_END, self._enqueue)

        self._thread = threading.Thread(target=self._loop, name="stt-worker", daemon=True)
        self._thread.start()
        log.info("stt.started", model=self.model_size)

    def stop(self) -> None:
        self._stop.set()

    def _enqueue(self, audio: np.ndarray) -> None:
        try:
            self._q.put_nowait(audio)
        except queue.Full:
            log.warning("stt.queue_full_drop")

    def _loop(self) -> None:
        bus = get_bus()
        while not self._stop.is_set():
            try:
                audio = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            x = audio.astype(np.float32) / 32768.0
            try:
                segments, _info = self._model.transcribe(  # type: ignore[union-attr]
                    x,
                    sampling_rate=settings.sample_rate,
                    beam_size=1,
                    vad_filter=False,
                )
                text = " ".join(s.text for s in segments).strip()
            except Exception as exc:  # noqa: BLE001
                log.warning("stt.error", error=str(exc))
                continue
            if text:
                bus.publish(EventType.TRANSCRIPT_FINAL, Transcript(text=text, is_final=True))
