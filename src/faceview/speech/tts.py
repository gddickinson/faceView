"""Text-to-speech via pyttsx3 (macOS NSSpeechSynthesizer under the hood).

Receives ``TTS_SPEAK`` text from the bus and synthesises in a background
thread. ``pyttsx3.runAndWait`` is blocking, so we keep one thread per worker
and serialise utterances through a queue to avoid overlapping speech.

Upgrade path: swap in ``kokoro-onnx`` for a much more natural voice; the
public surface (``speak(text)``) is intentionally identical.
"""

from __future__ import annotations

import queue
import threading

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("tts")


class TtsWorker:
    def __init__(self) -> None:
        self._engine = None
        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except ImportError as exc:
            raise MissingDependency("pyttsx3", "speech") from exc

        # init lazily on the worker thread — pyttsx3 on macOS needs the engine
        # to be created and used on the same thread.
        bus = get_bus()
        bus.subscribe(EventType.TTS_SPEAK, self.speak)

        self._thread = threading.Thread(target=self._loop, name="tts-worker", daemon=True)
        self._thread.start()
        log.info("tts.started")

    def speak(self, text: str) -> None:
        if not text:
            return
        self._q.put(str(text))

    def stop(self) -> None:
        self._q.put(None)

    def _loop(self) -> None:
        try:
            import pyttsx3  # type: ignore
        except ImportError:
            return  # already raised in start()

        engine = pyttsx3.init()
        engine.setProperty("rate", 190)
        bus = get_bus()
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                bus.publish(EventType.TTS_STARTED, item)
                engine.say(item)
                engine.runAndWait()
                bus.publish(EventType.TTS_FINISHED, item)
            except Exception as exc:  # noqa: BLE001
                log.warning("tts.error", error=str(exc))
