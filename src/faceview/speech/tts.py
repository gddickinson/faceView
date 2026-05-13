"""Text-to-speech worker.

Subscribes to ``TTS_SPEAK`` and serialises utterances through a thread
queue (so two concurrent replies don't overlap). The engine is one of:

- **kokoro** — neural TTS via :mod:`faceview.speech.tts_kokoro`.
  Selected when ``kokoro-onnx`` is installed and the model files exist.
  Much more natural than the macOS system voices.
- **pyttsx3** — macOS NSSpeechSynthesizer wrapper. Fallback when
  Kokoro is unavailable. Compact voices only unless the user has
  downloaded Premium variants from System Settings.

Engine selection follows ``FACEVIEW_TTS_ENGINE`` (``kokoro|pyttsx3``,
default ``auto``). Voice + speed come from ``FACEVIEW_TTS_VOICE`` and
``FACEVIEW_TTS_RATE`` (rate is words/min for pyttsx3, 0.5-2.0 multiplier
for Kokoro).
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Optional

from faceview.core.errors import MissingDependency
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType
from faceview.core.logger import get_logger


log = get_logger("tts")


def _select_engine_name() -> str:
    requested = (os.environ.get("FACEVIEW_TTS_ENGINE") or "auto").lower()
    if requested in ("kokoro", "pyttsx3"):
        return requested
    # auto: prefer kokoro if available + assets present.
    try:
        from faceview.speech.tts_kokoro import assets_present
        import kokoro_onnx  # noqa: F401
        if assets_present():
            return "kokoro"
    except ImportError:
        pass
    return "pyttsx3"


class TtsWorker:
    def __init__(self) -> None:
        self._engine = None
        self._engine_name = ""
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.TTS_SPEAK, self.speak)
        self._thread = threading.Thread(
            target=self._loop, name="tts-worker", daemon=True,
        )
        self._thread.start()
        log.info("tts.started")

    def speak(self, text: str) -> None:
        if not text:
            return
        self._q.put(str(text))

    def stop(self) -> None:
        self._q.put(None)

    # ── engine selection (lazy, on worker thread) ────────────

    def engine_name(self) -> str:
        return self._engine_name

    def voices(self) -> list[str]:
        # Called from GUI thread for the config picker. Build a transient
        # engine if needed; cheap for both backends.
        name = self._engine_name or _select_engine_name()
        if name == "kokoro":
            try:
                from faceview.speech.tts_kokoro import KokoroEngine
                return KokoroEngine().voices()
            except Exception:  # noqa: BLE001
                return []
        try:
            import pyttsx3  # type: ignore
            e = pyttsx3.init()
            return [v.name for v in e.getProperty("voices") if v.name]
        except Exception:  # noqa: BLE001
            return []

    def _make_engine(self):
        name = _select_engine_name()
        if name == "kokoro":
            try:
                from faceview.speech.tts_kokoro import KokoroEngine
                voice = os.environ.get("FACEVIEW_TTS_VOICE") or "af_sarah"
                speed = float(os.environ.get("FACEVIEW_TTS_RATE") or 1.0)
                eng = KokoroEngine(voice=voice, speed=speed)
                # Force a load so we fail fast if the model is bad.
                eng._ensure_engine()
                self._engine_name = "kokoro"
                return eng
            except Exception as exc:  # noqa: BLE001
                log.warning("kokoro.init_failed_falling_back", error=str(exc))
                # fall through to pyttsx3
        return self._make_pyttsx3_engine()

    def _make_pyttsx3_engine(self):
        try:
            import pyttsx3  # type: ignore
        except ImportError as exc:
            raise MissingDependency("pyttsx3", "speech") from exc
        e = pyttsx3.init()
        rate = int(os.environ.get("FACEVIEW_TTS_RATE_WPM") or 190)
        e.setProperty("rate", rate)
        voice = os.environ.get("FACEVIEW_TTS_VOICE")
        if voice:
            for v in e.getProperty("voices"):
                if v.name == voice or v.id == voice:
                    e.setProperty("voice", v.id)
                    break
        self._engine_name = "pyttsx3"

        class _PyttsxAdapter:
            def __init__(self, engine):
                self._e = engine

            def speak(self, text: str):
                self._e.say(text)
                self._e.runAndWait()
                return None

        return _PyttsxAdapter(e)

    # ── loop ──────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            self._engine = self._make_engine()
        except Exception as exc:  # noqa: BLE001
            log.warning("tts.engine_init_failed", error=str(exc))
            return
        log.info("tts.engine_ready", engine=self._engine_name)
        bus = get_bus()
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                bus.publish(EventType.TTS_STARTED, item)
                self._engine.speak(item)
                bus.publish(EventType.TTS_FINISHED, item)
            except Exception as exc:  # noqa: BLE001
                log.warning("tts.error", error=str(exc))
