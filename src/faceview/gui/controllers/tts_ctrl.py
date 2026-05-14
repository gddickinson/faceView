"""Text-to-speech worker lifecycle + LLM_REPLY → TTS_SPEAK bridge.

When TTS is on, every assistant reply is published as a TTS_SPEAK
event so the worker picks it up. When the user holds the push-to-
speak button (handled in :class:`AudioController`), the active
utterance is interrupted via :meth:`interrupt`.
"""

from __future__ import annotations

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, EventType
from faceview.gui.controllers.base import BaseController


class TtsController(BaseController):
    log_name = "tts_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._tts = None
        self._reply_bridge = None  # cached subscriber for unsubscribe

    # ── public API ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._tts is not None

    def set_enabled(self, on: bool) -> None:
        if on and self._tts is None:
            try:
                from faceview.speech.tts import TtsWorker
                self._tts = TtsWorker()
                self._tts.start()
                self._wire_reply_bridge()
                self.status("TTS started — Claude will speak replies")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("tts.start_failed", error=str(exc))
                self.status(f"TTS unavailable: {exc}")
                self._tts = None
        elif not on and self._tts is not None:
            try:
                self._tts.stop()
            except Exception:  # noqa: BLE001
                pass
            self._tts = None
            self.status("TTS stopped")

    def interrupt(self) -> None:
        """Kill the active utterance — used by push-to-speak."""
        w = self._tts
        if w is None:
            return
        try:
            if hasattr(w, "interrupt"):
                w.interrupt()
        except Exception as exc:  # noqa: BLE001
            self.log.warning("tts.interrupt_failed", error=str(exc))

    def set_voice(self, voice: str) -> None:
        """Apply a Kokoro voice name to the worker, if loaded."""
        w = self._tts
        if w is None:
            return
        try:
            if hasattr(w, "set_voice"):
                w.set_voice(voice)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("tts.set_voice_failed", error=str(exc))

    # ── internals ─────────────────────────────────────────────────

    def _wire_reply_bridge(self) -> None:
        """LLM_REPLY → TTS_SPEAK. Subscribed lazily so the worker
        only speaks while TTS is on."""
        bus = get_bus()

        def _say_reply(msg):
            text = (
                getattr(msg, "content", "")
                if isinstance(msg, ChatMessage)
                else str(msg)
            )
            if text:
                bus.publish(EventType.TTS_SPEAK, text)

        self._reply_bridge = _say_reply
        bus.subscribe(EventType.LLM_REPLY, _say_reply)
