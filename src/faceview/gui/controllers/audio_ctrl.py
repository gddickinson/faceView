"""Microphone + VAD + STT + echo-gate + push-to-speak.

Three concerns live together here because they share state:

1. **Audio capture** — `sounddevice` InputStream worker.
2. **STT bridge** — VAD gates `faster-whisper` which publishes
   ``TRANSCRIPT_FINAL`` events; we translate those into chat input.
3. **Echo gate** — `AudioCapture.muted` is flipped on/off around
   `TTS_STARTED` / `TTS_FINISHED` so the avatar's own voice doesn't
   come back through the mic and start an infinite self-loop.
4. **Push-to-speak** — holding the chat-panel button interrupts TTS,
   un-mutes the mic, and overrides the echo gate so the user can
   talk over Claude.
"""

from __future__ import annotations

import threading
import time

from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, EventType, Transcript
from faceview.gui.controllers.base import BaseController


_TTS_COOLDOWN_S = 2.5


class AudioController(BaseController):
    log_name = "audio_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._audio_worker = None
        self._vad = None
        self._stt = None
        self._stt_to_chat_wired = False
        # Echo-gate state.
        self._tts_busy = False
        self._tts_quiet_until = 0.0
        self._push_to_speak = False

    # ── public API ────────────────────────────────────────────────

    def is_running(self) -> bool:
        return self._audio_worker is not None

    def set_enabled(self, on: bool) -> None:
        if on and self._audio_worker is None:
            try:
                from faceview.speech.audio_capture import AudioCapture
                self._audio_worker = AudioCapture()
                self._audio_worker.start()
                self._start_stt_chain()
                self.status("Microphone started")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("audio.start_failed", error=str(exc))
                self.status(f"Microphone unavailable: {exc}")
                self._audio_worker = None
        elif not on and self._audio_worker is not None:
            try:
                self._audio_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._audio_worker = None
            self.status("Microphone stopped")

    # Push-to-speak — called from the chat panel button.
    def push_to_speak_pressed(self) -> None:
        self._push_to_speak = True
        # Interrupt any in-flight TTS so the user can talk over Claude.
        tts = self.window.tts_ctrl
        try:
            tts.interrupt()
        except Exception:  # noqa: BLE001
            pass
        # Force-clear the echo gate + un-mute the mic.
        self._tts_busy = False
        self._tts_quiet_until = 0.0
        self._set_muted(False)
        self.status("Listening…")

    def push_to_speak_released(self) -> None:
        self._push_to_speak = False
        self.clear_status()

    # ── internals ─────────────────────────────────────────────────

    def _set_muted(self, on: bool) -> None:
        w = self._audio_worker
        if w is not None:
            try:
                w.muted = bool(on)
            except Exception:  # noqa: BLE001
                pass

    def _start_stt_chain(self) -> None:
        """Bring up VAD + STT downstream of audio capture."""
        if self._vad is None:
            try:
                from faceview.speech.vad import VadGate
                self._vad = VadGate()
                self._vad.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("vad.start_failed", error=str(exc))
                self._vad = None
        if self._stt is None:
            try:
                from faceview.speech.stt import SttWorker
                self._stt = SttWorker()
                self._stt.start()
            except Exception as exc:  # noqa: BLE001
                self.log.warning("stt.start_failed", error=str(exc))
                self._stt = None
        if self._stt_to_chat_wired:
            return
        self._wire_bus()
        self._stt_to_chat_wired = True

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.TTS_STARTED, self._on_tts_started)
        bus.subscribe(EventType.TTS_FINISHED, self._on_tts_finished)
        bus.subscribe(EventType.TRANSCRIPT_FINAL, self._on_transcript_final)

    # ── bus handlers ──────────────────────────────────────────────

    def _on_tts_started(self, _payload) -> None:
        self._tts_busy = True
        # Drop AUDIO_CHUNKs at source so VAD/STT/transcript-panel
        # never see the avatar's own voice — fixes echo + duplicate
        # transcripts.
        self._set_muted(True)

    def _on_tts_finished(self, _payload) -> None:
        self._tts_busy = False
        self._tts_quiet_until = time.time() + _TTS_COOLDOWN_S
        # Brief post-playback hold so the speaker's trailing tail
        # doesn't leak through the mic in the moment before audio
        # fully stops.
        threading.Timer(0.25, lambda: self._set_muted(False)).start()

    def _on_transcript_final(self, payload) -> None:
        text = payload.text if isinstance(payload, Transcript) else str(payload)
        text = (text or "").strip()
        if not text or len(text) < 2:
            return
        # Drop transcripts captured while the avatar was speaking or
        # right after — those are the avatar's own voice. Push-to-
        # speak overrides the gate so the user can interrupt.
        if not self._push_to_speak and (
            self._tts_busy or time.time() < self._tts_quiet_until
        ):
            self.log.info("stt.dropped_echo", text=text[:80])
            return
        get_bus().publish(EventType.CHAT_USER_MESSAGE,
                          ChatMessage("user", text))
