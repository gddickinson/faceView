"""Avatar worker lifecycle + persona swap + cognition + mirror mode.

Owns the Claude-side :class:`SimCameraWorker` and the optional
:class:`MirrorState`. When the user swaps personas:

1. Avatar's appearance overlay is rebound in place (no worker
   restart, which used to race the moderngl GL context).
2. The :class:`CognitionStore` for the new persona is loaded and
   bound to ``ClaudeClient.bind_memory``.
3. The new character's preferred Kokoro voice is applied to the TTS
   worker.

The "Avatar window" View-menu toggle and the avatar window itself
remain on :class:`MainWindow` so menu actions can ``setChecked`` it
directly; the controller drives behaviour, not the menu state.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Optional

from faceview.core.event_bus import get_bus
from faceview.core.events import AudioAmplitude, EventType
from faceview.gui.controllers.base import BaseController

if TYPE_CHECKING:
    from faceview.vision.sim_camera import SimCameraWorker


class AvatarController(BaseController):
    log_name = "avatar_ctrl"

    def __init__(self, window) -> None:
        super().__init__(window)
        self._avatar_worker: Optional["SimCameraWorker"] = None
        self._mirror_state = None
        self._current_persona = self._default_persona()
        # A47 — audio-driven lip-sync bias. Updated by AUDIO_AMPLITUDE
        # events while TTS is playing; the avatar's render loop reads
        # this via :meth:`audio_jaw_bias` and folds it into jaw_open
        # so the mouth shape tracks the actual TTS waveform amplitude
        # rather than only the text-predicted phoneme stream.
        self._audio_amp: float = 0.0
        self._audio_amp_ts: float = 0.0
        get_bus().subscribe(
            EventType.AUDIO_AMPLITUDE, self._on_audio_amplitude,
        )

    # ── public API: avatar lifecycle ──────────────────────────────

    def is_running(self) -> bool:
        return self._avatar_worker is not None

    @property
    def worker(self):
        return self._avatar_worker

    def current_persona(self) -> str:
        return self._current_persona

    def set_enabled(self, on: bool) -> None:
        if on and self._avatar_worker is None:
            self._start_worker(self._current_persona)
            self.window.show_avatar_window()
            try:
                self.window._avatar_action.setChecked(True)
            except Exception:  # noqa: BLE001
                pass
        elif not on and self._avatar_worker is not None:
            self.stop()
            self.status("Avatar stopped")

    def stop(self) -> None:
        if self._avatar_worker is not None:
            try:
                self._avatar_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._avatar_worker = None

    def set_persona(self, name: str) -> None:
        prev = self._current_persona
        self._current_persona = name
        # In-place swap (avatar restart raced ICT GL context).
        if self._avatar_worker is not None:
            try:
                self._avatar_worker.avatar.set_persona(name)
                self.status(f"Avatar persona: {name}")
            except Exception as exc:  # noqa: BLE001
                self.log.warning("persona.swap_failed", error=str(exc))
                self.status(f"Persona swap failed: {exc}")
        # Re-bind cognition + TTS voice to the new persona.
        if prev != name:
            self.bind_memory_for_current_persona(save_previous=True)

    # ── cognition rebinding ───────────────────────────────────────

    def bind_memory_for_current_persona(
        self, *, save_previous: bool = False,
    ) -> None:
        client = getattr(self.window, "llm_client", None)
        if client is None or not hasattr(client, "bind_memory"):
            return
        if save_previous and getattr(client, "memory", None) is not None:
            try:
                client.memory.save()
            except Exception:  # noqa: BLE001
                pass
        try:
            from faceview.llm.cognition import CognitionStore
            store = CognitionStore.load(self._current_persona)
            client.bind_memory(store)
            self._sync_tts_voice(store.character)
        except Exception as exc:  # noqa: BLE001
            self.log.warning("memory.bind_failed", error=str(exc))

    def _sync_tts_voice(self, character) -> None:
        """Apply the character's preferred Kokoro voice."""
        voice = getattr(character, "voice", None)
        if not voice:
            return
        os.environ["FACEVIEW_TTS_VOICE"] = voice
        tts = self.window.tts_ctrl
        try:
            tts.set_voice(voice)
        except Exception:  # noqa: BLE001
            pass

    # ── mirror mode ───────────────────────────────────────────────

    def mirror_running(self) -> bool:
        return (self._mirror_state is not None
                and self._mirror_state.active)

    def set_mirror_enabled(self, on: bool) -> None:
        if on:
            if not self.window.camera_ctrl.is_running():
                self.status(
                    "Mirror mode needs the camera on "
                    "(Tools → Toggle camera)"
                )
                return
            if not self.is_running():
                self.set_enabled(True)
            if self._mirror_state is None:
                from faceview.vision.mirror import MirrorState
                self._mirror_state = MirrorState()
                self._mirror_state.attach_bus()
            self._mirror_state.active = True
            self._avatar_worker.set_mirror_provider(
                self._mirror_state.face_params,
            )
            self.status("Mirror mode: Claude mimics you")
        else:
            if self._mirror_state is not None:
                self._mirror_state.active = False
            if self._avatar_worker is not None:
                self._avatar_worker.set_mirror_provider(None)
            self.status("Mirror mode off")

    # ── internals ─────────────────────────────────────────────────

    def _start_worker(self, persona: str) -> None:
        from faceview.vision.sim_camera import SimCameraWorker
        self._avatar_worker = SimCameraWorker(
            scenario="avatar",
            emotion="happy",
            persona=persona,
            wire_to_llm=True,
            frame_channel=EventType.AVATAR_FRAME,
            publish_user_events=False,
        )
        self._avatar_worker.start()
        # Re-bind the service's avatar handle so MCP/HTTP ops keep working.
        try:
            from faceview.server.service import get_service
            get_service().bind_camera_worker(self._avatar_worker)
        except Exception:  # noqa: BLE001
            pass

    # ── A47 audio-driven lip-sync bias ───────────────────────────

    def _on_audio_amplitude(self, payload) -> None:
        if not isinstance(payload, AudioAmplitude):
            return
        # Light smoothing so a sub-30 ms drop to zero doesn't snap
        # the mouth shut between syllables.
        self._audio_amp = (
            0.6 * self._audio_amp + 0.4 * float(payload.amplitude)
        )
        self._audio_amp_ts = time.time()
        # If the avatar worker exposes a jaw-bias hook, push it.
        w = self._avatar_worker
        if w is None:
            return
        bias_setter = getattr(w, "set_audio_jaw_bias", None)
        if callable(bias_setter):
            try:
                bias_setter(self.audio_jaw_bias())
            except Exception:  # noqa: BLE001
                pass

    def audio_jaw_bias(self) -> float:
        """Return the current jaw-open bias in [0, 1] — 0 once the
        amplitude signal goes stale (>500 ms without an event)."""
        if time.time() - self._audio_amp_ts > 0.5:
            return 0.0
        return float(max(0.0, min(1.0, self._audio_amp)))

    @staticmethod
    def _default_persona() -> str:
        try:
            from faceview.vision.ict_face import _data_path as _ict_data
            return "ict_xray_young" if _ict_data().exists() else "claude"
        except Exception:  # noqa: BLE001
            return "claude"
