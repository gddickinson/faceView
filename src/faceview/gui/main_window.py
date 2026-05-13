"""Main window — webcam-of-the-user side of the conversation.

Layout:

    ┌──────────────────────────────────────────────────────────┐
    │ menu / status bar                                       │
    ├────────────────────┬────────────────────┬───────────────┤
    │                    │                    │  Status       │
    │   Camera           │   Chat             │  panel        │
    │   (the user's      │   (history+input)  ├───────────────┤
    │    webcam)         │                    │  Transcript   │
    └────────────────────┴────────────────────┴───────────────┘

A separate :class:`AvatarWindow` (Claude's face) lives next to this one
and is opened automatically at boot. MainWindow owns the worker
lifecycles (camera / mic / avatar / dual-Claude test mode) so the
config dialog can flip them on and off without restarting the app.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QStatusBar

from faceview.config import settings
from faceview.core.logger import get_logger
from faceview.gui.avatar_window import AvatarWindow
from faceview.gui.camera_panel import CameraPanel
from faceview.gui.chat_panel import ChatPanel
from faceview.gui.layout import LayoutManager
from faceview.gui.screenshotter import Screenshotter
from faceview.gui.status_panel import StatusPanel
from faceview.gui.transcript_panel import TranscriptPanel

if TYPE_CHECKING:
    from pathlib import Path

    from faceview.llm.test_conversation import TestConversation
    from faceview.vision.sim_camera import SimCameraWorker


log = get_logger("main_window")


class MainWindow(QMainWindow):
    """Shell that composes panels and owns worker lifecycle."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("faceView — You")
        self.resize(1280, 800)

        self.shotter = Screenshotter()

        self.camera = CameraPanel(self)
        self.chat = ChatPanel(self)
        self.status_panel = StatusPanel(self)
        self.transcript = TranscriptPanel(self)

        # Companion window for Claude's avatar.
        self.avatar_window: Optional[AvatarWindow] = None

        # Worker handles — created lazily so the GUI shell stays cheap.
        self._camera_worker = None
        self._audio_worker = None
        self._avatar_worker: Optional["SimCameraWorker"] = None
        self._user_avatar_worker: Optional["SimCameraWorker"] = None  # test mode
        self._test_orchestrator: Optional["TestConversation"] = None
        self._current_persona = self._default_avatar_persona()
        # Vision-pipeline analysers + TTS (lazy ML deps)
        self._presence = None
        self._mouth = None
        self._emotion = None
        self._identity = None
        self._tts = None
        self._stt = None
        self._vad = None
        # Monitor windows (created on demand)
        self._monitors: dict[str, Any] = {}

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("faceView ready")

    # ── layout ──────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Every panel is wrapped in a QDockWidget so it can detach,
        # tab, hide, or re-dock. LayoutManager also persists state via
        # QSettings and exposes save/reset to the menu.
        self.layout_mgr = LayoutManager(self)
        self.layout_mgr.build()

    def _build_menu(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        a_shot = QAction("Take screenshot", self)
        a_shot.setShortcut(QKeySequence("Ctrl+Shift+S"))
        a_shot.triggered.connect(lambda: self.take_screenshot("manual.png"))
        m_file.addAction(a_shot)
        a_quit = QAction("Quit", self)
        a_quit.setShortcut(QKeySequence.StandardKey.Quit)
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_view = self.menuBar().addMenu("&View")
        a_avatar = QAction("Avatar window", self)
        a_avatar.setShortcut(QKeySequence("Ctrl+Shift+A"))
        a_avatar.setCheckable(True)
        a_avatar.setChecked(True)
        a_avatar.toggled.connect(self._toggle_avatar_window)
        m_view.addAction(a_avatar)
        self._avatar_action = a_avatar

        a_fx = QAction("Effects panel…", self)
        a_fx.setShortcut(QKeySequence("Ctrl+E"))
        a_fx.triggered.connect(self._open_effects_panel)
        m_view.addAction(a_fx)

        a_styles = QAction("Avatar style…", self)
        a_styles.setShortcut(QKeySequence("Ctrl+Shift+P"))
        a_styles.triggered.connect(self._open_persona_picker)
        m_view.addAction(a_styles)

        a_chars = QAction("Edit personas…", self)
        a_chars.setShortcut(QKeySequence("Ctrl+Shift+I"))
        a_chars.triggered.connect(self._open_character_editor)
        m_view.addAction(a_chars)

        m_tools = self.menuBar().addMenu("&Tools")
        a_cfg = QAction("Configuration…", self)
        a_cfg.setShortcut(QKeySequence("Ctrl+,"))
        a_cfg.triggered.connect(self._open_config_dialog)
        m_tools.addAction(a_cfg)

        a_cam = QAction("Toggle camera", self)
        a_cam.setShortcut(QKeySequence("Ctrl+Shift+C"))
        a_cam.triggered.connect(lambda: self.set_camera_enabled(not self.camera_running()))
        m_tools.addAction(a_cam)

        a_mic = QAction("Toggle microphone", self)
        a_mic.setShortcut(QKeySequence("Ctrl+Shift+M"))
        a_mic.triggered.connect(lambda: self.set_audio_enabled(not self.audio_running()))
        m_tools.addAction(a_mic)

        a_tts = QAction("Toggle Claude voice (TTS)", self)
        a_tts.setShortcut(QKeySequence("Ctrl+Shift+T"))
        a_tts.triggered.connect(lambda: self.set_tts_enabled(not self.tts_running()))
        m_tools.addAction(a_tts)

        a_enroll = QAction("Enroll me as owner…", self)
        a_enroll.setShortcut(QKeySequence("Ctrl+Shift+E"))
        a_enroll.triggered.connect(self.enroll_owner)
        m_tools.addAction(a_enroll)

        a_mirror = QAction("Toggle mirror mode (avatar mimics user)", self)
        a_mirror.setShortcut(QKeySequence("Ctrl+Shift+R"))
        a_mirror.triggered.connect(lambda: self.set_mirror_mode_enabled(not self.mirror_running()))
        m_tools.addAction(a_mirror)

        m_window = self.menuBar().addMenu("&Window")
        self.layout_mgr.install_menu(m_window)

        m_monitor = self.menuBar().addMenu("&Monitor")
        for kind, label, shortcut in (
            ("audio",    "Audio waveform / VAD…",   "Ctrl+1"),
            ("emotion",  "Emotion scores…",          "Ctrl+2"),
            ("mouth",    "Mouth + visemes…",         "Ctrl+3"),
            ("transcript", "Transcript history…",   "Ctrl+4"),
        ):
            act = QAction(label, self)
            act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(lambda _checked=False, k=kind: self.open_monitor(k))
            m_monitor.addAction(act)

    # ── companion window ───────────────────────────────────────────

    def show_avatar_window(self) -> None:
        if self.avatar_window is None:
            self.avatar_window = AvatarWindow(self)
            # Place the avatar window to the right of the main window so
            # the user reads the layout as "me on the left, Claude on
            # the right".
            geo = self.geometry()
            self.avatar_window.move(geo.x() + geo.width() + 12, geo.y())
        self.avatar_window.show()
        self.avatar_window.raise_()

    def _toggle_avatar_window(self, on: bool) -> None:
        if on:
            self.show_avatar_window()
        elif self.avatar_window is not None:
            self.avatar_window.hide()

    # ── menu actions ────────────────────────────────────────────────

    def _open_effects_panel(self) -> None:
        from faceview.gui.effects_panel import EffectsPanel
        if not hasattr(self, "_fx_panel") or self._fx_panel is None:
            self._fx_panel = EffectsPanel(self)
        self._fx_panel.show()
        self._fx_panel.raise_()
        self._fx_panel.activateWindow()

    def _open_config_dialog(self) -> None:
        from faceview.gui.config_dialog import ConfigDialog
        if not hasattr(self, "_cfg_dialog") or self._cfg_dialog is None:
            self._cfg_dialog = ConfigDialog(self, self)
        self._cfg_dialog.show()
        self._cfg_dialog.raise_()
        self._cfg_dialog.activateWindow()

    def _open_character_editor(self) -> None:
        from faceview.gui.character_editor import CharacterEditor
        if not hasattr(self, "_char_editor") or self._char_editor is None:
            self._char_editor = CharacterEditor(self, self)
        else:
            self._char_editor._reload_from_disk()
        self._char_editor.show()
        self._char_editor.raise_()
        self._char_editor.activateWindow()

    def _open_persona_picker(self) -> None:
        from faceview.gui.persona_picker import PersonaPicker
        if not hasattr(self, "_persona_picker") or self._persona_picker is None:
            self._persona_picker = PersonaPicker(self, self)
        else:
            self._persona_picker._refresh_selection()
        self._persona_picker.show()
        self._persona_picker.raise_()
        self._persona_picker.activateWindow()

    # ── public helpers (used by ConfigDialog) ───────────────────────

    def take_screenshot(self, name: str) -> "Path":
        return self.shotter.capture_window(self, name)

    def seed_demo_state(self) -> None:
        self.chat.seed_demo_conversation()
        self.status_panel.seed_demo()
        self.transcript.seed_demo()
        self.statusBar().showMessage(
            "Demo state — owner recognised, mic idle, camera idle."
        )

    # ── lifecycle: camera ───────────────────────────────────────────

    def camera_running(self) -> bool:
        return self._camera_worker is not None

    def set_camera_enabled(self, on: bool) -> None:
        if on and self._camera_worker is None:
            try:
                from faceview.vision.camera import CameraWorker
                self._camera_worker = CameraWorker()
                self._camera_worker.start()
                self._start_vision_analysers()
                self.statusBar().showMessage("Camera started + vision analysers up")
            except Exception as exc:  # noqa: BLE001
                log.warning("camera.start_failed", error=str(exc))
                self.statusBar().showMessage(f"Camera unavailable: {exc}")
                self._camera_worker = None
        elif not on and self._camera_worker is not None:
            try:
                self._camera_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._camera_worker = None
            self.statusBar().showMessage("Camera stopped")

    def _start_vision_analysers(self) -> None:
        """Bring up presence/mouth/emotion/identity if their deps are present.

        Each analyser is started independently; missing ML libraries
        degrade gracefully (the corresponding status pill stays idle).
        """
        if self._presence is None:
            try:
                from faceview.vision.presence import PresenceDetector
                self._presence = PresenceDetector()
                self._presence.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("presence.start_failed", error=str(exc))
                self._presence = None
        if self._mouth is None:
            try:
                from faceview.vision.mouth import MouthAnalyzer
                self._mouth = MouthAnalyzer()
                self._mouth.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("mouth.start_failed", error=str(exc))
                self._mouth = None
        if self._emotion is None:
            try:
                from faceview.vision.emotion import EmotionAnalyzer
                self._emotion = EmotionAnalyzer()
                self._emotion.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("emotion.start_failed", error=str(exc))
                self._emotion = None
        if self._identity is None:
            try:
                from faceview.vision.identity import IdentityRecognizer
                self._identity = IdentityRecognizer()
                self._identity.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("identity.start_failed", error=str(exc))
                self._identity = None

    # ── lifecycle: microphone ──────────────────────────────────────

    def audio_running(self) -> bool:
        return self._audio_worker is not None

    def set_audio_enabled(self, on: bool) -> None:
        if on and self._audio_worker is None:
            try:
                from faceview.speech.audio_capture import AudioCapture
                self._audio_worker = AudioCapture()
                self._audio_worker.start()
                self._start_stt_chain()
                self.statusBar().showMessage("Microphone started")
            except Exception as exc:  # noqa: BLE001
                log.warning("audio.start_failed", error=str(exc))
                self.statusBar().showMessage(f"Microphone unavailable: {exc}")
                self._audio_worker = None
        elif not on and self._audio_worker is not None:
            try:
                self._audio_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._audio_worker = None
            self.statusBar().showMessage("Microphone stopped")

    def _start_stt_chain(self) -> None:
        """Bring up VAD + STT downstream of the audio capture worker."""
        if self._vad is None:
            try:
                from faceview.speech.vad import VadGate
                self._vad = VadGate()
                self._vad.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("vad.start_failed", error=str(exc))
                self._vad = None
        if self._stt is None:
            try:
                from faceview.speech.stt import SttWorker
                self._stt = SttWorker()
                self._stt.start()
            except Exception as exc:  # noqa: BLE001
                log.warning("stt.start_failed", error=str(exc))
                self._stt = None
        # Bridge final transcripts to the chat → LLM bus, just once.
        # Without this, STT output only ever reaches the transcript
        # panel; the user speaks and Claude never replies.
        if not getattr(self, "_stt_to_chat_wired", False):
            from faceview.core.event_bus import get_bus
            from faceview.core.events import (
                ChatMessage, EventType, Transcript,
            )
            bus = get_bus()

            def _stt_to_chat(payload) -> None:
                text = payload.text if isinstance(payload, Transcript) else str(payload)
                text = (text or "").strip()
                if not text:
                    return
                # Drop obvious filler (faster-whisper sometimes emits
                # silence transcriptions like "you" or "Thanks for watching").
                if len(text) < 2:
                    return
                bus.publish(EventType.CHAT_USER_MESSAGE,
                            ChatMessage("user", text))

            bus.subscribe(EventType.TRANSCRIPT_FINAL, _stt_to_chat)
            self._stt_to_chat_wired = True

    # ── lifecycle: TTS (Claude speaks) ─────────────────────────────

    def tts_running(self) -> bool:
        return self._tts is not None

    def set_tts_enabled(self, on: bool) -> None:
        if on and self._tts is None:
            try:
                from faceview.core.event_bus import get_bus
                from faceview.core.events import ChatMessage, EventType
                from faceview.speech.tts import TtsWorker
                self._tts = TtsWorker()
                self._tts.start()
                # Bridge LLM_REPLY → TTS_SPEAK so Claude actually speaks.
                bus = get_bus()
                def _say_reply(msg):
                    text = getattr(msg, "content", "") if isinstance(msg, ChatMessage) else str(msg)
                    if text:
                        bus.publish(EventType.TTS_SPEAK, text)
                self._tts_bridge = _say_reply
                bus.subscribe(EventType.LLM_REPLY, _say_reply)
                self.statusBar().showMessage("TTS started — Claude will speak replies")
            except Exception as exc:  # noqa: BLE001
                log.warning("tts.start_failed", error=str(exc))
                self.statusBar().showMessage(f"TTS unavailable: {exc}")
                self._tts = None
        elif not on and self._tts is not None:
            try:
                self._tts.stop()
            except Exception:  # noqa: BLE001
                pass
            self._tts = None
            self.statusBar().showMessage("TTS stopped")

    # ── lifecycle: mirror mode ────────────────────────────────────

    def mirror_running(self) -> bool:
        return getattr(self, "_mirror_state", None) is not None and self._mirror_state.active

    def set_mirror_mode_enabled(self, on: bool) -> None:
        if on:
            if not self._camera_worker:
                self.statusBar().showMessage(
                    "Mirror mode needs the camera on (Tools → Toggle camera)"
                )
                return
            if not self._avatar_worker:
                self.set_avatar_enabled(True)
            if getattr(self, "_mirror_state", None) is None:
                from faceview.vision.mirror import MirrorState
                self._mirror_state = MirrorState()
                self._mirror_state.attach_bus()
            self._mirror_state.active = True
            self._avatar_worker.set_mirror_provider(self._mirror_state.face_params)
            self.statusBar().showMessage("Mirror mode: Claude mimics you")
        else:
            if getattr(self, "_mirror_state", None) is not None:
                self._mirror_state.active = False
            if self._avatar_worker is not None:
                self._avatar_worker.set_mirror_provider(None)
            self.statusBar().showMessage("Mirror mode off")

    # ── owner enrollment ─────────────────────────────────────────

    def enroll_owner(self, n_samples: int = 10) -> None:
        """Capture N frames, embed each off-thread, save the averaged template.

        The handler subscribed to FRAME runs on the GUI thread, so it only
        grabs cheap numpy copies. The expensive InsightFace inference runs
        in a worker thread to avoid contending with the identity recogniser
        (which already embeds at 2 Hz on the same bus thread).
        """
        if self._identity is None or self._camera_worker is None:
            self.statusBar().showMessage(
                "Enroll: start the camera + identity first (Tools → Toggle camera)"
            )
            return

        import threading
        import time
        import numpy as np
        from faceview.core.event_bus import get_bus
        from faceview.core.events import EventType

        frames: list[np.ndarray] = []
        target_frames = max(n_samples * 3, 30)  # over-capture; some embeds will fail

        last_capture = [0.0]

        def _on_frame(frame) -> None:
            if frame is None or len(frames) >= target_frames:
                return
            # Sample ~5 fps so we get varied poses, not 30 near-duplicates.
            now = time.time()
            if now - last_capture[0] < 0.2:
                return
            last_capture[0] = now
            frames.append(frame.copy())

        bus = get_bus()
        bus.subscribe(EventType.FRAME, _on_frame)
        self.statusBar().showMessage(
            f"Enrolling… hold still while I grab {target_frames} frames"
        )

        def _finish() -> None:
            t0 = time.time()
            while len(frames) < target_frames and time.time() - t0 < 10.0:
                time.sleep(0.1)
            try:
                bus.unsubscribe(EventType.FRAME, _on_frame)
            except Exception:  # noqa: BLE001
                pass
            if not frames:
                self.statusBar().showMessage("Enroll failed: no frames captured")
                return
            log.info("enroll.frames_captured", count=len(frames))
            samples: list[np.ndarray] = []
            for i, frame in enumerate(frames):
                try:
                    emb = self._identity.embed(frame)
                except Exception as exc:  # noqa: BLE001
                    log.warning("enroll.embed_error", error=str(exc), idx=i)
                    continue
                if emb is not None:
                    samples.append(emb)
                if len(samples) >= n_samples:
                    break
            if not samples:
                self.statusBar().showMessage(
                    f"Enroll failed: no face detected in any of {len(frames)} frames"
                )
                return
            mean = np.stack(samples).mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            path = self._identity.save_owner_template(mean)
            self.statusBar().showMessage(
                f"Enrolled owner from {len(samples)}/{len(frames)} frames → {path.name}"
            )

        threading.Thread(target=_finish, name="enroll-owner", daemon=True).start()

    # ── monitor windows ────────────────────────────────────────────

    def open_monitor(self, kind: str) -> None:
        from faceview.gui import monitors
        cls = monitors.MONITORS.get(kind)
        if cls is None:
            return
        win = self._monitors.get(kind)
        if win is None:
            win = cls(self)
            self._monitors[kind] = win
        win.show()
        win.raise_()
        win.activateWindow()

    # ── lifecycle: avatar ─────────────────────────────────────────

    def avatar_running(self) -> bool:
        return self._avatar_worker is not None

    def current_persona(self) -> str:
        return self._current_persona

    def refresh_llm_pill(self) -> None:
        """Update the LLM status pill to reflect what's *actually* driving
        conversation right now. Test mode with a real engine wins over the
        main client — the visible state should match the running bots.
        """
        from faceview.gui.status_panel import _model_short
        colors = {"anthropic": "#3a8", "ollama": "#5e72e4", "demo": "#666"}

        # Test mode with an LLM engine overrides the main pill.
        if self.test_mode_running():
            test_engine = (os.environ.get("FACEVIEW_TEST_ENGINE") or "canned").lower()
            if test_engine in ("anthropic", "ollama", "demo"):
                model = os.environ.get("FACEVIEW_TEST_MODEL") or ""
                if test_engine == "anthropic":
                    label = _model_short(model or settings.anthropic_model)
                elif test_engine == "ollama":
                    label = (model or "ollama").split(":")[0]
                else:
                    label = "demo"
                self.status_panel.set_llm_label(
                    f"⇄ {label}", color=colors.get(test_engine, "#666"),
                )
                return
            if test_engine == "canned":
                self.status_panel.set_llm_label("⇄ canned", color="#666")
                return

        # Otherwise: the main ClaudeClient.
        client = getattr(self, "llm_client", None)
        engine = client.current_engine() if (
            client is not None and hasattr(client, "current_engine")
        ) else None
        if engine == "anthropic":
            label = _model_short(settings.anthropic_model)
        elif engine == "ollama":
            model = os.environ.get("FACEVIEW_OLLAMA_MODEL") or "ollama"
            label = model.split(":")[0]
        else:
            label = "demo mode"
            engine = engine or "demo"
        self.status_panel.set_llm_label(label, color=colors.get(engine, "#666"))

    def set_avatar_enabled(self, on: bool) -> None:
        if on and self._avatar_worker is None:
            self._start_avatar_worker(self._current_persona)
            self.show_avatar_window()
            self._avatar_action.setChecked(True)
        elif not on and self._avatar_worker is not None:
            self._stop_avatar_worker()
            self.statusBar().showMessage("Avatar stopped")

    def set_persona(self, name: str) -> None:
        prev = self._current_persona
        self._current_persona = name
        # Swap the persona in place rather than restarting the worker —
        # restarting raced the old + new render threads on the ICT mesh
        # caches and segfaulted the process. TalkingAvatar.set_persona
        # only rebinds the appearance overlay; the render loop keeps
        # ticking on the same thread.
        if self._avatar_worker is not None:
            try:
                self._avatar_worker.avatar.set_persona(name)
                self.statusBar().showMessage(f"Avatar persona: {name}")
            except Exception as exc:  # noqa: BLE001
                log.warning("persona.swap_failed", error=str(exc))
                self.statusBar().showMessage(f"Persona swap failed: {exc}")
        # Flip the LLM client to this persona's memory pool so the new
        # face also has a matching set of recollections + ledger.
        if prev != name:
            self._bind_memory_for_current_persona(save_previous=True)

    def _bind_memory_for_current_persona(self, *, save_previous: bool = False) -> None:
        client = getattr(self, "llm_client", None)
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
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.bind_failed", error=str(exc))

    def _start_avatar_worker(self, persona: str) -> None:
        from faceview.core.events import EventType as _ET
        from faceview.vision.sim_camera import SimCameraWorker
        self._avatar_worker = SimCameraWorker(
            scenario="avatar",
            emotion="happy",
            persona=persona,
            wire_to_llm=True,
            frame_channel=_ET.AVATAR_FRAME,
            publish_user_events=False,
        )
        self._avatar_worker.start()
        # Re-bind the service's avatar handle so MCP/HTTP ops keep working.
        try:
            from faceview.server.service import get_service
            get_service().bind_camera_worker(self._avatar_worker)
        except Exception:  # noqa: BLE001
            pass

    def _stop_avatar_worker(self) -> None:
        if self._avatar_worker is not None:
            try:
                self._avatar_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._avatar_worker = None

    # ── lifecycle: dual-Claude test mode ──────────────────────────

    def test_mode_running(self) -> bool:
        return self._test_orchestrator is not None

    def set_test_mode_enabled(self, on: bool) -> None:
        if on and self._test_orchestrator is None:
            self._start_test_mode()
        elif not on and self._test_orchestrator is not None:
            self._stop_test_mode()

    def _start_test_mode(self) -> None:
        from faceview.core.events import EventType as _ET
        from faceview.llm.test_conversation import TestConversation
        from faceview.vision.sim_camera import SimCameraWorker
        # Ensure the Claude-side avatar is running.
        if self._avatar_worker is None:
            self._start_avatar_worker(self._current_persona)
            self.show_avatar_window()
        # Replace the real webcam with a second bot avatar that fills
        # the camera window. Stop the camera worker first so the two
        # don't fight over the FRAME channel.
        self.set_camera_enabled(False)
        self._user_avatar_worker = SimCameraWorker(
            scenario="avatar",
            emotion="neutral",
            persona=self._test_mode_partner_persona(),
            wire_to_llm=False,
            frame_channel=_ET.FRAME,
            publish_user_events=False,
        )
        self._user_avatar_worker.start()

        engine_a, engine_b, engine_name = self._build_test_engines()
        persona_a = self._test_mode_partner_persona()
        persona_b = self._current_persona
        self._test_orchestrator = TestConversation(
            avatar_worker=self._avatar_worker,
            user_worker=self._user_avatar_worker,
            engine_a=engine_a,
            engine_b=engine_b,
            chat_panel=self.chat,
            persona_a=persona_a,
            persona_b=persona_b,
        )
        self._test_orchestrator.start()
        mode = "LLM (" + engine_name + ")" if engine_a is not None else "canned"
        self.statusBar().showMessage(f"Test mode: two bots conversing — {mode}")
        self.refresh_llm_pill()

    def _build_test_engines(self) -> tuple[object | None, object | None, str]:
        """Construct two engines for test mode based on env vars / settings.

        Honours ``FACEVIEW_TEST_ENGINE`` (canned / anthropic / ollama / demo)
        and ``FACEVIEW_TEST_MODEL`` (model id for the chosen engine).
        Returns ``(engine_a, engine_b, name)``; ``(None, None, "canned")``
        falls back to the seed-prompt loop.
        """
        engine_name = (os.environ.get("FACEVIEW_TEST_ENGINE") or "canned").lower()
        model = os.environ.get("FACEVIEW_TEST_MODEL") or None
        if engine_name in ("", "canned", "seed", "off"):
            return None, None, "canned"
        try:
            if engine_name == "anthropic":
                if not settings.has_claude_key:
                    raise RuntimeError("ANTHROPIC_API_KEY not set")
                from faceview.llm.claude_client import AnthropicEngine
                m = model or settings.anthropic_model
                return (
                    AnthropicEngine(api_key=settings.anthropic_api_key, model=m),  # type: ignore[arg-type]
                    AnthropicEngine(api_key=settings.anthropic_api_key, model=m),  # type: ignore[arg-type]
                    f"anthropic:{m}",
                )
            if engine_name == "ollama":
                from faceview.llm.ollama_client import OllamaEngine, pick_default_model
                m = model or pick_default_model()
                if not m:
                    raise RuntimeError("no ollama models installed")
                return OllamaEngine(model=m), OllamaEngine(model=m), f"ollama:{m}"
            if engine_name == "demo":
                from faceview.llm.claude_client import EchoEngine
                return EchoEngine(), EchoEngine(), "demo"
        except Exception as exc:  # noqa: BLE001
            log.warning("test_mode.engine_build_failed", engine=engine_name, error=str(exc))
            self.statusBar().showMessage(
                f"Test mode: {engine_name} unavailable — falling back to canned"
            )
        return None, None, "canned"

    def restart_test_mode(self) -> None:
        """Stop and re-start test mode so engine-config changes take effect."""
        if self._test_orchestrator is None:
            return
        self._stop_test_mode()
        self._start_test_mode()

    def _stop_test_mode(self) -> None:
        if self._test_orchestrator is not None:
            try:
                self._test_orchestrator.stop()
            except Exception:  # noqa: BLE001
                pass
            self._test_orchestrator = None
        if self._user_avatar_worker is not None:
            try:
                self._user_avatar_worker.stop()
            except Exception:  # noqa: BLE001
                pass
            self._user_avatar_worker = None
        self.statusBar().showMessage("Test mode stopped")
        self.refresh_llm_pill()

    def _test_mode_partner_persona(self) -> str:
        # Pick a different persona (and therefore character) for the
        # camera-window bot. Prefer lightweight (non-ICT-3D, non-GPU)
        # personas because running two heavy renderers in parallel races
        # on moderngl's GL context and segfaults the process.
        try:
            from faceview.llm.character import list_character_keys
            from faceview.vision.personas import load_persona
            keys = [k for k in list_character_keys()
                    if k != self._current_persona and not k.endswith("_fallback")]
        except Exception:  # noqa: BLE001
            keys = []
        # Filter to stylised render modes for safety, fall back to any
        # registered character if no stylised ones are available.
        safe: list[str] = []
        try:
            from faceview.vision.personas import load_persona
            for k in keys:
                try:
                    mode = getattr(load_persona(k), "render_mode", "stylised") or "stylised"
                except Exception:  # noqa: BLE001
                    mode = "stylised"
                if mode in {"stylised", "anatomical", "anatomy_overlay", "wireframe"}:
                    safe.append(k)
        except Exception:  # noqa: BLE001
            safe = keys
        pool = safe or keys
        if not pool:
            return "warm_tan" if self._current_persona != "warm_tan" else "claude"
        idx = abs(hash(self._current_persona)) % len(pool)
        return pool[idx]

    # ── lifecycle: clean shutdown ─────────────────────────────────

    def closeEvent(self, ev) -> None:  # noqa: N802 — Qt API
        # Persist the dock layout before workers shut down so the next
        # launch restores the user's last arrangement.
        try:
            self.layout_mgr.save(quiet=True)
        except Exception:  # noqa: BLE001
            pass
        # Flush LLM memory to disk so this session's turns survive.
        client = getattr(self, "llm_client", None)
        if client is not None and getattr(client, "memory", None) is not None:
            try:
                client.memory.save()
            except Exception:  # noqa: BLE001
                pass
        for stopper in (
            self._stop_test_mode,
            lambda: self.set_camera_enabled(False),
            lambda: self.set_audio_enabled(False),
            lambda: self.set_tts_enabled(False),
            self._stop_avatar_worker,
        ):
            try:
                stopper()
            except Exception:  # noqa: BLE001
                pass
        if self.avatar_window is not None:
            self.avatar_window.close()
        for win in list(self._monitors.values()):
            try:
                win.close()
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(ev)

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _default_avatar_persona() -> str:
        try:
            from faceview.vision.ict_face import _data_path as _ict_data
            return "ict_xray_young" if _ict_data().exists() else "claude"
        except Exception:  # noqa: BLE001
            return "claude"
