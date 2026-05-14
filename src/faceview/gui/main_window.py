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

A separate :class:`AvatarWindow` (Claude's face) lives next to this
one and is opened automatically at boot.

This class is intentionally a thin facade: panels + menu + layout
live here, but every worker lifecycle (camera, audio, TTS, avatar,
test mode, monitor windows, owner enrollment) lives in a dedicated
controller under :mod:`faceview.gui.controllers`. External callers
(``app.py``, ``server/service.py``, ``config_dialog.py``,
``chat_panel.py``) see the same public methods they always did —
they just delegate one hop into the appropriate controller.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMainWindow, QStatusBar

from faceview.config import settings
from faceview.core.logger import get_logger
from faceview.gui.avatar_window import AvatarWindow
from faceview.gui.camera_panel import CameraPanel
from faceview.gui.chat_panel import ChatPanel
from faceview.gui.controllers import (
    AudioController, AvatarController, CameraController,
    EnrollmentController, MonitorController, TestModeController,
    TtsController,
)
from faceview.gui.layout import LayoutManager
from faceview.gui.perception_panel import PerceptionPanel
from faceview.gui.screenshotter import Screenshotter
from faceview.gui.status_panel import StatusPanel
from faceview.gui.transcript_panel import TranscriptPanel

if TYPE_CHECKING:
    from pathlib import Path


log = get_logger("main_window")


class MainWindow(QMainWindow):
    """Shell that composes panels and delegates lifecycle to controllers."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("faceView — You")
        self.resize(1280, 800)

        # Panels (Qt widgets — must exist before LayoutManager.build).
        self.shotter = Screenshotter()
        self.camera = CameraPanel(self)
        self.chat = ChatPanel(self)
        self.status_panel = StatusPanel(self)
        self.transcript = TranscriptPanel(self)
        self.perception_panel = PerceptionPanel(self)

        # Companion window for Claude's avatar.
        self.avatar_window: Optional[AvatarWindow] = None

        # Lifecycle controllers — each owns its workers + state.
        # Order matters: AudioController calls into tts_ctrl on
        # push-to-speak, so tts_ctrl must exist first. AvatarController
        # reads camera_ctrl for mirror-mode preconditions; same idea.
        self.camera_ctrl = CameraController(self)
        self.tts_ctrl = TtsController(self)
        self.audio_ctrl = AudioController(self)
        self.avatar_ctrl = AvatarController(self)
        self.test_mode_ctrl = TestModeController(self)
        self.monitor_ctrl = MonitorController(self)
        self.enrollment_ctrl = EnrollmentController(self)

        # Room-map worker — runs at ~1 Hz only while the map window is
        # open. Lives here (not on a controller) since it has a single
        # owner and a single consumer.
        from faceview.vision.room_map import RoomMapWorker
        self.room_map_worker = RoomMapWorker()
        self.room_map_worker.start()
        self.room_map_window = None

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("faceView ready")

    # ── layout / menu ─────────────────────────────────────────────

    def _build_layout(self) -> None:
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
        a_map = QAction("Room map…", self)
        a_map.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        a_map.triggered.connect(self.open_room_map)
        m_view.addAction(a_map)

        m_tools = self.menuBar().addMenu("&Tools")
        a_cfg = QAction("Configuration…", self)
        a_cfg.setShortcut(QKeySequence("Ctrl+,"))
        a_cfg.triggered.connect(self._open_config_dialog)
        m_tools.addAction(a_cfg)
        a_cam = QAction("Toggle camera", self)
        a_cam.setShortcut(QKeySequence("Ctrl+Shift+C"))
        a_cam.triggered.connect(
            lambda: self.set_camera_enabled(not self.camera_running())
        )
        m_tools.addAction(a_cam)
        a_mic = QAction("Toggle microphone", self)
        a_mic.setShortcut(QKeySequence("Ctrl+Shift+M"))
        a_mic.triggered.connect(
            lambda: self.set_audio_enabled(not self.audio_running())
        )
        m_tools.addAction(a_mic)
        a_tts = QAction("Toggle Claude voice (TTS)", self)
        a_tts.setShortcut(QKeySequence("Ctrl+Shift+T"))
        a_tts.triggered.connect(
            lambda: self.set_tts_enabled(not self.tts_running())
        )
        m_tools.addAction(a_tts)
        a_enroll = QAction("Enroll me as owner…", self)
        a_enroll.setShortcut(QKeySequence("Ctrl+Shift+E"))
        a_enroll.triggered.connect(self.enroll_owner)
        m_tools.addAction(a_enroll)
        a_mirror = QAction(
            "Toggle mirror mode (avatar mimics user)", self,
        )
        a_mirror.setShortcut(QKeySequence("Ctrl+Shift+R"))
        a_mirror.triggered.connect(
            lambda: self.set_mirror_mode_enabled(not self.mirror_running())
        )
        m_tools.addAction(a_mirror)

        m_window = self.menuBar().addMenu("&Window")
        self.layout_mgr.install_menu(m_window)

        m_monitor = self.menuBar().addMenu("&Monitor")
        for kind, label, shortcut in (
            ("audio",      "Audio waveform / VAD…", "Ctrl+1"),
            ("emotion",    "Emotion scores…",        "Ctrl+2"),
            ("mouth",      "Mouth + visemes…",       "Ctrl+3"),
            ("transcript", "Transcript history…",    "Ctrl+4"),
        ):
            act = QAction(label, self)
            act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(
                lambda _checked=False, k=kind: self.open_monitor(k)
            )
            m_monitor.addAction(act)

    # ── companion window ───────────────────────────────────────────

    def show_avatar_window(self) -> None:
        if self.avatar_window is None:
            self.avatar_window = AvatarWindow(self)
            # Place the avatar window to the right of the main window.
            geo = self.geometry()
            self.avatar_window.move(geo.x() + geo.width() + 12, geo.y())
        self.avatar_window.show()
        self.avatar_window.raise_()

    def _toggle_avatar_window(self, on: bool) -> None:
        if on:
            self.show_avatar_window()
        elif self.avatar_window is not None:
            self.avatar_window.hide()

    # ── popup dialogs ──────────────────────────────────────────────

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

    def open_room_map(self) -> None:
        """Open the top-down room-map window. Worker runs only while
        the window is visible."""
        from faceview.gui.room_map_panel import RoomMapWindow
        if self.room_map_window is None:
            self.room_map_window = RoomMapWindow(self)
            # Place to the right of the avatar window when possible.
            geo = self.geometry()
            self.room_map_window.move(geo.x() + 80, geo.y() + 80)
        self.room_map_window.show()
        self.room_map_window.raise_()
        self.room_map_window.activateWindow()

    def _open_persona_picker(self) -> None:
        from faceview.gui.persona_picker import PersonaPicker
        if not hasattr(self, "_persona_picker") or self._persona_picker is None:
            self._persona_picker = PersonaPicker(self, self)
        else:
            self._persona_picker._refresh_selection()
        self._persona_picker.show()
        self._persona_picker.raise_()
        self._persona_picker.activateWindow()

    # ── public helpers (also used by ConfigDialog / Service) ──────

    def take_screenshot(self, name: str) -> "Path":
        return self.shotter.capture_window(self, name)

    def seed_demo_state(self) -> None:
        self.chat.seed_demo_conversation()
        self.status_panel.seed_demo()
        self.transcript.seed_demo()
        self.statusBar().showMessage(
            "Demo state — owner recognised, mic idle, camera idle."
        )

    # ── lifecycle facade — delegates to controllers ───────────────

    def camera_running(self) -> bool:
        return self.camera_ctrl.is_running()

    def set_camera_enabled(self, on: bool) -> None:
        self.camera_ctrl.set_enabled(on)

    def audio_running(self) -> bool:
        return self.audio_ctrl.is_running()

    def set_audio_enabled(self, on: bool) -> None:
        self.audio_ctrl.set_enabled(on)

    def tts_running(self) -> bool:
        return self.tts_ctrl.is_running()

    def set_tts_enabled(self, on: bool) -> None:
        self.tts_ctrl.set_enabled(on)

    def push_to_speak_pressed(self) -> None:
        self.audio_ctrl.push_to_speak_pressed()

    def push_to_speak_released(self) -> None:
        self.audio_ctrl.push_to_speak_released()

    def avatar_running(self) -> bool:
        return self.avatar_ctrl.is_running()

    def set_avatar_enabled(self, on: bool) -> None:
        self.avatar_ctrl.set_enabled(on)

    def current_persona(self) -> str:
        return self.avatar_ctrl.current_persona()

    def set_persona(self, name: str) -> None:
        self.avatar_ctrl.set_persona(name)

    def mirror_running(self) -> bool:
        return self.avatar_ctrl.mirror_running()

    def set_mirror_mode_enabled(self, on: bool) -> None:
        self.avatar_ctrl.set_mirror_enabled(on)

    def _bind_memory_for_current_persona(
        self, *, save_previous: bool = False,
    ) -> None:
        self.avatar_ctrl.bind_memory_for_current_persona(
            save_previous=save_previous,
        )

    def test_mode_running(self) -> bool:
        return self.test_mode_ctrl.is_running()

    def set_test_mode_enabled(self, on: bool) -> None:
        self.test_mode_ctrl.set_enabled(on)

    def restart_test_mode(self) -> None:
        self.test_mode_ctrl.restart()

    def enroll_owner(self, n_samples: int = 10) -> None:
        self.enrollment_ctrl.enroll_owner(n_samples=n_samples)

    def open_monitor(self, kind: str) -> None:
        self.monitor_ctrl.open(kind)

    # ── status pill (lives here because multiple controllers + the
    # config dialog hit it) ───────────────────────────────────────

    def refresh_llm_pill(self) -> None:
        """Update the LLM status pill to reflect what's actually
        driving conversation right now. Test mode with a real engine
        wins over the main client — the visible state should match
        the running bots."""
        from faceview.gui.status_panel import _model_short
        colors = {"anthropic": "#3a8", "ollama": "#5e72e4", "demo": "#666"}

        # Test mode with an LLM engine overrides the main pill.
        if self.test_mode_running():
            test_engine = (os.environ.get("FACEVIEW_TEST_ENGINE")
                           or "canned").lower()
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

        client = getattr(self, "llm_client", None)
        engine = (client.current_engine() if (
            client is not None and hasattr(client, "current_engine")
        ) else None)
        if engine == "anthropic":
            label = _model_short(settings.anthropic_model)
        elif engine == "ollama":
            model = os.environ.get("FACEVIEW_OLLAMA_MODEL") or "ollama"
            label = model.split(":")[0]
        else:
            label = "demo mode"
            engine = engine or "demo"
        self.status_panel.set_llm_label(
            label, color=colors.get(engine, "#666"),
        )

    # ── clean shutdown ────────────────────────────────────────────

    def closeEvent(self, ev) -> None:  # noqa: N802 — Qt API
        # Persist the dock layout before workers shut down so the
        # next launch restores the user's last arrangement.
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
            self.test_mode_ctrl._stop,
            lambda: self.camera_ctrl.set_enabled(False),
            lambda: self.audio_ctrl.set_enabled(False),
            lambda: self.tts_ctrl.set_enabled(False),
            self.avatar_ctrl.stop,
            self.monitor_ctrl.close_all,
            self.room_map_worker.stop,
        ):
            try:
                stopper()
            except Exception:  # noqa: BLE001
                pass
        if self.avatar_window is not None:
            self.avatar_window.close()
        if self.room_map_window is not None:
            try:
                self.room_map_window.close()
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(ev)
