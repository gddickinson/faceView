"""ConfigDialog — runtime settings for the faceView GUI.

Accessible from Tools → Configuration… in the main window. Lets the user:

- enable / disable the webcam capture worker (controls what Claude "sees")
- enable / disable the avatar worker (controls what the user "sees")
- pick a persona / render-mode for the avatar
- pick the LLM model
- pick a head-nod mode for animated 3-D heads
- toggle mic capture (placeholder for the STT pipeline wire-up)

Settings are applied live where possible and persisted to
:class:`faceview.config.settings` for the lifetime of the process. The
heavier reload paths (changing persona while the avatar is running)
restart the avatar worker through the parent ``MainWindow``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from faceview.config import settings

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class ConfigDialog(QDialog):
    """Modeless settings dialog."""

    def __init__(self, main_window: "MainWindow", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent or main_window)
        self.setWindowTitle("Configuration")
        self.setMinimumWidth(420)
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        intro = QLabel(
            "Configure the GUI: pick what Claude looks like, which model "
            "powers the conversation, and whether the camera/mic are on."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Hardware
        self.cam_box = QCheckBox("Enable webcam (Claude sees the user)")
        self.cam_box.setChecked(self.main_window.camera_running())
        self.cam_box.toggled.connect(self.main_window.set_camera_enabled)
        form.addRow("Camera", self.cam_box)

        self.mic_box = QCheckBox("Enable microphone (STT to chat input)")
        self.mic_box.setChecked(self.main_window.audio_running())
        self.mic_box.toggled.connect(self.main_window.set_audio_enabled)
        form.addRow("Microphone", self.mic_box)

        self.tts_box = QCheckBox("Speak Claude's replies aloud (TTS)")
        self.tts_box.setChecked(self.main_window.tts_running())
        self.tts_box.toggled.connect(self.main_window.set_tts_enabled)
        form.addRow("Claude voice", self.tts_box)

        # Avatar
        self.avatar_box = QCheckBox("Show Claude avatar (separate window)")
        self.avatar_box.setChecked(self.main_window.avatar_running())
        self.avatar_box.toggled.connect(self.main_window.set_avatar_enabled)
        form.addRow("Avatar", self.avatar_box)

        # Dual-Claude test mode
        self.test_box = QCheckBox(
            "Test mode: two bots talk to each other (camera + avatar windows)"
        )
        self.test_box.setChecked(self.main_window.test_mode_running())
        self.test_box.toggled.connect(self.main_window.set_test_mode_enabled)
        form.addRow("Test mode", self.test_box)

        # Mirror mode
        self.mirror_box = QCheckBox(
            "Mirror mode: avatar mimics user's expression + mouth"
        )
        self.mirror_box.setChecked(self.main_window.mirror_running())
        self.mirror_box.toggled.connect(self.main_window.set_mirror_mode_enabled)
        form.addRow("Mirror", self.mirror_box)

        # Persona
        self.persona_combo = QComboBox()
        for name in self._persona_names():
            self.persona_combo.addItem(name)
        current_persona = self.main_window.current_persona() or "default"
        idx = self.persona_combo.findText(current_persona)
        if idx >= 0:
            self.persona_combo.setCurrentIndex(idx)
        self.persona_combo.currentTextChanged.connect(self.main_window.set_persona)
        form.addRow("Persona", self.persona_combo)

        # Nod mode (ICT 3-D heads)
        self.nod_combo = QComboBox()
        for name in self._nod_modes():
            self.nod_combo.addItem(name)
        current_nod = os.environ.get("FACEVIEW_NOD_MODE", "head_block_neck_stretch")
        idx = self.nod_combo.findText(current_nod)
        if idx >= 0:
            self.nod_combo.setCurrentIndex(idx)
        self.nod_combo.currentTextChanged.connect(self._on_nod_changed)
        form.addRow("Head-nod mode", self.nod_combo)

        # Model
        self.model_combo = QComboBox()
        for m in _MODELS:
            self.model_combo.addItem(m)
        if settings.anthropic_model not in _MODELS:
            self.model_combo.addItem(settings.anthropic_model)
        self.model_combo.setCurrentText(settings.anthropic_model)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        form.addRow("LLM model", self.model_combo)

        root.addLayout(form)

        # API-key status (read-only — set via env var)
        key_state = "set" if settings.has_claude_key else "missing — using demo echo"
        key_label = QLabel(f"ANTHROPIC_API_KEY: <b>{key_state}</b>")
        key_label.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(key_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        root.addWidget(buttons)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _persona_names() -> list[str]:
        try:
            from faceview.vision.personas import list_personas
            return list_personas() or ["default"]
        except Exception:  # noqa: BLE001
            return ["default", "claude"]

    @staticmethod
    def _nod_modes() -> list[str]:
        try:
            from faceview.vision.ict_face import _NOD_MODES
            return list(_NOD_MODES.keys())
        except Exception:  # noqa: BLE001
            return ["head_block_neck_stretch"]

    # ── slots ────────────────────────────────────────────────────────

    def _on_nod_changed(self, name: str) -> None:
        os.environ["FACEVIEW_NOD_MODE"] = name

    def _on_model_changed(self, name: str) -> None:
        settings.anthropic_model = name
        os.environ["FACEVIEW_MODEL"] = name
        # Refresh the LLM pill in the main window's status panel.
        try:
            from faceview.gui.status_panel import _model_short
            label = _model_short(name) if settings.has_claude_key else f"demo · {_model_short(name)}"
            self.main_window.status_panel.set_llm_label(label)
        except Exception:  # noqa: BLE001
            pass
        # The Anthropic engine re-reads settings.anthropic_model on each
        # send_reply call, so the new model takes effect on the next message.
