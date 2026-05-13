"""ConfigDialog — runtime settings for the faceView GUI.

Accessible from Tools → Configuration… in the main window. Tabs:

- **General**: webcam / mic / TTS / avatar / test / mirror toggles.
- **LLM**:     engine picker (auto / anthropic / ollama / demo) plus the
               model combo for whichever engine is active, with a live
               status pill update.
- **Avatar**:  persona combo + persona-picker shortcut, head-nod cascade
               mode, body-rig weighting mode.

Settings apply live where possible: workers stop/start through the
``MainWindow`` lifecycle methods, the LLM engine is swapped on
``window.llm_client`` (no app restart), and rendering env-vars
(``FACEVIEW_NOD_MODE``, ``FACEVIEW_RIG_WEIGHT_MODE``) update in-place
for any new render thread.
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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from faceview.config import settings

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


_ANTHROPIC_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

_ENGINE_CHOICES = [
    ("auto",      "Auto — key → Anthropic, else Ollama, else demo"),
    ("anthropic", "Anthropic (cloud, needs ANTHROPIC_API_KEY)"),
    ("ollama",    "Ollama (local, needs `ollama serve`)"),
    ("demo",      "Demo echo (no network)"),
]

# For test-mode bots only (no "auto" — the user must pick deliberately).
_TEST_ENGINE_CHOICES = [
    ("canned",    "Canned seed prompts (no network, deterministic)"),
    ("ollama",    "Ollama (local) — two bots, same model"),
    ("anthropic", "Anthropic — two bots, same model (uses tokens)"),
    ("demo",      "Demo echo (stub, mostly for tests)"),
]

_RIG_MODES = ["graded_3ring", "hard"]


class ConfigDialog(QDialog):
    """Tabbed runtime-settings dialog."""

    def __init__(
        self,
        main_window: "MainWindow",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent or main_window)
        self.setWindowTitle("Configuration")
        self.setMinimumSize(520, 460)
        self.main_window = main_window
        self._build_ui()

    # ── layout ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        intro = QLabel(
            "Configure faceView: enable / disable workers, pick an LLM "
            "engine, choose how the avatar looks and moves."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#9aa3b2;")
        root.addWidget(intro)

        tabs = QTabWidget(self)
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_llm_tab(), "LLM")
        tabs.addTab(self._build_avatar_tab(), "Avatar")
        root.addWidget(tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        root.addWidget(buttons)

    def _build_general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        mw = self.main_window

        self.cam_box = QCheckBox("Enable webcam (Claude sees the user)")
        self.cam_box.setChecked(mw.camera_running())
        self.cam_box.toggled.connect(mw.set_camera_enabled)
        form.addRow("Camera", self.cam_box)

        self.mic_box = QCheckBox("Enable microphone (STT to chat input)")
        self.mic_box.setChecked(mw.audio_running())
        self.mic_box.toggled.connect(mw.set_audio_enabled)
        form.addRow("Microphone", self.mic_box)

        self.tts_box = QCheckBox("Speak Claude's replies aloud (TTS)")
        self.tts_box.setChecked(mw.tts_running())
        self.tts_box.toggled.connect(mw.set_tts_enabled)
        form.addRow("Claude voice", self.tts_box)

        # TTS engine + voice
        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItem("Auto (kokoro if available, else pyttsx3)", "auto")
        self.tts_engine_combo.addItem("Kokoro (neural, local)", "kokoro")
        self.tts_engine_combo.addItem("pyttsx3 (macOS system voices)", "pyttsx3")
        cur = (os.environ.get("FACEVIEW_TTS_ENGINE") or "auto").lower()
        idx = self.tts_engine_combo.findData(cur)
        self.tts_engine_combo.setCurrentIndex(max(0, idx))
        self.tts_engine_combo.currentIndexChanged.connect(self._on_tts_engine_changed)
        form.addRow("Voice engine", self.tts_engine_combo)

        self.tts_voice_combo = QComboBox()
        self.tts_voice_combo.setEditable(True)
        self.tts_voice_combo.setMinimumWidth(240)
        self._refresh_tts_voices()
        self.tts_voice_combo.currentTextChanged.connect(self._on_tts_voice_changed)
        form.addRow("Voice", self.tts_voice_combo)

        self.avatar_box = QCheckBox("Show Claude avatar (separate window)")
        self.avatar_box.setChecked(mw.avatar_running())
        self.avatar_box.toggled.connect(mw.set_avatar_enabled)
        form.addRow("Avatar", self.avatar_box)

        self.test_box = QCheckBox(
            "Test mode: two bots talk to each other (camera + avatar windows)"
        )
        self.test_box.setChecked(mw.test_mode_running())
        self.test_box.toggled.connect(mw.set_test_mode_enabled)
        form.addRow("Test mode", self.test_box)

        self.mirror_box = QCheckBox(
            "Mirror mode: avatar mimics user's expression + mouth"
        )
        self.mirror_box.setChecked(mw.mirror_running())
        self.mirror_box.toggled.connect(mw.set_mirror_mode_enabled)
        form.addRow("Mirror", self.mirror_box)

        return w

    def _build_llm_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.engine_combo = QComboBox()
        for key, label in _ENGINE_CHOICES:
            self.engine_combo.addItem(label, key)
        idx = self.engine_combo.findData(self._current_engine_key())
        self.engine_combo.setCurrentIndex(max(0, idx))
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine", self.engine_combo)

        self.anthropic_combo = QComboBox()
        for m in _ANTHROPIC_MODELS:
            self.anthropic_combo.addItem(m)
        if settings.anthropic_model not in _ANTHROPIC_MODELS:
            self.anthropic_combo.addItem(settings.anthropic_model)
        self.anthropic_combo.setCurrentText(settings.anthropic_model)
        self.anthropic_combo.currentTextChanged.connect(self._on_anthropic_model_changed)
        form.addRow("Anthropic model", self.anthropic_combo)

        ollama_row = QWidget()
        oh = QHBoxLayout(ollama_row)
        oh.setContentsMargins(0, 0, 0, 0)
        self.ollama_combo = QComboBox()
        self.ollama_combo.setMinimumWidth(220)
        refresh = QPushButton("Refresh")
        refresh.setMaximumWidth(80)
        refresh.clicked.connect(self._refresh_ollama_models)
        oh.addWidget(self.ollama_combo, 1)
        oh.addWidget(refresh, 0)
        self.ollama_combo.currentTextChanged.connect(self._on_ollama_model_changed)
        form.addRow("Ollama model", ollama_row)
        self._refresh_ollama_models()

        key_state = "set" if settings.has_claude_key else "missing — Anthropic disabled"
        self.key_label = QLabel(f"ANTHROPIC_API_KEY: <b>{key_state}</b>")
        self.key_label.setTextFormat(Qt.TextFormat.RichText)
        form.addRow("API key", self.key_label)

        hint = QLabel(
            "Engines swap live — no app restart. Auto re-detects on each "
            "selection. Demo never hits the network."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2;font-size:11px;")
        form.addRow(" ", hint)

        # ── Test-mode bot engine ─────────────────────────────────────
        from PySide6.QtWidgets import QFrame
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        form.addRow(sep)

        bots_header = QLabel("<b>Test-mode bots</b>")
        bots_header.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(bots_header)

        self.test_engine_combo = QComboBox()
        for key, label in _TEST_ENGINE_CHOICES:
            self.test_engine_combo.addItem(label, key)
        current_test = (os.environ.get("FACEVIEW_TEST_ENGINE") or "canned").lower()
        idx = self.test_engine_combo.findData(current_test)
        self.test_engine_combo.setCurrentIndex(max(0, idx))
        self.test_engine_combo.currentIndexChanged.connect(self._on_test_engine_changed)
        form.addRow("Bot engine", self.test_engine_combo)

        self.test_model_combo = QComboBox()
        self.test_model_combo.setEditable(True)
        self.test_model_combo.setMinimumWidth(220)
        self.test_model_combo.currentTextChanged.connect(self._on_test_model_changed)
        form.addRow("Bot model", self.test_model_combo)
        self._refresh_test_models()

        bot_hint = QLabel(
            "Drives the two bots when Test mode is enabled (General tab). "
            "Changing engine or model restarts test mode."
        )
        bot_hint.setWordWrap(True)
        bot_hint.setStyleSheet("color:#9aa3b2;font-size:11px;")
        form.addRow(" ", bot_hint)

        self._update_engine_widgets()
        return w

    def _build_avatar_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        persona_row = QWidget()
        ph = QHBoxLayout(persona_row)
        ph.setContentsMargins(0, 0, 0, 0)
        self.persona_combo = QComboBox()
        for name in self._persona_names():
            self.persona_combo.addItem(name)
        current_persona = self.main_window.current_persona() or "default"
        idx = self.persona_combo.findText(current_persona)
        if idx >= 0:
            self.persona_combo.setCurrentIndex(idx)
        self.persona_combo.currentTextChanged.connect(self.main_window.set_persona)
        picker_btn = QPushButton("Open picker…")
        picker_btn.setMaximumWidth(130)
        picker_btn.clicked.connect(self.main_window._open_persona_picker)
        ph.addWidget(self.persona_combo, 1)
        ph.addWidget(picker_btn, 0)
        form.addRow("Persona", persona_row)

        self.nod_combo = QComboBox()
        for name in self._nod_modes():
            self.nod_combo.addItem(name)
        current_nod = os.environ.get("FACEVIEW_NOD_MODE", "head_block_neck_stretch")
        idx = self.nod_combo.findText(current_nod)
        if idx >= 0:
            self.nod_combo.setCurrentIndex(idx)
        self.nod_combo.currentTextChanged.connect(self._on_nod_changed)
        form.addRow("Head-nod mode", self.nod_combo)

        self.rig_combo = QComboBox()
        for name in _RIG_MODES:
            self.rig_combo.addItem(name)
        current_rig = os.environ.get("FACEVIEW_RIG_WEIGHT_MODE", "graded_3ring")
        idx = self.rig_combo.findText(current_rig)
        if idx >= 0:
            self.rig_combo.setCurrentIndex(idx)
        self.rig_combo.currentTextChanged.connect(self._on_rig_changed)
        form.addRow("Body rig weights", self.rig_combo)

        hint = QLabel(
            "Avatar style picker (Cmd-Shift-P) groups the 41 bundled "
            "personas by renderer for easier browsing."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2;font-size:11px;")
        form.addRow(" ", hint)

        return w

    # ── data helpers ─────────────────────────────────────────────────

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

    def _current_engine_key(self) -> str:
        client = getattr(self.main_window, "llm_client", None)
        if client is not None and hasattr(client, "current_engine"):
            engine = client.current_engine()
            if engine in {"anthropic", "ollama", "demo"}:
                return engine
        return "auto"

    def _refresh_ollama_models(self) -> None:
        try:
            from faceview.llm.ollama_client import is_ollama_available, list_ollama_models
            available = is_ollama_available()
            models = list_ollama_models() if available else []
        except Exception:  # noqa: BLE001
            models = []
        self.ollama_combo.blockSignals(True)
        self.ollama_combo.clear()
        if not models:
            self.ollama_combo.addItem("(no models — start `ollama serve` and pull one)")
            self.ollama_combo.setEnabled(False)
        else:
            for m in models:
                self.ollama_combo.addItem(m)
            self.ollama_combo.setEnabled(True)
            preferred = os.environ.get("FACEVIEW_OLLAMA_MODEL") or models[0]
            idx = self.ollama_combo.findText(preferred)
            if idx >= 0:
                self.ollama_combo.setCurrentIndex(idx)
        self.ollama_combo.blockSignals(False)

    def _update_engine_widgets(self) -> None:
        engine = self.engine_combo.currentData()
        self.anthropic_combo.setEnabled(engine in {"anthropic", "auto"})
        # The ollama combo's enabled state was set by _refresh_ollama_models
        # based on availability; only force-disable when irrelevant.
        if engine in {"anthropic", "demo"}:
            self.ollama_combo.setEnabled(False)
        elif self.ollama_combo.count() > 0 and not self.ollama_combo.itemText(0).startswith("("):
            self.ollama_combo.setEnabled(True)

    # ── slots ────────────────────────────────────────────────────────

    def _on_engine_changed(self, _idx: int) -> None:
        engine = self.engine_combo.currentData()
        model: Optional[str] = None
        if engine == "anthropic":
            model = self.anthropic_combo.currentText()
        elif engine == "ollama" and self.ollama_combo.isEnabled():
            text = self.ollama_combo.currentText()
            if text and not text.startswith("("):
                model = text
        client = getattr(self.main_window, "llm_client", None)
        if client is not None and hasattr(client, "select_engine"):
            actual = client.select_engine(engine, model=model)
            self._update_status_pill(actual)
        self._update_engine_widgets()

    def _on_anthropic_model_changed(self, name: str) -> None:
        if not name:
            return
        settings.anthropic_model = name
        os.environ["FACEVIEW_MODEL"] = name
        client = getattr(self.main_window, "llm_client", None)
        # Only re-create the engine if anthropic is the active engine, so
        # changing the dropdown while on Ollama doesn't surprise-switch.
        if (client is not None and hasattr(client, "current_engine")
                and client.current_engine() == "anthropic"):
            client.select_engine("anthropic", model=name)
            self._update_status_pill("anthropic")
        else:
            self._refresh_status_pill_from_client()

    def _on_ollama_model_changed(self, name: str) -> None:
        if not name or name.startswith("("):
            return
        os.environ["FACEVIEW_OLLAMA_MODEL"] = name
        client = getattr(self.main_window, "llm_client", None)
        if (client is not None and hasattr(client, "current_engine")
                and client.current_engine() == "ollama"):
            client.select_engine("ollama", model=name)
            self._update_status_pill("ollama")

    def _on_nod_changed(self, name: str) -> None:
        os.environ["FACEVIEW_NOD_MODE"] = name

    # ── TTS engine + voice ──────────────────────────────────────────

    def _restart_tts_if_running(self) -> None:
        try:
            mw = self.main_window
            if mw.tts_running():
                mw.set_tts_enabled(False)
                mw.set_tts_enabled(True)
        except Exception:  # noqa: BLE001
            pass

    def _on_tts_engine_changed(self, _idx: int) -> None:
        os.environ["FACEVIEW_TTS_ENGINE"] = self.tts_engine_combo.currentData()
        self._refresh_tts_voices()
        self._restart_tts_if_running()

    def _on_tts_voice_changed(self, name: str) -> None:
        name = (name or "").strip()
        if not name or name.startswith("("):
            return
        os.environ["FACEVIEW_TTS_VOICE"] = name
        self._restart_tts_if_running()

    def _refresh_tts_voices(self) -> None:
        from faceview.speech.tts_kokoro import KokoroEngine, assets_present
        engine = self.tts_engine_combo.currentData()
        self.tts_voice_combo.blockSignals(True)
        self.tts_voice_combo.clear()
        added = False
        if engine in ("kokoro", "auto"):
            if assets_present():
                try:
                    for v in KokoroEngine().voices():
                        self.tts_voice_combo.addItem(v)
                    added = True
                except Exception:  # noqa: BLE001
                    self.tts_voice_combo.addItem("(kokoro install incomplete)")
            elif engine == "kokoro":
                self.tts_voice_combo.addItem(
                    "(no model — run `python -m faceview.speech.tts_kokoro --download`)")
        if engine in ("pyttsx3", "auto") and not added:
            try:
                import pyttsx3  # type: ignore
                for v in pyttsx3.init().getProperty("voices"):
                    if v.name:
                        self.tts_voice_combo.addItem(v.name)
            except Exception:  # noqa: BLE001
                pass
        if self.tts_voice_combo.count() == 0:
            self.tts_voice_combo.addItem("(no voices found)")
        preferred = os.environ.get("FACEVIEW_TTS_VOICE") or "af_sarah"
        idx = self.tts_voice_combo.findText(preferred)
        if idx >= 0:
            self.tts_voice_combo.setCurrentIndex(idx)
        self.tts_voice_combo.blockSignals(False)

    def _on_rig_changed(self, name: str) -> None:
        os.environ["FACEVIEW_RIG_WEIGHT_MODE"] = name

    # ── test-mode bot engine ─────────────────────────────────────────

    def _refresh_test_models(self) -> None:
        engine = self.test_engine_combo.currentData()
        self.test_model_combo.blockSignals(True)
        self.test_model_combo.clear()
        if engine == "anthropic":
            for m in _ANTHROPIC_MODELS:
                self.test_model_combo.addItem(m)
            preferred = os.environ.get("FACEVIEW_TEST_MODEL") or settings.anthropic_model
            idx = self.test_model_combo.findText(preferred)
            self.test_model_combo.setCurrentIndex(max(0, idx))
            self.test_model_combo.setEnabled(True)
        elif engine == "ollama":
            try:
                from faceview.llm.ollama_client import is_ollama_available, list_ollama_models
                models = list_ollama_models() if is_ollama_available() else []
            except Exception:  # noqa: BLE001
                models = []
            if not models:
                self.test_model_combo.addItem("(no models — start `ollama serve`)")
                self.test_model_combo.setEnabled(False)
            else:
                for m in models:
                    self.test_model_combo.addItem(m)
                preferred = os.environ.get("FACEVIEW_TEST_MODEL") or models[0]
                idx = self.test_model_combo.findText(preferred)
                self.test_model_combo.setCurrentIndex(max(0, idx))
                self.test_model_combo.setEnabled(True)
        else:
            self.test_model_combo.addItem("—")
            self.test_model_combo.setEnabled(False)
        self.test_model_combo.blockSignals(False)

    def _on_test_engine_changed(self, _idx: int) -> None:
        engine = self.test_engine_combo.currentData()
        os.environ["FACEVIEW_TEST_ENGINE"] = engine
        self._refresh_test_models()
        # Push the current model selection too (so restart picks it up).
        text = self.test_model_combo.currentText()
        if text and not text.startswith("(") and text != "—":
            os.environ["FACEVIEW_TEST_MODEL"] = text
        else:
            os.environ.pop("FACEVIEW_TEST_MODEL", None)
        self._restart_test_mode_if_running()

    def _on_test_model_changed(self, name: str) -> None:
        if not name or name.startswith("(") or name == "—":
            return
        os.environ["FACEVIEW_TEST_MODEL"] = name
        self._restart_test_mode_if_running()

    def _restart_test_mode_if_running(self) -> None:
        mw = self.main_window
        try:
            if mw.test_mode_running():
                mw.restart_test_mode()
        except Exception:  # noqa: BLE001
            pass

    # ── status-pill plumbing ─────────────────────────────────────────

    def _refresh_status_pill_from_client(self) -> None:
        self._update_status_pill(None)

    def _update_status_pill(self, _engine_hint) -> None:
        """Delegate to MainWindow's single source of truth so the pill
        consistently reflects test-mode engine when test mode is on."""
        try:
            self.main_window.refresh_llm_pill()
        except Exception:  # noqa: BLE001
            pass
