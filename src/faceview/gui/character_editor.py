"""Persona / character editor window.

Lets the user customise a persona's identity: name, occupation,
backstory, Big Five traits, conversation style, catchphrases, goals.
Writes to ``assets/config/characters.json``. The visual side of
personas (skin/hair/render_mode) is read-only here — change those via
the Avatar style picker, the appearance system is out of scope for
this editor.

Layout:

    ┌─────────────────────────────────────────────────────┐
    │  Personas        │  selected persona's character     │
    │  ──────────      │  ────────────────────────────     │
    │  • claude        │  Name: [_____________]            │
    │  • iris          │  Backstory: [textarea]            │
    │  • soraya        │  Traits: openness     [slider]    │
    │  • theo          │          conscient... [slider]    │
    │  • niko          │  Topics: [______, ______, ___]    │
    │  • bayard        │  Catchphrases: [textarea]         │
    │                  │  Goals: [textarea]                │
    │                  │  [Save]  [Revert]  [New persona]  │
    └─────────────────────────────────────────────────────┘

Closing the window persists nothing automatically — Save is explicit.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from faceview.llm.character import (
    Character,
    _DEFAULT_CHARACTER,
    load_character_registry,
)
from faceview.utils.paths import project_root

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


_CHARACTERS_JSON = (
    project_root() / "src" / "faceview" / "assets" / "config" / "characters.json"
)

_TRAITS = [
    ("openness",          "Openness"),
    ("conscientiousness", "Conscientious"),
    ("extraversion",      "Extraversion"),
    ("agreeableness",     "Agreeable"),
    ("neuroticism",       "Neuroticism"),
]


class CharacterEditor(QDialog):
    """Edit ``assets/config/characters.json`` interactively."""

    def __init__(self, main_window: "MainWindow",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent or main_window)
        self.main_window = main_window
        self.setWindowTitle("Persona editor")
        self.setMinimumSize(820, 560)

        self._registry: dict[str, dict] = {}
        self._current_key: Optional[str] = None
        self._dirty: bool = False
        self._loading: bool = False

        self._build_ui()
        self._reload_from_disk()

    # ── construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── left: persona list ───────────────────────────────────
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>Personas with characters</b>"))
        self.list = QListWidget()
        self.list.setMinimumWidth(220)
        self.list.currentTextChanged.connect(self._on_select)
        left.addWidget(self.list, 1)

        add_btn = QPushButton("New persona…")
        add_btn.clicked.connect(self._on_new_persona)
        left.addWidget(add_btn)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._on_delete)
        left.addWidget(del_btn)

        # ── right: edit form ─────────────────────────────────────
        right = QVBoxLayout()

        intro = QLabel(
            "Edit the character driving each persona. Save writes to "
            "<code>assets/config/characters.json</code>; the running "
            "avatar picks up changes on its next memory rebind."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        intro.setStyleSheet("color:#9aa3b2;")
        right.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)

        self.persona_label = QLabel("(none)")
        self.persona_label.setStyleSheet("font-weight:600;")
        form.addRow("Persona key", self.persona_label)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Name", self.name_edit)

        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setSpecialValueText("(unknown)")
        self.age_spin.valueChanged.connect(self._mark_dirty)
        form.addRow("Age", self.age_spin)

        self.occ_edit = QLineEdit()
        self.occ_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Occupation", self.occ_edit)

        self.backstory_edit = QTextEdit()
        self.backstory_edit.setMinimumHeight(80)
        self.backstory_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Backstory", self.backstory_edit)

        # Big Five sliders
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        form.addRow(sep)
        form.addRow(QLabel("<b>Big Five</b>"))
        self.trait_sliders: dict[str, QSlider] = {}
        self.trait_value_labels: dict[str, QLabel] = {}
        for key, label in _TRAITS:
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            val = QLabel("0.50")
            val.setMinimumWidth(40)
            slider.valueChanged.connect(
                lambda v, k=key: self._on_slider(k, v))
            self.trait_sliders[key] = slider
            self.trait_value_labels[key] = val
            rl.addWidget(slider, 1)
            rl.addWidget(val, 0)
            form.addRow(label, row)

        # Conversation style
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        form.addRow(sep2)
        form.addRow(QLabel("<b>Voice</b>"))

        self.topics_edit = QLineEdit()
        self.topics_edit.setPlaceholderText("comma-separated")
        self.topics_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Topics", self.topics_edit)

        self.catch_edit = QTextEdit()
        self.catch_edit.setPlaceholderText("one per line")
        self.catch_edit.setMaximumHeight(72)
        self.catch_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Catchphrases", self.catch_edit)

        self.outlook_edit = QLineEdit()
        self.outlook_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Outlook", self.outlook_edit)

        self.goals_edit = QTextEdit()
        self.goals_edit.setPlaceholderText("one per line")
        self.goals_edit.setMaximumHeight(80)
        self.goals_edit.textChanged.connect(self._mark_dirty)
        form.addRow("Goals", self.goals_edit)

        right.addLayout(form)
        right.addStretch(0)

        # Buttons
        btns = QDialogButtonBox()
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save)
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.clicked.connect(self._on_select)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btns.addButton(self.save_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        btns.addButton(self.revert_btn, QDialogButtonBox.ButtonRole.ResetRole)
        btns.addButton(self.close_btn, QDialogButtonBox.ButtonRole.RejectRole)
        right.addWidget(btns)

        root.addLayout(left, 1)
        root.addLayout(right, 3)

    # ── data plumbing ───────────────────────────────────────────────

    def _reload_from_disk(self) -> None:
        self._registry = load_character_registry()
        self.list.blockSignals(True)
        self.list.clear()
        for key in sorted(self._registry.keys()):
            if key.endswith("_fallback"):
                continue
            self.list.addItem(key)
        self.list.blockSignals(False)
        if self.list.count() > 0:
            self.list.setCurrentRow(0)
        else:
            self._clear_form()

    def _on_select(self, *_args) -> None:
        item = self.list.currentItem()
        if item is None:
            self._clear_form()
            return
        self._current_key = item.text()
        data = self._registry.get(self._current_key) or {}
        self._loading = True
        try:
            self.persona_label.setText(self._current_key)
            self.name_edit.setText(data.get("name", ""))
            self.age_spin.setValue(int(data.get("age") or 0))
            self.occ_edit.setText(data.get("occupation", ""))
            self.backstory_edit.setPlainText(data.get("backstory", ""))
            traits = data.get("traits") or {}
            for key, _label in _TRAITS:
                v = float(traits.get(key, 0.5))
                self.trait_sliders[key].setValue(int(v * 100))
                self.trait_value_labels[key].setText(f"{v:.2f}")
            conv = data.get("conversation") or {}
            self.topics_edit.setText(", ".join(conv.get("topics") or []))
            self.catch_edit.setPlainText("\n".join(conv.get("catchphrases") or []))
            self.outlook_edit.setText(conv.get("outlook")
                                       or conv.get("gamePhilosophy") or "")
            self.goals_edit.setPlainText("\n".join(data.get("goals") or []))
        finally:
            self._loading = False
        self._dirty = False
        self.save_btn.setEnabled(False)

    def _clear_form(self) -> None:
        self._current_key = None
        self.persona_label.setText("(none)")
        self.name_edit.clear()
        self.age_spin.setValue(0)
        self.occ_edit.clear()
        self.backstory_edit.clear()
        for s in self.trait_sliders.values():
            s.setValue(50)
        self.topics_edit.clear()
        self.catch_edit.clear()
        self.outlook_edit.clear()
        self.goals_edit.clear()

    def _collect(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "age": (self.age_spin.value() or None),
            "occupation": self.occ_edit.text().strip(),
            "backstory": self.backstory_edit.toPlainText().strip(),
            "traits": {
                key: self.trait_sliders[key].value() / 100.0
                for key, _ in _TRAITS
            },
            "conversation": {
                "topics": [t.strip() for t in self.topics_edit.text().split(",")
                           if t.strip()],
                "catchphrases": [c.strip() for c in
                                 self.catch_edit.toPlainText().splitlines()
                                 if c.strip()],
                "outlook": self.outlook_edit.text().strip(),
            },
            "goals": [g.strip() for g in
                      self.goals_edit.toPlainText().splitlines()
                      if g.strip()],
        }

    # ── slots ───────────────────────────────────────────────────────

    def _on_slider(self, key: str, v: int) -> None:
        self.trait_value_labels[key].setText(f"{v/100:.2f}")
        self._mark_dirty()

    def _mark_dirty(self, *_args) -> None:
        if self._loading:
            return
        self._dirty = True
        self.save_btn.setEnabled(True)

    def _on_save(self) -> None:
        if self._current_key is None:
            return
        payload = self._collect()
        self._registry[self._current_key] = payload
        try:
            _CHARACTERS_JSON.write_text(json.dumps(self._registry, indent=2))
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self._dirty = False
        self.save_btn.setEnabled(False)
        # If this persona is currently active, rebind so the new
        # character takes effect on the next reply.
        if self._current_key == getattr(self.main_window, "_current_persona", None):
            try:
                self.main_window._bind_memory_for_current_persona(save_previous=True)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.main_window.statusBar().showMessage(
                f"Saved character for {self._current_key}",
            )
        except Exception:  # noqa: BLE001
            pass

    def _on_new_persona(self) -> None:
        key, ok = QInputDialog.getText(
            self, "New persona",
            "Persona key (e.g. ict_male_young, my_custom):",
        )
        if not ok or not key.strip():
            return
        key = key.strip()
        if key in self._registry:
            QMessageBox.information(self, "Already exists",
                                    f"'{key}' is already in the registry.")
            return
        # Seed from defaults so the form has sensible starting values.
        base = asdict(_DEFAULT_CHARACTER)
        base["name"] = key.replace("_", " ").title()
        base["backstory"] = "Describe this character's life, voice, and quirks."
        self._registry[key] = base
        self.list.addItem(key)
        items = self.list.findItems(key, Qt.MatchFlag.MatchExactly)
        if items:
            self.list.setCurrentItem(items[0])

    def _on_delete(self) -> None:
        if self._current_key is None:
            return
        key = self._current_key
        if QMessageBox.question(
            self, "Delete character",
            f"Remove '{key}' from characters.json?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._registry.pop(key, None)
        try:
            _CHARACTERS_JSON.write_text(json.dumps(self._registry, indent=2))
        except OSError:
            return
        # Rebuild list.
        self._reload_from_disk()
