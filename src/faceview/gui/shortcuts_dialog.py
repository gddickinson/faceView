"""U5 — Help → Keyboard shortcuts inventory dialog.

A single source of truth for the keyboard shortcuts faceView wires
up. Edit :data:`SHORTCUTS` when you add or rename one; the dialog
re-renders from the list."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# (category, shortcut, action) — grouped so the dialog can section
# the table by category for scannability.
SHORTCUTS: list[tuple[str, str, str]] = [
    # Chat
    ("Chat",  "Enter",         "Send the typed message"),
    ("Chat",  "Cmd/Ctrl+F",    "Find in chat history"),
    ("Chat",  "Esc (in find)", "Dismiss the find bar"),
    # Workers
    ("Tools", "Cmd/Ctrl+Shift+C", "Toggle camera"),
    ("Tools", "Cmd/Ctrl+Shift+M", "Toggle microphone"),
    ("Tools", "Cmd/Ctrl+Shift+T", "Toggle TTS (Claude voice)"),
    ("Tools", "Cmd/Ctrl+Shift+R", "Toggle mirror mode"),
    ("Tools", "Cmd/Ctrl+Shift+E", "Enroll me as owner"),
    ("Tools", "Cmd/Ctrl+Shift+N", "Toggle incognito mode"),
    ("Tools", "Cmd/Ctrl+,",       "Open Configuration…"),
    # View
    ("View",  "Cmd/Ctrl+Shift+A", "Toggle avatar window"),
    ("View",  "Cmd/Ctrl+Shift+P", "Avatar style picker"),
    ("View",  "Cmd/Ctrl+Shift+I", "Edit personas…"),
    ("View",  "Cmd/Ctrl+Shift+Z", "Open room map"),
    ("View",  "Cmd/Ctrl+Shift+W", "Toggle screen capture"),
    ("View",  "Cmd/Ctrl+E",       "Effects panel"),
    # File
    ("File",  "Cmd/Ctrl+Shift+S", "Take a screenshot"),
    ("File",  "Cmd/Ctrl+Q",       "Quit"),
    # Window / layout
    ("Window", "Cmd/Ctrl+Shift+Y", "Save layout"),
    ("Window", "Cmd/Ctrl+Shift+L", "Reset layout to default"),
    # Monitor sub-windows
    ("Monitor", "Cmd/Ctrl+1", "Audio waveform / VAD…"),
    ("Monitor", "Cmd/Ctrl+2", "Emotion scores…"),
    ("Monitor", "Cmd/Ctrl+3", "Mouth + visemes…"),
    ("Monitor", "Cmd/Ctrl+4", "Transcript history…"),
    # Help
    ("Help",  "Cmd/Ctrl+/", "Show this dialog"),
]


class ShortcutsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("faceView — Keyboard shortcuts")
        self.setMinimumSize(520, 520)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QLabel("Keyboard shortcuts")
        f = QFont()
        f.setBold(True)
        f.setPointSize(14)
        header.setFont(f)
        root.addWidget(header)

        hint = QLabel(
            "Cmd on macOS, Ctrl elsewhere. Many shortcuts mirror the "
            "Tools / View menus — pulling up the menu always shows "
            "the live binding too."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa3b2;")
        root.addWidget(hint)

        table = QTableWidget(len(SHORTCUTS), 3, self)
        table.setHorizontalHeaderLabels(["Category", "Shortcut", "Action"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        for r, (cat, key, desc) in enumerate(SHORTCUTS):
            table.setItem(r, 0, QTableWidgetItem(cat))
            kc = QTableWidgetItem(key)
            kfont = QFont("Menlo")
            kc.setFont(kfont)
            table.setItem(r, 1, kc)
            table.setItem(r, 2, QTableWidgetItem(desc))
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        root.addWidget(table, 1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        root.addWidget(bb)
