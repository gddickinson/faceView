"""Detachable-panel layout for :class:`MainWindow`.

Each user-facing panel (camera, chat, status, transcript) lives inside a
:class:`QDockWidget` so users can:

- drag a panel into a floating window
- re-dock to any edge
- tab two panels together
- hide a panel and bring it back from the Window menu

The dock layout is persisted via ``QSettings`` so it survives restarts.
A "Reset layout" action restores the snapshot we took right after
:meth:`build`.

Kept in its own module so ``main_window.py`` stays focused on worker
lifecycle and stays under the 500-line guideline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QDockWidget, QMainWindow, QMenu, QWidget

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


_SETTINGS_ORG = "faceview"
_SETTINGS_APP = "main"

# (key, title, attribute on MainWindow)
_PANELS: list[tuple[str, str, str]] = [
    ("camera",     "Camera",     "camera"),
    ("chat",       "Chat",       "chat"),
    ("status",     "Status",     "status_panel"),
    ("transcript", "Transcript", "transcript"),
    ("perception", "Perception", "perception_panel"),
]


class LayoutManager:
    """Wraps :class:`MainWindow`'s panels in dock widgets + manages state."""

    def __init__(self, window: "MainWindow") -> None:
        self.window = window
        self.docks: dict[str, QDockWidget] = {}
        self._default_state: bytes | None = None
        self._default_geometry: bytes | None = None

    # ── build ─────────────────────────────────────────────────────────

    def build(self) -> None:
        w = self.window
        w.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.AnimatedDocks
        )

        # The MainWindow still needs a central widget; we make it
        # invisible so the four panels (all docks) own the whole area.
        placeholder = QWidget(w)
        placeholder.setMaximumSize(0, 0)
        w.setCentralWidget(placeholder)

        for key, title, attr in _PANELS:
            panel = getattr(w, attr)
            self.docks[key] = self._make_dock(title, key, panel)

        left = Qt.DockWidgetArea.LeftDockWidgetArea
        right = Qt.DockWidgetArea.RightDockWidgetArea

        w.addDockWidget(left, self.docks["camera"])
        w.addDockWidget(left, self.docks["chat"])
        w.splitDockWidget(
            self.docks["camera"], self.docks["chat"], Qt.Orientation.Horizontal,
        )
        w.addDockWidget(right, self.docks["status"])
        w.addDockWidget(right, self.docks["transcript"])
        w.addDockWidget(right, self.docks["perception"])
        w.splitDockWidget(
            self.docks["status"], self.docks["transcript"], Qt.Orientation.Vertical,
        )
        # Tab the perception panel behind the transcript so it stays
        # out of the way until the user clicks the tab.
        w.tabifyDockWidget(self.docks["transcript"], self.docks["perception"])
        self.docks["transcript"].raise_()

        # Give camera + chat the wider columns and status a small strip.
        w.resizeDocks(
            [self.docks["camera"], self.docks["chat"]],
            [520, 520],
            Qt.Orientation.Horizontal,
        )
        w.resizeDocks(
            [self.docks["status"], self.docks["transcript"]],
            [220, 360],
            Qt.Orientation.Vertical,
        )

        # Snapshot the just-built layout so "Reset layout" has something
        # to restore. saveState/saveGeometry return QByteArray; cast to
        # bytes so we own a stable copy.
        self._default_state = bytes(w.saveState())
        self._default_geometry = bytes(w.saveGeometry())

        self.restore()

    def _make_dock(self, title: str, key: str, panel: QWidget) -> QDockWidget:
        dock = QDockWidget(title, self.window)
        # objectName is required for saveState() to identify the dock.
        dock.setObjectName(f"dock_{key}")
        dock.setWidget(panel)
        dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable,
        )
        return dock

    # ── menu wiring ───────────────────────────────────────────────────

    def install_menu(self, menu: QMenu) -> None:
        for key, _title, _attr in _PANELS:
            menu.addAction(self.docks[key].toggleViewAction())
        menu.addSeparator()
        act_save = QAction("Save layout", self.window)
        act_save.setShortcut(QKeySequence("Ctrl+Shift+Y"))
        act_save.triggered.connect(self.save)
        menu.addAction(act_save)
        act_reset = QAction("Reset layout to default", self.window)
        act_reset.setShortcut(QKeySequence("Ctrl+Shift+L"))
        act_reset.triggered.connect(self.reset)
        menu.addAction(act_reset)

    # ── persistence ───────────────────────────────────────────────────

    def _settings(self) -> QSettings:
        return QSettings(_SETTINGS_ORG, _SETTINGS_APP)

    def save(self, *, quiet: bool = False) -> None:
        s = self._settings()
        s.setValue("layout/state", self.window.saveState())
        s.setValue("layout/geometry", self.window.saveGeometry())
        if not quiet:
            try:
                self.window.statusBar().showMessage("Layout saved")
            except Exception:  # noqa: BLE001
                pass

    def restore(self) -> None:
        s = self._settings()
        state = s.value("layout/state")
        geom = s.value("layout/geometry")
        if state is not None:
            self.window.restoreState(state)
        if geom is not None:
            self.window.restoreGeometry(geom)

    def reset(self) -> None:
        if self._default_state is not None:
            self.window.restoreState(self._default_state)
        if self._default_geometry is not None:
            self.window.restoreGeometry(self._default_geometry)
        # Drop persisted overrides so the next launch is also default.
        s = self._settings()
        s.remove("layout/state")
        s.remove("layout/geometry")
        try:
            self.window.statusBar().showMessage("Layout reset to default")
        except Exception:  # noqa: BLE001
            pass
