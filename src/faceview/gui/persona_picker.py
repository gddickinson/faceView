"""Persona / head-type picker — separate window listing all avatars.

The 41 bundled personas span very different renderers (stylised
cartoon, anatomical 2D, anatomy-layer composites, faceforge meshes,
ICT-FaceKit 26 k-vertex 3D head, MakeHuman base mesh, image warps).
Cramming them into a single combo box is unfriendly, so this window
groups them by render-mode category and shows them as clickable
tiles — pick one and the avatar updates immediately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from faceview.gui.main_window import MainWindow


# Render-mode → friendly category label. Order here drives tab order.
_CATEGORY_FOR_MODE: dict[str, str] = {
    "stylised":           "Cartoon (2D)",
    "anatomical":         "Anatomy (2D)",
    "anatomy_overlay":    "Anatomy (2D)",
    "wireframe":          "Anatomy (2D)",
    "anatomy_layers":     "Anatomy layers",
    "anatomy_skull":      "Anatomy layers",
    "anatomy_brain":      "Anatomy layers",
    "anatomy_muscles":    "Anatomy layers",
    "anatomy_xray":       "Anatomy layers",
    "anatomy_eyeballs":   "Anatomy layers",
    "ict_face_3d":        "ICT-FaceKit 3D",
    "makehuman_3d":       "MakeHuman 3D",
    "faceforge_3d":       "Faceforge 3D",
    "faceforge_3d_gpu":   "Faceforge 3D",
    "head_decimated_3d":  "BP3D-decimated",
    "head_decimated_3d_gpu": "BP3D-decimated",
    "face_warp_2d":       "Photo warp",
    "face_warp_3d":       "Photo warp",
}

_CATEGORY_ORDER = [
    "ICT-FaceKit 3D",
    "MakeHuman 3D",
    "Faceforge 3D",
    "Photo warp",
    "BP3D-decimated",
    "Anatomy layers",
    "Anatomy (2D)",
    "Cartoon (2D)",
]


class PersonaPicker(QDialog):
    def __init__(self, main_window: "MainWindow", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent or main_window)
        self.setWindowTitle("Avatar style")
        self.setMinimumSize(620, 520)
        self.main_window = main_window
        self._buttons: dict[str, QPushButton] = {}
        self._build_ui()
        self._refresh_selection()

    # ── construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        head = QLabel(
            "Pick a head type for the Claude avatar. The change applies "
            "to the avatar window immediately."
        )
        head.setWordWrap(True)
        head.setStyleSheet("color:#9aa3b2;")
        root.addWidget(head)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        tabs = QTabWidget(self)
        root.addWidget(tabs, 1)

        groups = self._group_personas()
        for cat in _CATEGORY_ORDER:
            personas = groups.get(cat)
            if not personas:
                continue
            tabs.addTab(self._build_tab(cat, personas), cat)

        # Footer with current selection.
        self._current_label = QLabel(f"current: {self.main_window.current_persona()}")
        self._current_label.setStyleSheet("color:#cdd3e0;font-weight:600;")
        root.addWidget(self._current_label)

    @staticmethod
    def _group_personas() -> dict[str, list[str]]:
        from faceview.vision.personas import list_personas, load_persona
        groups: dict[str, list[str]] = defaultdict(list)
        for name in list_personas():
            try:
                mode = getattr(load_persona(name), "render_mode", "stylised") or "stylised"
            except Exception:  # noqa: BLE001
                mode = "stylised"
            cat = _CATEGORY_FOR_MODE.get(mode, "Cartoon (2D)")
            groups[cat].append(name)
        for cat in groups:
            groups[cat].sort()
        return groups

    def _build_tab(self, cat: str, personas: list[str]) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        cols = 3
        for idx, name in enumerate(personas):
            btn = QPushButton(name.replace("_", " "))
            btn.setCheckable(True)
            btn.setMinimumHeight(56)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setToolTip(name)
            btn.clicked.connect(lambda _checked=False, n=name: self._pick(n))
            self._buttons[name] = btn
            grid.addWidget(btn, idx // cols, idx % cols)
        scroll.setWidget(inner)
        return scroll

    # ── slots ───────────────────────────────────────────────────────

    def _pick(self, name: str) -> None:
        self.main_window.set_persona(name)
        self._current_label.setText(f"current: {name}")
        self._refresh_selection()

    def _refresh_selection(self) -> None:
        current = self.main_window.current_persona()
        for name, btn in self._buttons.items():
            btn.setChecked(name == current)
