"""Effects panel — trigger transient FX + adjust persistent sliders.

A QDialog accessed from the main menu (View → Effects…). Has one
tab per effect category (warp / lighting / sci-fi / smoke / anatomy
/ comic / emotional) and a final "Sliders" tab for always-on
persona tweaks (skin hue, eye colour, head morph blends, etc.).

Triggering an effect schedules it through
:func:`faceview.vision.effects_runtime.get_runtime`, which the
camera worker reads on its next tick.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QColorDialog, QDialog, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider, QTabWidget,
    QVBoxLayout, QWidget,
)

from faceview.vision.effects import REGISTRY, specs_by_category
from faceview.vision.effects_runtime import get_runtime


_CATEGORY_ICONS = {
    "warp":     "📐",
    "lighting": "💡",
    "scifi":    "🛸",
    "smoke":    "💨",
    "anatomy":  "🦴",
    "comic":    "💥",
    "emotional":"❤️",
}


class _TriggerButton(QPushButton):
    def __init__(self, name: str, label: str, parent=None):
        super().__init__(label, parent)
        self.name = name


class EffectsPanel(QDialog):
    """Main effects-control dialog."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Avatar Effects")
        self.setMinimumSize(640, 520)
        self.runtime = get_runtime()
        self._intensity = 0.85

        root = QVBoxLayout(self)

        # Intensity + duration controls applied to next trigger.
        top = QHBoxLayout()
        top.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(10, 100)
        self.intensity_slider.setValue(85)
        self.intensity_slider.valueChanged.connect(
            lambda v: setattr(self, "_intensity", v / 100.0))
        top.addWidget(self.intensity_slider, 2)
        self.intensity_label = QLabel("0.85")
        self.intensity_slider.valueChanged.connect(
            lambda v: self.intensity_label.setText(f"{v/100:.2f}"))
        top.addWidget(self.intensity_label)
        top.addSpacing(20)
        stop_all = QPushButton("Stop all FX")
        stop_all.clicked.connect(self.runtime.stop_all)
        top.addWidget(stop_all)
        root.addLayout(top)

        # Tabs by category + a Sliders tab.
        self.tabs = QTabWidget(self)
        for cat, specs in specs_by_category().items():
            tab = self._build_category_tab(cat, specs)
            icon = _CATEGORY_ICONS.get(cat, "")
            self.tabs.addTab(tab, f"{icon} {cat}".strip())
        self.tabs.addTab(self._build_sliders_tab(), "🎚 Sliders")
        root.addWidget(self.tabs, 1)

    # ── helpers ────────────────────────────────────────────────

    def _build_category_tab(self, category: str, specs: list) -> QWidget:
        w = QWidget(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget(self)
        grid = QGridLayout(inner)
        grid.setSpacing(8)
        for i, spec in enumerate(specs):
            r, c = divmod(i, 3)
            btn = _TriggerButton(spec.name, spec.label)
            btn.setMinimumHeight(48)
            btn.setToolTip(spec.description or spec.name)
            btn.clicked.connect(lambda _checked=False, n=spec.name:
                                  self.runtime.trigger(
                                      n, intensity=self._intensity))
            grid.addWidget(btn, r, c)
        inner.setLayout(grid)
        scroll.setWidget(inner)
        outer = QVBoxLayout(w)
        outer.addWidget(scroll)
        return w

    def _build_sliders_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Helper that builds a labelled slider for a numeric slider key.
        def add_slider(key: str, label: str, lo: float, hi: float,
                          step: float = 0.01, default: float = 0.0,
                          fmt: str = "{:.2f}") -> None:
            wrap = QWidget(self)
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            slider = QSlider(Qt.Orientation.Horizontal)
            n_steps = int((hi - lo) / step)
            slider.setRange(0, n_steps)
            cur = ((default - lo) / step)
            slider.setValue(int(cur))
            value_label = QLabel(fmt.format(default))
            value_label.setMinimumWidth(50)
            reset = QPushButton("⌖")
            reset.setFixedWidth(24)
            reset.setToolTip("Reset to default")

            def on_change(val: int) -> None:
                v = lo + val * step
                value_label.setText(fmt.format(v))
                self.runtime.set_slider(key, v)

            def on_reset() -> None:
                slider.setValue(int(cur))
                self.runtime.set_slider(key, default)

            slider.valueChanged.connect(on_change)
            reset.clicked.connect(on_reset)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            row.addWidget(reset)
            form.addRow(label, wrap)

        add_slider("skin_hue", "Skin hue (°)", 0, 360, 1, 28, "{:.0f}")
        add_slider("skin_saturation", "Skin saturation", 0, 1, 0.01, 0.32)
        add_slider("skin_value", "Skin brightness", 0.4, 1.0, 0.01, 0.86)
        add_slider("bloom_amp", "Bloom strength", 0, 1, 0.01, 0.45)
        add_slider("emit_pulse_scale", "Eye glow strength", 0, 3, 0.05, 1.0)

        form.addRow(_separator())

        add_slider("eye_open_bias", "Eye open bias", -0.5, 0.5, 0.01, 0.0)
        add_slider("smile_bias", "Smile bias", -0.5, 0.5, 0.01, 0.0)
        add_slider("brow_raise_bias", "Brow raise bias", -0.5, 0.5, 0.01, 0.0)
        add_slider("head_pitch_bias", "Head pitch bias", -0.3, 0.3, 0.01, 0.0)
        add_slider("head_yaw_bias", "Head yaw bias", -0.3, 0.3, 0.01, 0.0)

        form.addRow(_separator())

        add_slider("head_age", "Head age (young → elder)",
                     -1.0, 1.0, 0.05, 0.0)
        add_slider("head_gender", "Head gender (♀ → ♂)",
                     -1.0, 1.0, 0.05, 0.0)

        form.addRow(_separator())

        # Direct ICT blendshape sliders.
        add_slider("pupil_dilate", "Pupil dilate", 0.0, 1.0, 0.02, 0.0)
        add_slider("jaw_forward", "Jaw forward (underbite)",
                     0.0, 1.0, 0.02, 0.0)
        add_slider("jaw_left_right", "Jaw left ↔ right",
                     -1.0, 1.0, 0.02, 0.0)
        add_slider("mouth_left_right", "Mouth pose left ↔ right",
                     -1.0, 1.0, 0.02, 0.0)
        add_slider("mouth_funnel", "Mouth funnel (lip horn)",
                     0.0, 1.0, 0.02, 0.0)
        add_slider("mouth_close", "Mouth close (active purse)",
                     0.0, 1.0, 0.02, 0.0)
        add_slider("cheek_puff", "Cheeks puff", 0.0, 1.0, 0.02, 0.0)

        layout.addLayout(form)

        # Eye colour picker.
        eye_row = QHBoxLayout()
        eye_row.addWidget(QLabel("Eye glow colour:"))
        self.eye_swatch = QLabel("    ")
        self.eye_swatch.setMinimumWidth(60)
        self.eye_swatch.setStyleSheet(
            "background:#5a3818; border:1px solid #888;")
        pick = QPushButton("Pick…")
        pick.clicked.connect(self._pick_eye_color)
        eye_row.addWidget(self.eye_swatch)
        eye_row.addWidget(pick)
        eye_row.addStretch(1)
        layout.addLayout(eye_row)

        # Reset all sliders.
        reset_all = QPushButton("Reset all sliders")
        reset_all.clicked.connect(self._reset_sliders)
        layout.addWidget(reset_all)
        layout.addStretch(1)
        return w

    def _pick_eye_color(self) -> None:
        col = QColorDialog.getColor(parent=self)
        if not col.isValid():
            return
        hex_ = col.name()
        self.eye_swatch.setStyleSheet(
            f"background:{hex_}; border:1px solid #888;")
        self.runtime.set_slider("eye_color_hex", hex_)

    def _reset_sliders(self) -> None:
        self.runtime.reset_sliders()
        # Reset GUI by closing + reopening (simplest path).
        self.eye_swatch.setStyleSheet(
            "background:#5a3818; border:1px solid #888;")


def _separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f
