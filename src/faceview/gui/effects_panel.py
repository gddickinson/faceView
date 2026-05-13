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
    QColorDialog, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
    QTabWidget, QVBoxLayout, QWidget,
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
        self.tabs.addTab(self._build_camera_tab(), "🎥 Camera")
        self.tabs.addTab(self._build_colours_tab(), "🎨 Colours")
        self.tabs.addTab(self._build_tongue_tab(), "👅 Tongue")
        self.tabs.addTab(self._build_body_tab(), "🧍 Body")
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
        add_slider("head_gender", "Head sex (♀ ↔ ♂)",
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

        # Reset all sliders.
        reset_all = QPushButton("Reset all sliders")
        reset_all.clicked.connect(self._reset_sliders)
        layout.addWidget(reset_all)
        layout.addStretch(1)
        return w

    # ── Camera tab ─────────────────────────────────────────────

    def _build_camera_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        info = QLabel(
            "Camera orbit — rotates the whole head around its centre at\n"
            "fixed distance, independent of the head's own talking-pose.")
        info.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info)

        self._cam_sliders = []  # so reset can re-set them
        for key, label, lo, hi, step, default in [
            ("camera_yaw", "Camera yaw (orbit ↺ ↻)", -3.14, 3.14, 0.05, 0.0),
            ("camera_pitch", "Camera pitch (above ↕ below)", -1.2, 1.2, 0.05, 0.0),
            ("camera_zoom", "Camera zoom (out ↔ in)", 0.3, 5.0, 0.05, 1.0),
            ("camera_focus_y", "Focus offset (down ↕ up)", -1.0, 1.0, 0.02, 0.0),
        ]:
            slider, value_label, default_val = self._labelled_slider(
                key, lo, hi, step, default, "{:.2f}",
            )
            wrap = QWidget(self)
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            form.addRow(label, wrap)
            self._cam_sliders.append((slider, default_val))

        layout.addLayout(form)

        # Snap-zoom buttons — preset (zoom, focus_y) combinations
        # for common framings. Each pushes values into the runtime
        # AND moves the corresponding sliders so they stay in sync.
        snaps_label = QLabel("Frame:")
        snaps_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(snaps_label)
        snap_row = QHBoxLayout()
        for label, zoom, focus_y in [
            ("Head", 4.5, 0.78),
            ("Portrait", 2.5, 0.40),
            ("Full body", 1.0, 0.0),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _checked=False, z=zoom, fy=focus_y:
                    self._snap_zoom(z, fy)
            )
            snap_row.addWidget(btn)
        layout.addLayout(snap_row)

        # View-angle buttons — preset (yaw, pitch) for common camera
        # angles. Front / back / left / right circle the avatar; top
        # and bottom look down / up at it.
        import math as _math
        angles_label = QLabel("View:")
        angles_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(angles_label)
        angle_row1 = QHBoxLayout()
        for label, yaw, pitch in [
            ("Front",   0.0,           0.0),
            ("Back",    _math.pi,      0.0),
            ("Left",   -_math.pi / 2,  0.0),
            ("Right",   _math.pi / 2,  0.0),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, y=yaw, p=pitch:
                    self._snap_view(y, p)
            )
            angle_row1.addWidget(btn)
        layout.addLayout(angle_row1)
        angle_row2 = QHBoxLayout()
        for label, yaw, pitch in [
            ("Top",    0.0,  -1.0),
            ("Bottom", 0.0,   1.0),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, y=yaw, p=pitch:
                    self._snap_view(y, p)
            )
            angle_row2.addWidget(btn)
        layout.addLayout(angle_row2)

        # Reset view button.
        reset_view = QPushButton("Reset view (front)")
        reset_view.clicked.connect(self._reset_camera)
        layout.addWidget(reset_view)
        layout.addStretch(1)
        return w

    # ── Colours tab ────────────────────────────────────────────

    def _build_colours_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)

        info = QLabel(
            "Hair style + colour, eye glow colour. Drag the\n"
            "Skin hue / saturation sliders on the Sliders tab\n"
            "for full skin recolour.")
        info.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info)

        # Hair style dropdown.
        hair_row = QHBoxLayout()
        hair_row.addWidget(QLabel("Hair style:"))
        self.hair_combo = QComboBox(self)
        try:
            from faceview.vision.hair_3d import list_styles
            for s in list_styles():
                self.hair_combo.addItem(s)
        except Exception:
            self.hair_combo.addItems(["none"])
        self.hair_combo.currentTextChanged.connect(
            lambda s: self.runtime.set_slider("hair_style", s))
        hair_row.addWidget(self.hair_combo, 1)
        layout.addLayout(hair_row)

        # Hair colour picker.
        hair_color_row = QHBoxLayout()
        hair_color_row.addWidget(QLabel("Hair colour:"))
        self.hair_swatch = QLabel("    ")
        self.hair_swatch.setMinimumWidth(60)
        self.hair_swatch.setStyleSheet(
            "background:#3a2418; border:1px solid #888;")
        pick_hair = QPushButton("Pick…")
        pick_hair.clicked.connect(self._pick_hair_color)
        hair_color_row.addWidget(self.hair_swatch)
        hair_color_row.addWidget(pick_hair)
        hair_color_row.addStretch(1)
        layout.addLayout(hair_color_row)

        # Eye glow colour picker.
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

        layout.addStretch(1)
        return w

    # ── Body tab ───────────────────────────────────────────────

    def _build_body_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        info = QLabel(
            "Full-body avatar (CC0 from MakeHuman base mesh bundle).\n"
            "Toggle on to attach a body below the head. The Body sex\n"
            "slider blends body topology only; the head has its own\n"
            "Head sex slider on the Sliders tab so you can mix-and-\n"
            "match (or set both for a coherent look). The head still\n"
            "tracks expression / pose; the body rotates with the\n"
            "camera only.")
        info.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info)

        for key, label, lo, hi, step, default in [
            ("show_body",  "Show body (off ↔ on)",     0.0, 1.0, 1.0, 0.0),
            ("body_morph", "Body sex (♀ ↔ ♂)",         -1.0, 1.0, 2.0, 1.0),
        ]:
            slider, value_label, _ = self._labelled_slider(
                key, lo, hi, step, default, "{:.2f}",
            )
            wrap = QWidget(self)
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            form.addRow(label, wrap)

        layout.addLayout(form)

        # Body pose effects — full-figure rigging.
        pose_label = QLabel("Pose effects (transient):")
        pose_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(pose_label)
        pose_row1 = QHBoxLayout()
        for label, name in [
            ("Bow",        "body_bow"),
            ("Lean back",  "body_lean_back"),
            ("Sway",       "body_sway"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, n=name: self.runtime.trigger(n))
            pose_row1.addWidget(btn)
        layout.addLayout(pose_row1)
        pose_row2 = QHBoxLayout()
        for label, name in [
            ("Lean L",   "body_lean_left"),
            ("Lean R",   "body_lean_right"),
            ("Twist L",  "body_twist_left"),
            ("Twist R",  "body_twist_right"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, n=name: self.runtime.trigger(n))
            pose_row2.addWidget(btn)
        layout.addLayout(pose_row2)

        limbs_label = QLabel("Limb effects:")
        limbs_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(limbs_label)
        limb_row1 = QHBoxLayout()
        for label, name in [
            ("Wave L",  "wave_left"),
            ("Wave R",  "wave_right"),
            ("Arms up", "arms_up"),
            ("Arms out","arms_out"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, n=name: self.runtime.trigger(n))
            limb_row1.addWidget(btn)
        layout.addLayout(limb_row1)
        limb_row2 = QHBoxLayout()
        for label, name in [
            ("Kick L",  "kick_left"),
            ("Kick R",  "kick_right"),
            ("Squat",   "squat"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda _c=False, n=name: self.runtime.trigger(n))
            limb_row2.addWidget(btn)
        layout.addLayout(limb_row2)

        layout.addStretch(1)
        return w

    # ── Tongue tab ─────────────────────────────────────────────

    def _build_tongue_tab(self) -> QWidget:
        w = QWidget(self)
        layout = QVBoxLayout(w)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        info = QLabel(
            "Dynamic 3D tongue. Rooted at the back of the mouth, bends\n"
            "along a Bézier curve. Set Extend = -1 to retract — when\n"
            "retracted (and Talking-tongue is on), the tongue is driven\n"
            "by the active speech viseme so it moves naturally inside\n"
            "the mouth as the avatar talks.")
        info.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(info)

        self._tongue_sliders: list[tuple[QSlider, int]] = []
        for key, label, lo, hi, default in [
            ("tongue_extend",   "Extend (in ↔ out, -1 hidden)", -1.0, 1.0, -1.0),
            ("tongue_lateral",  "Lateral (left ↔ right)",       -1.0, 1.0,  0.0),
            ("tongue_vertical", "Vertical (over lower ↔ upper lip)",
                                                                  -1.0, 1.0,  0.0),
            ("tongue_curl",     "Curl (droop ↔ arch)",          -1.0, 1.0,  0.0),
            ("tongue_taper",    "Taper (flat ↔ pointed)",        0.0, 1.0,  0.4),
            ("talking_tongue",  "Talking tongue (off ↔ on)",     0.0, 1.0,  1.0),
        ]:
            slider, value_label, default_step = self._labelled_slider(
                key, lo, hi, 0.05, default, "{:.2f}",
            )
            wrap = QWidget(self)
            row = QHBoxLayout(wrap)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(slider, 1)
            row.addWidget(value_label)
            form.addRow(label, wrap)
            self._tongue_sliders.append((slider, default_step))

        layout.addLayout(form)

        reset = QPushButton("Hide tongue / reset")
        reset.clicked.connect(self._reset_tongue)
        layout.addWidget(reset)
        layout.addStretch(1)
        return w

    def _reset_tongue(self) -> None:
        for slider, default_step in self._tongue_sliders:
            slider.setValue(default_step)
        self.runtime.set_slider("tongue_extend", -1.0)
        self.runtime.set_slider("tongue_lateral", 0.0)
        self.runtime.set_slider("tongue_vertical", 0.0)
        self.runtime.set_slider("tongue_curl", 0.0)
        self.runtime.set_slider("tongue_taper", 0.4)

    def _labelled_slider(self, key: str, lo: float, hi: float,
                            step: float, default: float, fmt: str
                            ) -> tuple[QSlider, QLabel, int]:
        """Helper used by camera tab — returns (slider, value_label,
        default_step_position) so callers can also wire reset.

        Stores ``lo`` and ``step`` as Qt properties so callers can
        round-trip a float value back into a slider position.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        n_steps = int((hi - lo) / step)
        slider.setRange(0, n_steps)
        cur = int((default - lo) / step)
        slider.setValue(cur)
        slider.setProperty("float_lo", float(lo))
        slider.setProperty("float_step", float(step))
        slider.setProperty("float_fmt", fmt)
        value_label = QLabel(fmt.format(default))
        value_label.setMinimumWidth(50)

        def on_change(val: int) -> None:
            v = lo + val * step
            value_label.setText(fmt.format(v))
            self.runtime.set_slider(key, v)

        slider.valueChanged.connect(on_change)
        return slider, value_label, cur

    def _reset_camera(self) -> None:
        """Reset camera-orbit sliders to centred / front view."""
        for slider, default_step in self._cam_sliders:
            slider.setValue(default_step)
        self.runtime.set_slider("camera_yaw", 0.0)
        self.runtime.set_slider("camera_pitch", 0.0)
        self.runtime.set_slider("camera_zoom", 1.0)
        self.runtime.set_slider("camera_focus_y", 0.0)

    def _snap_view(self, yaw: float, pitch: float) -> None:
        """Snap to a preset camera view-angle. Camera sliders for
        yaw and pitch live at indices 0 and 1 of ``_cam_sliders``."""
        if len(self._cam_sliders) < 2:
            return
        yaw_slider = self._cam_sliders[0][0]
        pitch_slider = self._cam_sliders[1][0]
        for slider, value in ((yaw_slider, yaw), (pitch_slider, pitch)):
            lo = slider.property("float_lo")
            step = slider.property("float_step")
            if lo is None or step is None or step == 0:
                continue
            pos = int(round((float(value) - float(lo)) / float(step)))
            pos = max(slider.minimum(), min(slider.maximum(), pos))
            slider.setValue(pos)  # triggers on_change → runtime

    def _snap_zoom(self, zoom: float, focus_y: float) -> None:
        """Apply a preset (zoom, focus_y) pair from the snap buttons
        and move the matching sliders to reflect the new values.
        Camera sliders are stored as (widget, step) pairs in
        ``_cam_sliders`` in creation order (yaw, pitch, zoom, focus_y).
        """
        if len(self._cam_sliders) < 4:
            return
        zoom_slider = self._cam_sliders[2][0]
        focus_slider = self._cam_sliders[3][0]
        for slider, value in ((zoom_slider, zoom), (focus_slider, focus_y)):
            lo = slider.property("float_lo")
            step = slider.property("float_step")
            if lo is None or step is None or step == 0:
                continue
            pos = int(round((float(value) - float(lo)) / float(step)))
            pos = max(slider.minimum(), min(slider.maximum(), pos))
            slider.setValue(pos)  # triggers on_change → runtime update

    def _pick_hair_color(self) -> None:
        col = QColorDialog.getColor(parent=self)
        if not col.isValid():
            return
        hex_ = col.name()
        self.hair_swatch.setStyleSheet(
            f"background:{hex_}; border:1px solid #888;")
        self.runtime.set_slider("hair_color", hex_)


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
