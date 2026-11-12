"""Camera and Body tabs of the Effects panel (split out of effects_panel.py
to keep it under the 500-line budget).

:class:`CameraBodyTabsMixin` is mixed into :class:`EffectsPanel`; its methods
use the panel's helpers (``_labelled_slider``, ``_snap_view``, ``_snap_zoom``,
``_reset_camera``, ``_effect_button``) and ``self.runtime``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)


class CameraBodyTabsMixin:
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
            btn = self._effect_button(label, name)
            pose_row1.addWidget(btn)
        layout.addLayout(pose_row1)
        pose_row2 = QHBoxLayout()
        for label, name in [
            ("Lean L",   "body_lean_left"),
            ("Lean R",   "body_lean_right"),
            ("Twist L",  "body_twist_left"),
            ("Twist R",  "body_twist_right"),
        ]:
            btn = self._effect_button(label, name)
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
            btn = self._effect_button(label, name)
            limb_row1.addWidget(btn)
        layout.addLayout(limb_row1)
        limb_row2 = QHBoxLayout()
        for label, name in [
            ("Kick L",  "kick_left"),
            ("Kick R",  "kick_right"),
            ("Squat",   "squat"),
        ]:
            btn = self._effect_button(label, name)
            limb_row2.addWidget(btn)
        layout.addLayout(limb_row2)

        layout.addStretch(1)
        return w

    # ── Tongue tab ─────────────────────────────────────────────
