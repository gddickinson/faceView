"""Debug panel: 'What the LLM is seeing right now'.

Shows the live perception snapshot the chat bots have access to via
the system prompt. Two sections:

1. The narrated paragraph (exactly what gets prepended to the system
   prompt) — so you can verify what the model receives.
2. A structured grid of the individual signals so you can spot which
   ones are stale or missing.

Refreshed by a Qt timer at 4 Hz; reads from the singleton
:class:`PerceptionStore`. Cheap — no bus subscriptions of its own.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from faceview.vision.perception import PerceptionStore


_FIELDS: list[tuple[str, str]] = [
    ("Presence",      "presence"),
    ("Identity",      "identity"),
    ("Emotion",       "emotion"),
    ("Mouth",         "mouth"),
    ("Head pose",     "head_pose"),
    ("Gaze",          "gaze"),
    ("Distance",      "distance"),
    ("Blink",         "blink"),
    ("Gesture",       "gesture"),
    ("Scene",         "scene"),
    ("Scene caption", "scene_caption"),
    ("Objects",       "objects"),
]


class PerceptionPanel(QWidget):
    """Live view of the perception snapshot fed to the LLM."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._store = PerceptionStore.shared()
        self._labels: dict[str, QLabel] = {}
        self._build_ui()

        # Refresh at 4 Hz — fast enough to feel live, cheap enough to
        # not eat the GUI thread.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(250)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Perception (what the LLM sees)")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        # Narrated paragraph.
        self.narrative = QPlainTextEdit(self)
        self.narrative.setReadOnly(True)
        self.narrative.setObjectName("perception_narrative")
        self.narrative.setMaximumHeight(120)
        self.narrative.setPlaceholderText(
            "Live status that gets prepended to the LLM system prompt "
            "on every turn will appear here once the vision workers "
            "publish data."
        )
        root.addWidget(self.narrative)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        form = QFormLayout()
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for title, key in _FIELDS:
            lab = QLabel("—")
            lab.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Preferred)
            lab.setStyleSheet("color:#9aa3b2;")
            self._labels[key] = lab
            form.addRow(title, lab)
        root.addLayout(form, 1)

    # ── refresh ───────────────────────────────────────────────────────

    def _tick(self) -> None:
        snap = self._store.snapshot_dict()
        narr = self._store.narrate_now()
        if narr != self.narrative.toPlainText():
            self.narrative.setPlainText(narr)

        for _title, key in _FIELDS:
            lab = self._labels[key]
            payload = snap.get(key)
            lab.setText(_format_field(key, payload))
            stale = payload is not None and not payload.get("fresh", True)
            if payload is None:
                lab.setStyleSheet("color:#666;")
            elif stale:
                lab.setStyleSheet("color:#aaa;font-style:italic;")
            else:
                lab.setStyleSheet("color:#22a36b;")


def _format_field(key: str, payload: dict | None) -> str:
    if payload is None:
        return "—"
    if key == "presence":
        return f"{payload.get('face_count', 0)} face(s)"
    if key == "identity":
        if payload.get("is_owner"):
            return f"owner {payload.get('similarity', 0.0):.2f}"
        return f"{payload.get('label','unknown')} " \
               f"{payload.get('similarity', 0.0):.2f}"
    if key == "emotion":
        return f"{payload.get('label','neutral')} " \
               f"{payload.get('confidence', 0.0):.0%}"
    if key == "mouth":
        if payload.get("speaking"):
            return payload.get("viseme") or "speaking"
        return "silent"
    if key == "head_pose":
        return (f"yaw {payload.get('yaw', 0.0):+.2f}, "
                f"pitch {payload.get('pitch', 0.0):+.2f}, "
                f"roll {payload.get('roll', 0.0):+.2f}")
    if key == "gaze":
        return (f"{payload.get('direction','?')} "
                f"(att {payload.get('attention', 0.0):.2f})")
    if key == "distance":
        return (f"{payload.get('label','?')} "
                f"({payload.get('bbox_ratio', 0.0):.2f})")
    if key == "blink":
        return (f"{payload.get('state','?')} "
                f"EAR {payload.get('eye_open', 0.0):.2f}, "
                f"{payload.get('rate_per_min', 0.0):.0f}/min")
    if key == "gesture":
        label = payload.get("label") or "none"
        hand = payload.get("hand") or "?"
        return f"{label} ({hand})"
    if key == "scene":
        return (f"{payload.get('brightness_label','?')} / "
                f"{payload.get('motion_label','?')}")
    if key == "scene_caption":
        text = (payload.get("text") or "").strip()
        if not text:
            return "(awaiting first caption)"
        model = payload.get("model") or "?"
        lat = payload.get("latency_s", 0.0)
        # Truncate to keep the row short — the full text is in the
        # narrative box above.
        snippet = text if len(text) < 90 else text[:87] + "..."
        return f'"{snippet}"  [{model}, {lat:.1f}s]'
    if key == "objects":
        items = payload.get("detections") or []
        if not items:
            return "(none detected)"
        return ", ".join(d.get("label", "?") for d in items[:5])
    return str(payload)
