"""Status panel: live readouts of presence/identity/emotion/mouth/etc.

The panel only consumes events from the bus — it doesn't reach into the
vision modules — so it works equally well with real models, mocked workers,
or seeded demo state for screenshots.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from faceview.core.event_bus import get_bus
from faceview.core.events import (
    Emotion,
    EventType,
    Identity,
    MouthActivity,
    Presence,
)


def _model_short(name: str) -> str:
    """Compact label for the LLM pill: 'claude-opus-4-7' → 'opus 4.7'."""
    n = name.lower()
    if "opus" in n:
        return "opus " + n.split("opus-")[-1].replace("-", ".")[:3]
    if "sonnet" in n:
        return "sonnet " + n.split("sonnet-")[-1].replace("-", ".")[:3]
    if "haiku" in n:
        return "haiku " + n.split("haiku-")[-1].replace("-", ".")[:3]
    return name


class _Pill(QLabel):
    def __init__(self, text: str = "—", color: str = "#444"):
        super().__init__(text)
        self.set_color(color)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(110)
        self.setMaximumHeight(26)

    def set_color(self, color: str) -> None:
        self.setStyleSheet(
            f"background:{color};color:white;border-radius:13px;padding:2px 10px;"
            f"font-weight:600;"
        )


class StatusPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._wire_bus()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        header = QLabel("Status")
        f = QFont()
        f.setBold(True)
        f.setPointSize(13)
        header.setFont(f)
        root.addWidget(header)

        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.presence = _Pill("idle", "#666")
        self.identity = _Pill("unknown", "#666")
        self.emotion = _Pill("—", "#666")
        self.mouth = _Pill("silent", "#666")
        self.audio = _Pill("idle", "#666")
        self.llm = _Pill(self._llm_initial_label(), "#3a8")

        form.addRow("Presence", self.presence)
        form.addRow("Identity", self.identity)
        form.addRow("Emotion", self.emotion)
        form.addRow("Mouth", self.mouth)
        form.addRow("Audio", self.audio)
        form.addRow("LLM", self.llm)

        root.addLayout(form)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        self.summary = QLabel("Waiting for events…")
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pal = self.summary.palette()
        pal.setColor(QPalette.ColorRole.WindowText, QColor("#9aa3b2"))
        self.summary.setPalette(pal)
        root.addWidget(self.summary, 1)

    def _wire_bus(self) -> None:
        bus = get_bus()
        bus.subscribe(EventType.PRESENCE, self._on_presence)
        bus.subscribe(EventType.IDENTITY, self._on_identity)
        bus.subscribe(EventType.EMOTION, self._on_emotion)
        bus.subscribe(EventType.MOUTH_ACTIVITY, self._on_mouth)
        bus.subscribe(EventType.STATUS, self._on_status)
        bus.subscribe(EventType.VAD_SPEECH_START, lambda _p: self.audio.setText("speech") or self.audio.set_color("#22a"))
        bus.subscribe(EventType.VAD_SPEECH_END, lambda _p: self.audio.setText("idle") or self.audio.set_color("#666"))

    @staticmethod
    def _llm_initial_label() -> str:
        from faceview.config import settings
        if not settings.has_claude_key:
            return "demo mode"
        return _model_short(settings.anthropic_model)

    def set_llm_label(
        self,
        text: str,
        *,
        has_key: bool | None = None,
        color: str | None = None,
    ) -> None:
        """Update the LLM pill (called when the user changes engines / models).

        Pass ``color`` to override the pill colour explicitly — used by the
        config dialog to distinguish anthropic / ollama / demo at a glance.
        """
        self.llm.setText(text)
        if color is None:
            if has_key is None:
                from faceview.config import settings
                has_key = settings.has_claude_key
            color = "#3a8" if has_key else "#666"
        self.llm.set_color(color)

    # ── slots ────────────────────────────────────────────────────────

    def _on_presence(self, p: Presence) -> None:
        if p.face_count == 0:
            self.presence.setText("absent")
            self.presence.set_color("#666")
        elif p.face_count == 1:
            self.presence.setText("present")
            self.presence.set_color("#28a745")
        else:
            self.presence.setText(f"{p.face_count} faces")
            self.presence.set_color("#fd7e14")

    def _on_identity(self, i: Identity) -> None:
        if i.is_owner:
            self.identity.setText(f"owner {i.similarity:.2f}")
            self.identity.set_color("#1a73e8")
        else:
            self.identity.setText(f"{i.label} {i.similarity:.2f}")
            self.identity.set_color("#9aa3b2")

    def _on_emotion(self, e: Emotion) -> None:
        self.emotion.setText(f"{e.label} {e.confidence:.0%}")
        color = {
            "happy": "#22a36b",
            "neutral": "#666",
            "sad": "#5066c0",
            "surprise": "#e8a23a",
            "angry": "#c0392b",
            "fear": "#8e44ad",
            "disgust": "#7d6608",
        }.get(e.label, "#666")
        self.emotion.set_color(color)

    def _on_mouth(self, m: MouthActivity) -> None:
        if m.speaking:
            label = m.viseme or "speaking"
            self.mouth.setText(label)
            self.mouth.set_color("#1a73e8")
        else:
            self.mouth.setText("silent")
            self.mouth.set_color("#666")

    def _on_status(self, payload) -> None:
        msg = getattr(payload, "message", str(payload))
        self.summary.setText(msg)

    # ── seeded demo state for screenshots ───────────────────────────

    def seed_demo(self) -> None:
        self._on_presence(Presence(face_count=1))
        self._on_identity(Identity(is_owner=True, similarity=0.71, label="owner"))
        self._on_emotion(
            Emotion(
                label="neutral",
                confidence=0.78,
                scores={"neutral": 0.78, "happy": 0.12, "surprise": 0.05},
            )
        )
        self._on_mouth(
            MouthActivity(
                speaking=False,
                jaw_open=0.04,
                mouth_funnel=0.02,
                mouth_pucker=0.01,
                viseme=None,
            )
        )
        self.audio.setText("idle")
        self.summary.setText(
            "1 face present — owner recognised (sim 0.71). Neutral. "
            "Mic idle, ready to listen."
        )
