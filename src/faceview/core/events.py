"""Event types and payload dataclasses for the in-process bus.

Every event published on :class:`faceview.core.event_bus.EventBus` carries one
of the dataclasses below as its payload. Keeping payloads as explicit
dataclasses (rather than free dicts) lets pylance/mypy and pytest verify the
contract end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import time
from typing import Optional


class EventType(Enum):
    # Audio pipeline
    AUDIO_CHUNK = auto()
    VAD_SPEECH_START = auto()
    VAD_SPEECH_END = auto()
    TRANSCRIPT_PARTIAL = auto()
    TRANSCRIPT_FINAL = auto()

    # LLM
    CHAT_USER_MESSAGE = auto()
    LLM_TOKEN = auto()
    LLM_REPLY = auto()
    LLM_ERROR = auto()

    # TTS
    TTS_SPEAK = auto()
    TTS_STARTED = auto()
    TTS_FINISHED = auto()

    # Vision
    FRAME = auto()           # real webcam frame (the user)
    AVATAR_FRAME = auto()    # rendered avatar frame (Claude)
    PRESENCE = auto()
    IDENTITY = auto()
    EMOTION = auto()
    MOUTH_ACTIVITY = auto()
    HEAD_POSE = auto()       # yaw/pitch/roll from face-mesh landmarks

    # Lifecycle / generic
    SCREENSHOT_TAKEN = auto()
    STATUS = auto()
    ERROR = auto()


# ── Payloads ─────────────────────────────────────────────────────────────


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    ts: float = field(default_factory=time)


@dataclass
class Transcript:
    text: str
    is_final: bool
    ts: float = field(default_factory=time)


@dataclass
class FrameInfo:
    width: int
    height: int
    ts: float = field(default_factory=time)
    fps: float = 0.0


@dataclass
class Presence:
    face_count: int
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    ts: float = field(default_factory=time)


@dataclass
class Identity:
    is_owner: bool
    similarity: float  # cosine similarity to owner template, in [-1, 1]
    label: str = "unknown"
    ts: float = field(default_factory=time)


@dataclass
class Emotion:
    label: str  # happy | sad | angry | surprise | fear | disgust | neutral
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    ts: float = field(default_factory=time)


@dataclass
class HeadPose:
    yaw: float       # left/right rotation, -1..1 (positive = turning right)
    pitch: float     # up/down, -1..1 (positive = chin up)
    roll: float      # tilt, -1..1 (positive = right ear toward shoulder)
    ts: float = field(default_factory=time)


@dataclass
class MouthActivity:
    speaking: bool
    jaw_open: float
    mouth_funnel: float
    mouth_pucker: float
    viseme: Optional[str] = None  # crude class: AA, EE, OO, MM, ...
    ts: float = field(default_factory=time)


@dataclass
class StatusEvent:
    source: str
    message: str
    level: str = "info"  # info | warning | error
    ts: float = field(default_factory=time)


@dataclass
class ErrorEvent:
    source: str
    message: str
    detail: Optional[str] = None
    ts: float = field(default_factory=time)
