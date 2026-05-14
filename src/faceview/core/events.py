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
    CHAT_LOG = auto()        # any line that hit the chat panel (incl. test mode)

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
    GAZE = auto()            # iris-derived gaze direction + attention
    FACE_DISTANCE = auto()   # face bbox area / frame area → close|near|far
    BLINK = auto()           # eye aspect ratio + rolling blink rate
    GESTURE = auto()         # MP Gesture Recognizer label (thumbs_up, …)
    SCENE = auto()           # frame brightness + motion level
    SCENE_CAPTION = auto()   # ambient VLM caption (moondream, ~15 s cadence)
    OBJECTS = auto()         # MP Object Detector results
    PIXELS_LEAVING = auto()  # webcam frame being sent off-device (privacy)
    TURN_RECORDED = auto()   # one LLM turn's cost/latency telemetry

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
class ChatLogEntry:
    """One rendered line in the chat panel (any source)."""
    who: str          # "You", "Claude", "Camera bot", "Avatar bot", "error", ...
    text: str
    color: str = "#666"
    ts: float = field(default_factory=time)


@dataclass
class ErrorEvent:
    source: str
    message: str
    detail: Optional[str] = None
    ts: float = field(default_factory=time)


# ── Perception signals (read by PerceptionStore) ─────────────────────────


@dataclass
class Gaze:
    """Where the eyes are pointing, derived from refined face-mesh iris."""
    direction: str               # camera | left | right | up | down | away
    yaw: float                   # -1..1 horizontal iris offset
    pitch: float                 # -1..1 vertical iris offset
    attention: float             # 0..1, 1 = looking straight at the camera
    ts: float = field(default_factory=time)


@dataclass
class FaceDistance:
    """How close the user is to the camera (bbox area relative to frame)."""
    label: str                   # close | near | normal | far
    bbox_ratio: float            # 0..1, fraction of frame the face fills
    ts: float = field(default_factory=time)


@dataclass
class Blink:
    """Eye-aspect-ratio state + a rolling blink rate."""
    eye_open: float              # 0..~0.35; <0.18 ≈ closed
    state: str                   # open | closed | drowsy
    rate_per_min: float          # blinks per minute, rolling 30 s window
    ts: float = field(default_factory=time)


@dataclass
class Gesture:
    """Hand-gesture classification from MP Gesture Recognizer."""
    label: str                   # thumbs_up | thumbs_down | open_palm | ...
    hand: str                    # left | right | both | none
    confidence: float
    ts: float = field(default_factory=time)


@dataclass
class SceneInfo:
    """Coarse scene descriptors that aren't tied to the user's face."""
    brightness: float            # 0..1, mean luminance of a downscaled frame
    brightness_label: str        # dark | dim | lit | bright
    motion: float                # 0..1, normalised inter-frame difference
    motion_label: str            # still | moving | active
    ts: float = field(default_factory=time)


@dataclass
class DetectedObject:
    label: str
    score: float
    bbox: tuple[int, int, int, int]  # x, y, w, h (pixels)


@dataclass
class ObjectsSeen:
    detections: list[DetectedObject] = field(default_factory=list)
    ts: float = field(default_factory=time)


@dataclass
class SceneCaption:
    """Ambient VLM caption — one short sentence describing the scene."""
    text: str
    model: str = ""              # which VLM produced it (e.g. "moondream")
    latency_s: float = 0.0       # round-trip seconds (logged + shown)
    ts: float = field(default_factory=time)


@dataclass
class TurnRecord:
    """Per-turn telemetry: latency + token usage + $ cost."""
    engine: str                  # "anthropic" | "ollama" | "demo"
    model: str
    duration_s: float
    prompt_tokens: int           # 0 when unknown
    completion_tokens: int
    usd_cost: float              # 0 for local engines
    ts: float = field(default_factory=time)


@dataclass
class PixelTransmission:
    """Frame leaves the host. Privacy indicator.

    ``active=True`` is published at the moment we hand a frame to
    something that might transmit it; ``active=False`` when known
    finished. Subscribers (status panel) flash a red dot while
    active. The ``destination`` distinguishes Anthropic (real
    off-machine) from a local Ollama VLM (stays on machine but
    user still wants visibility into compute use)."""
    active: bool
    destination: str             # "anthropic" | "ollama:<model>" | ...
    tool: str = ""               # which tool triggered the send
    ts: float = field(default_factory=time)
