"""Core primitives shared by all subsystems.

- :mod:`faceview.core.event_bus` — Qt-signal pub/sub hub
- :mod:`faceview.core.events`    — event types and payload dataclasses
- :mod:`faceview.core.logger`    — structlog config
- :mod:`faceview.core.errors`    — exception hierarchy
"""

from faceview.core.event_bus import EventBus, get_bus
from faceview.core.events import EventType, ChatMessage, Transcript, FrameInfo
from faceview.core.errors import FaceViewError, MissingDependency

__all__ = [
    "EventBus",
    "get_bus",
    "EventType",
    "ChatMessage",
    "Transcript",
    "FrameInfo",
    "FaceViewError",
    "MissingDependency",
]
