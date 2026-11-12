"""Core primitives shared by all subsystems.

- :mod:`faceview.core.event_bus` — Qt-signal pub/sub hub
- :mod:`faceview.core.events`    — event types and payload dataclasses
- :mod:`faceview.core.logger`    — structlog config
- :mod:`faceview.core.errors`    — exception hierarchy

Exports are resolved lazily (PEP 562) so the Qt-free modules in this package
(e.g. the animation and character code used by the web interface) import without PySide6.
"""

from importlib import import_module

_EXPORTS = {
    "EventBus": "faceview.core.event_bus",
    "get_bus": "faceview.core.event_bus",
    "EventType": "faceview.core.events",
    "ChatMessage": "faceview.core.events",
    "Transcript": "faceview.core.events",
    "FrameInfo": "faceview.core.events",
    "FaceViewError": "faceview.core.errors",
    "MissingDependency": "faceview.core.errors",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
