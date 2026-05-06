"""In-process pub/sub bus built on Qt signals.

Why a Qt signal bus rather than blinker/pypubsub: signals integrate natively
with ``Qt.QueuedConnection`` so cross-thread delivery is safe and ordered, and
this lets us schedule slot calls back on the main GUI thread for free.

Usage::

    from faceview.core.event_bus import get_bus
    from faceview.core.events import EventType, ChatMessage

    bus = get_bus()
    bus.subscribe(EventType.CHAT_USER_MESSAGE, on_user_msg)
    bus.publish(EventType.CHAT_USER_MESSAGE, ChatMessage("user", "hi"))
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, Signal


_Handler = Callable[[Any], None]


class EventBus(QObject):
    """A small QObject that fans events out to subscribers.

    A single ``payload`` Signal carries an :class:`~faceview.core.events.EventType`
    plus the matching dataclass payload. We then dispatch in-process to
    per-type subscriber lists so subscribers don't have to filter themselves.
    """

    payload = Signal(object, object)  # (EventType, payload-dataclass)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._subs: dict[Any, list[_Handler]] = {}
        self.payload.connect(self._dispatch)

    # ── pub/sub API ──────────────────────────────────────────────────

    def subscribe(self, event_type: Any, handler: _Handler) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: Any, handler: _Handler) -> None:
        try:
            self._subs.get(event_type, []).remove(handler)
        except ValueError:
            pass

    def publish(self, event_type: Any, payload: Any = None) -> None:
        """Emit the underlying Qt signal.

        Use a ``Qt.QueuedConnection``-style flow when called from a worker
        thread by emitting the Qt signal — Qt marshals the call back onto the
        receiver's thread automatically.
        """
        self.payload.emit(event_type, payload)

    def _dispatch(self, event_type: Any, payload: Any) -> None:
        for handler in list(self._subs.get(event_type, [])):
            try:
                handler(payload)
            except Exception as exc:  # noqa: BLE001 — bus must never crash
                # Avoid recursive ERROR re-entry: log via stderr only.
                import sys
                print(
                    f"[event_bus] handler error on {event_type}: {exc!r}",
                    file=sys.stderr,
                )


# ── module-level singleton ───────────────────────────────────────────────

_bus: EventBus | None = None


def get_bus() -> EventBus:
    """Return the process-wide :class:`EventBus`, creating it on first call."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_bus_for_tests() -> None:
    """Drop the singleton — call from test fixtures only."""
    global _bus
    _bus = None
