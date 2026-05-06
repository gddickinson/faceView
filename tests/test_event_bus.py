"""EventBus pub/sub semantics."""

from __future__ import annotations

from faceview.core.event_bus import EventBus
from faceview.core.events import ChatMessage, EventType


def test_subscribe_and_publish(qtbot):
    bus = EventBus()
    seen: list[ChatMessage] = []

    def handler(msg):
        seen.append(msg)

    bus.subscribe(EventType.CHAT_USER_MESSAGE, handler)
    msg = ChatMessage("user", "hello")
    bus.publish(EventType.CHAT_USER_MESSAGE, msg)

    qtbot.wait(20)  # let the queued signal flush
    assert seen == [msg]


def test_unsubscribe_stops_delivery(qtbot):
    bus = EventBus()
    received: list[str] = []

    def h1(msg):
        received.append("h1")

    bus.subscribe(EventType.STATUS, h1)
    bus.publish(EventType.STATUS, "ping")
    qtbot.wait(20)
    assert received == ["h1"]

    bus.unsubscribe(EventType.STATUS, h1)
    bus.publish(EventType.STATUS, "ping")
    qtbot.wait(20)
    assert received == ["h1"]  # still 1


def test_handler_exception_does_not_crash_bus(qtbot, capsys):
    bus = EventBus()
    seen: list[str] = []

    def bad(msg):
        raise RuntimeError("boom")

    def good(msg):
        seen.append(msg)

    bus.subscribe(EventType.STATUS, bad)
    bus.subscribe(EventType.STATUS, good)
    bus.publish(EventType.STATUS, "x")
    qtbot.wait(20)

    assert seen == ["x"]
    out = capsys.readouterr().err
    assert "handler error" in out
