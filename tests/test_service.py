"""Service-layer ops: state, screenshot, send_chat, list_events."""

from __future__ import annotations

from PySide6.QtCore import QEventLoop, QTimer

from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, Presence
from faceview.gui.main_window import MainWindow
from faceview.server.service import Service


def _spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_camera_state_reflects_presence(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    qtbot.waitExposed(win)
    svc = Service(win)

    get_bus().publish(EventType.PRESENCE, Presence(face_count=2))
    _spin(20)
    state = svc.get_camera_state()
    assert state["presence"]["face_count"] == 2


def test_screenshot_via_service_writes_png(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr("faceview.server.service.docs_image_dir", lambda: tmp_path)
    monkeypatch.setattr("faceview.gui.screenshotter.docs_image_dir", lambda: tmp_path)

    win = MainWindow()
    qtbot.addWidget(win)
    qtbot.waitExposed(win)
    svc = Service(win)

    res = svc.screenshot("svc_test.png", encode_b64=True)
    assert res["ok"] is True
    assert res["path"].endswith("svc_test.png")
    assert "png_b64" in res and len(res["png_b64"]) > 100


def test_send_chat_emits_user_event(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)

    received: list = []
    get_bus().subscribe(EventType.CHAT_USER_MESSAGE, received.append)

    res = svc.send_chat("hello from test")
    _spin(20)
    assert res == {"ok": True, "queued": True}
    assert any(getattr(m, "content", "") == "hello from test" for m in received)


def test_send_chat_rejects_empty(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    res = svc.send_chat("   ")
    assert res["ok"] is False


def test_list_events_returns_jsonable(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)

    get_bus().publish(EventType.PRESENCE, Presence(face_count=1))
    _spin(20)
    evs = svc.list_events(n=10)
    assert any(e["type"] == "PRESENCE" for e in evs)
