"""New Service ops: set_emotion, set_persona, avatar_say, list_personas."""

from __future__ import annotations

from faceview.gui.main_window import MainWindow
from faceview.server.service import Service


class _FakeWorker:
    """Mimics SimCameraWorker.avatar surface for the service ops."""

    def __init__(self):
        from faceview.vision.avatar import TalkingAvatar
        self.avatar = TalkingAvatar(emotion="neutral", persona="default", seed=0)


def test_set_emotion_without_avatar_returns_error(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    res = svc.set_emotion("happy")
    assert res["ok"] is False
    assert "no avatar" in res["error"]


def test_set_emotion_changes_avatar_baseline(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    worker = _FakeWorker()
    svc.bind_camera_worker(worker)

    res = svc.set_emotion("happy")
    assert res == {"ok": True, "emotion": "happy"}
    assert worker.avatar.emotion == "happy"


def test_set_persona_changes_avatar_persona(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    worker = _FakeWorker()
    svc.bind_camera_worker(worker)

    res = svc.set_persona("auburn")
    assert res == {"ok": True, "persona": "auburn"}
    assert worker.avatar.persona.name == "auburn"


def test_avatar_say_returns_duration_and_phoneme_count(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    worker = _FakeWorker()
    svc.bind_camera_worker(worker)

    res = svc.avatar_say("Hello world.")
    assert res["ok"] is True
    assert res["duration"] > 0
    assert res["phonemes"] >= 1


def test_avatar_say_rejects_empty(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    res = svc.avatar_say("   ")
    assert res["ok"] is False


def test_list_personas_includes_known(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    svc = Service(win)
    names = svc.list_personas()
    assert "default" in names and "claude" in names
