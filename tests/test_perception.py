"""Coverage for the perception aggregator + narrator."""

from __future__ import annotations


def _reset_singleton(monkeypatch, tmp_path=None):
    # Isolate the people-store path too — PerceptionStore.narrate_now
    # reads the roster, and the dev's real on-disk people would leak
    # into "empty" assertions otherwise.
    if tmp_path is not None:
        import faceview.config as cfg
        monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
        monkeypatch.setattr(cfg.settings, "owner_dir", tmp_path / "owner")
    import faceview.vision.people as people
    monkeypatch.setattr(people.PeopleStore, "_instance", None)
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    return p


def test_narrate_now_empty_when_no_signals(fresh_bus, monkeypatch, tmp_path):
    p = _reset_singleton(monkeypatch, tmp_path)
    store = p.PerceptionStore.shared()
    # Nothing on the bus yet → safe empty string so adding the provider
    # never breaks the engine call.
    assert store.narrate_now() == ""


def test_narrate_now_includes_signals(fresh_bus, monkeypatch):
    p = _reset_singleton(monkeypatch)
    from faceview.core.events import (
        EventType, Emotion, Gaze, Gesture, Presence, SceneInfo,
    )
    store = p.PerceptionStore.shared()
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=1, bboxes=[]))
    fresh_bus.publish(EventType.EMOTION,
                      Emotion(label="happy", confidence=0.84,
                              scores={"happy": 0.84}))
    fresh_bus.publish(EventType.GAZE,
                      Gaze(direction="camera", yaw=0.0, pitch=0.0,
                           attention=0.95))
    fresh_bus.publish(EventType.GESTURE,
                      Gesture(label="thumbs_up", hand="right",
                              confidence=0.92))
    fresh_bus.publish(EventType.SCENE,
                      SceneInfo(brightness=0.6, brightness_label="lit",
                                motion=0.1, motion_label="moving"))

    text = store.narrate_now()
    assert text.startswith("Live perception")
    assert "one face is visible" in text
    assert "happy" in text
    assert "looking at the camera" in text
    assert "thumbs_up" in text
    assert "lit" in text


def test_snapshot_dict_marks_freshness(fresh_bus, monkeypatch):
    p = _reset_singleton(monkeypatch)
    from faceview.core.events import EventType, Presence
    store = p.PerceptionStore.shared()
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=2, bboxes=[]))
    snap = store.snapshot_dict()
    assert snap["presence"]["face_count"] == 2
    assert snap["presence"]["fresh"] is True
    assert snap["emotion"] is None


def test_conversation_composes_multiple_extras_providers():
    from faceview.llm.conversation import Conversation
    conv = Conversation(system="BASE")
    conv.add_system_extras_provider(lambda: "PERCEPTION")
    conv.add_system_extras_provider(lambda: "MEMORY")
    full = conv.effective_system()
    # Order: registered providers first (in order), then base.
    assert full == "PERCEPTION\n\nMEMORY\n\nBASE"


def test_conversation_skips_empty_providers():
    from faceview.llm.conversation import Conversation
    conv = Conversation(system="BASE")
    conv.add_system_extras_provider(lambda: "")
    conv.add_system_extras_provider(lambda: "REAL")
    assert conv.effective_system() == "REAL\n\nBASE"


def test_conversation_remove_provider():
    from faceview.llm.conversation import Conversation
    conv = Conversation(system="BASE")

    def mem() -> str:
        return "MEM"

    conv.add_system_extras_provider(mem)
    assert "MEM" in conv.effective_system()
    conv.remove_system_extras_provider(mem)
    assert "MEM" not in conv.effective_system()
