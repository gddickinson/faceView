"""PeopleStore + remember_person tool coverage.

These tests deliberately don't touch InsightFace — they inject a fake
embedding function so we can verify the persistence + matching logic
without the heavy identity dep installed.
"""

from __future__ import annotations

import numpy as np


def _store_with_tmp_dir(tmp_path, monkeypatch):
    """Build a PeopleStore that reads/writes inside ``tmp_path``."""
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    monkeypatch.setattr(cfg.settings, "owner_dir", tmp_path / "owner_data")
    from faceview.vision.people import PeopleStore
    PeopleStore.reset_for_tests()
    return PeopleStore.shared()


def test_empty_store_matches_stranger(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    name, sim, known = store.match(np.ones(512, dtype=np.float32))
    assert name == "stranger"
    assert sim == 0.0
    assert known is False


def test_remember_persists_to_disk(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)

    # Fake embed_fn returns a deterministic vector per "frame".
    def fake_embed(_frame):
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec

    store.set_embed_fn(fake_embed)
    ok, msg = store.remember("Alice", np.zeros((10, 10, 3), dtype=np.uint8))
    assert ok is True
    assert "Alice" in msg
    assert "alice.npz" in {p.name for p in (tmp_path / "people").iterdir()}

    # Re-instantiate and confirm Alice is loaded from disk.
    from faceview.vision.people import PeopleStore
    PeopleStore.reset_for_tests()
    s2 = PeopleStore.shared()
    assert "Alice" in s2.list_people()


def test_match_recognises_remembered_person(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0
    store.set_embed_fn(lambda _f: vec.copy())
    store.remember("Bob", np.zeros((10, 10, 3), dtype=np.uint8))

    name, sim, known = store.match(vec)
    assert name == "Bob"
    assert known is True
    assert sim > 0.95


def test_match_rejects_dissimilar(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    v_bob = np.zeros(512, dtype=np.float32)
    v_bob[0] = 1.0
    store.set_embed_fn(lambda _f: v_bob.copy())
    store.remember("Bob", np.zeros((10, 10, 3), dtype=np.uint8))

    v_other = np.zeros(512, dtype=np.float32)
    v_other[1] = 1.0
    name, _sim, known = store.match(v_other)
    assert name == "stranger"
    assert known is False


def test_remember_rejects_empty_name(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    store.set_embed_fn(lambda _f: np.zeros(512, dtype=np.float32))
    ok, _msg = store.remember("   ", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok is False


def test_remember_fails_without_embed_fn(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    ok, msg = store.remember("Carol", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok is False
    assert "Identity recognizer" in msg


def test_remember_fails_when_embed_returns_none(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    store.set_embed_fn(lambda _f: None)
    ok, msg = store.remember("Dave", np.zeros((4, 4, 3), dtype=np.uint8))
    assert ok is False
    assert "face" in msg.lower()


def test_forget_removes_entry(tmp_path, monkeypatch):
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    store.set_embed_fn(lambda _f: np.eye(1, 512, dtype=np.float32).ravel())
    store.remember("Eve", np.zeros((4, 4, 3), dtype=np.uint8))
    assert "Eve" in store.list_people()
    assert store.forget("Eve") is True
    assert "Eve" not in store.list_people()


def test_tool_schemas_have_required_name():
    from faceview.llm.vision_tool import (
        REMEMBER_TOOL_ANTHROPIC, REMEMBER_TOOL_OLLAMA,
    )
    assert REMEMBER_TOOL_ANTHROPIC["name"] == "remember_person"
    assert REMEMBER_TOOL_ANTHROPIC["input_schema"]["required"] == ["name"]
    f = REMEMBER_TOOL_OLLAMA["function"]
    assert f["name"] == "remember_person"
    assert f["parameters"]["required"] == ["name"]


def test_run_remember_person_routes_through_store(
    tmp_path, monkeypatch, fresh_bus,
):
    """End-to-end: tool helper calls PeopleStore.remember with the
    cached frame from FrameGrabber, succeeds when an embed_fn is wired."""
    store = _store_with_tmp_dir(tmp_path, monkeypatch)
    store.set_embed_fn(
        lambda _f: np.eye(1, 512, dtype=np.float32).ravel()
    )

    import faceview.llm.vision_tool as vt
    monkeypatch.setattr(vt.FrameGrabber, "_instance", None)
    g = vt.FrameGrabber.shared()
    fresh_bus.publish(vt.EventType.FRAME,
                      np.zeros((20, 20, 3), dtype=np.uint8))
    msg = vt.run_remember_person(g, "Frankie")
    assert "Frankie" in msg
    assert "Frankie" in store.list_people()


def test_perception_nudges_after_stranger_visible(
    tmp_path, monkeypatch, fresh_bus,
):
    """After a stranger has been visible for >2 s the narration should
    prompt the LLM to ask their name."""
    import time as _time
    _store_with_tmp_dir(tmp_path, monkeypatch)
    import faceview.vision.perception as perc
    perc.PerceptionStore._instance = None
    store = perc.PerceptionStore.shared()

    from faceview.core.events import EventType, Identity, Presence
    fresh_bus.publish(EventType.PRESENCE, Presence(face_count=1, bboxes=[]))
    fresh_bus.publish(EventType.IDENTITY,
                      Identity(is_owner=False, similarity=0.1,
                               label="stranger"))
    # Immediately, the run is < 2 s so no nudge yet.
    narr_early = store.narrate_now()
    assert "unfamiliar person" not in narr_early
    # Backdate the run so the 2 s threshold has been crossed.
    store._stranger_since = _time.time() - 4.0
    narr = store.narrate_now()
    assert "unfamiliar person" in narr
    assert "remember_person" in narr
