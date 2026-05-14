"""C3 — per-person memory branches."""

from __future__ import annotations

import json


def _store(tmp_path, monkeypatch):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    return cog.CognitionStore("testpersona")


def test_record_routes_to_per_person_when_speaker_known(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("George")
    store.record_chat_turn("Hey it's me", "Hi George")
    assert store.episodic == []
    assert "George" in store.per_person
    assert len(store.per_person["George"]) == 1
    assert store.per_person["George"][0]["speaker"] == "George"


def test_record_routes_to_global_when_stranger(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("stranger")
    store.record_chat_turn("Hi who is this", "Hello there")
    assert store.per_person == {}
    assert len(store.episodic) == 1
    assert "speaker" not in store.episodic[0]


def test_no_speaker_means_global(tmp_path, monkeypatch):
    """When neither override nor PerceptionStore reports a name, the
    turn goes to the shared episodic list."""
    store = _store(tmp_path, monkeypatch)
    # Don't set an override; reset PerceptionStore so current_speaker
    # returns None.
    import faceview.vision.perception as p
    monkeypatch.setattr(p.PerceptionStore, "_instance", None)
    store.record_chat_turn("hi", "hello")
    assert len(store.episodic) == 1
    assert store.per_person == {}


def test_narrate_includes_per_person_history(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Alice")
    store.record_chat_turn("My favourite colour is teal", "Noted.")
    store.record_chat_turn("Did I tell you I'm a violinist?", "Wow!")
    out = store.narrate_for_prompt()
    assert "Conversation history with Alice" in out
    assert "teal" in out


def test_recall_by_embedding_sees_speaker_bucket(tmp_path, monkeypatch):
    """The semantic recall should pull from the current speaker's
    bucket AND the shared episodic list."""
    import math

    class _Fake:
        _PETS = ("cat", "dog", "pet")
        def available(self) -> bool: return True
        def embed(self, text: str):
            v = [0.0, 0.0]
            t = (text or "").lower()
            v[0] = 1.0 if any(p in t for p in self._PETS) else 0.0
            v[1] = 1.0 if "food" in t else 0.0
            if sum(v) == 0: return None
            n = math.sqrt(sum(x*x for x in v))
            return [x / n for x in v]

    import faceview.llm.embeddings as emb
    monkeypatch.setattr(emb.EmbeddingService, "shared",
                        classmethod(lambda cls: _Fake()))

    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Bob")
    store.record_chat_turn("My dog Rex needs walking", "Sweet.")
    # An unrelated turn in the shared episodic list.
    store.set_current_speaker(None)
    store.record_chat_turn("Pizza for dinner", "Yum.")
    # Now Bob is speaking again; query about pets.
    store.set_current_speaker("Bob")
    found = store.recall_by_embedding("Tell me about your cat", limit=3)
    labels = [m.get("text", "") for m in found]
    # Bob's own dog turn should beat the unrelated pizza turn.
    assert any("dog Rex" in t for t in labels)


def test_schema_migration_v2_to_v3(tmp_path, monkeypatch):
    """An on-disk v2 file (no per_person key) should load cleanly and
    treat per_person as empty."""
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "v2legacy.json").write_text(json.dumps({
        "schema": 2,
        "persona": "v2legacy",
        "first_seen": "2025-01-01",
        "session_count": 3,
        "relationship_score": 5,
        "episodic": [{
            "ts": 1700000000.0, "type": "chat", "text": "old",
            "significance": 4, "emotion": "neutral", "session_id": 0,
            "recalled": 0,
        }],
        "semantic": {}, "emotional": {},
    }))
    store = cog.CognitionStore.load("v2legacy")
    assert len(store.episodic) == 1
    assert store.per_person == {}


def test_save_round_trips_per_person(tmp_path, monkeypatch):
    """Persist a per-person bucket and reload it."""
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Carol")
    store.record_chat_turn("hi from Carol", "hi Carol")
    store.save()
    # Re-instantiate via the same data_dir.
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    reloaded = cog.CognitionStore.load("testpersona")
    assert "Carol" in reloaded.per_person
    assert len(reloaded.per_person["Carol"]) == 1


def test_summary_includes_per_person_counts(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Dave")
    store.record_chat_turn("hi", "hello")
    store.record_chat_turn("how are you", "good")
    s = store.summary()
    assert s["per_person"] == {"Dave": 2}


def test_clear_wipes_per_person(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Eve")
    store.record_chat_turn("hi", "hi")
    assert store.per_person
    store.clear()
    assert store.per_person == {}
