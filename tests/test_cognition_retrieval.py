"""Retrieval-augmented cognition (C4) coverage.

We don't pull in sentence-transformers for CI — instead we inject a
fake encoder so the embed/cosine/retrieve logic can be exercised
without the heavy dep. The fallback paths (missing dep) are also
covered.
"""

from __future__ import annotations

import math
from typing import Optional

import pytest


# ── fake embedding service ────────────────────────────────────────


class _FakeService:
    """Toy 3-D embedder — words map to canonical axes so we can
    write semantic tests that are actually predictable."""

    _AXIS = {
        # axis 0 = pets, 1 = food, 2 = travel
        "cat":     (1, 0, 0),
        "dog":     (1, 0, 0),
        "pet":     (1, 0, 0),
        "animal":  (1, 0, 0),
        "pizza":   (0, 1, 0),
        "food":    (0, 1, 0),
        "dinner":  (0, 1, 0),
        "eat":     (0, 1, 0),
        "trip":    (0, 0, 1),
        "travel":  (0, 0, 1),
        "flight":  (0, 0, 1),
        "vacation": (0, 0, 1),
    }

    def available(self) -> bool:
        return True

    def embed(self, text: str) -> Optional[list[float]]:
        if not text or not text.strip():
            return None
        vec = [0.0, 0.0, 0.0]
        for w in text.lower().split():
            for axis, name in [(0, "pets"), (1, "food"), (2, "travel")]:
                pass
            for needle, ax in self._AXIS.items():
                if needle in w:
                    for i, v in enumerate(ax):
                        vec[i] += v
        # L2-normalise.
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return None
        return [v / norm for v in vec]


def _inject_fake(monkeypatch) -> _FakeService:
    """Patch EmbeddingService.shared() to return a deterministic fake."""
    import faceview.llm.embeddings as emb
    fake = _FakeService()
    monkeypatch.setattr(emb.EmbeddingService, "shared", classmethod(
        lambda cls: fake
    ))
    return fake


def _store_with_tmp(tmp_path, monkeypatch):
    """Build a CognitionStore that writes its JSON under tmp_path."""
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    return cog.CognitionStore("testpersona")


# ── tests ─────────────────────────────────────────────────────────


def test_embedding_service_unavailable_returns_none(monkeypatch):
    """When sentence-transformers isn't installed, embed() returns
    None and the cognition layer must handle it gracefully."""
    import faceview.llm.embeddings as emb
    emb.EmbeddingService.reset_for_tests()
    monkeypatch.setattr(emb, "_sentence_transformers_available",
                        lambda: False)
    svc = emb.EmbeddingService()
    assert svc.available() is False
    assert svc.embed("hello") is None
    assert svc.embed_batch(["a", "b"]) == [None, None]


def test_cosine_corner_cases():
    from faceview.llm.embeddings import cosine
    assert cosine(None, [1.0]) == 0.0
    assert cosine([1.0], None) == 0.0
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0  # length mismatch
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0  # zero norm
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_record_chat_turn_stores_embedding(tmp_path, monkeypatch):
    fake = _inject_fake(monkeypatch)
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.record_chat_turn("My cat loves the windowsill.", "Sweet.")
    assert len(store.episodic) == 1
    emb = store.episodic[0].get("embedding")
    assert emb is not None
    # "cat" was in the input → axis 0 should dominate.
    assert emb[0] > 0.99


def test_record_chat_turn_embedding_optional(tmp_path, monkeypatch):
    """When EmbeddingService.embed returns None (e.g. dep missing),
    record_chat_turn must still succeed and just omit the field."""
    import faceview.llm.embeddings as emb_mod

    class _NoneSvc:
        def available(self):
            return False
        def embed(self, _t):
            return None

    monkeypatch.setattr(
        emb_mod.EmbeddingService, "shared",
        classmethod(lambda cls: _NoneSvc()),
    )
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.record_chat_turn("Hi there", "Hello back")
    assert len(store.episodic) == 1
    assert "embedding" not in store.episodic[0]


def test_recall_by_embedding_returns_relevant(tmp_path, monkeypatch):
    _inject_fake(monkeypatch)
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.record_chat_turn("My dog is a labrador.", "Cute.")
    store.record_chat_turn("I love pizza on Friday.", "Same.")
    store.record_chat_turn("Booking a trip to Lisbon.", "Have fun!")

    pet_q = "Tell me about a cat I once had."
    relevant = store.recall_by_embedding(pet_q, limit=2)
    assert len(relevant) == 1
    assert "dog" in relevant[0]["text"]

    food_q = "What did we eat for dinner?"
    relevant = store.recall_by_embedding(food_q, limit=2)
    assert len(relevant) == 1
    assert "pizza" in relevant[0]["text"]


def test_recall_by_embedding_fallback_when_no_dep(tmp_path, monkeypatch):
    """No embedding service → empty list, no exception."""
    import faceview.llm.embeddings as emb_mod

    class _Down:
        def available(self):
            return False
        def embed(self, _t):
            return None

    monkeypatch.setattr(
        emb_mod.EmbeddingService, "shared",
        classmethod(lambda cls: _Down()),
    )
    store = _store_with_tmp(tmp_path, monkeypatch)
    # Manually populate one episode without embeddings.
    store.record_episode("chat", "Old memory", significance=5)
    assert store.recall_by_embedding("anything") == []


def test_narrate_for_prompt_injects_relevant_section(tmp_path, monkeypatch):
    _inject_fake(monkeypatch)
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.record_chat_turn("My dog is a labrador.", "Cute.")
    store.record_chat_turn("I love pizza on Friday.", "Same.")
    store.record_chat_turn("Booking a trip to Lisbon.", "Have fun!")

    store.set_query_context("Do you remember my pet?")
    out = store.narrate_for_prompt()
    assert "[Relevant past memories" in out
    assert "dog" in out

    # Clearing the context drops the section.
    store.set_query_context(None)
    out = store.narrate_for_prompt()
    assert "[Relevant past memories" not in out


def test_schema_migration_loads_v2_without_embeddings(tmp_path, monkeypatch):
    """Old memory JSON (v2 schema, pre-embedding) should load
    cleanly — no embeddings present, no crash."""
    import json
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    CognitionStore = cog.CognitionStore
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 2,
        "persona": "legacy",
        "first_seen": "2026-01-01",
        "session_count": 7,
        "relationship_score": 14,
        "episodic": [{
            "ts": 1700000000.0,
            "type": "chat",
            "text": "User: hi — You: hello",
            "significance": 3,
            "emotion": "neutral",
            "session_id": 0,
            "recalled": 0,
        }],
        "semantic": {},
        "emotional": {},
    }
    (mem_dir / "legacy.json").write_text(json.dumps(payload))
    store = CognitionStore.load("legacy")
    assert len(store.episodic) == 1
    assert "embedding" not in store.episodic[0]
    # Retrieval against the old entry returns empty (no embedding).
    assert store.recall_by_embedding("anything") == []


def test_query_context_setter_clears_cleanly(tmp_path, monkeypatch):
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.set_query_context("hello")
    assert store._query_context == "hello"
    store.set_query_context(None)
    assert store._query_context is None


def test_relevant_section_dedupes_against_keyword_block(tmp_path, monkeypatch):
    """If the embedding retrieval picks an episode that the keyword-
    based recall would also pick, narrate_for_prompt should not show
    it twice."""
    _inject_fake(monkeypatch)
    store = _store_with_tmp(tmp_path, monkeypatch)
    store.record_chat_turn("My dog Sam is great.", "Aw.")
    store.set_query_context("Tell me about my dog Sam.")
    out = store.narrate_for_prompt()
    # The episode appears in the "[Relevant past memories...]" section;
    # the keyword block (if any) should not duplicate it.
    assert out.count("My dog Sam") == 1
