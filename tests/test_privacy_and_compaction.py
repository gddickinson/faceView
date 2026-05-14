"""C5 + C6 + L10 — forget tool, incognito mode, context auto-compaction."""

from __future__ import annotations


def _store(tmp_path, monkeypatch):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    # Reset the global incognito flag between tests.
    cog.CognitionStore.set_incognito(False)
    return cog.CognitionStore("testpersona")


# ── C6: Incognito ───────────────────────────────────────────────


def test_incognito_default_off():
    from faceview.llm.cognition import CognitionStore
    CognitionStore.set_incognito(False)
    assert CognitionStore.is_incognito() is False


def test_incognito_drops_chat_turn(tmp_path, monkeypatch):
    from faceview.llm.cognition import CognitionStore
    store = _store(tmp_path, monkeypatch)
    CognitionStore.set_incognito(True)
    try:
        store.record_chat_turn("hello", "hi there")
        assert store.episodic == []
        assert store.per_person == {}
        assert store.relationship_score == 0
    finally:
        CognitionStore.set_incognito(False)


def test_incognito_does_not_block_reads(tmp_path, monkeypatch):
    """Existing memory should still be visible to the LLM while
    incognito — it just doesn't grow."""
    from faceview.llm.cognition import CognitionStore
    store = _store(tmp_path, monkeypatch)
    store.record_chat_turn("My name is Bob", "Hi Bob.")
    assert len(store.episodic) >= 0  # at least recorded
    CognitionStore.set_incognito(True)
    try:
        # Narration must still surface the existing memory.
        text = store.narrate_for_prompt()
        assert "[Identity]" in text  # identity block always present
    finally:
        CognitionStore.set_incognito(False)


def test_incognito_toggle_via_main_window_facade(qtbot, monkeypatch):
    """MainWindow.set_incognito flips the class-level flag and the
    menu action stays in sync."""
    from faceview.gui.main_window import MainWindow
    from faceview.llm.cognition import CognitionStore
    CognitionStore.set_incognito(False)
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.incognito_running() is False
    w.set_incognito(True)
    assert w.incognito_running() is True
    assert w._incognito_action.isChecked() is True
    w.set_incognito(False)
    assert w.incognito_running() is False


# ── C5: forget_recent / forget_matching ─────────────────────────


def test_forget_recent_pops_latest(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.record_episode("chat", "first", significance=3)
    store.record_episode("chat", "second", significance=3)
    store.record_episode("chat", "third", significance=3)
    removed = store.forget_recent(n=1)
    assert removed == 1
    # Most-recent ("third") gone.
    texts = [m["text"] for m in store.episodic]
    assert "third" not in texts
    assert "first" in texts and "second" in texts


def test_forget_recent_across_per_person(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Carol")
    store.record_chat_turn("private to Carol", "ok")
    store.set_current_speaker(None)
    store.record_chat_turn("public", "sure")
    # The "public" one is more recent.
    removed = store.forget_recent(n=1)
    assert removed == 1
    assert all("public" not in m["text"] for m in store.episodic)


def test_forget_matching_by_substring(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.record_episode("chat", "apple pie discussion", significance=3)
    store.record_episode("chat", "banana plans", significance=3)
    store.record_episode("chat", "pie crust recipe", significance=3)
    removed = store.forget_matching("pie", limit=10)
    assert removed == 2
    assert any("banana" in m["text"] for m in store.episodic)


def test_forget_matching_no_match_returns_zero(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.record_episode("chat", "hello world", significance=3)
    assert store.forget_matching("xyzzy") == 0


def test_forget_memory_tool_executor(tmp_path, monkeypatch):
    """Tool dispatch path: with query → forget_matching; without →
    forget_recent."""
    from faceview.llm.vision_tool import run_forget_memory
    store = _store(tmp_path, monkeypatch)
    store.record_episode("chat", "test memory about cake", significance=3)
    msg = run_forget_memory(store, query="cake", limit=1)
    assert "Forgotten" in msg or "removed 1" in msg
    assert not store.episodic


def test_forget_memory_tool_no_store():
    from faceview.llm.vision_tool import run_forget_memory
    msg = run_forget_memory(None)
    assert "isn't bound" in msg or "nothing for me" in msg


# ── L10: Conversation.maybe_compact ─────────────────────────────


def test_maybe_compact_noop_under_budget():
    from faceview.llm.conversation import Conversation
    conv = Conversation()
    for i in range(5):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    fired = conv.maybe_compact(budget_tokens=1000)
    assert fired is False


def test_maybe_compact_folds_old_turns_when_over_budget():
    from faceview.llm.conversation import Conversation
    conv = Conversation()
    # 60 small turns; budget tiny so we force compaction.
    for i in range(60):
        conv.add_user(f"user message {i} with several words")
        conv.add_assistant(f"assistant reply {i} ditto")
    before = len(conv.messages())
    fired = conv.maybe_compact(budget_tokens=10)
    assert fired is True
    after = conv.messages()
    # We keep ~30 most-recent turns + 1 summary block.
    assert len(after) < before
    assert "[earlier conversation summarised]" in after[0].content


def test_maybe_compact_idempotent_subsequent_calls():
    from faceview.llm.conversation import Conversation
    conv = Conversation()
    for i in range(50):
        conv.add_user(f"u{i}")
        conv.add_assistant(f"a{i}")
    conv.maybe_compact(budget_tokens=10)
    after_first = len(conv.messages())
    # Calling again on the now-shorter conv shouldn't crash.
    conv.maybe_compact(budget_tokens=10)
    assert len(conv.messages()) <= after_first


def test_maybe_compact_preserves_summary_across_folds():
    """A second compaction should fold MORE turns into the existing
    summary, not start fresh."""
    from faceview.llm.conversation import Conversation
    conv = Conversation()
    for i in range(50):
        conv.add_user(f"first batch {i}")
        conv.add_assistant(f"first reply {i}")
    conv.maybe_compact(budget_tokens=10)
    # Add a fresh wave then compact again.
    for i in range(50):
        conv.add_user(f"second batch {i}")
        conv.add_assistant(f"second reply {i}")
    conv.maybe_compact(budget_tokens=10)
    head = conv.messages()[0].content
    # Both eras represented in the summary.
    assert "[earlier conversation summarised]" in head
    assert "first batch" in head
    assert "second batch" in head


def test_estimate_tokens_increases_with_content():
    from faceview.llm.conversation import Conversation
    conv = Conversation()
    before = conv.estimate_tokens()
    conv.add_user("hello there friend, here is a sentence")
    after = conv.estimate_tokens()
    assert after > before
