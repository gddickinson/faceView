"""PR2 + I3 + C9 — consent dial, webhooks, memory forgetting pass."""

from __future__ import annotations


# ── PR2 — consent ───────────────────────────────────────────────


def _fresh_consent(tmp_path, monkeypatch):
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    import faceview.core.consent as c
    monkeypatch.setattr(c.ConsentStore, "_instance", None)
    return c


def test_default_allows_local_tools(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    # Local on-device tools default-allow.
    assert store.is_allowed("read_text") is True
    assert store.is_allowed("describe_pose") is True
    assert store.is_allowed("scan_qr") is True


def test_default_prompts_remote_tools_via_anthropic(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    # Remote-risk tool via Anthropic defaults to NOT allowed
    # (trust_remote off by default).
    assert store.is_allowed("look_at_camera", engine="anthropic") is False
    assert store.is_allowed("look_at_screen", engine="anthropic") is False


def test_ollama_engine_treats_remote_risk_as_local(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    # On Ollama, the same tool stays on-device → allowed.
    assert store.is_allowed("look_at_camera", engine="ollama") is True


def test_trust_remote_unlocks_anthropic_path(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    store.set_trust_remote(True)
    assert store.is_allowed("look_at_camera", engine="anthropic") is True
    assert store.trust_remote() is True


def test_explicit_decision_overrides_default(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    store.set_tool_decision("read_text", "block")
    assert store.is_allowed("read_text") is False
    store.set_tool_decision("read_text", "allow")
    assert store.is_allowed("read_text") is True


def test_consent_persists_to_disk(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    store = c.ConsentStore.shared()
    store.set_tool_decision("scan_qr", "block")
    store.set_trust_remote(True)
    # Reload via a fresh instance.
    import faceview.core.consent as c2
    monkeypatch.setattr(c2.ConsentStore, "_instance", None)
    s2 = c2.ConsentStore.shared()
    assert s2.get_tool_decision("scan_qr") == "block"
    assert s2.trust_remote() is True


def test_refuse_message_includes_tool_name(tmp_path, monkeypatch):
    c = _fresh_consent(tmp_path, monkeypatch)
    msg = c.ConsentStore.shared().refuse_message("read_text")
    assert "read_text" in msg
    assert "Tools" in msg


# ── I3 — webhooks ───────────────────────────────────────────────


def _fresh_webhooks(tmp_path, monkeypatch):
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    import faceview.core.webhooks as wh
    monkeypatch.setattr(wh.WebhookManager, "_instance", None)
    return wh


def test_register_validates_url_scheme(tmp_path, monkeypatch):
    wh = _fresh_webhooks(tmp_path, monkeypatch)
    mgr = wh.WebhookManager.shared()
    import pytest
    with pytest.raises(ValueError):
        mgr.register("not-a-url", ["EMOTION"])


def test_register_filters_unknown_event_names(tmp_path, monkeypatch):
    wh = _fresh_webhooks(tmp_path, monkeypatch)
    mgr = wh.WebhookManager.shared()
    sub = mgr.register("http://example.invalid/hook",
                        ["EMOTION", "fictional_event"])
    # Unknown event silently filtered out.
    assert sub.events == ["EMOTION"]


def test_register_and_unregister(tmp_path, monkeypatch):
    wh = _fresh_webhooks(tmp_path, monkeypatch)
    mgr = wh.WebhookManager.shared()
    sub = mgr.register("http://example.invalid/hook", ["GESTURE"])
    assert len(mgr.list_subs()) == 1
    assert mgr.unregister(sub.id) is True
    assert mgr.list_subs() == []
    # Unregistering twice → False.
    assert mgr.unregister(sub.id) is False


def test_webhook_subs_persist(tmp_path, monkeypatch):
    wh = _fresh_webhooks(tmp_path, monkeypatch)
    mgr = wh.WebhookManager.shared()
    mgr.register("http://example.invalid/a", ["EMOTION"])
    mgr.register("http://example.invalid/b", ["GESTURE", "OBJECTS"])
    # Reload.
    import faceview.core.webhooks as wh2
    monkeypatch.setattr(wh2.WebhookManager, "_instance", None)
    m2 = wh2.WebhookManager.shared()
    urls = sorted(s.url for s in m2.list_subs())
    assert urls == ["http://example.invalid/a",
                    "http://example.invalid/b"]


def test_dispatcher_posts_to_interested_subs_only(
    tmp_path, monkeypatch, fresh_bus,
):
    """Spy on _post — verify only subs that listed the event get
    a delivery attempt."""
    wh = _fresh_webhooks(tmp_path, monkeypatch)

    posted: list = []
    def _spy_post(url, event_name, body):
        posted.append((url, event_name))

    monkeypatch.setattr(wh, "_post", _spy_post)
    # Force the dispatcher thread to run synchronously by replacing
    # threading.Thread with a synchronous shim.
    import threading as _th
    class _SyncThread:
        def __init__(self, target=None, args=(), **kw):
            self._t = target
            self._a = args
        def start(self): self._t(*self._a)
    monkeypatch.setattr(wh.threading, "Thread", _SyncThread)

    mgr = wh.WebhookManager.shared()
    mgr.register("http://example.invalid/emo", ["EMOTION"])
    mgr.register("http://example.invalid/all", [])  # empty → wildcard

    from faceview.core.events import Emotion, EventType, Gesture
    fresh_bus.publish(EventType.EMOTION,
                      Emotion(label="happy", confidence=0.8))
    fresh_bus.publish(EventType.GESTURE,
                      Gesture(label="wave", hand="right",
                              confidence=0.7))

    # EMOTION → both subs (specific + wildcard).
    # GESTURE → wildcard sub only.
    urls = [u for u, _e in posted]
    events = [e for _u, e in posted]
    assert "EMOTION" in events
    assert "GESTURE" in events
    assert urls.count("http://example.invalid/emo") == 1
    assert urls.count("http://example.invalid/all") == 2


# ── C9 — forgetting pass ────────────────────────────────────────


def _store(tmp_path, monkeypatch):
    import faceview.llm.cognition as cog
    monkeypatch.setattr(cog, "data_dir", lambda: tmp_path)
    cog.CognitionStore.set_incognito(False)
    return cog.CognitionStore("p")


def test_forgetting_pass_drops_old_unrehearsed_low_sig(tmp_path, monkeypatch):
    import time as _time
    store = _store(tmp_path, monkeypatch)
    # Inject one old + low-sig + un-rehearsed memory.
    store.episodic.append({
        "ts": _time.time() - 200 * 86400,  # >180 days old
        "type": "chat", "text": "trivial chitchat",
        "significance": 2, "emotion": "neutral",
        "session_id": 0, "recalled": 0,
    })
    # And one that should survive (recent OR rehearsed OR high-sig).
    store.episodic.append({
        "ts": _time.time(),
        "type": "chat", "text": "important recent thing",
        "significance": 2, "emotion": "neutral",
        "session_id": 0, "recalled": 0,
    })
    dropped = store.run_forgetting_pass()
    assert dropped == 1
    assert all("trivial" not in m["text"] for m in store.episodic)


def test_forgetting_pass_promotes_rehearsed_memories(tmp_path, monkeypatch):
    import time as _time
    store = _store(tmp_path, monkeypatch)
    store.episodic.append({
        "ts": _time.time(),
        "type": "chat", "text": "user always asks about coffee",
        "significance": 5, "emotion": "neutral",
        "session_id": 0, "recalled": 7,
    })
    store.run_forgetting_pass()
    # Semantic fact under "self" promoted.
    self_facts = store.all_facts("self")
    assert any("coffee" in str(v) for v in self_facts.values())


def test_forgetting_pass_keeps_significant_memories(tmp_path, monkeypatch):
    import time as _time
    store = _store(tmp_path, monkeypatch)
    store.episodic.append({
        "ts": _time.time() - 250 * 86400,
        "type": "chat", "text": "high-significance milestone",
        "significance": 9, "emotion": "joy",
        "session_id": 0, "recalled": 0,
    })
    dropped = store.run_forgetting_pass()
    assert dropped == 0
    assert any("milestone" in m["text"] for m in store.episodic)


def test_forgetting_pass_sweeps_per_person_buckets(tmp_path, monkeypatch):
    import time as _time
    store = _store(tmp_path, monkeypatch)
    store.set_current_speaker("Alice")
    store.per_person["Alice"] = [{
        "ts": _time.time() - 200 * 86400,
        "type": "chat", "text": "ancient throwaway with Alice",
        "significance": 2, "emotion": "neutral",
        "session_id": 0, "recalled": 0,
    }, {
        "ts": _time.time(),
        "type": "chat", "text": "Alice's recent thing",
        "significance": 3, "emotion": "neutral",
        "session_id": 0, "recalled": 0,
    }]
    store.run_forgetting_pass()
    texts = [m["text"] for m in store.per_person["Alice"]]
    assert "ancient throwaway with Alice" not in texts
    assert "Alice's recent thing" in texts
