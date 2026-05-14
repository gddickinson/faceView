"""I4 + C8 + P6 — plugin system, emotion feedback loop, multi-face identity."""

from __future__ import annotations


# ── I4 — plugin registry ────────────────────────────────────────


def test_register_and_lookup_plugin_tool():
    from faceview.llm.plugins import (
        clear_registry, register_tool, get_plugin_tool,
        list_plugin_tools,
    )
    clear_registry()

    def _exec(args: dict) -> str:
        return f"hi {args.get('name', 'world')}"

    register_tool(
        name="greet",
        description="say hello",
        schema={"type": "object", "properties": {}},
        executor=_exec,
    )
    t = get_plugin_tool("greet")
    assert t is not None
    assert t.description == "say hello"
    assert t.executor({"name": "Alice"}) == "hi Alice"
    assert any(p.name == "greet" for p in list_plugin_tools())
    clear_registry()


def test_register_validates_name_and_executor():
    from faceview.llm.plugins import clear_registry, register_tool
    import pytest
    clear_registry()
    with pytest.raises(ValueError):
        register_tool("", "x", {}, lambda a: "")
    with pytest.raises(ValueError):
        register_tool("ok", "x", {}, "not_callable")


def test_anthropic_and_ollama_dicts_shape():
    from faceview.llm.plugins import (
        clear_registry, register_tool,
        anthropic_tool_dicts, ollama_tool_dicts,
    )
    clear_registry()
    register_tool("ping", "respond pong",
                  {"type": "object", "properties": {}},
                  lambda a: "pong")
    a = anthropic_tool_dicts()[0]
    o = ollama_tool_dicts()[0]
    assert a["name"] == "ping"
    assert a["input_schema"]["type"] == "object"
    assert o["type"] == "function"
    assert o["function"]["name"] == "ping"
    clear_registry()


def test_run_plugin_tool_unknown_name():
    from faceview.llm.plugins import clear_registry, run_plugin_tool
    clear_registry()
    msg = run_plugin_tool("nope", {})
    assert "Unknown plugin tool" in msg


def test_run_plugin_tool_catches_executor_exceptions():
    from faceview.llm.plugins import (
        clear_registry, register_tool, run_plugin_tool,
    )
    clear_registry()

    def _boom(_a):
        raise RuntimeError("intentional")

    register_tool("boom", "explode", {"type": "object"}, _boom)
    msg = run_plugin_tool("boom", {})
    assert "raised" in msg or "intentional" in msg
    clear_registry()


def test_discover_loads_plugin_files(tmp_path, monkeypatch):
    """End-to-end: write a plugin file under data_dir/plugins, run
    discover, the registered tool shows up."""
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    pdir = tmp_path / "plugins"
    pdir.mkdir(parents=True)
    (pdir / "hello.py").write_text(
        "from faceview.llm.plugins import register_tool\n"
        "register_tool('hello_plugin', 'say hi',\n"
        "              {'type':'object','properties':{}},\n"
        "              lambda a: 'hi from plugin')\n"
    )
    from faceview.llm.plugins import (
        discover_and_load_plugins, get_plugin_tool, clear_registry,
    )
    clear_registry()
    n = discover_and_load_plugins()
    assert n >= 1
    t = get_plugin_tool("hello_plugin")
    assert t is not None
    assert t.executor({}) == "hi from plugin"
    clear_registry()


# ── C8 — emotion feedback loop ─────────────────────────────────


class _StubMemory:
    def __init__(self):
        self.bumps: list = []
    def set_emotion(self, label, intensity, trigger=""):
        self.bumps.append((label, intensity, trigger))


class _StubClient:
    def __init__(self, mem):
        self.memory = mem


def test_emotion_feedback_skips_neutral(fresh_bus, monkeypatch):
    import faceview.vision.emotion_feedback as ef
    ef.EmotionFeedback.reset_for_tests()
    fb = ef.EmotionFeedback.shared()
    mem = _StubMemory()
    fb.attach(_StubClient(mem))
    from faceview.core.events import Emotion, EventType
    for _ in range(10):
        fresh_bus.publish(EventType.EMOTION,
                          Emotion(label="neutral", confidence=0.9))
    # No mapping for neutral → no bump.
    assert mem.bumps == []


def test_emotion_feedback_bumps_persona_on_sustained_sad(
    fresh_bus, monkeypatch,
):
    import faceview.vision.emotion_feedback as ef
    ef.EmotionFeedback.reset_for_tests()
    fb = ef.EmotionFeedback.shared()
    mem = _StubMemory()
    fb.attach(_StubClient(mem))
    from faceview.core.events import Emotion, EventType
    # Strong sustained sad readings.
    for _ in range(5):
        fresh_bus.publish(EventType.EMOTION,
                          Emotion(label="sad", confidence=0.8))
    # Should have triggered one tenderness bump.
    assert len(mem.bumps) == 1
    label, _intensity, trigger = mem.bumps[0]
    assert label == "tenderness"
    assert "sad" in trigger


def test_emotion_feedback_cooldown_prevents_spam(
    fresh_bus, monkeypatch,
):
    import faceview.vision.emotion_feedback as ef
    ef.EmotionFeedback.reset_for_tests()
    fb = ef.EmotionFeedback.shared()
    mem = _StubMemory()
    fb.attach(_StubClient(mem))
    from faceview.core.events import Emotion, EventType
    # Two waves of strong-sad readings; only the first triggers a bump.
    for _ in range(5):
        fresh_bus.publish(EventType.EMOTION,
                          Emotion(label="sad", confidence=0.8))
    for _ in range(5):
        fresh_bus.publish(EventType.EMOTION,
                          Emotion(label="sad", confidence=0.8))
    assert len(mem.bumps) == 1


def test_emotion_feedback_low_confidence_no_bump(fresh_bus):
    import faceview.vision.emotion_feedback as ef
    ef.EmotionFeedback.reset_for_tests()
    fb = ef.EmotionFeedback.shared()
    mem = _StubMemory()
    fb.attach(_StubClient(mem))
    from faceview.core.events import Emotion, EventType
    # Sustained "sad" but confidence below threshold.
    for _ in range(8):
        fresh_bus.publish(EventType.EMOTION,
                          Emotion(label="sad", confidence=0.40))
    assert mem.bumps == []


# ── P6 — multi-face identity (event payload shape) ──────────────


def test_identities_multi_payload_shape():
    from faceview.core.events import IdentitiesMulti, IdentityHit
    m = IdentitiesMulti(hits=[
        IdentityHit(is_owner=True, similarity=0.9, label="owner",
                    bbox=(10, 20, 100, 100)),
        IdentityHit(is_owner=False, similarity=0.5, label="stranger",
                    bbox=(200, 30, 80, 80)),
    ])
    assert len(m.hits) == 2
    assert m.hits[0].is_owner is True
    assert m.hits[1].label == "stranger"


def test_identity_recognizer_publishes_both_events(fresh_bus, monkeypatch):
    """Stub out InsightFace's FaceAnalysis and PeopleStore so we can
    verify the recogniser emits IDENTITY (single) + IDENTITIES_MULTI."""
    from faceview.core.events import EventType
    from faceview.vision.identity import IdentityRecognizer
    import faceview.vision.people as people_mod
    import time as _time
    import numpy as np

    # Stub PeopleStore.match.
    class _StubPeople:
        def __init__(self): self.threshold = 0.42
        def set_embed_fn(self, fn): pass
        def count(self): return 1
        def match(self, emb):
            return ("owner" if emb[0] > 0.5 else "stranger",
                    0.7 if emb[0] > 0.5 else 0.2,
                    bool(emb[0] > 0.5))

    monkeypatch.setattr(people_mod.PeopleStore, "shared",
                        classmethod(lambda cls: _StubPeople()))

    # Stub the InsightFace app's .get() output.
    class _Face:
        def __init__(self, x, y, w, h, ax):
            self.bbox = (x, y, x + w, y + h)
            self.normed_embedding = np.zeros(512, dtype=np.float32)
            self.normed_embedding[0] = ax  # >0.5 → owner per stub

    rec = IdentityRecognizer()
    rec._app = type("App", (), {
        "get": lambda self, frame: [
            _Face(10, 10, 200, 200, 0.9),     # owner (largest)
            _Face(300, 30, 80, 80, 0.1),       # stranger
        ],
    })()
    rec._people = _StubPeople()
    rec._last_emit = 0.0

    received_multi: list = []
    received_single: list = []
    fresh_bus.subscribe(EventType.IDENTITIES_MULTI,
                        received_multi.append)
    fresh_bus.subscribe(EventType.IDENTITY, received_single.append)

    rec._on_frame(np.zeros((480, 640, 3), dtype=np.uint8))

    assert len(received_multi) == 1
    assert len(received_multi[0].hits) == 2
    # Single-face IDENTITY reports the largest (the owner one).
    assert received_single[-1].label == "owner"
    assert received_multi[0].hits[0].label == "owner"
    assert received_multi[0].hits[1].label == "stranger"
