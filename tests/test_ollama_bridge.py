"""Ollama LLM fallback bridge."""

from __future__ import annotations

from faceview.llm.ollama_client import (
    DEFAULT_HOST,
    OllamaEngine,
    is_ollama_available,
    list_ollama_models,
    pick_default_model,
)


def test_imports():
    assert DEFAULT_HOST == "http://127.0.0.1:11434"


def test_is_ollama_available_returns_bool():
    """Reachability check should never raise — just return False."""
    assert isinstance(is_ollama_available(timeout=0.1), bool)


def test_list_models_returns_list():
    """List should never raise; returns [] on any failure."""
    models = list_ollama_models(timeout=0.1)
    assert isinstance(models, list)


def test_pick_default_model_returns_str_or_none():
    m = pick_default_model()
    assert m is None or isinstance(m, str)


def test_engine_constructor():
    """OllamaEngine init shouldn't make network calls."""
    eng = OllamaEngine(model="llama3:8b")
    assert eng.model == "llama3:8b"
    assert eng.host == DEFAULT_HOST


def test_chained_fallback_when_no_anthropic_key(monkeypatch, fresh_bus):
    """Without ANTHROPIC_API_KEY + no Ollama, should land on EchoEngine."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import faceview.config as config
    monkeypatch.setattr(config, "settings",
                         type(config.settings)(anthropic_api_key=None))
    import faceview.llm.ollama_client as oc
    monkeypatch.setattr(oc, "is_ollama_available", lambda *_a, **_k: False)
    from faceview.llm.claude_client import ClaudeClient, EchoEngine
    import faceview.llm.claude_client as cc
    monkeypatch.setattr(cc, "settings", config.settings)
    client = ClaudeClient()
    assert isinstance(client.engine, EchoEngine)
    client.stop()


def test_chained_fallback_picks_ollama_when_available(monkeypatch, fresh_bus):
    """If Ollama is reachable + has a model, ClaudeClient picks OllamaEngine."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import faceview.config as config
    monkeypatch.setattr(config, "settings",
                         type(config.settings)(anthropic_api_key=None))
    import faceview.llm.claude_client as cc
    monkeypatch.setattr(cc, "settings", config.settings)
    import faceview.llm.ollama_client as oc
    monkeypatch.setattr(oc, "is_ollama_available", lambda *_a, **_k: True)
    monkeypatch.setattr(oc, "pick_default_model", lambda *_a, **_k: "llama3:8b")
    from faceview.llm.claude_client import ClaudeClient
    from faceview.llm.ollama_client import OllamaEngine
    client = ClaudeClient()
    assert isinstance(client.engine, OllamaEngine)
    assert client.engine.model == "llama3:8b"
    client.stop()
