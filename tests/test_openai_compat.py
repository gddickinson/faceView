"""OpenAI ``/v1/chat/completions`` compatibility shim coverage.

Uses FastAPI's TestClient and a fake ClaudeClient so the tests don't
spin up uvicorn or hit a real LLM. The shim's job is wire-format
translation; the real LLM behaviour is exercised elsewhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── fakes ─────────────────────────────────────────────────────────


class _FakeConversation:
    """Drop-in for the live Conversation — just records messages."""

    def __init__(self) -> None:
        self.system = ""
        self._messages: list = []
        self._extras_providers: list = []

    def add_user(self, text: str) -> None:
        self._messages.append(("user", text))

    def add_assistant(self, text: str) -> None:
        self._messages.append(("assistant", text))

    def add_system_extras_provider(self, fn) -> None:
        self._extras_providers.append(fn)

    def messages(self) -> list:
        return list(self._messages)

    def effective_system(self) -> str:
        parts = [p() for p in self._extras_providers]
        return "\n\n".join([self.system, *(p for p in parts if p)]).strip()

    def for_anthropic(self) -> list[dict]:
        return [{"role": r, "content": c} for r, c in self._messages]


class _FakeEngine:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.last_conv = None
        self.last_user = None

    def stream_reply(self, conv, user_text: str):
        self.last_conv = conv
        self.last_user = user_text
        # Yield in chunks so the collector exercise its loop.
        for c in [self.reply[i : i + 5]
                  for i in range(0, len(self.reply), 5)]:
            yield c


class _FakeClient:
    def __init__(self, reply: str = "hello!") -> None:
        self.engine = _FakeEngine(reply)
        self.conversation = _FakeConversation()

    def current_engine(self) -> str:
        return "fake"


class _FakeService:
    def __init__(self, client) -> None:
        self.window = SimpleNamespace(llm_client=client)


# ── fixtures ──────────────────────────────────────────────────────


def _make_client(reply: str = "Hi there"):
    from faceview.server.openai_compat import build_router

    client = _FakeClient(reply=reply)
    service = _FakeService(client)
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app), client


# ── tests ─────────────────────────────────────────────────────────


def test_models_endpoint_lists_faceview_aliases():
    api, _ = _make_client()
    resp = api.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    ids = {m["id"] for m in body["data"]}
    assert "faceview" in ids
    assert "faceview-anthropic" in ids
    assert "faceview-ollama" in ids
    assert body["object"] == "list"


def test_chat_completion_round_trip():
    api, fake = _make_client(reply="The cat sat on the mat.")
    resp = api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello!"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "faceview"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "The cat sat on the mat."
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] >= 1
    assert body["usage"]["completion_tokens"] >= 1
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    # The engine should have seen the last user message.
    assert fake.engine.last_user == "Hello!"


def test_chat_completion_passes_history_to_engine():
    """Multi-turn history should reach the engine in order."""
    api, fake = _make_client(reply="ack")
    api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
    })
    # The shim builds a real faceview Conversation; the engine sees
    # ChatMessage instances with role+content+ts.
    pairs = [(m.role, m.content) for m in fake.engine.last_conv.messages()]
    assert pairs == [
        ("user", "first"),
        ("assistant", "ok"),
        ("user", "second"),
    ]
    assert fake.engine.last_user == "second"


def test_streaming_not_implemented():
    api, _ = _make_client()
    resp = api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert resp.status_code == 501
    assert "streaming" in resp.json()["detail"].lower()


def test_empty_messages_rejected():
    api, _ = _make_client()
    resp = api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [],
    })
    assert resp.status_code == 400


def test_extras_providers_copied_from_live_conversation():
    """Perception + cognition narratives plumbed through the live
    conversation should reach the ephemeral conversation built per
    request — otherwise the OpenAI endpoint loses faceView's
    distinctive value."""
    api, fake = _make_client(reply="ack")
    fake.conversation._extras_providers.append(
        lambda: "LIVE_PERCEPTION_BLOCK",
    )
    api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [{"role": "user", "content": "test"}],
    })
    sys_text = fake.engine.last_conv.effective_system()
    assert "LIVE_PERCEPTION_BLOCK" in sys_text


def test_no_llm_client_returns_503():
    """Service hits boot ordering — fail fast before MainWindow.llm_client
    is bound rather than crashing inside engine."""
    from faceview.server.openai_compat import build_router

    service = _FakeService(client=None)
    # Explicitly clear the llm_client attribute.
    service.window = SimpleNamespace(llm_client=None)
    app = FastAPI()
    app.include_router(build_router(service))
    api = TestClient(app)
    resp = api.post("/v1/chat/completions", json={
        "model": "faceview",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 503
