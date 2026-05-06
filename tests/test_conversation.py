"""Conversation/history helpers."""

from __future__ import annotations

from faceview.llm.conversation import Conversation


def test_round_trip_for_anthropic():
    conv = Conversation(system="be brief")
    conv.add_user("hi")
    conv.add_assistant("hello")
    conv.add_user("how are you?")

    msgs = conv.for_anthropic()
    assert msgs == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]
    assert len(conv) == 3


def test_jsonable_round_trip():
    conv = Conversation()
    conv.add_user("ping")
    out = conv.to_jsonable()
    assert out["system"]
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][0]["content"] == "ping"


def test_demo_engine_streams_chunks():
    """Without an API key, ``ClaudeClient`` should fall back to EchoEngine."""
    from faceview.llm.claude_client import EchoEngine

    eng = EchoEngine()
    chunks = list(eng.stream_reply(Conversation(), "say hi"))
    assert chunks
    text = "".join(chunks)
    assert "say hi" in text
    assert "demo mode" in text
