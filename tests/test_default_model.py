"""Default Anthropic model and request shape (no network: the SDK client is faked)."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from faceview.config import Settings
from faceview.llm.conversation import Conversation


def test_default_model_is_current_sonnet(monkeypatch):
    monkeypatch.delenv("FACEVIEW_MODEL", raising=False)
    assert Settings().anthropic_model == "claude-sonnet-5-5"


def test_model_env_override_kept(monkeypatch):
    monkeypatch.setenv("FACEVIEW_MODEL", "claude-opus-4-7")
    assert Settings().anthropic_model == "claude-opus-4-7"


def test_config_dialog_offers_default_model():
    import ast
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src/faceview/gui/config_dialog.py"
    tree = ast.parse(src.read_text())
    models = next(ast.literal_eval(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_ANTHROPIC_MODELS")
    assert "claude-sonnet-5-5" in models


def test_telemetry_prices_default_model():
    from faceview.llm.telemetry import _price_for
    assert _price_for("claude-sonnet-5-5") == (2.0, 10.0)


class _FakeStream:
    text_stream = iter(["Hi."])

    def get_final_message(self):
        return SimpleNamespace(stop_reason="end_turn", content=[],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def test_anthropic_request_leaves_room_for_adaptive_thinking(monkeypatch):
    """Sonnet 5.5 thinks by default and thinking counts toward max_tokens, so a 1024 cap
    could truncate or empty a reply; the request also must not send temperature or
    thinking={'type': 'disabled'}, which Sonnet 5.5 rejects."""
    monkeypatch.setenv("FACEVIEW_VISION_TOOL", "0")
    from faceview.llm.claude_client import AnthropicEngine
    sent = {}

    @contextmanager
    def stream(**kw):
        sent.update(kw)
        yield _FakeStream()

    eng = AnthropicEngine(api_key="test", model="claude-sonnet-5-5")
    eng._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    conv = Conversation()
    conv.add_user("hello")
    assert "".join(eng.stream_reply(conv, "hello")) == "Hi."
    assert sent["max_tokens"] >= 16000
    assert "temperature" not in sent and "thinking" not in sent
