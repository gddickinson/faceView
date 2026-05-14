"""L9 — cost/latency telemetry."""

from __future__ import annotations

import json


def _reset_recorder(tmp_path, monkeypatch):
    import faceview.config as cfg
    monkeypatch.setattr(cfg.settings, "data_dir", tmp_path)
    import faceview.llm.telemetry as tel
    tel.TelemetryRecorder.reset_for_tests()
    return tel


def test_price_lookup_known_models():
    from faceview.llm.telemetry import _price_for
    assert _price_for("claude-opus-4-7")[0] == 15.0
    assert _price_for("claude-sonnet-4-6") == (3.0, 15.0)
    assert _price_for("claude-haiku-4-5") == (1.0, 5.0)
    # Unknown / local — no cost reported.
    assert _price_for("qwen2.5:14b") == (0.0, 0.0)
    assert _price_for("") == (0.0, 0.0)


def test_estimate_tokens_floor():
    from faceview.llm.telemetry import _estimate_tokens
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("hello world") == 2
    assert _estimate_tokens("a b c d e f g h i j") == 10


def test_record_uses_real_tokens_when_provided(tmp_path, monkeypatch):
    tel = _reset_recorder(tmp_path, monkeypatch)
    rec = tel.TelemetryRecorder.shared().record(
        engine="anthropic",
        model="claude-sonnet-4-6",
        duration_s=1.23,
        prompt_text="hello",
        completion_text="hi",
        prompt_tokens=100,
        completion_tokens=50,
    )
    assert rec.prompt_tokens == 100
    assert rec.completion_tokens == 50
    # cost = (100 * 3 + 50 * 15) / 1e6 = 0.001050
    assert abs(rec.usd_cost - 0.00105) < 1e-7


def test_record_falls_back_to_word_count(tmp_path, monkeypatch):
    tel = _reset_recorder(tmp_path, monkeypatch)
    rec = tel.TelemetryRecorder.shared().record(
        engine="ollama",
        model="qwen2.5:14b",
        duration_s=0.5,
        prompt_text="one two three",
        completion_text="four five",
    )
    assert rec.prompt_tokens == 3
    assert rec.completion_tokens == 2
    assert rec.usd_cost == 0.0


def test_record_persists_jsonl(tmp_path, monkeypatch):
    tel = _reset_recorder(tmp_path, monkeypatch)
    rec_a = tel.TelemetryRecorder.shared().record(
        engine="demo", model="echo", duration_s=0.1,
        prompt_text="a", completion_text="b",
    )
    rec_b = tel.TelemetryRecorder.shared().record(
        engine="demo", model="echo", duration_s=0.2,
        prompt_text="c", completion_text="d",
    )
    path = tmp_path / "telemetry.jsonl"
    assert path.exists()
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["duration_s"] == rec_a.duration_s
    assert parsed[1]["duration_s"] == rec_b.duration_s


def test_lifetime_totals_accumulate(tmp_path, monkeypatch):
    tel = _reset_recorder(tmp_path, monkeypatch)
    rec_one = tel.TelemetryRecorder.shared()
    rec_one.record(engine="anthropic", model="claude-sonnet-4-6",
                   duration_s=1.0, prompt_tokens=200, completion_tokens=100)
    rec_one.record(engine="anthropic", model="claude-sonnet-4-6",
                   duration_s=2.0, prompt_tokens=300, completion_tokens=200)
    life = rec_one.lifetime
    assert life["turns"] == 2
    assert life["duration_s"] == 3.0
    assert life["prompt_tokens"] == 500
    assert life["completion_tokens"] == 300


def test_last_turn_publishes_event(tmp_path, monkeypatch, fresh_bus):
    tel = _reset_recorder(tmp_path, monkeypatch)
    received: list = []
    from faceview.core.events import EventType
    fresh_bus.subscribe(EventType.TURN_RECORDED, received.append)
    tel.TelemetryRecorder.shared().record(
        engine="demo", model="echo", duration_s=0.05,
        prompt_text="hi", completion_text="hello",
    )
    assert len(received) == 1
    assert received[0].engine == "demo"


def test_extract_anthropic_usage_handles_missing():
    from faceview.llm.telemetry import extract_anthropic_usage
    from types import SimpleNamespace
    assert extract_anthropic_usage(None) == (0, 0)
    assert extract_anthropic_usage(SimpleNamespace(usage=None)) == (0, 0)
    msg = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=42, output_tokens=17,
    ))
    assert extract_anthropic_usage(msg) == (42, 17)


def test_extract_ollama_usage():
    from faceview.llm.telemetry import extract_ollama_usage
    assert extract_ollama_usage(None) == (0, 0)
    assert extract_ollama_usage({}) == (0, 0)
    assert extract_ollama_usage({"prompt_eval_count": 100,
                                  "eval_count": 50}) == (100, 50)
