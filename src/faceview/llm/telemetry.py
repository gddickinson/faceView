"""Per-turn cost + latency telemetry.

Records one JSONL line per LLM turn to ``~/.faceview/telemetry.jsonl``
and publishes a :data:`EventType.TURN_RECORDED` event so the status
bar can show "last turn: 0.8s · 240 tok · $0.003".

Anthropic engines return real token usage via the SDK's
``stream.get_final_message().usage``. Ollama's ``/api/chat`` returns
``prompt_eval_count`` + ``eval_count`` in the final streamed chunk.
Both flavours store the counts on ``engine.last_usage`` for the
recorder to pick up. If neither is available (demo engine, or a
custom engine that doesn't bother) we estimate from word counts.

Costs are USD only for Anthropic models. Local engines are recorded
as $0.00 but their latency still goes through so users can compare
*time*-cost of e.g. qwen2.5:14b vs qwen2.5:7b empirically.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

from faceview.config import settings
from faceview.core.event_bus import get_bus
from faceview.core.events import EventType, TurnRecord
from faceview.core.logger import get_logger


log = get_logger("telemetry")


# Per-million-token USD prices — update as the Anthropic catalogue
# moves. Keyed by substring so we match "claude-opus-4-7" and
# "claude-opus-4-7-20260101" identically. Defaults to (0, 0) for
# unknown models (no false cost numbers).
_PRICES_PER_MILLION: list[tuple[str, float, float]] = [
    ("opus-4-7",        15.0, 75.0),
    ("opus-4-6",        15.0, 75.0),
    ("opus-4",          15.0, 75.0),
    ("sonnet-4-6",       3.0, 15.0),
    ("sonnet-4-5",       3.0, 15.0),
    ("sonnet-4",         3.0, 15.0),
    ("sonnet-3-5",       3.0, 15.0),
    ("haiku-4-5",        1.0,  5.0),
    ("haiku-3-5",        0.8,  4.0),
    ("haiku-3",          0.25, 1.25),
]


def _price_for(model: str) -> tuple[float, float]:
    name = (model or "").lower()
    for needle, inp, out in _PRICES_PER_MILLION:
        if needle in name:
            return inp, out
    return 0.0, 0.0


def _estimate_tokens(text: str) -> int:
    """Cheap whitespace word-count proxy when real counts aren't
    available. English ~0.75 words/token; we just use word count."""
    return max(1, len((text or "").split()))


# ── recorder singleton ────────────────────────────────────────────


class TelemetryRecorder:
    _instance: "TelemetryRecorder | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "TelemetryRecorder":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = TelemetryRecorder()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last: Optional[TurnRecord] = None
        # Lifetime totals so the status bar can show a running tally.
        self.lifetime = {
            "turns": 0,
            "duration_s": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "usd_cost": 0.0,
        }

    # ── public API ────────────────────────────────────────────

    def last_turn(self) -> Optional[TurnRecord]:
        return self._last

    def record(
        self,
        *,
        engine: str,
        model: str,
        duration_s: float,
        prompt_text: str = "",
        completion_text: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> TurnRecord:
        """Build a TurnRecord, persist to JSONL, publish, return it.

        ``prompt_tokens`` / ``completion_tokens`` win when non-zero;
        otherwise we estimate from the text counts."""
        if prompt_tokens <= 0:
            prompt_tokens = _estimate_tokens(prompt_text)
        if completion_tokens <= 0:
            completion_tokens = _estimate_tokens(completion_text)
        in_per_m, out_per_m = _price_for(model)
        usd = (prompt_tokens * in_per_m
               + completion_tokens * out_per_m) / 1_000_000.0
        rec = TurnRecord(
            engine=engine,
            model=model,
            duration_s=round(duration_s, 3),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            usd_cost=round(usd, 6),
        )
        with self._lock:
            self._last = rec
            self.lifetime["turns"] += 1
            self.lifetime["duration_s"] += rec.duration_s
            self.lifetime["prompt_tokens"] += rec.prompt_tokens
            self.lifetime["completion_tokens"] += rec.completion_tokens
            self.lifetime["usd_cost"] += rec.usd_cost
        self._persist(rec)
        try:
            get_bus().publish(EventType.TURN_RECORDED, rec)
        except Exception:  # noqa: BLE001
            pass
        return rec

    # ── internals ─────────────────────────────────────────────

    def _persist(self, rec: TurnRecord) -> None:
        """Append one JSONL line. Best-effort — never raise."""
        try:
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({
                "ts": rec.ts,
                "engine": rec.engine,
                "model": rec.model,
                "duration_s": rec.duration_s,
                "prompt_tokens": rec.prompt_tokens,
                "completion_tokens": rec.completion_tokens,
                "usd_cost": rec.usd_cost,
            })
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            log.warning("telemetry.write_failed", error=str(exc))

    def _path(self) -> Path:
        return settings.data_dir / "telemetry.jsonl"


# ── helpers used by engines to surface real counts ────────────────


def extract_anthropic_usage(final_message) -> tuple[int, int]:
    """Pull ``(input_tokens, output_tokens)`` from an Anthropic SDK
    Message object. Returns ``(0, 0)`` on any failure so the
    estimator kicks in."""
    try:
        usage = getattr(final_message, "usage", None)
        if usage is None:
            return 0, 0
        return (int(getattr(usage, "input_tokens", 0) or 0),
                int(getattr(usage, "output_tokens", 0) or 0))
    except Exception:  # noqa: BLE001
        return 0, 0


def extract_ollama_usage(final_chunk: dict) -> tuple[int, int]:
    """Pull token counts out of Ollama's final streaming chunk."""
    if not isinstance(final_chunk, dict):
        return 0, 0
    return (int(final_chunk.get("prompt_eval_count") or 0),
            int(final_chunk.get("eval_count") or 0))
