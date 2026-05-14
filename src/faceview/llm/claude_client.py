"""Anthropic Claude client with a demo-mode fallback.

If ``ANTHROPIC_API_KEY`` is set, real streaming responses are emitted to the
event bus. If not, an ``EchoEngine`` returns a friendly stubbed reply so the
GUI is fully usable without a key.

The send loop runs on a single background thread per client (kept alive
between messages) — the Anthropic SDK's streaming context manager is
synchronous, so a thread is the right shape.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

from faceview.config import settings
from faceview.core.event_bus import get_bus
from faceview.core.events import ChatMessage, EventType
from faceview.core.logger import get_logger
from faceview.llm.conversation import Conversation


log = get_logger("claude")


# ── Demo engine ──────────────────────────────────────────────────────────


class EchoEngine:
    """Fallback used when no API key is set; produces a deterministic reply."""

    def stream_reply(self, conv: Conversation, user_text: str):
        # Simulate streaming so the UI animation looks the same as real Claude.
        reply = (
            f"(demo mode — set $ANTHROPIC_API_KEY for real Claude)\n"
            f"You said: {user_text}"
        )
        for chunk in _chunked(reply, 12):
            yield chunk
            time.sleep(0.02)


def _chunked(s: str, n: int):
    for i in range(0, len(s), n):
        yield s[i : i + n]


# ── Real Anthropic engine (lazy import) ──────────────────────────────────


class AnthropicEngine:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None  # lazy
        # Set by stream_reply after each final response so the
        # telemetry recorder can read real token counts. None when no
        # call has been made yet.
        self.last_usage: tuple[int, int] | None = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic  # type: ignore
            except ImportError as exc:  # pragma: no cover
                from faceview.core.errors import MissingDependency
                raise MissingDependency("anthropic", "dev") from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def stream_reply(self, conv: Conversation, user_text: str):
        client = self._ensure_client()
        # Re-read from settings on each call so the config dialog can
        # switch models live without restarting the client.
        from faceview.config import settings as _s
        active_model = _s.anthropic_model or self.model

        # Optional vision tools — Claude can call any of these to
        # gather extra information about the camera scene. See
        # llm/vision_tool.py for the full catalogue.
        from faceview.llm.vision_tool import (
            FrameGrabber, vision_tool_enabled,
            TIER1_TOOLS_ANTHROPIC, TIER23_TOOLS_ANTHROPIC,
            run_look_anthropic, run_remember_person,
            run_read_text, run_track_object, run_check_visible,
            run_describe_color, run_describe_pose, run_face_attributes,
            run_scan_qr, run_estimate_depth, run_gaze_target,
            run_segment_object, run_describe_room_layout,
            run_forget_memory, run_look_at_screen_anthropic,
        )
        use_tools = vision_tool_enabled()
        # I4 — plugin tools registered via .faceview/plugins/*.py
        # auto-mix into the catalogue.
        try:
            from faceview.llm.plugins import anthropic_tool_dicts
            plugin_tools = anthropic_tool_dicts()
        except Exception:  # noqa: BLE001
            plugin_tools = []
        tools_arg: list[dict] = (
            list(TIER1_TOOLS_ANTHROPIC)
            + list(TIER23_TOOLS_ANTHROPIC)
            + list(plugin_tools)
            if use_tools else []
        )
        grabber = FrameGrabber.shared() if use_tools else None

        # We mutate a local messages list so any tool-use round-trip
        # lives only in this turn; Conversation only sees the final
        # text reply via add_assistant() in the consumer.
        messages: list = list(conv.for_anthropic())
        system = conv.effective_system()

        log.info("anthropic.send", model=active_model,
                 messages=len(messages), tools=len(tools_arg),
                 system_preview=system[:400])
        # Cap on tool loops so a misbehaving model can't burn the API.
        for _step in range(4):
            kwargs: dict = {
                "model": active_model,
                "max_tokens": 1024,
                "system": system,
                "messages": messages,
            }
            if tools_arg:
                kwargs["tools"] = tools_arg
            with client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            log.info("anthropic.stream_done", step=_step,
                     stop_reason=str(final.stop_reason))
            # Capture token usage from the last response so the
            # telemetry recorder can pick it up after the loop.
            try:
                from faceview.llm.telemetry import extract_anthropic_usage
                self.last_usage = extract_anthropic_usage(final)
            except Exception:  # noqa: BLE001
                self.last_usage = None
            if final.stop_reason != "tool_use":
                return
            # Append the assistant turn (text + tool_use blocks) and
            # the user-side tool_result messages, then re-stream.
            assistant_blocks: list = []
            tool_results: list = []
            for block in final.content:
                blk = block.model_dump() if hasattr(block, "model_dump") \
                    else dict(block)
                assistant_blocks.append(blk)
                if blk.get("type") != "tool_use":
                    continue
                tu_id = blk.get("id", "")
                name = blk.get("name", "")
                log.info("anthropic.tool_use", tool=name,
                         input=blk.get("input") or {})
                if name == "look_at_camera" and grabber is not None:
                    inp = blk.get("input") or {}
                    content = run_look_anthropic(
                        grabber,
                        question=str(inp.get("question") or ""),
                        region=str(inp.get("region") or "full"),
                    )
                elif name == "remember_person" and grabber is not None:
                    arg_name = (blk.get("input") or {}).get("name", "")
                    msg = run_remember_person(grabber, arg_name)
                    content = [{"type": "text", "text": msg}]
                    log.info("anthropic.tool_result",
                             tool=name, msg=msg[:120])
                elif name == "read_text" and grabber is not None:
                    inp = blk.get("input") or {}
                    msg = run_read_text(
                        grabber, region=str(inp.get("region") or "full"),
                    )
                    content = [{"type": "text", "text": msg}]
                    log.info("anthropic.tool_result",
                             tool=name, msg=msg[:120])
                elif name == "track_object":
                    inp = blk.get("input") or {}
                    msg = run_track_object(
                        label=str(inp.get("label") or ""),
                        duration_s=float(inp.get("duration_s") or 10),
                    )
                    content = [{"type": "text", "text": msg}]
                    log.info("anthropic.tool_result",
                             tool=name, msg=msg[:120])
                elif name == "check_visible" and grabber is not None:
                    inp = blk.get("input") or {}
                    msg = run_check_visible(
                        grabber,
                        query=str(inp.get("query") or ""),
                        region=str(inp.get("region") or "full"),
                    )
                    content = [{"type": "text", "text": msg}]
                    log.info("anthropic.tool_result",
                             tool=name, msg=msg[:120])
                elif name == "describe_color" and grabber is not None:
                    inp = blk.get("input") or {}
                    msg = run_describe_color(
                        grabber,
                        region=str(inp.get("region") or "full"),
                    )
                    content = [{"type": "text", "text": msg}]
                elif name == "describe_pose" and grabber is not None:
                    msg = run_describe_pose(grabber)
                    content = [{"type": "text", "text": msg}]
                elif name == "face_attributes" and grabber is not None:
                    msg = run_face_attributes(grabber)
                    content = [{"type": "text", "text": msg}]
                elif name == "scan_qr" and grabber is not None:
                    msg = run_scan_qr(grabber)
                    content = [{"type": "text", "text": msg}]
                elif name == "estimate_depth" and grabber is not None:
                    inp = blk.get("input") or {}
                    msg = run_estimate_depth(
                        grabber,
                        region=str(inp.get("region") or "full"),
                    )
                    content = [{"type": "text", "text": msg}]
                elif name == "gaze_target":
                    msg = run_gaze_target()
                    content = [{"type": "text", "text": msg}]
                elif name == "segment_object" and grabber is not None:
                    inp = blk.get("input") or {}
                    msg = run_segment_object(
                        grabber, label=str(inp.get("label") or ""),
                    )
                    content = [{"type": "text", "text": msg}]
                elif name == "describe_room_layout":
                    msg = run_describe_room_layout()
                    content = [{"type": "text", "text": msg}]
                elif name == "forget_memory":
                    inp = blk.get("input") or {}
                    msg = run_forget_memory(
                        self.memory,
                        query=str(inp.get("query") or ""),
                        limit=int(inp.get("limit") or 1),
                    )
                    content = [{"type": "text", "text": msg}]
                elif name == "look_at_screen":
                    inp = blk.get("input") or {}
                    content = run_look_at_screen_anthropic(
                        question=str(inp.get("question") or ""),
                        region=str(inp.get("region") or "full"),
                    )
                else:
                    # Try the plugin registry (I4) before giving up.
                    try:
                        from faceview.llm.plugins import (
                            get_plugin_tool, run_plugin_tool,
                        )
                        plugin = get_plugin_tool(name)
                    except Exception:  # noqa: BLE001
                        plugin = None
                    if plugin is not None:
                        msg = run_plugin_tool(
                            name, blk.get("input") or {},
                        )
                        content = [{"type": "text", "text": msg}]
                        log.info("anthropic.plugin_dispatched", tool=name)
                    else:
                        content = [{"type": "text",
                                    "text": f"Unknown tool: {name}"}]
                        log.warning("anthropic.unknown_tool", tool=name)
                log.info("anthropic.tool_dispatched", tool=name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": content,
                })
            messages.append({"role": "assistant",
                             "content": assistant_blocks})
            messages.append({"role": "user", "content": tool_results})
        # Hit the cap without resolving — bail out gracefully.
        yield "\n(I tried to use a tool too many times. Stopping.)"


# ── Client facade ────────────────────────────────────────────────────────


class ClaudeClient:
    """Thread-backed client; emits LLM_TOKEN tokens then LLM_REPLY on done."""

    def __init__(
        self,
        conversation: Optional[Conversation] = None,
        engine=None,
    ) -> None:
        self.bus = get_bus()
        self.conversation = conversation or Conversation()
        self._engine_name = "custom" if engine is not None else ""
        # Memory store (persistent per-persona). Bound after construction
        # via :meth:`bind_memory`. Until then, the system prompt is just
        # the default + whatever live providers (perception) are added.
        self.memory = None
        # We track our own memory provider so swapping persona doesn't
        # clobber unrelated extras providers (e.g. PerceptionStore).
        self._memory_provider = None
        if engine is not None:
            self.engine = engine
        else:
            self.select_engine("auto")

        self._q: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="claude-worker", daemon=True)
        self._thread.start()

    # ── memory wiring ────────────────────────────────────────────────

    def bind_memory(self, store) -> None:
        """Attach a :class:`MemoryStore`. Its narrate_for_prompt() output
        will be appended to the system extras on every turn, regardless
        of which engine (Anthropic / Ollama / Demo) is active.

        Other extras providers (e.g. live perception added in app.py
        before bind_memory is called) are preserved across persona
        swaps — we only swap our own memory slot."""
        if self._memory_provider is not None:
            self.conversation.remove_system_extras_provider(
                self._memory_provider
            )
            self._memory_provider = None
        self.memory = store
        # Stash on the conversation so engines can reach it without
        # the engine knowing about ClaudeClient (used by the Ollama
        # forget_memory dispatch path).
        try:
            setattr(self.conversation, "_bound_memory", store)
        except Exception:  # noqa: BLE001
            pass
        if store is not None:
            self._memory_provider = store.narrate_for_prompt
            self.conversation.add_system_extras_provider(
                self._memory_provider
            )

    # ── live engine selection ────────────────────────────────────────

    def current_engine(self) -> str:
        return self._engine_name or "demo"

    def select_engine(self, name: str, *, model: Optional[str] = None) -> str:
        """Swap the active LLM engine. Returns the engine actually selected.

        ``name`` is ``"auto"``, ``"anthropic"``, ``"ollama"``, or ``"demo"``.
        ``"auto"`` picks anthropic when a key is set, otherwise ollama if a
        local server is reachable, otherwise demo. Falls back to demo if a
        requested engine is unavailable (e.g. anthropic without a key).
        """
        requested = (name or "auto").lower()
        actual = requested
        if requested == "auto":
            if settings.has_claude_key:
                actual = "anthropic"
            else:
                try:
                    from faceview.llm.ollama_client import is_ollama_available
                    actual = "ollama" if is_ollama_available() else "demo"
                except Exception:  # noqa: BLE001
                    actual = "demo"

        if actual == "anthropic":
            if not settings.has_claude_key:
                actual = "demo"
            else:
                self.engine = AnthropicEngine(
                    api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
                    model=model or settings.anthropic_model,
                )
                log.info("llm.engine", engine="anthropic",
                         model=settings.anthropic_model)

        if actual == "ollama":
            try:
                from faceview.llm.ollama_client import OllamaEngine, pick_default_model
                chosen = model or pick_default_model()
            except Exception:  # noqa: BLE001
                chosen = None
            if not chosen:
                actual = "demo"
            else:
                self.engine = OllamaEngine(model=chosen)
                log.info("llm.engine", engine="ollama", model=chosen)

        if actual == "demo":
            self.engine = EchoEngine()
            log.info("llm.engine", engine="demo")

        self._engine_name = actual
        return actual

    # ── public ──────────────────────────────────────────────────────

    def send_async(self, msg: ChatMessage | str) -> None:
        text = msg.content if isinstance(msg, ChatMessage) else str(msg)
        self._q.put(text)

    def send_sync(self, text: str) -> str:
        """Block until reply complete; collects tokens. Mostly for tests."""
        self.conversation.add_user(text)
        chunks: list[str] = []
        try:
            for tok in self.engine.stream_reply(self.conversation, text):
                chunks.append(str(tok))
        except Exception as exc:  # noqa: BLE001
            return f"[error: {exc}]"
        reply = "".join(chunks)
        self.conversation.add_assistant(reply)
        self._record_turn(text, reply)
        return reply

    def _record_turn(self, user_text: str, assistant_text: str) -> None:
        if self.memory is None:
            return
        try:
            self.memory.record_chat_turn(user_text, assistant_text)
            self.memory.maybe_decay_and_compact()
        except Exception as exc:  # noqa: BLE001
            log.warning("memory.record_failed", error=str(exc))

    def _record_telemetry(
        self, user_text: str, assistant_text: str, duration_s: float,
    ) -> None:
        """Hand the turn's wall-time + token counts to TelemetryRecorder.

        Real token counts come from ``engine.last_usage`` when the
        engine exposes them (Anthropic, Ollama). Demo engine doesn't,
        so we fall back to word-count estimates."""
        try:
            from faceview.llm.telemetry import TelemetryRecorder
            usage = getattr(self.engine, "last_usage", None) or (0, 0)
            in_tok, out_tok = usage
            # Best guess at the model name for cost lookup.
            model = (getattr(self.engine, "model", None)
                     or self._engine_name
                     or "demo")
            TelemetryRecorder.shared().record(
                engine=self._engine_name or "demo",
                model=str(model),
                duration_s=duration_s,
                prompt_text=user_text,
                completion_text=assistant_text,
                prompt_tokens=in_tok,
                completion_tokens=out_tok,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("telemetry.record_failed", error=str(exc))

    def stop(self) -> None:
        self._q.put(None)

    # ── worker thread ───────────────────────────────────────────────

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            user_text = item
            try:
                # L10 — fold older turns into a summary if we're
                # approaching the model's context window. No-op until
                # the budget is actually hit.
                try:
                    budget_raw = os.environ.get(
                        "FACEVIEW_CONTEXT_TOKEN_BUDGET",
                    )
                    budget = int(budget_raw) if budget_raw else 100_000
                except Exception:  # noqa: BLE001
                    budget = 100_000
                try:
                    self.conversation.maybe_compact(budget_tokens=budget)
                except Exception as exc:  # noqa: BLE001
                    log.warning("conversation.compact_failed",
                                error=str(exc))
                self.conversation.add_user(user_text)
                # Thread the user message down to cognition so its
                # narrate_for_prompt can retrieve semantically similar
                # past episodes for this turn only. Cleared in the
                # finally so the next inference (e.g. a tool-loop
                # continuation) doesn't reuse stale context.
                mem = self.memory
                if mem is not None and hasattr(mem, "set_query_context"):
                    try:
                        mem.set_query_context(user_text)
                    except Exception:  # noqa: BLE001
                        pass
                chunks: list[str] = []
                t0 = time.time()
                try:
                    for tok in self.engine.stream_reply(self.conversation, user_text):
                        chunks.append(str(tok))
                        self.bus.publish(EventType.LLM_TOKEN, str(tok))
                finally:
                    if mem is not None and hasattr(mem, "set_query_context"):
                        try:
                            mem.set_query_context(None)
                        except Exception:  # noqa: BLE001
                            pass
                final = "".join(chunks)
                duration_s = time.time() - t0
                self.conversation.add_assistant(final)
                self._record_turn(user_text, final)
                self._record_telemetry(user_text, final, duration_s)
                self.bus.publish(EventType.LLM_REPLY, ChatMessage("assistant", final))
                # TTS_SPEAK is routed via MainWindow's set_tts_enabled
                # subscription (conditional on the TTS worker being on);
                # publishing here would cause the worker to speak twice
                # because both subscribers fire.
            except Exception as exc:  # noqa: BLE001
                log.error("llm.error", error=str(exc))
                self.bus.publish(EventType.LLM_ERROR, str(exc))
