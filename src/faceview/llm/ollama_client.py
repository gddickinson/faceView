"""Ollama backend — local LLM fallback when no Anthropic key is set.

[Ollama](https://ollama.com) runs LLMs locally and exposes them on
``http://127.0.0.1:11434`` by default. This module provides a
streaming engine compatible with our ``EchoEngine`` /
``AnthropicEngine`` interface so the existing ``ClaudeClient``
worker thread can use it without changes.

Auto-fallback chain (set up in `claude_client.ClaudeClient`):

    1. ``ANTHROPIC_API_KEY`` set → Anthropic
    2. Ollama reachable on localhost:11434 → Ollama (default model
       picked from ``FACEVIEW_OLLAMA_MODEL`` env var or first
       installed model)
    3. Otherwise → :class:`EchoEngine` demo

No new runtime dependency — pure stdlib `urllib` + `json`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Optional

from faceview.core.logger import get_logger


log = get_logger("ollama")


DEFAULT_HOST = "http://127.0.0.1:11434"


def is_ollama_available(host: str = DEFAULT_HOST, timeout: float = 0.5) -> bool:
    """Quick reachability check for a running ``ollama serve``."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def list_ollama_models(host: str = DEFAULT_HOST, timeout: float = 1.0) -> list[str]:
    """List installed Ollama models. Returns ``[]`` on any failure."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def pick_default_model(host: str = DEFAULT_HOST) -> Optional[str]:
    """Pick a sensible default model — env-var override, then first installed.

    Prefers text-only chat models (skips ``-vision`` / ``llava`` variants
    whose chat endpoint expects multimodal input we don't supply).
    """
    env = os.environ.get("FACEVIEW_OLLAMA_MODEL")
    if env:
        return env
    models = list_ollama_models(host)
    if not models:
        return None
    # Prefer text-only chat-tuned models. Skip vision / llava variants.
    candidates = [m for m in models
                  if "vision" not in m.lower() and "llava" not in m.lower()]
    if not candidates:
        candidates = models
    for needle in ("llama3", "llama2", "qwen", "mistral", "phi", "gemma"):
        for m in candidates:
            if needle in m.lower():
                return m
    return candidates[0]


def pick_vision_model(host: str = DEFAULT_HOST) -> Optional[str]:
    """Pick a locally-installed VLM for the ``look_at_camera`` tool.

    Override via ``FACEVIEW_OLLAMA_VISION_MODEL``. Otherwise prefers
    small / fast models first: moondream, llava-phi, minicpm-v,
    llama3.2-vision, llava. Returns ``None`` if no vision model is
    installed — in that case the tool simply isn't offered to the chat
    model."""
    env = os.environ.get("FACEVIEW_OLLAMA_VISION_MODEL")
    if env:
        return env
    models = list_ollama_models(host)
    if not models:
        return None
    for needle in (
        "moondream", "llava-phi", "minicpm-v",
        "llama3.2-vision", "llama3-vision", "bakllava", "llava",
    ):
        for m in models:
            if needle in m.lower():
                return m
    return None


class OllamaEngine:
    """Streaming Ollama engine matching the ClaudeClient engine protocol."""

    def __init__(
        self,
        model: str,
        host: str = DEFAULT_HOST,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        # Token usage from the most recent /api/chat final chunk.
        # Picked up by TelemetryRecorder. None before the first call.
        self.last_usage: tuple[int, int] | None = None

    def stream_reply(self, conv, user_text: str) -> Iterator[str]:
        """Yield text chunks from Ollama's streaming /api/chat endpoint.

        If a vision model is installed and ``FACEVIEW_VISION_TOOL`` is
        on, ``look_at_camera`` is offered as a tool. When the chat model
        chooses to call it, we run the local VLM via /api/generate, push
        a ``tool`` message with the description, and re-stream — capped
        at three tool round-trips per turn.
        """
        # Lazy imports so test_imports doesn't pull cv2 / bus.
        from faceview.llm.vision_tool import (
            LOOK_TOOL_OLLAMA, FrameGrabber, pick_deep_vision_model,
            TIER1_TOOLS_OLLAMA, TIER23_TOOLS_OLLAMA,
            run_look_ollama, run_remember_person,
            run_read_text, run_track_object, run_check_visible,
            run_describe_color, run_describe_pose, run_face_attributes,
            run_scan_qr, run_estimate_depth, run_gaze_target,
            run_segment_object,
            vision_tool_enabled,
        )

        # Build the initial message list from Conversation.
        messages: list[dict] = self._build_messages(conv)

        tools_on = vision_tool_enabled()
        # The on-demand look_at_camera tool prefers a *capable* VLM
        # (llama3.2-vision / llava:13b) rather than the small ambient
        # captioner. Falls back through to moondream if that's all
        # that's installed.
        vlm = pick_deep_vision_model(self.host) if tools_on else None
        if vlm is None and tools_on:
            vlm = pick_vision_model(self.host)
        sys_msg = messages[0]["content"] if messages and messages[0].get(
            "role") == "system" else ""
        log.info("ollama.send", model=self.model, vlm=vlm or "—",
                 msgs=len(messages), system_preview=sys_msg[:400])
        # remember_person works whether or not a VLM is installed (no
        # local captioning needed), so we offer it any time tools are
        # globally on. look_at_camera only when a VLM exists.
        tool_set: list[dict] = []
        if tools_on and vlm is not None:
            tool_set.append(LOOK_TOOL_OLLAMA)
        if tools_on:
            # Tier 1 (without look_at_camera, that's added above only
            # when a VLM is installed) + Tier 2/3.
            for t in TIER1_TOOLS_OLLAMA:
                if t is LOOK_TOOL_OLLAMA:
                    continue
                tool_set.append(t)
            tool_set.extend(TIER23_TOOLS_OLLAMA)
        use_tools = bool(tool_set)
        grabber = FrameGrabber.shared() if use_tools else None

        for _step in range(3):
            payload: dict = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"num_predict": 512},
            }
            if use_tools:
                payload["tools"] = tool_set

            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
            )

            text_buf: list[str] = []
            tool_calls: list[dict] = []
            final_chunk: dict = {}
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for line in resp:
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except ValueError:
                            continue
                        msg = chunk.get("message") or {}
                        content = msg.get("content") or ""
                        if content:
                            text_buf.append(content)
                            yield content
                        calls = msg.get("tool_calls")
                        if calls:
                            tool_calls.extend(calls)
                        if chunk.get("done"):
                            final_chunk = chunk
                            break
            except (urllib.error.URLError, ConnectionError,
                    TimeoutError, OSError) as exc:
                log.warning("ollama.stream_failed", error=str(exc))
                return
            # Persist token usage from the final chunk for telemetry.
            try:
                from faceview.llm.telemetry import extract_ollama_usage
                self.last_usage = extract_ollama_usage(final_chunk)
            except Exception:  # noqa: BLE001
                self.last_usage = None

            if not tool_calls:
                log.info("ollama.stream_done", step=_step,
                         text_chars=sum(len(t) for t in text_buf))
                return

            log.info("ollama.tool_calls", step=_step,
                     count=len(tool_calls),
                     names=[(c.get('function') or {}).get('name')
                            for c in tool_calls])
            # Append the assistant turn that requested the tool(s).
            messages.append({
                "role": "assistant",
                "content": "".join(text_buf),
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                log.info("ollama.tool_invoke", tool=name, args=args)
                if (name == "look_at_camera"
                        and grabber is not None and vlm is not None):
                    result = run_look_ollama(
                        grabber, vlm, host=self.host,
                        question=str(args.get("question") or ""),
                        region=str(args.get("region") or "full"),
                    )
                elif name == "remember_person" and grabber is not None:
                    result = run_remember_person(grabber,
                                                 args.get("name", ""))
                elif name == "read_text" and grabber is not None:
                    result = run_read_text(
                        grabber,
                        region=str(args.get("region") or "full"),
                    )
                elif name == "track_object":
                    result = run_track_object(
                        label=str(args.get("label") or ""),
                        duration_s=float(args.get("duration_s") or 10),
                    )
                elif name == "check_visible" and grabber is not None:
                    result = run_check_visible(
                        grabber,
                        query=str(args.get("query") or ""),
                        region=str(args.get("region") or "full"),
                    )
                elif name == "describe_color" and grabber is not None:
                    result = run_describe_color(
                        grabber, region=str(args.get("region") or "full"),
                    )
                elif name == "describe_pose" and grabber is not None:
                    result = run_describe_pose(grabber)
                elif name == "face_attributes" and grabber is not None:
                    result = run_face_attributes(grabber)
                elif name == "scan_qr" and grabber is not None:
                    result = run_scan_qr(grabber)
                elif name == "estimate_depth" and grabber is not None:
                    result = run_estimate_depth(
                        grabber, region=str(args.get("region") or "full"),
                    )
                elif name == "gaze_target":
                    result = run_gaze_target()
                elif name == "segment_object" and grabber is not None:
                    result = run_segment_object(
                        grabber, label=str(args.get("label") or ""),
                    )
                else:
                    result = f"Unknown tool: {name}"
                    log.warning("ollama.unknown_tool", tool=name)
                log.info("ollama.tool_result", tool=name,
                         result=result[:120])
                messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                })

    def _build_messages(self, conv) -> list[dict]:
        """Convert a ``Conversation`` into Ollama's chat format."""
        messages: list[dict] = []
        sys_fn = getattr(conv, "effective_system", None)
        sys_text = sys_fn() if callable(sys_fn) else getattr(conv, "system", None)
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        msg_list = (conv.messages()
                    if callable(getattr(conv, "messages", None))
                    else getattr(conv, "_messages", []))
        for m in msg_list:
            messages.append({
                "role": "user" if m.role == "user" else "assistant",
                "content": m.content,
            })
        return messages
