"""OpenAI ``/v1/chat/completions`` compatibility shim.

Lets external tools (Cursor, OpenAI SDK callers, langchain pointed at
``http://127.0.0.1:8765/v1``) treat faceView as a local OpenAI
endpoint. The chat completion routes through the live
:class:`ClaudeClient`, so the response is whatever engine is active
(Anthropic / Ollama / demo) — *plus* the live perception narrative
and cognition narrative get prepended to the system prompt, which
is the point: faceView as an OpenAI endpoint that already knows
what the camera sees.

Scope (minimum viable):

* Non-streaming chat completions only — streaming SSE is a TODO.
* Tool-use is **not** translated; the chat goes straight to the
  engine. The engine's own tool-use loop still fires (look_at_camera
  etc.) and the model's textual reply is what comes back. The OpenAI
  caller doesn't see the intermediate tool calls.
* ``model`` in the request is honoured for *naming* only — we don't
  hot-swap engines per request. Use the ``/llm/engine`` endpoint or
  the GUI config dialog to switch.
* No real token-usage accounting; we report a rough word-count
  estimate so SDKs don't choke on missing ``usage``.

Concurrency: the chat completion handler builds an ephemeral
:class:`Conversation` to avoid polluting the GUI's chat history.
Running the GUI chat + the OpenAI endpoint simultaneously shares one
underlying engine; the Anthropic SDK + our Ollama client are
thread-safe for streaming, but the model itself is only one
connection so back-pressure is real if you fan out heavily.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from faceview.core.logger import get_logger


log = get_logger("openai_compat")


# ── request / response models ─────────────────────────────────────


class OpenAIMessage(BaseModel):
    """One element of the ``messages`` array.

    We accept ``role`` and ``content`` only — ``name``, ``tool_call_id``
    etc. are tolerated but ignored."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str | None = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Subset of the OpenAI request shape we actually use."""
    model: str = "faceview"
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # OpenAI extensions we ignore: top_p, n, stop, presence_penalty,
    # frequency_penalty, logit_bias, user, tools, tool_choice,
    # response_format, seed.


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats


class ModelEntry(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "faceview"


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelEntry]


# ── helpers ───────────────────────────────────────────────────────


def _ephemeral_conversation_from_request(
    req: ChatCompletionRequest,
    client,
):
    """Build a one-shot :class:`Conversation` from the request.

    System messages from the request become the base system prompt;
    everything else (user / assistant) is appended in order. Live
    perception + cognition providers from the GUI's conversation are
    copied across so the LLM sees the same camera/memory context an
    in-app chat would."""
    from faceview.llm.conversation import Conversation

    system_msgs = [m.content for m in req.messages
                   if m.role == "system" and m.content]
    base_system = "\n\n".join(system_msgs) if system_msgs else None
    conv = Conversation(system=base_system)
    # Borrow the live extras providers so perception + cognition still
    # show up.
    src = getattr(client, "conversation", None)
    providers = getattr(src, "_extras_providers", []) if src else []
    for p in providers:
        conv.add_system_extras_provider(p)
    for m in req.messages:
        if m.role == "system":
            continue
        text = m.content or ""
        if m.role == "user":
            conv.add_user(text)
        elif m.role == "assistant":
            conv.add_assistant(text)
        # tool messages from OpenAI's tool-use flow: ignore for now
    return conv


def _stream_and_collect(client, conv, last_user_text: str) -> str:
    """Run ``client.engine.stream_reply`` and return the joined text.

    Bypasses :meth:`ClaudeClient.send_sync` so we don't pollute the
    GUI's chat history or trigger LLM_REPLY bus events (which would
    write to the chat panel and speak through TTS)."""
    chunks: list[str] = []
    try:
        for tok in client.engine.stream_reply(conv, last_user_text):
            chunks.append(str(tok))
    except Exception as exc:  # noqa: BLE001
        log.warning("openai_compat.stream_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"engine error: {exc}",
        ) from exc
    return "".join(chunks)


def _approx_tokens(text: str) -> int:
    """Cheap whitespace word-count as a stand-in for real tokenisation.

    OpenAI SDKs read ``usage`` for cost-tracking; returning 0 is
    fine but some clients warn. Word count gets us in the right
    order of magnitude (a real token is ~0.75 words for English)."""
    return max(1, len((text or "").split()))


def _list_model_entries(client) -> list[ModelEntry]:
    """Surface the active engine plus the two faceview-* aliases."""
    now = int(time.time())
    entries: list[ModelEntry] = []
    active = getattr(client, "current_engine", lambda: "demo")()
    entries.append(ModelEntry(id="faceview", created=now,
                              owned_by=f"faceview/{active}"))
    entries.append(ModelEntry(id="faceview-anthropic", created=now,
                              owned_by="faceview/anthropic"))
    entries.append(ModelEntry(id="faceview-ollama", created=now,
                              owned_by="faceview/ollama"))
    return entries


# ── router ────────────────────────────────────────────────────────


def build_router(service) -> APIRouter:
    """Return an APIRouter to mount under ``/v1`` on the main FastAPI app."""
    router = APIRouter(prefix="/v1", tags=["openai-compat"])

    @router.get("/models")
    def list_models() -> ModelsResponse:
        client = getattr(service.window, "llm_client", None)
        return ModelsResponse(data=_list_model_entries(client))

    @router.post("/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if req.stream:
            # Streaming would emit text/event-stream — supported by
            # the engines but not by this minimum-viable shim.
            raise HTTPException(
                status_code=501,
                detail=("streaming not implemented in this shim — "
                        "set stream=false"),
            )
        if not req.messages:
            raise HTTPException(
                status_code=400,
                detail="messages must contain at least one entry",
            )
        client = getattr(service.window, "llm_client", None)
        if client is None:
            raise HTTPException(
                status_code=503,
                detail="no LLM client bound — faceView is still starting",
            )
        last_user = next(
            (m.content for m in reversed(req.messages)
             if m.role == "user" and m.content),
            "",
        )
        conv = _ephemeral_conversation_from_request(req, client)
        reply = _stream_and_collect(client, conv, last_user)
        return ChatCompletionResponse(
            id="chatcmpl-" + uuid.uuid4().hex[:24],
            created=int(time.time()),
            model=req.model,
            choices=[ChatCompletionChoice(
                message=OpenAIMessage(role="assistant", content=reply),
                finish_reason="stop",
            )],
            usage=UsageStats(
                prompt_tokens=sum(_approx_tokens(m.content or "")
                                  for m in req.messages),
                completion_tokens=_approx_tokens(reply),
                total_tokens=(sum(_approx_tokens(m.content or "")
                                  for m in req.messages)
                              + _approx_tokens(reply)),
            ),
        )

    return router
