# faceView — Model choice guide

faceView is engine-agnostic: every chat turn flows through the same
ClaudeClient/Conversation/CognitionStore stack regardless of which
LLM is on the back end. But the choice of *which* models you point
it at materially changes latency, cost, and capability — especially
once tool-use + an ambient VLM enter the picture.

This page is the cheat-sheet we wish we had when first wiring up
local backends.

---

## TL;DR — recommended setup on a 32 GB M-series Mac

| Slot | Recommended | Why |
|---|---|---|
| Chat (cloud) | `claude-sonnet-4-6` | Best tool-use reasoning we've tested; $3/1M in. |
| Chat (local) | `qwen2.5:14b` *(or `qwen2.5:7b` for faster)* | Tool-capable; 14B is more reliable on multi-step. |
| Ambient VLM | `moondream` | 1.7 GB, ~1 s/caption, runs continuously. |
| On-demand deep VLM | `llava:13b` *(or `llama3.2-vision`)* | Heavier captions for `look_at_camera` calls. |

```bash
ollama pull qwen2.5:14b
ollama pull moondream
ollama pull llava:13b
```

After that, faceView's auto-picker should find each model without
any env var hints. The status bar **LLM** pill will land on the
chat model, and the **Vision** pill will flash orange when a local
VLM is running.

---

## Chat-model picker (Ollama)

`pick_default_model` in `llm/ollama_client.py` walks `ollama list`
in this preference order, taking the first match:

```
llama3 → llama2 → qwen → mistral → phi → gemma
```

It explicitly **skips models whose name contains `vision` or `llava`**
because their `/api/chat` endpoint expects multimodal input the chat
layer doesn't supply.

| Model | Tool-use? | First-token (M1 Max) | Notes |
|---|---|---|---|
| `qwen2.5:14b` | ✅ | ~1.2 s | Default pick. Solid tool reasoning. |
| `qwen2.5:7b` | ✅ | ~0.7 s | Snappier; occasionally weaker on chained tool calls. |
| `llama3.1:8b` | ✅ | ~0.8 s | Meta's tool-capable 8B. Roughly on par with qwen 7B. |
| `mistral-nemo:12b` | ✅ | ~1.0 s | 12B Mistral with tool support; similar quality to qwen:14b. |
| `llama3.1:70b` | ✅ | 3–4 s | Overkill on consumer hardware; worth it only on workstation. |
| `llama3:latest` | ❌ | — | **No tool support.** Avoid for faceView; we'll auto-skip but it's a footgun. |
| `qwen2.5-coder` | ✅ | similar to qwen2.5 | Coder-tuned; good if your chats are code-heavy. |

Pin a specific chat model with `FACEVIEW_OLLAMA_MODEL=qwen2.5:14b`.

---

## VLM picker (Ollama)

Two separate slots — and two different env-var overrides:

### Ambient captioner — `pick_vision_model`

Picks **small + fast** for the every-15-s background caption. Preference order:

```
moondream → llava-phi → minicpm-v → llama3.2-vision → llava
```

Override with `FACEVIEW_OLLAMA_VISION_MODEL=<name>`.

| Model | Size | Latency | Caption quality |
|---|---|---|---|
| `moondream` | 1.7 GB | ~1 s | Adequate for "what's in the room" — short and direct. |
| `llava-phi3` | 2.9 GB | ~1.5 s | Slightly better detail than moondream. |
| `minicpm-v` | 5.5 GB | ~3 s | Better quality but at the edge of "ambient". |

### On-demand deep VLM — `pick_deep_vision_model`

Picks **capability over speed** for the `look_at_camera` tool. Preference order:

```
llama3.2-vision → llava:13b → llava-llama3 → llava:7b → minicpm-v → llava → moondream
```

Auto-health-checks each candidate against `/api/generate` once per
process (L7) — broken / incompatible models are demoted automatically.

Override with `FACEVIEW_OLLAMA_DEEP_VISION_MODEL=<name>`.

| Model | Size | Latency | Best for |
|---|---|---|---|
| `moondream` | 1.7 GB | ~1 s | Quick scene Q&A. |
| `llava:7b` | 4.7 GB | ~3 s | Reliable middle ground. |
| `llava:13b` | 8 GB | ~5 s | Good OCR-ish detail; clean object identification. |
| `llama3.2-vision:11b` | 7.9 GB | ~6 s | High-quality long captions, OCR on signs/screens. |

### Known footguns

- **`llama3.2-vision` pulled with an old Ollama version returns
  HTTP 500.** The deep-VLM picker now health-checks and demotes
  the model automatically, but you'll see warnings in the log.
  Fix: `ollama pull llama3.2-vision` to refresh, or
  `ollama pull moondream` for a smaller working alternative.
- **Pinning a chat model that doesn't support tools** (e.g.
  `FACEVIEW_OLLAMA_MODEL=llama3:latest`) silently drops every tool
  invocation. The chat reply still streams; you just never see the
  model call `look_at_camera`. The status pill stays the right
  colour either way.
- **Running an Anthropic-vision request while in test mode.** Test
  mode shares the same engine; concurrent VLM calls back into
  Anthropic can rate-limit. Keep test mode off when iterating on
  tool use.

---

## Anthropic chat models

| Model | Input $/1M | Output $/1M | Notes |
|---|---|---|---|
| `claude-opus-4-7` | 15 | 75 | Most capable; pricey. Reserve for hard reasoning. |
| `claude-sonnet-4-6` (default) | 3 | 15 | Best price/perf. Tool use is excellent. |
| `claude-haiku-4-5` | 1 | 5 | Fast + cheap; works for chat, weaker on tools. |

`FACEVIEW_MODEL=claude-sonnet-4-6` pins the chat model. Status pill
shows `sonnet 4.6` / `opus 4.7` / `haiku 4.5` shortened.

---

## Latency expectations

These are end-to-end (request → final token) on M1 Max, after first-
token warmup, for a single short user message ("What can you see?"):

| Setup | Chat-only turn | With `look_at_camera` |
|---|---|---|
| Anthropic sonnet 4.6 | ~1.0 s | ~2.5 s (native vision) |
| qwen2.5:14b + moondream | ~1.2 s | ~2.5 s (chat + 1 s VLM) |
| qwen2.5:7b + moondream | ~0.7 s | ~2.0 s |
| qwen2.5:14b + llava:13b | ~1.2 s | ~6 s |

First-call cold load is much slower (~10–15 s for CLIP, ~6 s for
EasyOCR, ~12 s for sentence-transformers) — those numbers above
assume the singleton is already loaded.

---

## What to do if a model isn't picked up

1. `ollama list` — confirm the model is actually installed.
2. Restart faceView so the in-process `list_ollama_models` cache
   re-reads. Health-check results are cached per process; stale
   models recover after a restart.
3. Force an engine via the **Tools → Configuration → LLM** dialog
   if auto-pick is wrong.
4. Pin via env var (`FACEVIEW_OLLAMA_MODEL`,
   `FACEVIEW_OLLAMA_VISION_MODEL`, `FACEVIEW_OLLAMA_DEEP_VISION_MODEL`)
   when starting from a launch script.

See also `docs/TROUBLESHOOTING.md` for failure-mode details.
