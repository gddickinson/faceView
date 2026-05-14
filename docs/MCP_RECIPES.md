# faceView — MCP recipes for Claude Code

The faceView MCP server adapter (run via `python -m tools.run_mcp_server`)
exposes the same operation set as the HTTP API to a Claude Code
session: lifecycle control, chat injection, screenshots, persona
swaps, perception snapshots. This file collects a few concrete
prompts that demonstrate driving the GUI from outside.

Each "recipe" assumes:

1. faceView is running locally (HTTP API on `127.0.0.1:8765`).
2. The MCP server is registered in your Claude Code configuration:

   ```jsonc
   // ~/Library/Application Support/Claude/claude_desktop_config.json
   {
     "mcpServers": {
       "faceview": {
         "command": "/opt/anaconda3/envs/faceview/bin/python",
         "args": ["-m", "tools.run_mcp_server"]
       }
     }
   }
   ```

3. You've started a Claude Code session that lists `faceview` as
   a connected MCP server.

---

## Recipe 1 — "Show me the room from where I'm sitting"

```text
You:    Open the room map and tell me what's on my desk.
Claude: → faceview: lifecycle(name="camera", on=true)
Claude: → faceview: avatar_say(text="Opening the room map…")
Claude: → faceview: room_map_open()       # (planned MCP op)
Claude: → faceview: describe_room_layout()
Claude: "There's a laptop 1.2 m ahead, a coffee cup 60 cm to your
         right, and what looks like a notebook a metre to your left."
```

Pieces in play:
- `lifecycle` brings up the camera worker + vision stack.
- The room-map worker only ticks when the panel is visible, so
  opening it via MCP wakes the depth path.
- `describe_room_layout` reads the latest published `ROOM_MAP`
  event and formats it as prose.

---

## Recipe 2 — "Capture my error, then remember the fix"

```text
You:    I keep hitting this Python error. Watch my screen and tell
        me when you see the next traceback.
Claude: → faceview: lifecycle(name="screen_capture", on=true)
Claude: → faceview: look_at_screen(question="is there a traceback
                                            visible? if so, paste the
                                            top frame.")
Claude: "Top frame: KeyError: 'user_id' in handlers.py:42."
You:    The fix is to pass `user_id=request.user.id` from the
        caller. Remember that.
Claude: → faceview: chat(text="Got it — I'll remember the
                              user_id-keyword fix for the next time
                              this comes up.")
        # The cognition store auto-records significant turns; nothing
        # special needed here.
```

Pieces in play:
- `lifecycle("screen_capture", on=true)` turns on the SCREEN_FRAME
  publisher.
- `look_at_screen` with a `question` lets Claude phrase the VLM
  prompt specifically.
- Subsequent chat turns get folded into the persona's episodic
  memory automatically — no explicit save call needed.

---

## Recipe 3 — "Watch for me, ping when something happens"

You want Claude Code to alert you when someone walks into your
office while you're heads-down on something else.

```text
You:    Tell me when someone shows up at the camera, or 30 minutes
        from now if nobody does. I'll go finish this task.
Claude: → faceview: lifecycle(name="camera", on=true)
Claude: → set a webhook listener: POST /webhooks
                  body={"url": "https://your-pushover-relay.example/notify",
                        "events": ["PRESENCE"]}
        # OR: poll /state every minute, watch for face_count > 0
```

The HTTP-side `/webhooks` (I3) lets external services subscribe to
bus events; for an in-flight Claude Code chat the polling pattern
is usually simpler. Either way, the LLM stays out of the
event loop until something interesting happens.

---

## Recipe 4 — "Two-bot critique with shared memory"

Test mode runs two LLM-backed avatars chatting with each other.
You can use this for prompt-iteration loops:

```text
You:    Use test mode with the qwen2.5:14b engine; have the
        partner critique my essay draft. The full draft is in
        ~/Desktop/essay.md.
Claude: → read essay.md via MCP filesystem tools
Claude: → faceview: chat(text="Avatar A: please summarise the essay
                              you're about to receive in two
                              sentences. Avatar B: critique it
                              specifically on argument structure.")
Claude: → faceview: chat(text=<the essay text>)
Claude: → faceview: lifecycle(name="test_mode", on=true)
        # Test mode picks up both bot personas + the seeded
        # conversation context.
```

This is the same engine path as solo chat — every persona's
`CognitionStore` keeps growing, so the two bots accumulate
distinct memories of the critique session.

---

## Recipe 5 — "Read me what's on this paper"

```text
You:    Hold up a printed page to the camera. Read it to me.
Claude: → faceview: lifecycle(name="camera", on=true)
Claude: → faceview: chat(text="Hold the paper still while I focus
                              on it.")
Claude: → wait briefly, then look_at_camera(question="what text is
                              visible on the page in the centre of
                              the frame?", region="center")
        # OR: read_text(region="center") if OCR is preferred over a
        # VLM caption.
Claude: → speak the extracted text via TTS
```

Pieces in play:
- `region="center"` crops to the middle of the frame so the OCR /
  VLM doesn't waste tokens on background.
- `read_text` (EasyOCR) is more reliable than `look_at_camera` for
  dense / small / structured text; `look_at_camera` is better when
  the model also needs to *interpret* the text (summarise, answer
  questions about it).
- TTS routes via the existing avatar pipeline (no extra MCP op
  beyond `chat`).

---

## Recipe 6 — "Switch persona, keep my name"

```text
You:    My name is George.
Claude: # CognitionStore.record_chat_turn extracts the name
        # and writes it to BOTH the current persona's semantic.player.name
        # AND the cross-persona shared bag (.faceview/memory/_shared.json)
        # so every persona sees it.

You:    Switch to Iris.
Claude: → faceview: persona(name="ict_xray")
        # Iris's CognitionStore loads, sees the shared "player.name"
        # entry in its narration, knows you're George without you
        # having to re-introduce yourself.
Claude: "Hi George — I'm Iris. Last time we talked you were
         debugging that segmentation issue…"
```

Pieces in play:
- C7's shared facts file means the most important user-level facts
  (name, key preferences) cross persona boundaries even though
  each persona's `episodic` + `per_person` memory stays private.
- `persona()` rebinds the cognition store + TTS voice atomically.

---

## Recipe 7 — "Watch a code review, summarise at the end"

```text
You:    Capture my screen for the next 15 minutes while I review
        this PR. At the end summarise what I focused on.
Claude: → faceview: lifecycle(name="screen_capture", on=true)
Claude: → start a poll loop:
            every 60 seconds → look_at_screen(question="what file
                              and line is currently focused?")
            collect (timestamp, observation) pairs
Claude: ... 15 minutes later ...
Claude: "You spent the first 8 minutes on validators.py:140-200,
         then jumped to auth.py around line 80, where you stayed
         for the rest of the session. Most of your scroll-back
         was around the user_id keyword we discussed earlier."
Claude: → faceview: lifecycle(name="screen_capture", on=false)
```

Pieces in play:
- The polling loop is implemented entirely in the Claude Code
  session — faceView just answers `look_at_screen` calls.
- 15-minute observation patterns + summary works equally well for
  whiteboarding, paired programming, or any "ambient observer"
  use case.

---

## Where each MCP op maps to

The MCP server adapter wraps `server/service.py`. Today's surface:

| MCP op | Service method | HTTP equivalent |
|---|---|---|
| `chat(text)` | `send_chat(text)` | `POST /chat` |
| `speak(text)` | `speak(text)` | `POST /speak` |
| `screenshot(name)` | `screenshot(name)` | `POST /screenshot` |
| `persona(name)` | `set_persona(name)` | `POST /avatar/persona` |
| `emotion(name)` | `set_emotion(name)` | `POST /avatar/emotion` |
| `lifecycle(name, on)` | `set_lifecycle(name, on)` | `POST /lifecycle` |
| `engine(name, model?)` | `set_engine(...)` | `POST /llm/engine` |
| `state()` | `get_camera_state()` | `GET /state` |
| `memory()` | `get_memory()` | `GET /memory` |
| `export_chat(n?)` | `export_chat(n)` | `GET /chat/export` |

Anything not yet wired into the MCP adapter is still reachable via
the HTTP control plane — Claude Code can `curl` it the same way.
