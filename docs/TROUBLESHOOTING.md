# faceView — Troubleshooting

Common issues and their fixes, ordered roughly by how often they bite.

The first stop for any problem is the **status bar** at the bottom of
the main window. It surfaces worker failures (camera unavailable,
TTS unavailable, …) and the active LLM engine. The second stop is
**View → Perception**: it shows exactly what the LLM is being told
about the camera scene on every turn.

If neither of those tells you what's wrong, the structured log on
stderr is the source of truth — launch with `python -m faceview` so
log lines stream live.

---

## LLM picks "demo mode" / Claude only echoes me

The engine selector looks for, in order:

1. `ANTHROPIC_API_KEY` env var → Anthropic
2. Ollama reachable at `127.0.0.1:11434` → Ollama
3. Otherwise → demo echo

Two common causes:

- **Anthropic key not exported in the shell that launched faceView.**
  Verify with `echo $ANTHROPIC_API_KEY | head -c 8` before launching.
  We recommend storing the key in macOS Keychain and exporting on
  shell start; do *not* paste the key into a config dialog or commit.
- **Ollama running but no models installed.** `ollama list` should
  return at least one chat-capable model. We recommend
  `qwen2.5:14b` (or `qwen2.5:7b` for snappier turns) — both support
  tool calling.

To force an engine regardless of auto-detection, use **Tools →
Configuration → LLM** at runtime, or set
`FACEVIEW_ENGINE=anthropic|ollama|demo` before launch.

---

## `look_at_camera` tool fails with HTTP 500

Symptom: chat says *"Vision lookup failed: HTTP Error 500: Internal
Server Error"*, log shows
`vision.tool.ollama_failed error='HTTP Error 500: Internal Server Error'`.

Cause: the local VLM (typically `llama3.2-vision`) was pulled with
an older Ollama version and is no longer compatible.

Fix:

```bash
ollama pull moondream         # small + fast (1.7 GB), recommended
# OR
ollama pull llama3.2-vision   # re-pulls the broken one (~6 GB)
```

The auto-picker (`pick_deep_vision_model`) now health-checks each
candidate and demotes broken ones, so this should be self-healing on
next launch — but the model has to actually exist on disk in a
working form first.

To pin a specific VLM:

```bash
export FACEVIEW_OLLAMA_DEEP_VISION_MODEL=moondream     # on-demand
export FACEVIEW_OLLAMA_VISION_MODEL=moondream          # ambient
```

---

## Camera shows the idle placeholder forever

Click **Tools → Toggle camera** — does the status bar say
*"Camera unavailable: ..."* afterwards?

- **macOS camera permission denied.** Settings → Privacy & Security
  → Camera → enable the terminal/IDE you launched faceView from.
  faceView itself isn't bundled as an `.app`, so the permission is
  granted to whatever process forked it (Terminal, iTerm,
  PyCharm, VS Code).
- **Camera in use by another app.** Quit Zoom, Photo Booth, OBS.
- **`opencv-python` missing.** `pip install -e ".[vision]"` from the
  project root inside the `faceview` conda env.
- **Wrong camera index.** Default is index 0; some setups need
  `FACEVIEW_CAMERA=1` (e.g. iPhone Continuity Camera).

---

## Microphone never lights up the speech pill

- **macOS mic permission denied.** Same Settings → Privacy & Security
  → Microphone fix as the camera section. Required for the terminal
  that forked faceView.
- **`sounddevice` missing.** `pip install -e ".[speech]"`.
- **Wrong default input.** Check `python -c "import sounddevice;
  print(sounddevice.query_devices())"`. The default device is
  usually fine; if not, set `FACEVIEW_AUDIO_DEVICE=<index>`.

---

## TTS doesn't speak / pyttsx3 errors

faceView prefers Kokoro neural TTS; falls back to pyttsx3 if Kokoro
assets are missing.

- **Kokoro model not downloaded.** Run
  `python -m faceview.speech.tts_kokoro --download` once. Stores
  ~340 MB at `~/.faceview/tts/`.
- **`afplay` missing (non-macOS).** Currently the Kokoro path uses
  `afplay` for playback. On Linux/Windows the fallback to pyttsx3
  applies. Cross-platform TTS playback is on the roadmap.

---

## STT (faster-whisper) crashes on first speech

- **First-run model download blocked.** faster-whisper downloads
  `base`/`small`/etc. models on first use. If you're offline, the
  download fails. Pull manually:
  `huggingface-cli download Systran/faster-whisper-base`.
- **CTranslate2 missing CoreML.** On Apple Silicon, install with the
  `--no-build-isolation` flag if pip resolves the wrong wheel:
  `pip install --force-reinstall ctranslate2`.

---

## `Port 8765 already in use`

Symptom: HTTP API doesn't start; `api.start_failed` in the log.

Cause: a previous faceView instance crashed without releasing the
port, or another tool grabbed it.

Fix: `lsof -ti :8765 | xargs kill -9`. Or pick a different port with
`FACEVIEW_API_PORT=8766` before launch.

---

## Persona swap freezes the GUI / crashes

Cause: in earlier versions, swapping persona restarted the avatar
renderer, which raced the previous moderngl GL context and could
segfault. The current code does *in-place* persona swaps via
`TalkingAvatar.set_persona(name)`.

If this still happens, capture the stack trace
(`PYTHONDEVMODE=1 faceview`) and file an issue — it's likely a
reintroduced regression.

---

## Vision tools refuse to fire / chat ignores them

Two checks:

1. **Is `FACEVIEW_VISION_TOOL=0` set?** That disables all on-demand
   tools globally. Unset it.
2. **Is the chat model tool-capable?** With Ollama, the chat model
   must support tool calling. `llama3:latest` does *not* —
   `qwen2.5:14b`, `llama3.1:8b`, `mistral-nemo` all do. Switch via
   **Tools → Configuration → LLM** or
   `FACEVIEW_OLLAMA_MODEL=qwen2.5:14b`.

The Perception panel doubles as a sanity check: if the narrative
block at the top mentions a tool name (e.g. *"tracking: cup"*), the
LLM has the context it needs.

---

## InsightFace age estimate is wildly off

The `genderage` model in InsightFace's `buffalo_l` bundle is known to
skew high on low-light or low-resolution webcams. Treat the output
as ±10 years. We surface that caveat in the tool's reply text. The
roadmap item *PR2 — Per-tool consent dial* will let you turn off
attribute estimates entirely if you prefer.

---

## Memory grows over a long session

Known causes:

- **Frame ring isn't bounded.** PerceptionStore caches one of each
  signal — that's bounded. But the bus retains subscribers; long
  sessions can accrete event handlers if a worker is start/stopped
  many times. Status bar restart counter (when it exists, see R7 on
  the roadmap) will flag it.
- **CLIP / EasyOCR model weights stay resident.** Expected — the
  point of the singleton is to avoid the reload cost. Restart
  faceView to release.

---

## "It works for George but not for me" — multi-person ID

PeopleStore loads everyone in `.faceview/people/*.npz` on boot. To
inspect:

```bash
ls ~/.faceview/people/      # actually: <project>/.faceview/people/
```

The `owner_data/owner.npy` legacy template is loaded as a synthetic
`"owner"` entry — if you previously ran
`tools/enroll_owner.py`, that face is recognised as `owner`.

To re-enroll: delete the relevant `.npz` and ask the LLM (in chat)
*"Please call remember_person with name=<you>"* while facing the
camera.

---

## Anthropic vision content blocks failing

If `look_at_camera` succeeds via Ollama but fails via Anthropic, two
suspects:

- **API key has no vision quota.** Tier-1 Anthropic keys allow vision;
  some free trials don't.
- **Image too large.** We resize to ≤768 px on the long edge, ~80%
  JPEG — well under the 5 MB / 8000 px limits. If you see
  `image is too large` errors anyway, set
  `FACEVIEW_VISION_TOOL=0` to bypass and report the exact frame
  dimensions.

---

## Where to get more help

- **`SESSION_Log.md`** — chronological notes of design decisions
  and the bugs they fixed.
- **`INTERFACE.md`** — module map; tells you which file owns which
  symbol.
- **`ROADMAP.md`** — what's planned. Bugs we know about often have
  a tracked item; check before filing.
- **GitHub issues** —
  [gddickinson/faceView](https://github.com/gddickinson/faceView)
  is where new ones live.
