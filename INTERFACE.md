# faceView — Interface map

Top-level navigation. Read this before opening source.

faceView is a desktop GUI that gives an LLM (Anthropic / Ollama / demo)
a persistent face, a webcam view of the user, a microphone, and a
natural neural voice. Per-persona cognition + character system keeps
the avatar consistent across sessions and across engines.

## Layout

```
faceView/
├── README.md                    Public docs with screenshots
├── CLAUDE.md                    Claude project notes (refs this file)
├── INTERFACE.md                 ← you are here
├── SESSION_Log.md               Running progress log
├── pyproject.toml               Package metadata + optional ML extras
├── docs/images/                 README screenshots (auto-captured)
├── owner_data/                  Stored face embeddings (git-ignored)
├── .faceview/                   Per-user data dir (git-ignored)
│   ├── memory/<persona>.json    CognitionStore JSON per persona
│   └── tts/                     Kokoro neural-TTS model + voices
│
├── src/faceview/
│   ├── __init__.py
│   ├── __main__.py              `python -m faceview` → main()
│   ├── app.py                   QApplication wiring; loads cognition + LLM + workers
│   ├── config.py                Env vars, paths, runtime flags
│   │
│   ├── core/
│   │   ├── event_bus.py         EventBus(QObject) — Qt-signal pub/sub hub
│   │   ├── events.py            EventType enum + payload dataclasses
│   │   ├── logger.py            structlog setup
│   │   └── errors.py            FaceViewError hierarchy
│   │
│   ├── gui/
│   │   ├── main_window.py       MainWindow facade — panels + menu + layout;
│   │   │                        delegates worker lifecycle to controllers
│   │   ├── controllers/         Per-concern lifecycle controllers
│   │   │   ├── camera_ctrl.py     Webcam + presence/mouth/emotion/identity/
│   │   │   │                       scene/gestures/objects/captioner
│   │   │   ├── audio_ctrl.py      Mic + VAD + STT + echo-gate + push-to-speak
│   │   │   ├── tts_ctrl.py        TTS worker + LLM_REPLY→TTS_SPEAK bridge
│   │   │   ├── avatar_ctrl.py     Avatar worker + persona swap + cognition rebind
│   │   │   ├── test_mode_ctrl.py  Dual-LLM test mode + partner-persona picker
│   │   │   ├── enrollment_ctrl.py Owner enrollment flow (N-frame capture)
│   │   │   └── monitor_ctrl.py    Audio/emotion/mouth/transcript monitor windows
│   │   ├── layout.py            LayoutManager — wraps panels in QDockWidgets;
│   │   │                        Save/Reset layout via QSettings
│   │   ├── chat_panel.py        Chat history + input + Send + push-to-talk
│   │   ├── camera_panel.py      Live camera preview + idle placeholder
│   │   ├── avatar_window.py     Standalone window for Claude's face
│   │   ├── avatar_panel.py      Renders AVATAR_FRAME events
│   │   ├── status_panel.py      Presence/identity/emotion/mouth/audio + LLM pill
│   │   ├── transcript_panel.py  Streaming STT transcripts
│   │   ├── config_dialog.py     Tabbed config (General / LLM / Avatar)
│   │   ├── character_editor.py  Edit assets/config/characters.json live
│   │   ├── persona_picker.py    Tabbed avatar-style picker (41 personas)
│   │   ├── screenshotter.py     widget.grab() → PNG, live + offscreen
│   │   └── monitors/
│   │       ├── audio.py         Rolling waveform + VAD pill
│   │       ├── emotion.py       Emotion scores + dominant history
│   │       ├── mouth.py         Jaw-open trace + viseme strip
│   │       └── transcript.py    Full STT log
│   │
│   ├── speech/
│   │   ├── audio_capture.py     sounddevice mic worker + .muted flag
│   │   ├── vad.py               silero-vad gating (512-sample windowing)
│   │   ├── stt.py               faster-whisper STT worker
│   │   ├── tts.py               Engine selector (kokoro | pyttsx3) + interrupt
│   │   └── tts_kokoro.py        Kokoro neural TTS via ONNX + afplay
│   │
│   ├── vision/
│   │   ├── camera.py            cv2 AVFoundation capture worker (joined teardown)
│   │   ├── presence.py          MediaPipe face detection
│   │   ├── identity.py          InsightFace ArcFace, matches against
│   │   │                        every name in PeopleStore
│   │   ├── people.py            PeopleStore — name→embedding template
│   │   │                        disk store + match() / remember() API
│   │   ├── emotion.py           DeepFace 7-class emotion
│   │   ├── mouth.py             Mouth-activity + viseme + head-pose + gaze +
│   │   │                        face-distance + blink (one face mesh, many outputs)
│   │   ├── scene.py             Whole-frame brightness + motion at ~5 Hz
│   │   ├── gestures.py          MP Gesture Recognizer (thumbs_up / open_palm / …)
│   │   ├── objects.py           MP Object Detector (EfficientDet-Lite0, ~80 COCO)
│   │   ├── perception.py        PerceptionStore aggregator + narrate_now() for
│   │   │                        LLM system-prompt injection (always-on ambient)
│   │   ├── scene_caption.py     Ambient VLM captioner (moondream by default,
│   │   │                        ~15 s cadence, gated on presence + motion)
│   │   ├── ocr.py               EasyOCR singleton for read_text tool
│   │   ├── tracker.py           IoU-based object tracker (track_object tool)
│   │   ├── clip_query.py        OpenCLIP open-vocab visibility (check_visible)
│   │   ├── color.py             Dominant-colour analysis (describe_color)
│   │   ├── pose.py              MediaPipe Pose posture analysis (describe_pose)
│   │   ├── face_attr.py         InsightFace age+gender reuse (face_attributes)
│   │   ├── qr.py                cv2 QR code scanner (scan_qr)
│   │   ├── depth.py             MiDaS-small depth estimation (estimate_depth)
│   │   ├── gaze_target.py       Heuristic gaze-target reader (gaze_target)
│   │   ├── segment.py           GrabCut foreground mask (segment_object)
│   │   ├── mirror.py            MirrorState — user → avatar real-time mimic
│   │   ├── sim_face.py          Procedural face renderer (FaceParams)
│   │   ├── sim_camera.py        SimCameraWorker — synthetic frames; AVATAR_FRAME
│   │   ├── face_state.py        12 FACS AUs → FaceParams bridge
│   │   ├── expressions.py       Loads expression presets from JSON
│   │   ├── visemes.py           15-class viseme alphabet → AU targets
│   │   ├── speech.py            Text → ARPAbet phonemes → timed visemes
│   │   ├── personas.py          Persona overlay (skin/hair/lip/render_mode)
│   │   ├── avatar.py            TalkingAvatar — idle + coarticulated lip-sync
│   │   ├── effects.py           PreFX/PostFX runtime + effect specs
│   │   ├── effects_runtime.py   Effect scheduler + slider state
│   │   ├── effects_pre.py       Pre-render mutators on FaceParams
│   │   ├── effects_post.py      Post-render overlays (tears, blush, …)
│   │   ├── ict_face.py          USC ICT-FaceKit 26K-vert blendshape head
│   │   ├── head_3d_lite.py      ~105-vert animatable 3D head (CPU fast)
│   │   ├── head_decimated.py    BP3D skin mesh decimated
│   │   ├── face_warp.py         Image-warp realistic face (2D)
│   │   ├── face_warp_atlas.py   5-yaw atlas blending (3D)
│   │   ├── makehuman_mesh.py    MakeHuman base.obj (CC0) loader
│   │   ├── faceforge_bridge.py  Photo-anatomical render entry
│   │   ├── gpu_renderer.py      moderngl Apple-Metal Phong renderer
│   │   ├── body_3d.py           Procedural human body
│   │   ├── body_rig.py          Per-vertex BPF bone-driven rig
│   │   ├── anatomy*.py          Skull/brain/muscles/eyeballs anatomy renderers
│   │   ├── sim_face_*.py        Stylised + anatomical renderer parts
│   │   ├── mediapipe_capture.py Webcam → 52 ARKit blendshapes
│   │   ├── arkit_blendshapes.py 52 ARKit blendshapes ↔ 12 AUs mapping
│   │   └── openfacs_bridge.py   UDP bridge → openFACS Unreal
│   │
│   ├── llm/
│   │   ├── claude_client.py     Engine selector (anthropic|ollama|demo);
│   │   │                        live select_engine + bind_memory hooks
│   │   ├── conversation.py      Conversation + effective_system() with
│   │   │                        memory-context provider
│   │   ├── ollama_client.py     Local Ollama backend (auto-fallback) +
│   │   │                        tool-use loop + pick_vision_model()
│   │   ├── embeddings.py        EmbeddingService (lazy sentence-transformers)
│   │   │                        for retrieval-augmented cognition
│   │   ├── vision_tool.py       look_at_camera tool — FrameGrabber +
│   │   │                        Anthropic image content + Ollama VLM bridge
│   │   ├── character.py         Character dataclass + characters.json registry
│   │   ├── cognition.py         CognitionStore (episodic + semantic +
│   │   │                        emotional + relationship); JSON-persisted
│   │   │                        per-persona at .faceview/memory/<p>.json
│   │   └── test_conversation.py Two-bot orchestrator (canned or LLM-driven)
│   │
│   ├── server/
│   │   ├── service.py           Shared service (HTTP + MCP); _GuiBridge slots
│   │   ├── api.py               FastAPI on 127.0.0.1 (control surface)
│   │   ├── openai_compat.py     /v1/chat/completions + /v1/models shim —
│   │   │                        faceView as a drop-in local OpenAI endpoint
│   │   └── mcp_server.py        stdio MCP server adapter
│   │
│   ├── utils/
│   │   ├── headless.py          QT_QPA_PLATFORM=offscreen helpers
│   │   └── paths.py             data_dir / owner_dir / docs_image_dir
│   │
│   └── assets/
│       ├── config/
│       │   ├── au_definitions.json     12 FACS AU id→name map
│       │   ├── expressions.json        12 emotion presets
│       │   ├── expression_muscles.json 43 expression muscles
│       │   ├── personas.json           41 bundled appearance presets
│       │   ├── characters.json         8 character personalities
│       │   │                           (name + backstory + Big Five +
│       │   │                           voice + goals + relationship levels)
│       │   └── anatomy/                Faceforge head-anatomy configs
│       ├── body_part_labels_{male,female}.npz   Baked BPF labels for rig
│       └── data/
│           └── cmu_dict_compact.json   CMU pronouncing dict (150 words)
│
├── tests/                       pytest + pytest-qt suite (158 tests)
│   ├── conftest.py
│   ├── test_event_bus.py
│   ├── test_conversation.py
│   ├── test_screenshot.py
│   ├── test_service*.py
│   └── test_smoke_headless.py
│
└── tools/
    ├── faceview_monitor.py      Read-only CLI: status / chat / events /
    │                            memory / watch / screenshot
    ├── faceview_drive.py        Write CLI: launch / stop / chat / say /
    │                            persona / emotion / engine / test /
    │                            lifecycle / memory
    ├── run_headless.py          Offscreen launch + smoke screenshot
    ├── capture_gui_screenshots.py  Drives GUI states for README
    ├── animate_*.py             GIF + grid renderers (talking, anatomy, …)
    ├── build_ict_blendshapes.py Compile USC ICT-FaceKit → 23 MB npz
    ├── enroll_owner.py          One-time face-enrollment routine
    └── run_mcp_server.py        Standalone MCP entry for Claude Code
```

CI: `.github/workflows/test.yml` runs pytest + the headless smoke on
every push, archiving the screenshot as a build artefact.

## Key types

| Symbol | File | Notes |
|---|---|---|
| `EventBus` | `core/event_bus.py` | Singleton `QObject` with Qt signals; thread-safe via `Qt.QueuedConnection` |
| `EventType` | `core/events.py` | `AUDIO_CHUNK`, `VAD_SPEECH_START/END`, `TRANSCRIPT_PARTIAL/FINAL`, `CHAT_USER_MESSAGE`, `LLM_TOKEN`, `LLM_REPLY`, `LLM_ERROR`, `CHAT_LOG`, `TTS_SPEAK/STARTED/FINISHED`, `FRAME`, `AVATAR_FRAME`, `PRESENCE`, `IDENTITY`, `EMOTION`, `MOUTH_ACTIVITY`, `HEAD_POSE`, `GAZE`, `FACE_DISTANCE`, `BLINK`, `GESTURE`, `SCENE`, `OBJECTS`, `STATUS`, `ERROR`. |
| `PerceptionStore` | `vision/perception.py` | Singleton bus subscriber caching every structured vision signal. `narrate_now()` → one-paragraph live status that's prepended to the LLM system prompt every turn (added in `app.py` via `Conversation.add_system_extras_provider`). `snapshot_dict()` powers the Perception debug panel. |
| `FrameGrabber` | `llm/vision_tool.py` | Singleton subscribed to `FRAME` + `AVATAR_FRAME`. `latest_jpeg_b64()` → base-64 JPEG of the most-recent frame (preferring real camera over avatar render). Used by both Anthropic (returns image content block in tool_result) and Ollama (POST'd to /api/generate against a local VLM) backends of the `look_at_camera` tool. Also feeds raw BGR into the `remember_person` tool's PeopleStore.remember call. |
| `PeopleStore` | `vision/people.py` | Process-wide singleton mapping display-name → InsightFace embedding, persisted to `~/.faceview/people/<slug>.npz`. `IdentityRecognizer.start()` injects its `embed()` function so the LLM `remember_person` tool can save a face without owning a second InsightFace model. The legacy `owner_data/owner.npy` is loaded as a synthetic `"owner"` entry. |
| `PerceptionPanel` | `gui/perception_panel.py` | Debug dock that shows the narrated paragraph the LLM sees plus a structured grid of every signal — fresh signals in green, stale in grey, missing in dim. Refreshes at 4 Hz from `PerceptionStore`. |
| `MainWindow` | `gui/main_window.py` | Owns worker lifecycles (camera, mic, TTS, avatar, test mode, mirror); routes persona swap → cognition rebind → TTS voice swap. |
| `LayoutManager` | `gui/layout.py` | Wraps the four panels in `QDockWidget`s; persists Save/Reset state via `QSettings`. |
| `ChatPanel` | `gui/chat_panel.py` | Streaming chat + "🎤 Hold to talk" push-to-speak button (interrupts TTS). |
| `AvatarWindow` | `gui/avatar_window.py` | Standalone face window subscribing to `AVATAR_FRAME` + `EMOTION` + `LLM_REPLY`. |
| `Character` | `llm/character.py` | Stable identity: name, backstory, Big Five traits, conversation style, catchphrases, goals, voice, relationship levels. |
| `CognitionStore` | `llm/cognition.py` | Persisted per-persona. Layers: episodic (significance + emotion + rehearsal recall), semantic (facts/beliefs by subject with confidence), emotional (current emotions with ~6h half-life decay), relationship score → level. Builds `narrate_for_prompt()` for system-prompt injection on every turn. |
| `ClaudeClient` | `llm/claude_client.py` | Engine-agnostic facade (`anthropic` / `ollama` / `demo`). `select_engine(name, model)` swaps live; `bind_memory(store)` adds the cognition narrative as an extras provider (without disturbing the perception provider). Anthropic + Ollama engines run a `look_at_camera` tool-use loop — see `vision_tool.py`. |
| `Conversation` | `llm/conversation.py` | Message history + `effective_system()` that prepends *all* registered system-extras providers (live perception + cognition narrative) before the base prompt. `add_system_extras_provider()` appends; `set_system_extras_provider()` replaces. |
| `TestConversation` | `llm/test_conversation.py` | Two-bot orchestrator. `mode="llm"` when both engines supplied — each bot uses its character's `narrate_identity()` as system prompt and its own `Conversation`. |
| `TtsWorker` | `speech/tts.py` | Picks Kokoro if installed + assets present, else pyttsx3. `interrupt()` kills the active utterance; `set_voice(name)` swaps voice live. |
| `KokoroEngine` | `speech/tts_kokoro.py` | Kokoro-onnx synth → temp WAV → `afplay` (avoids `sounddevice.play` clashing with the mic InputStream). Tracks subprocess for interruption. |
| `AudioCapture` | `speech/audio_capture.py` | `sounddevice` mic worker. `.muted` flag dropped chunks at source, used by MainWindow to suppress TTS echo during/after playback. |
| `Persona` | `vision/personas.py` | Visual overlay (skin / hair / lip / background / render_mode) applied to every `FaceParams`. |
| `SimCameraWorker` | `vision/sim_camera.py` | Avatar render thread. `set_mirror_provider(fn)` for mirror mode; persona swap is in-place to avoid ICT GL-context races. |
| `TalkingAvatar` | `vision/avatar.py` | Owns `FaceState`; ticks combine baseline emotion + idle + utterance lip-sync. |
| `MirrorState` | `vision/mirror.py` | Aggregates EMOTION + MOUTH_ACTIVITY + HEAD_POSE + PRESENCE → synthetic `FaceParams` so avatar mimics the user. |
| `Service` | `server/service.py` | Shared layer (HTTP + MCP). `_GuiBridge` slots marshal lifecycle / persona / pill-refresh / shutdown ops onto the GUI thread. |
| `FastAPI app` | `server/api.py` | `127.0.0.1:8765`. Endpoints: `/healthz`, `/state`, `/events`, `/chat/log`, `/monitor`, `/memory`, `/chat`, `/speak`, `/screenshot`, `/avatar/*`, `/llm/engine`, `/test/engine`, `/lifecycle`, `/shutdown`, `/effects/*`. |

## Cross-module flow

```
mic ─► AudioCapture ──► VAD ──► STT ──► EventBus(TRANSCRIPT_FINAL)
       (muted during TTS)                 │
                                          ▼
                                   MainWindow STT-bridge  ──┐
                                   (echo gate +              │
                                    push-to-speak override)  │
                                                             ▼
chat input ─► ChatPanel ────────────────► EventBus(CHAT_USER_MESSAGE)
                                                             │
                                                             ▼
                                                       ClaudeClient
                                                          │
                                  ┌───────────────────────┤
                                  │   read CognitionStore.narrate_for_prompt()
                                  │   prepended to system; engine streams reply
                                  │
                                  ▼
              EventBus(LLM_TOKEN → LLM_REPLY) + record_chat_turn(...)
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
       ChatPanel              TtsWorker             SimCameraWorker
       (display +             (kokoro →             (avatar.say →
        CHAT_LOG)             afplay)                lip-sync)

cam ─► Camera ─► Presence / Identity / Emotion / Mouth + HeadPose
                       │
                       ▼
           StatusPanel pills + (mirror mode) → SimCameraWorker

HTTP / MCP ─► Service ─► _GuiBridge slots ─► MainWindow handlers
                                  │
                                  ▼
                          (same event bus as above)
```

## Lazy-import conventions

Heavy ML libs (`mediapipe`, `insightface`, `deepface`, `faster_whisper`,
`silero_vad`, `pyttsx3`, `kokoro_onnx`, `cv2`, `sounddevice`,
`onnxruntime`, `torch`) are imported **inside** the functions/classes
that need them, with a `try/except ImportError` that raises
`MissingDependency` from `core.errors` with the install hint.

The minimum install (`pip install -e ".[dev]"`) is enough to boot the
GUI shell, run all unit tests, and take screenshots — which is what
CI runs. The neural TTS model + voices (`.faceview/tts/`) are fetched
on demand via `python -m faceview.speech.tts_kokoro --download`.
