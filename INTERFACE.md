# faceView — Interface map

The top-level navigation map for the project. Read this before opening source.

## Layout

```
faceView/
├── README.md                    User-facing docs with screenshots
├── CLAUDE.md                    Claude project notes (refs this file)
├── INTERFACE.md                 ← you are here
├── SESSION_Log.md               Running progress log
├── pyproject.toml               Package metadata + optional ML extras
├── docs/images/                 README screenshots (auto-captured)
├── owner_data/                  Stored face embeddings (git-ignored)
├── src/faceview/
│   ├── __init__.py
│   ├── __main__.py              `python -m faceview` → main()
│   ├── app.py                   QApplication wiring; assembles modules
│   ├── config.py                Env vars, paths, runtime flags
│   ├── core/
│   │   ├── event_bus.py         EventBus(QObject) — Qt-signal pub/sub hub
│   │   ├── events.py            EventType enum + payload dataclasses
│   │   ├── logger.py            structlog setup
│   │   └── errors.py            FaceViewError hierarchy
│   ├── gui/
│   │   ├── main_window.py       MainWindow — assembles panels
│   │   ├── chat_panel.py        Chat history + input + send
│   │   ├── camera_panel.py      Live camera preview + overlays
│   │   ├── status_panel.py      Presence/identity/emotion/mouth indicators
│   │   ├── transcript_panel.py  Streaming STT transcripts
│   │   └── screenshotter.py     widget.grab() → PNG, live + offscreen
│   ├── speech/
│   │   ├── audio_capture.py     sounddevice mic worker (PCM stream)
│   │   ├── vad.py               silero-vad gating (lazy import)
│   │   ├── stt.py               faster-whisper STT worker (lazy import)
│   │   └── tts.py               pyttsx3 TTS worker (lazy import)
│   ├── vision/
│   │   ├── camera.py            cv2 AVFoundation capture worker
│   │   ├── presence.py          MediaPipe face detection (count + bbox)
│   │   ├── identity.py          InsightFace ArcFace owner-vs-stranger
│   │   ├── emotion.py           DeepFace 7-class emotion (optional)
│   │   ├── mouth.py             Mouth-activity + viseme from blendshapes
│   │   ├── sim_face.py          Procedural face renderer (FaceParams)
│   │   ├── sim_camera.py        SimCameraWorker — synthetic frames + events
│   │   ├── face_state.py        FACS FaceState (12 AUs) + → FaceParams bridge
│   │   ├── expressions.py       Loads expression presets from JSON (FACS)
│   │   ├── visemes.py           15-class viseme alphabet → AU targets
│   │   ├── speech.py            Text → ARPAbet phonemes → timed visemes;
│   │   │                        viseme_blend_at coarticulation envelope
│   │   ├── personas.py          Persona overlay (skin/hair/lip/bg/render_mode) + loader
│   │   ├── sim_face_parts.py    Brow/eye/cheek/nose/mouth helpers (stylised)
│   │   ├── anatomy.py           86-pt landmarks + 43 expression muscles +
│   │   │                        AU-driven landmark deformation
│   │   ├── sim_face_anatomical.py  Anatomical renderer entry + dispatcher
│   │   ├── sim_face_anatomical_parts.py Anatomical feature drawers
│   │   │                        (skin/cheeks/brows/eyes/nose/mouth/hair)
│   │   ├── sim_face_anatomy_overlay.py  Muscle activation overlay +
│   │   │                        wireframe debug renderer
│   │   ├── anatomy_skull.py     Stylised skull (cranium / orbits /
│   │   │                        pyriform aperture / mandible / teeth)
│   │   ├── anatomy_brain.py     Stylised cerebrum (4 lobes + cerebellum +
│   │   │                        brainstem) with gyri/sulci texture
│   │   ├── anatomy_eyeballs.py  Full eye globes + iris + optic nerve
│   │   ├── anatomy_muscle_masses.py  Solid expression muscles (43)
│   │   │                        oriented along fiber direction
│   │   ├── sim_face_layered.py  Compositor: stack skull→brain→
│   │   │                        eyeballs→muscles→skin with per-layer alpha
│   │   ├── anatomy_meshes.py    BodyParts3D STL loader + Phong raster
│   │   │                        with per-mesh materials and draw-order
│   │   ├── anatomy_catalog.py   Unified head-anatomy MeshSpec catalog
│   │   │                        (20 bones / 100+ muscles / 8 features /
│   │   │                        7 vertebrae / 1 skin) lifted from faceforge
│   │   ├── faceforge_bridge.py  Photo-anatomical render entry (CPU);
│   │   │                        layer sets: skull_only / muscles /
│   │   │                        features / lifelike / xray / vertebrae
│   │   ├── gpu_renderer.py      Same head, Apple Metal-backed OpenGL
│   │   │                        via moderngl. ~36 fps lifelike on M1.
│   │   ├── head_3d_lite.py      ~105-vertex animatable 3D head;
│   │   │                        Delaunay front + hand-tri back; AU-
│   │   │                        deformable; ~55 fps on CPU.
│   │   ├── bp3d_landmarks.py    Measure anatomical landmark positions
│   │   │                        from the BP3D skull (refines 2D template)
│   │   ├── face_warp.py         Image-warp realistic face — warps a
│   │   │                        GPU-rendered neutral texture per-frame
│   │   ├── head_decimated.py    BP3D skin mesh decimated via vertex
│   │   │                        clustering; real anatomical head
│   │   │                        topology at lite-3D polygon count
│   │   └── avatar.py            TalkingAvatar — idle (blink/breath/saccade)
│   │                            + coarticulated lip-sync from text
│   │                            + persona overlay applied per tick
│   └── assets/
│       ├── config/
│       │   ├── au_definitions.json     12 FACS AU id→name map
│       │   ├── expressions.json        12 emotion presets (AU dicts)
│       │   ├── expression_muscles.json 43 expression muscles + AU maps
│       │   ├── personas.json           Bundled appearance presets
│       │   └── anatomy/                Faceforge head-anatomy configs
│       │       ├── skull_bones.json    20 cranial bones + colors
│       │       ├── face_features.json  Eyes / ears / nose / eyebrows
│       │       ├── expression_muscles.json (catalog form, with FMA)
│       │       ├── jaw_muscles.json    22 mastication muscles
│       │       ├── neck_muscles.json   38 neck muscles
│       │       ├── cervical_vertebrae.json  C1-C7
│       │       ├── eye_colors.json     Brown/blue/green/hazel/grey
│       │       └── skin.json           Face skin (FMA7163)
│       └── data/
│           └── cmu_dict_compact.json   150-word CMU pronouncing dict
│   ├── llm/
│   │   ├── claude_client.py     anthropic SDK; demo fallback if no key
│   │   └── conversation.py      Message-history dataclass + serialization
│   ├── server/
│   │   ├── service.py           Shared service layer (used by HTTP + MCP)
│   │   ├── api.py               FastAPI on 127.0.0.1 in QThread
│   │   └── mcp_server.py        stdio MCP server adapter
│   └── utils/
│       ├── headless.py          QT_QPA_PLATFORM=offscreen helpers
│       └── paths.py             XDG-style data dirs
├── tests/
│   ├── conftest.py              Qt app fixture, headless setup
│   ├── test_event_bus.py
│   ├── test_conversation.py
│   ├── test_screenshot.py       grab() works headless
│   ├── test_service.py          Service layer ops
│   └── test_smoke_headless.py   Boots GUI offscreen, takes a screenshot
└── tools/
    ├── run_headless.py          Offscreen launch + smoke screenshot
    ├── capture_gui_screenshots.py  Drives GUI states for README images
    ├── animate_talking.py       Talking-avatar GIF + strip + monitor PNG
    ├── animate_anatomical.py    Anatomical-mode GIFs + emotion grid
    ├── animate_anatomy_layers.py  Layered-anatomy grid + peel-away GIF +
    │                            BP3D rotating head (when meshes present)
    ├── animate_3d_modes.py      Lite-3D talking GIF + emotion grid +
    │                            three-modes comparison panel
    ├── render_neutral_face_texture.py  Generate the BP3D photo-anatomical
    │                            face texture for face_warp_2d (one-time)
    ├── copy_anatomy_meshes.py   Copy head+neck STLs from a BodyParts3D dump
    ├── render_personas.py       Persona contact sheet (docs/images/personas.png)
    ├── enroll_owner.py          One-time face-enrollment routine
    └── run_mcp_server.py        Standalone MCP entry for Claude Code config
```

CI: `.github/workflows/test.yml` runs pytest + the headless smoke on
every push, archiving the screenshot as a build artefact.

## Key types

| Symbol | File | Notes |
|---|---|---|
| `EventBus` | `core/event_bus.py` | Singleton `QObject` with Qt signals; thread-safe via `Qt.QueuedConnection` |
| `EventType` | `core/events.py` | enum: `AudioChunk`, `VadSpeechStart`, `VadSpeechEnd`, `Transcript`, `LlmTokenStream`, `LlmReplyComplete`, `TtsSpeak`, `Frame`, `Presence`, `Identity`, `Emotion`, `MouthActivity`, `Screenshot`, `ChatMessage`, `Error` |
| `MainWindow` | `gui/main_window.py` | Composes panels; calls `Screenshotter` |
| `Screenshotter` | `gui/screenshotter.py` | `capture(widget, path)` works in live + offscreen modes |
| `ClaudeClient` | `llm/claude_client.py` | `async stream(messages)` → token chunks; demo fallback |
| `Service` | `server/service.py` | `send_chat`, `screenshot`, `camera_state`, `speak`, `list_events`, plus avatar ops `set_emotion`, `set_persona`, `avatar_say`, `list_personas`. Used by both HTTP and MCP adapters. |
| `Persona` | `vision/personas.py` | Static appearance overlay (skin_hue / hair / lip / background / render_mode) applied to every `FaceParams` at render time. |
| `Muscle` | `vision/anatomy.py` | One of 43 expression muscles. Centroid + fiber direction + AU map drive landmark displacement during anatomical rendering. |
| `Landmark` | `vision/anatomy.py` | 86 anatomically-positioned points in a normalised face box. Drives the anatomical renderer. |
| `FaceState` | `vision/face_state.py` | 12 FACS Action Units + head pose + gaze + blink. The animation pipeline's primary state. |
| `TalkingAvatar` | `vision/avatar.py` | Owns FaceState; ticks combine baseline emotion + idle (blink/breath/saccade) + utterance lip-sync. |
| `SpeechEngine` | `vision/speech.py` | Text → ARPAbet phonemes (CMU dict + letter rules) → timed visemes → AU targets. |
| `FaceViewService (FastAPI app)` | `server/api.py` | Wraps `Service`; cross-thread via `QMetaObject.invokeMethod` / signals |

## Cross-module flow

```
mic ─► AudioCapture ─► VAD ─► STT ─► EventBus(Transcript)
                                      │
chat input ─► ChatPanel ──────────────┴─► ClaudeClient ─► EventBus(LlmTokenStream → LlmReplyComplete)
                                                          │
                                                          ▼
                                                          ChatPanel (display) + TTS (speak)

cam ─► Camera ─► Presence ─► EventBus(Presence)
                ├─► Identity ─► EventBus(Identity)
                ├─► Emotion  ─► EventBus(Emotion)
                └─► Mouth    ─► EventBus(MouthActivity)
                ▼
                CameraPanel (overlay) + StatusPanel (indicators)

HTTP / MCP ─► Service ─► (signals into GUI thread) ─► same handlers
```

## Lazy-import conventions

Heavy ML libs (`mediapipe`, `insightface`, `deepface`, `faster_whisper`,
`silero_vad`, `pyttsx3`, `cv2`, `sounddevice`) are imported **inside** the
functions/classes that need them, with a `try/except ImportError` that raises
`MissingDependency` from `core.errors` with the install hint. The minimum
install (`pip install -e ".[dev]"`) is enough to boot the GUI shell, run all
unit tests, and take screenshots — which is what CI runs.
