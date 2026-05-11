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
│   │   ├── face_warp_atlas.py   5-yaw atlas blending — face_warp_3d
│   │   │                        rotates AND deforms with FACS
│   │   ├── makehuman_mesh.py    MakeHuman base.obj (CC0) loader +
│   │   │                        decimation; render mode makehuman_3d
│   │   ├── arkit_blendshapes.py 52 ARKit blendshapes (industry std,
│   │   │                        used by MediaPipe / iOS / MetaHumans)
│   │   │                        + two-way mapping to/from our 12 AUs
│   │   ├── skeleton.py          Full BP3D skeleton (~231 bones across
│   │   │                        cervical/thoracic/lumbar/skull/jaw/
│   │   │                        ribs/pelvis/upper_limb/hand/lower_limb/
│   │   │                        foot). Loads STLs, transforms BP3D→ICT.
│   │   ├── skeleton_fit.py      Region-aware skeleton-to-skin fit:
│   │   │                        per-region bbox + chain-rotation +
│   │   │                        Y-piecewise face fit. Two-segment
│   │   │                        arm/leg chains rotate independently
│   │   │                        around shoulder/elbow and hip/knee.
│   │   ├── skeleton_landmarks.py  Avatar-skin landmarks measured off
│   │   │                        the body + face meshes (torso widths
│   │   │                        per Y band, arm centroid X at three
│   │   │                        levels, ICT crown/eye/mouth/chin).
│   │   ├── ict_face.py          USC ICT-FaceKit blendshape head —
│   │   │                        26K verts, 157 ARKit-aligned shapes,
│   │   │                        per-material colours + SSS shader +
│   │   │                        eye specular. Owns the head-pitch
│   │   │                        `_apply_cervical_cascade` + the
│   │   │                        `_NOD_MODES` registry. Realistic-
│   │   │                        animated endpoint of the project.
│   │   ├── body_3d.py           Procedural human body (male + female
│   │   │                        10.5K-vert OBJs, blended by
│   │   │                        `body_morph` ∈ ±1). Strips above the
│   │   │                        chin so the ICT head transplants on
│   │   │                        top. Snaps intermediate morphs to the
│   │   │                        nearest baked NPZ extreme.
│   │   ├── body_rig.py          Hierarchical body rig: skin →
│   │   │                        per-vertex BPF labels (16 regions)
│   │   │                        → per-effect bone deformations. Hard
│   │   │                        or graded skinning weights via
│   │   │                        `FACEVIEW_RIG_WEIGHT_MODE`. Manual
│   │   │                        per-vert overrides honoured before
│   │   │                        phantom-triangle filtering.
│   │   ├── openfacs_bridge.py   UDP bridge: emit our AU stream as
│   │   │                        JSON to a phuselab/openFACS Unreal
│   │   │                        instance.
│   │   ├── mediapipe_capture.py MediaPipe FaceLandmarker live capture —
│   │   │                        webcam frames → 52 ARKit blendshapes.
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
│       ├── body_part_labels_{male,female}.npz
│       │                            Baked per-vertex BPF labels
│       │                            (16-region body-part painting)
│       │                            for the body rig. Created via
│       │                            `tools/import_part_painting.py`
│       │                            + `tools/skeleton_voxel_relabel`.
│       │                            Originals preserved as
│       │                            `..._orig.npz`.
│       ├── body_label_overrides_{male,female}_baked.json
│       │                            JSON record of manual per-vert
│       │                            overrides baked into the NPZs.
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
    ├── build_ict_blendshapes.py Compile ICT-FaceKit OBJ tree (~386 MB
    │                            from a local clone) → compressed 23 MB
    │                            npz with neutral + 157 blendshape deltas
    ├── animate_3d_modes.py      Lite-3D talking GIF + emotion grid +
    │                            three-modes comparison panel
    ├── render_neutral_face_texture.py  Generate the BP3D photo-anatomical
    │                            face texture for face_warp_2d (one-time)
    ├── copy_anatomy_meshes.py   Copy head+neck STLs from a BodyParts3D dump
    ├── render_skeleton_overlay.py  Front+side avatar with the fitted
    │                            BP3D skeleton overlaid as colour-coded
    │                            dots — for eyeballing the per-region fit
    ├── render_body_parts.py     Front+side avatar tinted by the 16
    │                            fine body-part labels (neck/chest/
    │                            abdomen/pelvis/upper-arm/forearm/
    │                            hand/thigh/shin/foot, L/R per limb)
    ├── render_personas.py       Persona contact sheet (docs/images/personas.png)
    ├── enroll_owner.py          One-time face-enrollment routine
    ├── run_mcp_server.py        Standalone MCP entry for Claude Code config
    │
    │   --- Body-rig diagnostic + relabel tools ---
    ├── skeleton_voxel_relabel.py  Stick-figure-driven voxel relabel:
    │                            move skeleton bones with the rig and
    │                            measure per-pose bone-to-vertex
    │                            distances. Reassigns systematically-
    │                            mislabelled verts (~700 male / 500
    │                            female caught on first pass).
    ├── bake_label_overrides.py  Merge JSON per-vert overrides into the
    │                            body_part_labels_{male,female}.npz
    │                            files (backs up originals as _orig).
    ├── highlight_problem_voxels.py  Visualize spatial outliers + mesh
    │                            label-islands per painting NPZ.
    ├── paint_body_parts.py      Manual painting tool (Pygame canvas)
    │                            with diagnostic overlay support.
    ├── import_part_painting.py  Import painted images → NPZ labels.
    ├── diagnose_body_rig.py     Per-effect dispersion stats.
    │
    │   --- Head-nod (cervical cascade) diagnostic tools ---
    ├── _nod_motion_overlay.py   Cyan-rest / red-pitched side-view
    │                            overlay per FACEVIEW_NOD_MODE.
    │                            Reveals where motion lives in
    │                            rendered avatars (see
    │                            docs/images/nod_overlay_*.png).
    ├── _quadrant_motion_assess.py  Counts cyan/red pixels per
    │                            above-ear × front/back quadrant
    │                            with 3-px erosion to discount
    │                            anti-aliasing edge noise.
    ├── _neck_base_sweep.py      Parameter sweep over cascade configs:
    │                            tracks per-Y-band displacement on
    │                            BOTH ICT head + body meshes; reports
    │                            chin Z/Y delta and base motion for
    │                            ranking. Outputs neck_sweep.json.
    ├── _nod_drift_measure.py    Per-Y-band quick measurement at
    │                            full pitch (older diagnostic).
    ├── _nod_drift_inspect.py    Dump cascade params + per-disc
    │                            Y/cumul values for a single render.
    ├── _capture_nod_sideview.py Baseline side-view grid at varying
    │                            pitch.
    ├── _compare_nod_modes.py    All-mode visual side-by-side grid.
    ├── _nod_final_compare.py    Before/after 2×2 with reference lines.
    ├── _nod_table.py            Numerical comparison table image.
    └── _gui_tour.sh             Drive GUI through 28 body effects
                                 per gender + capture screenshots.
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
| `_NOD_MODES` | `vision/ict_face.py` | Registry of head-nod cascade profiles. Each entry: `pitch`/`yaw` cumul fractions over 12 spine levels, `fade`, `anchor_y_norm`, `anchor_fade_band`, `pivot_z_offset`, optional `single_pivot_y_norm`. Selected via `FACEVIEW_NOD_MODE`. Default `head_block_neck_stretch`: single ear-level pivot, whole head rotates rigidly, throat stretches. |
| `_apply_cervical_cascade` | `vision/ict_face.py` | Applies head pitch / yaw / roll. Single-pivot path runs when a mode supplies `single_pivot_y_norm` (whole-head block); otherwise iterates over per-vertebra discs with smoothstep falloff. Always followed by the optional post-anchor smoothstep blend back to rest below `anchor_y_norm`. |
| `gen_body_mesh` | `vision/body_3d.py` | Returns `BodyMesh` for a given `body_morph`. Snaps intermediate morphs to nearest baked extreme (±1) because BPF labels are only baked at the two ends. |
| `apply_body_rig_v2` | `vision/body_rig.py` | Bone-driven body deformation. Per-vert BPF label drives which bones influence it; weight mode (`hard` / `graded_3ring`) chosen via env var. |

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
