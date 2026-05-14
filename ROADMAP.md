# faceView — Roadmap

A living document. Items move between **planned**, **in progress**, **done**.
Priorities reflect impact on the demo (a Claude Code session that can see,
hear, speak through an animated face) over engineering breadth.

Track legend:
- **R** Reliability & infra — CI, types, packaging, install hygiene.
- **L** Real-time loop — STT ↔ LLM ↔ TTS ↔ avatar end-to-end.
- **A** Avatar depth — renderer, animation, personas, coarticulation.
- **S** Server / control surface — HTTP + MCP parity, settings UI.
- **P** Perception — vision pipeline, LLM-callable image-analysis tools.
- **C** Cognition — memory, character, retrieval, multi-person ledgers.
- **U** UX & GUI polish — panels, themes, keyboard, markdown rendering.
- **I** Integration ecosystem — OpenAI shim, MCP recipes, webhooks, plugins.
- **PR** Privacy & security — local-only auth, encrypted templates, incognito.
- **X** Stretch — local LLMs, real VSR, web UI.

---

## Now (this iteration)

| ID | Track | Item | Status |
|---|---|---|---|
| R1 | R | GitHub Actions CI: pytest + headless smoke | done |
| A1 | A | Persona presets (skin/hair/lip JSON) + loader | done |
| A2 | A | Coarticulation: blended viseme windows (attack + release) | done |
| S1 | S | Service ops: `set_emotion`, `set_persona`, `avatar_say` (HTTP + MCP) | done |
| R2 | R | Persona showcase render + README block | done |
| A7 | A | Anatomical renderer (86 landmarks + 43 muscles, 4 render modes) | done |
| A10 | A | Layered illustrative anatomy (skull / brain / eyes / muscles / skin) — 6 modes | done |
| A11 | A | Photo-anatomical via BodyParts3D STL meshes (faceforge_3d mode) | done |
| A14 | A | Lift faceforge head catalog (145 STLs, per-mesh materials, lifelike skin) | done |
| A15 | A | Lite 3D animatable head (~105 verts, Delaunay-tri, AU-deformable, ~55 fps CPU) | done |
| A16 | A | GPU rendering via moderngl + Apple Metal (~36 fps lifelike on M1) | done |
| A17 | A | BP3D-derived landmark refinement (measure anatomy off the skull) | infrastructure done; 2D template integration pending |
| A18 | A | Smooth lite 3D (ellipsoidal Z + midpoint inserts + Loop subdivision + per-vertex Phong) | done |
| A19 | A | BP3D-aligned 2D landmark proportions (eye line midpoint, narrower head) | done |
| A20 | A | Image-warp realistic face (`face_warp_2d`) — texture warped per-frame | done |
| A21 | A | MediaPipe canonical face mesh (468 verts, proper feature topology) | candidate next |
| A22 | A | Decimated BP3D skin mesh (real anatomy, fewer polys, animatable) | candidate |
| A23 | A | Multi-angle texture atlas for `face_warp_2d` to support rotation | future |
| A24 | A | TMJ jaw rotation in landmark deformation pipeline | done |
| A25 | A | Decimated BP3D skin head (real anatomy, vertex-cluster decimated) | done |
| A26 | A | GPU path for `head_decimated_3d` (current CPU path is ~8 fps) | future |
| A27 | A | Open-mouth texture variant for face_warp (blend by AU26) | crude composite shipped, proper render via mandible rotation pending |
| A28 | A | Multi-angle texture atlas (`face_warp_3d`) — 5 yaws blended | done |
| A29 | A | ARKit 52-blendshape compatibility layer | done |
| A30 | A | MakeHuman CC0 base mesh integration (`makehuman_3d`) | done |
| A31 | A | USC ICT / ProductionCrate 150+ MIT-licensed blendshape pack — real mesh deltas | candidate, replaces synthetic FACS displacement vectors |
| A32 | A | MetaHuman Head FBX (52 ARKit blendshapes, Gumroad CC) — drop-in better base | candidate |
| A33 | A | Subsurface scattering / dual specular skin shader (per MetaHuman) | future GPU-only |
| A34 | A | MediaPipe FaceLandmarker as live capture input — drive avatar from webcam | future |
| A35 | A | CMU mocap library (BVH/ASF/AMC) for full-body avatar motion if scope expands | future, body-focused |
| A36 | A | openFACS UDP bridge — emit our AU stream to an Unreal-rendered avatar | future bridge, MIT-licensed [phuselab/openFACS](https://github.com/phuselab/openFACS) |
| A37 | A | **ICT-FaceKit blendshape head** (`ict_face_3d`) — research-grade animated avatar | **done** ✅ |
| A38 | A | FLAME PyTorch — differentiable head model for image-fitting capture path | bridge done — `vision/flame_face.py` (model download required) |
| A39 | A | Basel Face Model 2017 via `eos-py` — pip-installable 3DMM | bridge done — `vision/bfm_face.py` (eos-py is x86_64 only — needs Rosetta on Apple Silicon) |
| A40 | A | FaceScape / FaceVerse non-commercial scans — pore-level detail | bridge done — `vision/facescape_face.py` (data download required) |
| A41 | A | Ready Player Me GLB loader for end-user avatar customization | done — `vision/rpm_avatar.py` |
| A42 | A | Skin texture map + GLSL SSS / dual specular shader on ICT mesh | next realism tier |
| A43 | A | Eye-specific specular material (wet-eye look) | next realism tier |
| A44 | A | DECA/EMOCA capture pipeline — image → ARKit coefficients → ICT renderer | bridge done — `vision/deca_capture.py` (DECA repo required) |
| A45 | A | MakeHuman gendered targets (CC0 male_young / female_young) | done — `vision/makehuman_mesh.load_target` + `makehuman_male/female` personas |
| A46 | A | MetaHuman head FBX loader (Gumroad CC) | bridge done — `vision/metahuman_face.py` |
| L5 | L | Echo-loop fix + per-character voices + push-to-speak | done — `AudioCapture.muted` flag, Kokoro voices, chat-panel button |
| L6 | L | Tool-use loop in both engines (Anthropic + Ollama) | done — multi-turn dispatch with tool_result feedback |
| C1 | C | CognitionStore: episodic + semantic + emotional + relationship | done — `llm/cognition.py`, JSON-persisted per persona |
| C2 | C | Conversation composes cognition + live perception extras | done — `Conversation.add_system_extras_provider` |
| P1 | P | PerceptionStore aggregator + live narrate_now into system prompt | done — `vision/perception.py` |
| P2 | P | Cheap publishers: scene / gestures / objects / gaze / distance / blink | done — `vision/scene.py`, `gestures.py`, `objects.py`, extended `mouth.py` |
| P3 | P | Multi-person face ID: PeopleStore + `remember_person` tool | done — `vision/people.py` + LLM-driven enrollment |
| P4 | P | Two-tier VLM: moondream ambient (~15 s) + deep on-demand | done — `vision/scene_caption.py` + `pick_deep_vision_model` |
| P5 | P | 12 LLM-callable image-analysis tools — look_at_camera, read_text, track_object, check_visible, describe_color, describe_pose, face_attributes, scan_qr, estimate_depth, gaze_target, segment_object, remember_person | done — `llm/vision_tool.py` + `vision/{ocr,clip_query,tracker,color,pose,face_attr,qr,depth,gaze_target,segment}.py` |
| U1 | U | Perception debug panel — shows what the LLM sees | done — `gui/perception_panel.py`, tabbed behind Transcript |

## Next (clear winners, queued)

| ID | Track | Item | Notes |
|---|---|---|---|
| L1 | L | TTS audio in lockstep with avatar visemes | Use viseme stream as the clock for both audio playback and rendering. Today TTS_SPEAK and avatar.say() are independent — fix that. |
| L2 | L | STT → chat input wire-up (auto-send on VAD-end) | Already produces transcripts to TranscriptPanel; thread to ChatPanel.input_box on VAD speech-end with manual edit grace. |
| L3 | L | Emotion-aware avatar from Claude reply tone | Naive: keyword/sentiment heuristic on the reply text → set_emotion. Optional: a small tag schema Claude can emit. |
| S2 | S | Settings menu in MainWindow | File menu items: toggle camera, toggle audio, choose persona, choose model. |
| S3 | S | Conversation persistence (sqlite or JSON) | Save chat history per-day; load on startup; expose `clear_history` and `export_history`. |
| A3 | A | Subtle head motion during speech | Small yaw/pitch nod tied to phoneme stress + word boundaries. |
| A4 | A | Eye-contact / target gaze | Avatar can be told "look at point" or "look at user"; saccades respect. |
| R3 | R | Pre-commit hooks (ruff + mypy) | Configure once; speeds future PRs. |
| R4 | R | macOS install troubleshooting in README | Camera/mic permission prompts, conda env steps, common faster-whisper download issues. |
| R5 | R | Decompose `gui/main_window.py` (978 lines) into controllers | Lifecycle controllers (camera / audio / tts / avatar / test-mode), echo-gate, push-to-speak, monitor windows. Target ≤ 300 lines per file. |
| R6 | R | Type-check CI (pyright in pre-commit) | `pyproject.toml` already has pyright dep; turn it on in CI and fix existing failures. |
| R7 | R | Crash recovery for worker threads | Each worker (`CameraWorker`, `SceneCaptioner`, `ObjectDetector`, …) registers with a supervisor that restarts on death up to N times. |
| L7 | L | Default deep VLM picker handles incompatible models | `pick_deep_vision_model` should probe each candidate against `/api/generate` once; demote any that return HTTP 500 so users don't hit the llama3.2-vision footgun. |
| L8 | L | Recording indicator while frames leave the machine | A red dot + "image sent to Anthropic / local VLM" pill while `look_at_camera`-style tools are running. Privacy hygiene. |
| L9 | L | Cost / latency telemetry per turn | Track Anthropic tokens × $, Ollama wall-time, tool round-trip times. Persist to `~/.faceview/telemetry.jsonl` + show "last turn: 1.4 s, $0.003" in the status bar. |
| L10 | L | Conversation auto-summarisation on context overflow | When `messages` would exceed model context, condense the oldest N turns into one assistant-side summary block; surface "[earlier conversation summarised]" in the chat. |
| C3 | C | Per-person memory branches | When `PeopleStore.match()` resolves a known person mid-conversation, save the turn into a per-person sub-store so each interlocutor builds their own thread. |
| C4 | C | Retrieval-augmented memory | Embed each chat turn (local sentence-transformers); on every new user message, retrieve top-K similar past turns and inject as "you previously discussed:" hints. Replaces or augments the fixed cognition narrative. |
| C5 | C | "Forget that" / memory editing API | LLM tool + HTTP endpoint to redact a specific episodic memory or a semantic fact, with undo. |
| C6 | C | Incognito mode | Toggle in Tools menu: chat without writing to CognitionStore or chat history. Status bar shows the mode. |
| U2 | U | Markdown rendering in chat | Code fences, tables, math (KaTeX), syntax highlighting. Today the panel is plain HTML. |
| U3 | U | Conversation search | Ctrl+F over the chat history with highlight + next/prev. **done** — overlay find bar with wrap-around, pre-fill from selection. |
| U4 | U | Keyboard shortcuts inventory | Cmd+Enter to send, push-to-talk global hotkey, persona-switch (Cmd+1..8), camera/mic/TTS toggles. Document the list in a `Help → Shortcuts` dialog. |
| U5 | U | Live "Claude is working" indicator | While a tool is mid-flight, surface `vision.tool.* in progress (n s)` in the status bar + a translucent overlay on the chat panel. |
| S7 | S | API token authentication | Opt-in `FACEVIEW_API_TOKEN`; any local process can hit the HTTP API today. |
| I1 | I | OpenAI-compat shim on `/v1/chat/completions` | Lets external tools (Cursor, OpenAI SDK callers) treat faceView as a local LLM endpoint that has webcam tools built in. |
| I2 | I | Published MCP recipes for Claude Code | Tutorials + sample MCP-driven workflows: "Claude Code sees my error message via webcam", "Claude Code captures screen region and reads code". The plumbing exists, the docs don't. |
| D1 | R | Troubleshooting page | Camera permission, mic permission, missing API key, ollama down, stale VLM, missing extras. Add to `docs/TROUBLESHOOTING.md` + linked from README. |
| D2 | R | Model-choice guide for Ollama | Recommended chat + VLM combos with first-token latencies measured on M-series. Reduces the trial-and-error of picking models. |

## Later

| ID | Track | Item | Notes |
|---|---|---|---|
| L4 | L | Tool-use for Claude inside the chat panel | Render tool calls as collapsed cards; let Claude drive the avatar. |
| S4 | S | Per-session state inspection endpoint | `/inspect` returns full FaceState + utterance + persona + recent events. |
| S5 | S | MCP `set_face_params` raw passthrough | For experiments — bypass FACS, set FaceParams directly. |
| A5 | A | Multiple face-shape presets (round / oval / heart) | Geometry params on FaceParams. |
| A6 | A | Auto-AVSR ONNX VSR upgrade (true lip reading) | Keep current visemes path; add real VSR as opt-in. Heavy. |
| A8 | A | True 3D head via QOpenGLWidget + faceforge STL meshes | Bring across BodyParts3D meshes + skinning. Multi-week. |
| A9 | A | Per-muscle activation curves (not just max) for richer overlays | Replace `muscle_activation` max with weighted-sum + clip. |
| A12 | A | Phong shading + skin/bone material differentiation in `faceforge_3d` | Real specular + per-mesh material; today's renderer is flat Lambert. |
| A13 | A | AU-driven mesh deformation in `faceforge_3d` (skinning) | Lift faceforge's delta-matrix soft-tissue skinning. |
| X1 | X | Local LLM backend (Ollama) | Pluggable client behind `llm.client` interface. **done** — `llm/ollama_client.py` |
| X2 | X | Streaming TTS (Kokoro / Piper) | Replace pyttsx3 demo; keep pyttsx3 as fallback. **done** — Kokoro is the default engine |
| X3 | X | Web UI mode (server + browser frontend) | Headless faceView, browser renders via WS. |
| P6 | P | Multi-face identity tracking | Today `IdentityRecognizer` returns the largest face's label only. Publish a list of `(bbox, name, sim)` per face so the LLM can answer "who is on the left?". |
| P7 | P | Rolling video frame buffer | Keep the last N seconds of frames in a ring buffer so tools (action recognition, "what just happened?") can read short clips, not just stills. |
| P8 | P | Real action recognition | Small video classifier (MoViNet-stream / VideoMAE-small) on the frame buffer. Returns "typing", "drinking", "waving" etc. |
| P9 | P | Layout-aware OCR + document understanding | Replace EasyOCR with PaddleOCR / Surya; preserve column order, table layout. Adds a `summarise_document` tool that runs the result through the chat LLM. |
| P10 | P | Screen-region capture as a vision source | New `vision/screen.py` worker — capture a chosen monitor or rectangle and publish as a second FRAME stream (`SCREEN_FRAME`). Lets the LLM see code, slides, browser tabs. |
| P11 | P | Audio scene classification | YAMNet via ONNX — `is_there_music()`, `is_there_speech()`, `was_there_a_loud_sound()`. Different sensor modality but same on-demand pattern. |
| P12 | P | Better gaze: solvePnP + iris combined | Today head pose is approximate from landmark geometry. PnP with a canonical 3D model gives real yaw/pitch; combine with iris for accurate gaze targets. |
| P13 | P | Voice activity beyond binary VAD | Speaker diarisation — *"the user just spoke"* vs *"someone else just spoke"*. Useful when two people are in front of the camera. |
| **P14** | **P** | **Room-map worker — `vision/room_map.py`** | Run MiDaS-small depth on the latest FRAME at ~1 Hz; for every OBJECTS detection sample the depth at the bbox centre; project `(image_x, image_y, depth)` through estimated camera intrinsics (default 65° HFOV; env override `FACEVIEW_CAMERA_HFOV_DEG`) into a camera-relative 3-D point; smooth per-label position with a short EMA so dots don't jitter; publish `EventType.ROOM_MAP` carrying the list of `(label, position_xyz, last_seen_ts)`. Only runs while the Map window is open (lazy on-demand to save CPU). **done** |
| **P15** | **P** | **Room-map UI — `gui/room_map_panel.py` + View menu** | Standalone window showing a top-down plan view: camera at origin, field-of-view cone, persistent dots for each detected/tracked object labelled with class + distance, motion trails per object, the user's face direction (head-pose yaw) as a heading arrow. Accessed from **View → Room map…** with shortcut `Ctrl+Shift+Z`. Subscribes to `ROOM_MAP` for updates. Distance reported in "rel units" by default; switches to metres once P16 calibration is done. **done** |
| **P16** | **P** | **Camera intrinsics calibration** | One-number scale calibration: pick an object on the live map, type its real distance in metres; we compute `scale = metres / rel_units` and persist to `.faceview/camera_calibration.json`. RoomMap published in metres thereafter. Surfaced as "Calibrate camera…" in the Map window. **done** |
| **P17** | **P** | **LLM tool: `describe_room_layout()`** | New tool routed through both engines. Reads `RoomMapStore`, sorts items by distance, classifies direction (directly ahead / to the right / behind / etc.), formats as "Room layout: cup is 0.8 m ahead; laptop is 1.5 m to the right; …". No new VLM call. **done** |
| C7 | C | Cross-persona factual transfer | Facts about the user (their name, what they do) live above the persona layer; only stylistic memories are persona-scoped. |
| C8 | C | Persona emotional feedback loop | User's emotion (deepface) influences the persona's emotional ledger over time — a kind person notices when the user is sad. |
| C9 | C | Memory consolidation / forgetting curve | Background sweep that downgrades stale episodic memories, promotes frequently-rehearsed ones to semantic facts. |
| C10 | C | Conversation export | Save a single chat (with memory + perception traces) to JSON / PDF. Companion to incognito mode. |
| A47 | A | Audio-driven lip-sync | Decode visemes from the generated TTS waveform rather than from the text phonemes — more accurate timing for emphasised syllables. |
| A48 | A | Shared canvas as a second avatar output | Avatar can draw, annotate a screenshot, or render markdown into a side surface — multimodal output, not just speech. |
| U6 | U | Dark mode + theme system | Honour macOS appearance + an explicit override. Per-pill colour tokens. |
| U7 | U | Status bar dashboard | Live perception preview + last-turn latency + per-tool spend; one-line densification of what's now spread across the Perception panel + LLM pill. |
| U8 | U | Accessibility labels (VoiceOver) | All dock widgets + pills get descriptive labels; ensure keyboard navigation reaches every control. |
| U9 | U | Mobile/tablet companion (read-only) | Phone shows the chat + perception narrative streamed over the existing HTTP API. No new servers — just a SwiftUI / React Native client. |
| I3 | I | Webhook subscriptions for bus events | Third-party tools register a URL; faceView POSTs PRESENCE/EMOTION/GESTURE/etc payloads on change. Enables Home Assistant / Stream Deck integrations. |
| I4 | I | Plugin system for third-party tools | Drop-in `.faceview/plugins/*.py` declares a tool name, schema, executor; auto-registered into both engine tool catalogues. |
| PR1 | PR | Encrypted face templates at rest | `.faceview/people/*.npz` encrypted with a Keychain-stored key. Same for the legacy `owner.npy`. |
| PR2 | PR | Per-tool consent dial | Default-on for cheap tools, default-off for ones that send pixels off-device (Anthropic vision content blocks). Surface a one-click "trust this tool" toggle. |
| PR3 | PR | Content filter on user input | Block obvious prompt-injection / jailbreak patterns before they reach the LLM. Off by default; on for "shared device" mode. |
| X4 | X | Whisper-large STT option | Pluggable STT backend so non-English / accented users get higher-quality transcription. Cost: another big model. |
| X5 | X | Cross-platform (Linux + Windows) | Mostly works on Linux already (camera + Qt); needs CI matrix + AVFoundation-equivalent backends for capture worker. |

---

## Vision (longer-term, exploratory)

The current direction makes faceView a perception-first multimodal LLM
agent. Stretching that further, the questions worth asking next:

- **Embodied agent loop.** Can the LLM act on its own without per-call
  approval, given a structured trust model? E.g. it autonomously
  pulls `look_at_camera` every 30 s while you're cooking, narrates
  what it sees, stays quiet when nothing changes. Needs: a "consent
  budget" abstraction, careful UX so the user is never surprised.
- **Multi-user / multi-room.** Today one faceView serves one user
  (with multi-person face ID). Imagine a kitchen instance + a desk
  instance sharing a CognitionStore — a family LLM that has
  per-room context and per-person memory threads.
- **Differentiable capture pipeline.** DECA/EMOCA bridge exists
  (A44); integrating it would mean: user's face → ARKit blendshape
  coefficients → drive the ICT avatar in real time. The avatar
  becomes a *digital twin* of the user, not a separate character.
- **Open ecosystem.** A spec for third-party tools (name + schema +
  executor) that drops into both engine tool catalogues. faceView
  becomes a hub that any local ML model can plug into.
- **Edge-only / no-cloud mode.** Anthropic remains an option but
  defaults to Ollama + local Whisper + local Kokoro + local CLIP +
  local moondream. Recording indicator never has to go red.
- **Long-horizon character agency.** Today personas are stateless
  responders. Layer goals + plans + project memories so a character
  pursues an arc across many sessions ("you said you wanted to learn
  Spanish — let's practise for ten minutes").

---

## Non-goals (explicit)

To keep direction tight:

- **Generic chat client.** There are plenty. faceView's value is the
  webcam + perception loop; don't compete on chat UX alone.
- **Real-time visual speech recognition (VSR).** Documented in
  CLAUDE.md as impractical from Python in 2026. Visemes from
  blendshapes are good enough; revisit if a high-quality ONNX VSR
  model lands.
- **Universal multimodal model.** Don't write our own VLM. Use the
  best one Ollama has + Anthropic native vision; treat both as
  interchangeable backends.
- **Production deployment.** This is a single-user desktop app, not
  a SaaS. Authentication, rate-limiting, observability stay
  minimal.

---

## Done

### 2026-05-11 — Head nod: rigid head + neck stretch
- Earlier cervical-cascade work made the head look like a "rigid
  block rotating in space" because pivots were at the mesh
  centerline (`pivot_z=0`) and cumulative pitch was concentrated at
  the top of the neck.
- Added three new cascade dimensions: `pivot_z_offset` (back-of-neck
  shift), `single_pivot_y_norm` (replaces the cascade with one
  rotation around an ear-level pivot), and `anchor_fade_band`
  (per-mode smoothstep width).
- New default `head_block_neck_stretch`: single ear-level pivot at
  y_norm=+0.30 (atlanto-occipital joint), pivot_z back at -0.20,
  whole head rotates rigidly above y_norm=-0.10, throat at -0.30 to
  -0.10 stretches, body below stays at rest.
- Diagnostic tools added: `tools/_nod_motion_overlay.py` (cyan-rest
  / red-pitched side-view overlays), `tools/_quadrant_motion_assess`
  (counts cyan/red px in above-ear × front/back quadrants with
  3-px erosion), `tools/_neck_base_sweep.py` (parameter sweep over
  cascade configs).

### 2026-05-10 — Body rig clean-up
- Skeleton-driven voxel relabel (`tools/skeleton_voxel_relabel`)
  fixed ~700 male / 500 female systematically mislabelled verts on
  the procedural body OBJ. JSON overrides baked into
  `body_part_labels_{male,female}.npz` via `tools/bake_label_overrides`.
- Fixed phantom-filter ordering in `ict_face.py`: apply
  `_apply_manual_overrides` BEFORE `filter_phantom_triangles` so
  re-labeled verts don't keep stale bridge triangles to their old
  region (root cause of the "necklace" artifact at shoulders).
- `FACEVIEW_RIG_WEIGHT_MODE` default switched from `hard` to
  `graded_3ring`.
- `gen_body_mesh` snaps intermediate `body_morph` values to the
  nearest baked extreme (±1) — labels NPZ only exists at the two
  ends, so blended meshes were falling back to the threshold
  classifier and producing flyaway voxels at the GUI default
  (slider was 0.0).
- New regression test `tests/test_body_rig_regression.py` — 32
  arm/leg-effect cases assert that only the expected BPF labels
  move > 1.0 ICT unit during each effect.

### 2026-05-06 — Session 5
- A7 anatomical renderer: 86-point landmark template + 43 expression
  muscles (lifted from faceforge) drive AU-based 2D vertex
  displacement. Three new render modes (`anatomical`,
  `anatomy_overlay`, `wireframe`) toggleable via `Persona.render_mode`.
- New tools/animate_anatomical.py renders comparison demo and emotion
  grid into docs/images/.
- Tests: 48 → 63.

### 2026-05-06 — Sessions 1-3
- Project scaffolding, conda env, package metadata, INTERFACE.md, CLAUDE.md.
- Core PySide6 GUI shell: chat / camera / status / transcript panels.
- Event bus with Qt-signal pub/sub.
- LLM client (Anthropic SDK) with demo-mode echo fallback.
- FastAPI control plane on 127.0.0.1:8765.
- stdio MCP server adapter (5 tools).
- Sim camera worker + procedural face renderer (`vision/sim_face`).
- FACS-based talking avatar: 12 AUs, 15-class viseme alphabet, ARPAbet
  phoneme pipeline, expression presets, blink/breath/saccade idle systems.
- Layered renderer rewrite: ears, hair with strand highlights, almond
  eyes with eyelashes, AU-driven brows, asymmetric smile/frown mouth,
  cupid's bow, teeth strip with dividers.
- 31 → 35+ pytest tests.
