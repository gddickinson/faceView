# faceView — Roadmap

A living document. Items move between **planned**, **in progress**, **done**.
Priorities reflect impact on the demo (a Claude Code session that can see,
hear, speak through an animated face) over engineering breadth.

Track legend:
- **R** Reliability & infra — CI, types, packaging, install hygiene.
- **L** Real-time loop — STT ↔ Claude ↔ TTS ↔ avatar end-to-end.
- **A** Avatar depth — renderer, animation, personas, coarticulation.
- **S** Server / control surface — HTTP + MCP parity, settings UI.
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
| X1 | X | Local LLM backend (Ollama) | Pluggable client behind `llm.client` interface. |
| X2 | X | Streaming TTS (Kokoro / Piper) | Replace pyttsx3 demo; keep pyttsx3 as fallback. |
| X3 | X | Web UI mode (server + browser frontend) | Headless faceView, browser renders via WS. |

---

## Done

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
