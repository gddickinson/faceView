# faceView — Session Log

## 2026-05-06 — Session 1: Scaffold + research

- Spec: multimodal GUI (chat / STT / TTS / face presence / identity / emotion /
  mouth-activity + visemes) drivable by Claude Code via local HTTP and MCP.
- Research (parallel agents): confirmed PySide6 over PyQt6; MediaPipe + ArcFace
  for vision; faster-whisper + silero-vad + pyttsx3/Kokoro for speech; honest
  scope on lip-reading (visemes only, not VSR).
- Decisions: private `gddickinson/faceView` repo, fresh `faceview` conda env,
  ANTHROPIC_API_KEY with demo-mode fallback, MCP + FastAPI both shipped.
- Created project tree, `pyproject.toml` with optional ML extras, `CLAUDE.md`,
  `INTERFACE.md` (full module map), this log, and `.gitignore`.
- Conda env `faceview` created (Python 3.11).

## 2026-05-06 — Session 4: Roadmap + personas + coarticulation + CI

- Added `ROADMAP.md` — five tracks (R/L/A/S/X) covering reliability, the
  real-time loop, avatar depth, server surface, and stretch goals. Marks
  what's now in flight vs queued vs later.
- New `vision/personas.py` + `assets/config/personas.json` — 7 bundled
  appearance presets (default / claude / warm_tan / fair_blonde /
  cool_pale / warm_dark / auburn). `Persona` is a static overlay
  applied to every `FaceParams` after `face_state_to_params`, so
  appearance is fully decoupled from the FACS animation.
- Coarticulation in `vision/speech.py`: new `viseme_blend_at(timeline,
  t, attack=0.04, release=0.06)` returns a per-AU weighted-max blend
  across active viseme envelopes. `TalkingAvatar` now uses the blend
  as its mouth-AU target instead of stepped `viseme_at`, giving
  continuous trajectories across phoneme boundaries.
- New Service ops `set_emotion`, `set_persona`, `avatar_say`,
  `list_personas` reach the avatar through `Service.bind_camera_worker`.
  Wired through HTTP (`POST /avatar/{emotion,persona,say}`,
  `GET /avatar/personas`) and MCP (`set_emotion`, `set_persona`,
  `avatar_say`, `list_personas` — total MCP tool count: 9).
- New `tools/render_personas.py` produces `docs/images/personas.png`
  (4-col contact sheet with persona name labels).
- New `.github/workflows/test.yml` runs pytest + headless smoke on
  every push and PR; uploads the headless smoke PNG as an artefact.
- Tests: 31 → 48. New `test_personas.py` (6), `test_coarticulation.py`
  (5), `test_service_avatar.py` (6). All green.

## 2026-05-06 — Session 3: Richer renderer

- User asked for a better renderer. Split `vision/sim_face.py` into
  `sim_face.py` (303 lines, top-level layered draw) +
  `sim_face_parts.py` (492 lines, brow/eye/cheek/nose/mouth helpers)
  to stay under the 500-line budget.
- Extended `FaceParams` with 9 AU-grade fields (mouth_pucker,
  mouth_stretch, cheek_raise, nose_wrinkle, upper_lid_raise,
  inner/outer_brow_raise, brow_lower, lip_corner_drop) so visemes
  and expression presets reach the renderer with full per-AU
  intensity instead of being collapsed into smile/jaw_open.
- New layered drawing: background vignette → ears with inner shadow →
  head skin (radial gradient + side shading + rim light) → AU6 cheek
  apples → hair cap + fringe path with strand highlights → tangent-
  aligned brow strokes (12 hairs + solid body) → almond eyes (radial
  iris, eyelashes, AU6 lid crease) → nose bridge with AU9 wrinkle →
  mouth with cupid's bow, asymmetric smile/frown, teeth strip with
  vertical dividers, chin shadow.
- Mouth geometry settled after several iterations: separate
  `corner_dy` (capped) and `mid_dy` (asymmetric pos/neg) plus
  `upper_h_scale` floor of 0.30 so frowns no longer wedge into
  pointed triangles and smiles get a proper ∪ curve.
- All 31 tests still pass. Re-rendered `docs/images/` (main, happy,
  speaking, surprised, face_neutral/happy/sad/surprised, avatar
  GIF + strip + monitor).
- Committed as `84bd56b`. Push to remote pending user authorisation.

## 2026-05-06 — Session 2: FACS-based talking avatar

- User pointed out the related `face_app/faceforge` project — pulled the
  FACS model, expression presets, viseme table, and a compact CMU
  pronouncing dictionary as bundled assets. Replaces my hand-rolled
  char→viseme heuristic with proper ARPAbet phoneme lookup + 15-class
  viseme alphabet keyed to AU activations.
- Added `vision/face_state.py` (FACS FaceState with 12 AUs + pose + gaze),
  `vision/expressions.py` (preset loader), `vision/speech.py` (SpeechEngine),
  `vision/avatar.py` (TalkingAvatar with AutoBlink + AutoBreathing +
  AutoSaccade + lip-sync). Renderer unchanged — `face_state_to_params()`
  bridges the AU model to the existing `FaceParams` renderer.
- `SimCameraWorker` gained an "avatar" scenario and an LLM_REPLY hook so
  the camera panel can become Claude's animated face when
  `FACEVIEW_AVATAR=1` is set.
- New `tools/animate_talking.py` records `avatar_talking.gif`,
  `avatar_strip.png`, `avatar_monitor.png` for the README — visible
  lip-sync + blinks + breathing.
- Test suite: 17 → 31. New tests cover FACS preset loading, AU→FaceParams
  bridge, CMU dict + letter-rule fallback, viseme mapping, idle blink
  occurrence, jaw-motion during speech, frame variation across an utterance.

## 2026-05-06 — Session 1, continued: working build

- Core, GUI, LLM, speech, vision, server all implemented per `INTERFACE.md`.
- Procedural face renderer added (`vision.sim_face` + `SimCameraWorker`)
  after user request: full pipeline now runs without a real webcam.
- 17 pytest tests passing (event bus, conversation, screenshot helper,
  service layer, sim-face determinism, offscreen smoke).
- Captured 5 GUI scenes + 4 face crops into `docs/images/` via
  `tools.capture_gui_screenshots`.
- README written with embedded screenshots, honest lip-reading scope, and
  full HTTP+MCP usage docs.

## Next

- `git init`, push to private `gddickinson/faceView`.
- Optional follow-ups: real Auto-AVSR ONNX upgrade for true VSR; Kokoro TTS;
  enrol-owner CLI; auto-start camera/audio toggles in the GUI menu.
