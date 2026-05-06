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
