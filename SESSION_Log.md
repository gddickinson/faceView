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
