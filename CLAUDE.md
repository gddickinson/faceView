# faceView — Claude project notes

@INTERFACE.md

## Project intent
A multimodal desktop GUI for interacting with LLMs and Claude Code. Voice in
(STT) and out (TTS), live camera with face presence/identity/emotion/mouth
detection, real-time chat panel, plus a local control API + stdio MCP server
so a Claude Code session can drive the GUI itself.

## Conventions
- **Entry points**: `python -m faceview` (live), `python -m tools.run_headless`
  (offscreen smoke), `python -m tools.capture_gui_screenshots` (README shots).
- **Threading**: GUI thread owns Qt widgets; one `QThread` per heavy stage
  (audio, video, STT, vision, LLM, server). Cross-thread comms via
  `core.event_bus.EventBus` Qt signals (`Qt.QueuedConnection`).
- **Imports**: heavy ML deps (torch, mediapipe, insightface, deepface,
  faster-whisper) are **lazy** — imported inside functions, gated with a clear
  `ImportError` message pointing to `pip install -e ".[full]"`.
- **API key**: `ANTHROPIC_API_KEY` env var. If absent, `llm.claude_client`
  falls back to a demo-mode echo so the GUI is usable without setup.
- **Environment**: dedicated conda env `faceview` (Python 3.11). Do not pollute
  `flika`.
- **File size**: keep every file under 500 lines (split first if needed).
- **Lip reading**: visemes/mouth-activity from MediaPipe blendshapes only —
  true VSR is impractical from Python in 2026. Document this honestly.

## Where things live
See `INTERFACE.md` for the full module map. Read it before opening source.

## Running
```bash
conda activate faceview
pip install -e ".[dev,speech,vision]"   # add identity,emotion,mcp as wanted
faceview                                  # launch GUI
python -m tools.run_headless              # offscreen smoke + screenshot
pytest                                    # tests
```

## Runtime env-var switches
- `FACEVIEW_AVATAR=1` — enable the procedural talking avatar (sim
  camera worker) instead of a real webcam. Default persona is
  `ict_xray_young` when ICT-FaceKit data is present.
- `FACEVIEW_NOD_MODE=<mode>` — selects the head-nod cascade.
  Default is `head_block_neck_stretch` (single ear-level pivot,
  whole-head rigid block, neck stretches). See `_NOD_MODES` in
  `src/faceview/vision/ict_face.py` for the full list.
- `FACEVIEW_RIG_WEIGHT_MODE=hard|graded_3ring` — skinning weight
  mode for the body rig. Default `graded_3ring` smooths
  shoulder/armpit seams.
- `FACEVIEW_KINK_FIX=below_chin|legacy` — body-mesh strip cut.
- `FACEVIEW_DEBUG_PARTS=1` — log per-part displacement during
  cervical cascade for debugging.

## Updating session log
Update `SESSION_Log.md` whenever you finish a meaningful chunk of work.
