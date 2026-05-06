"""faceView — multimodal face/voice/chat GUI for LLMs and Claude Code.

Subpackages:
- ``core``    — event bus, event types, logging, errors
- ``gui``     — PySide6 widgets (main window, panels, screenshot helper)
- ``speech``  — audio capture, VAD, STT, TTS workers (lazy ML imports)
- ``vision``  — camera capture + face presence/identity/emotion/mouth workers
- ``llm``     — Anthropic Claude client + conversation history
- ``server``  — local FastAPI control API + stdio MCP server (shared service)
- ``utils``   — paths, headless helpers
"""

__version__ = "0.1.0"
