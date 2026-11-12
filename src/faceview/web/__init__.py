"""Web interface for faceView (FastAPI + plain ES-module front end; also exported as a static GitHub Pages site).

- app.py   -- FastAPI app: static files, read-only JSON API (STATIC_ENDPOINTS), POST /api/speech, POST /api/chat
- data.py  -- API payloads built from the Qt-free animation, persona and character modules
- chat.py  -- local-only Claude reply in a character's voice (key from the server environment, never the browser)
- static/  -- front end: face_engine.js / avatar_engine.js / mt19937.js / qcolor.js (ports of the Python animation
              pipeline, node-tested), face_render.js + face_parts.js (canvas port of the stylised renderer), pages
"""
