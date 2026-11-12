"""FastAPI web app for faceView: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8199, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below.
The face, expressions and lip-sync run in the browser (JS ports checked against the Python code); the server adds
the Python speech timeline (POST /api/speech) and, when ANTHROPIC_API_KEY is set for this process, a live Claude
reply (POST /api/chat). Neither POST is part of the static site.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from faceview.web import chat, data

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="faceView")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if data.IMAGES.exists():
    app.mount("/media", StaticFiles(directory=data.IMAGES), name="media")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Counts and summary for the home page."""
    return data.info()


@app.get("/api/expressions")
def expressions():
    """The 12 FACS Action Units and the expression presets (assets/config/expressions.json)."""
    return data.expressions()


@app.get("/api/visemes")
def visemes():
    """The 15-class viseme alphabet: AU targets, tongue pose and phonemes per viseme."""
    return data.visemes()


@app.get("/api/dictionary")
def dictionary():
    """The bundled CMU pronouncing dictionary subset used for text -> phonemes."""
    return data.dictionary()


@app.get("/api/personas")
def personas():
    """All bundled appearance presets (only render_mode 'stylised' is drawn in the browser)."""
    return data.personas()


@app.get("/api/characters")
def characters():
    """The character sheets in assets/config/characters.json."""
    return data.characters()


@app.get("/api/demo_lines")
def demo_lines():
    """Scripted lines for the demos: test mode's canned prompts and the characters' catchphrases."""
    return data.demo_lines()


@app.get("/api/gallery")
def gallery():
    """Desktop-app renders committed in docs/images."""
    return data.gallery()


@app.get("/api/reference")
def reference():
    """Python reference values for the Verify page (tests/fixtures/face_reference.json)."""
    return data.reference()


class SpeechIn(BaseModel):
    text: str = Field(max_length=2000)
    speed: float = Field(1.0, ge=0.1, le=4.0)


@app.post("/api/speech")
def speech(body: SpeechIn):
    """Text -> phonemes -> timed visemes from the Python SpeechEngine (the browser has the same engine)."""
    return data.speech(body.text, body.speed)


@app.get("/api/chat/status")
def chat_status():
    """Whether live chat is available (a key is set on the server); never returns the key."""
    return chat.status()


class Turn(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatIn(BaseModel):
    persona: str = "claude"
    history: list[Turn] = Field(max_length=60)


@app.post("/api/chat")
def chat_reply(body: ChatIn):
    """One in-character Claude reply (local app only)."""
    try:
        return chat.reply(body.persona, [t.model_dump() for t in body.history])
    except chat.ChatUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/expressions", "/visemes", "/dictionary", "/personas", "/characters", "/demo_lines",
                    "/gallery", "/reference"]
