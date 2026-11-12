"""Payloads for the web API, built from faceView's own Qt-free modules and bundled JSON.

Everything here is read-only and deterministic, so build_static.py exports it to JSON for the static site.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from faceview.assets import assets_dir
from faceview.llm.canned_lines import SEED_PROMPTS, _EMOTION_KEYWORDS
from faceview.llm.character import character_for, load_character_registry
from faceview.vision.expressions import expression_names, get_expression
from faceview.vision.personas import list_personas, load_persona
from faceview.vision.speech import SpeechEngine, _load_cmu_dict
from faceview.vision.visemes import PHONEME_TO_VISEME, TONGUE_POSE, VISEMES

ROOT = Path(__file__).resolve().parents[3]
IMAGES = ROOT / "docs" / "images"
FIXTURE = ROOT / "tests" / "fixtures" / "face_reference.json"
COLORS = ROOT / "tests" / "fixtures" / "face_colors.json"

# The viseme table from the visemes.py docstring.
VISEME_INFO = {
    "REST": "silence", "PP": "P / B / M: bilabial closure", "FF": "F / V: labiodental", "TH": "TH / DH: dental",
    "DD": "T / D / N / L: alveolar", "SS": "S / Z: sibilant", "SH": "SH / ZH / CH / JH: palatal",
    "KK": "K / G / NG: velar", "RR": "R / ER: retroflex", "AA": "AA / AE / AH: open vowel",
    "EH": "EH / EY: mid vowel", "IH": "IH / IY: close-front vowel", "OH": "AO / OW / OY: round-back vowel",
    "UH": "UH / UW: close-round vowel", "WW": "W / Y / HH: glide",
}

# Reference PNGs of the Python stylised renderer in static/ref/ (scripts/make_face_fixture.py).
REF_RENDERS = ["neutral", "happy", "sad", "surprised", "angry", "speaking_AA", "speaking_UH"]

# Desktop-app renders committed in docs/images, captions from the README.
GALLERY = [
    ("readme_avatar_claude.png", "Default Claude avatar",
     "Max-capability Claude, the default boot persona: USC ICT-FaceKit 3D head with an x-ray glow shader."),
    ("readme_avatar_iris.png", "Iris (neuroscience PhD) avatar",
     "Iris, neuroscience PhD student, with her own voice (af_nicole) and conversation memory."),
    ("readme_avatar_playful.png", "Playful Claude (cartoon)",
     "Playful Claude on the stylised cartoon face: the renderer this web site ports to canvas."),
    ("readme_avatar_bayard.png", "Bayard (retired guitarist)", "Bayard the retired classical guitarist."),
    ("readme_avatar_soraya.png", "Soraya (ER nurse)", "Soraya the ER nurse."),
    ("readme_main_chat.png", "faceView main window with chat",
     "The desktop main window: camera, chat, status pills and transcript as detachable docks."),
    ("readme_test_mode.png", "Two-bot test mode with Ollama",
     "Test mode running two Llama-3 bots (Theo and Claude); both replies are real Ollama output."),
]


def _json(rel: str):
    return json.loads((assets_dir() / rel).read_text())


def info() -> dict:
    personas = [load_persona(n) for n in list_personas()]
    return {
        "title": "faceView",
        "summary": "A face-to-face chat GUI for LLMs: Claude (or a local Ollama model) speaks through an animated "
                   "avatar with a neural voice, sees you via the webcam, listens through the microphone and "
                   "remembers you across sessions.",
        "action_units": len(_json("config/au_definitions.json")),
        "expressions": len(expression_names()),
        "visemes": len(VISEMES),
        "phonemes": len({p.rstrip("012") for p in PHONEME_TO_VISEME if p != "SIL"}),
        "dictionary_words": len(_load_cmu_dict()),
        "personas": len(personas),
        "stylised_personas": sum(p.render_mode == "stylised" for p in personas),
        "characters": len(load_character_registry()),
        "demo_lines": len(SEED_PROMPTS),
    }


def expressions() -> dict:
    return {
        "action_units": _json("config/au_definitions.json"),
        "presets": {name: get_expression(name) for name in expression_names()},
    }


def visemes() -> dict:
    return {
        "visemes": [{"name": k, "info": VISEME_INFO.get(k, ""), "au": v.to_dict(), "tongue": list(TONGUE_POSE[k]),
                     "phonemes": sorted({p.rstrip("012") for p, vv in PHONEME_TO_VISEME.items() if vv == k})}
                    for k, v in VISEMES.items()],
        "phoneme_duration": SpeechEngine().phoneme_duration,
        "word_gap": SpeechEngine().word_gap,
    }


def dictionary() -> dict:
    return _load_cmu_dict()


def personas() -> list[dict]:
    raw = _json("config/personas.json")
    out = []
    for name in list_personas():
        p = load_persona(name)
        out.append({"name": name, "skin_hue": p.skin_hue, "hair_color": p.hair_color, "lip_color": p.lip_color,
                    "background": p.background, "render_mode": p.render_mode,
                    "character": character_for(name).name if name in load_character_registry() else None,
                    "fields": sorted(raw.get(name, {}))})
    return out


def characters() -> list[dict]:
    out = []
    for key, data in load_character_registry().items():
        c = character_for(key)
        conv = c.conversation or {}
        out.append({"persona": key, "name": c.name, "age": c.age, "occupation": c.occupation,
                    "backstory": c.backstory, "traits": c.traits, "topics": conv.get("topics", []),
                    "catchphrases": conv.get("catchphrases", []), "outlook": conv.get("outlook", ""),
                    "goals": c.goals, "voice": c.voice,
                    "render_mode": load_persona(key).render_mode})
    return out


def demo_lines() -> dict:
    """Scripted lines only: test mode's canned SEED_PROMPTS and the characters' catchphrases."""
    return {
        "seed_prompts": list(SEED_PROMPTS),
        "emotion_keywords": _EMOTION_KEYWORDS,
        "catchphrases": [{"persona": c["persona"], "name": c["name"], "line": line}
                         for c in characters() for line in c["catchphrases"]],
    }


def gallery() -> list[dict]:
    return [{"file": f, "alt": alt, "caption": cap} for f, alt, cap in GALLERY if (IMAGES / f).exists()]


@lru_cache(maxsize=1)
def reference() -> dict:
    """Python reference values (scripts/make_face_fixture.py) for the in-browser Verify page."""
    ref = json.loads(FIXTURE.read_text())
    ref["colors"] = json.loads(COLORS.read_text()) if COLORS.exists() else None
    ref["renders"] = REF_RENDERS
    return ref


def speech(text: str, speed: float = 1.0) -> dict:
    eng = SpeechEngine()
    tl = eng.generate_au_sequence(text, speed=speed)
    return {"text": text, "speed": speed, "phonemes": eng.text_to_phonemes(text),
            "timeline": [{"viseme": t.viseme, "start": t.start_time, "end": t.end_time} for t in tl],
            "duration": tl[-1].end_time if tl else 0.0}
