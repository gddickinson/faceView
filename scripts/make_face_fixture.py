"""Write the Python-vs-JavaScript reference fixtures for the web interface.

tests/fixtures/face_reference.json  (pure Python, no Qt): text -> phonemes -> viseme timelines, coarticulated AU
    blends and tongue poses, FACS expression -> FaceParams, persona overlays, Mersenne Twister draws and a seeded
    TalkingAvatar run (idle blink/breath/saccade + a scripted utterance + an emotion change).
tests/fixtures/face_colors.json     (needs PySide6): QColor HSV skin tones and darker()/lighter() results that the
    canvas renderer reimplements.
src/faceview/web/static/ref/*.png   (needs PySide6): the desktop app's stylised renderer at 320x240, shown next to the
    canvas port on the Verify page.

Run: python scripts/make_face_fixture.py            (all three; Qt parts skipped if PySide6 is missing)
The JS engine test (static/face_engine.test.mjs) and tests/test_web.py read these files.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from faceview.llm.character import load_character_registry  # noqa: E402
from faceview.llm.canned_lines import SEED_PROMPTS  # noqa: E402
from faceview.vision.avatar import TalkingAvatar  # noqa: E402
from faceview.vision.expressions import apply_expression, expression_names  # noqa: E402
from faceview.vision.face_state import FaceState, face_state_to_params  # noqa: E402
from faceview.vision.personas import apply_persona, list_personas, load_persona  # noqa: E402
from faceview.vision.face_params import FaceParams  # noqa: E402
from faceview.vision.speech import SpeechEngine, tongue_pose_at, viseme_blend_at  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
REF_DIR = ROOT / "src" / "faceview" / "web" / "static" / "ref"

# FaceParams fields the canvas renderer reads (the rest are for the 3D renderers).
PARAM_FIELDS = [
    "yaw", "pitch", "eye_open", "jaw_open", "smile", "brow_raise", "pupil_x", "pupil_y", "skin_hue",
    "mouth_pucker", "mouth_stretch", "cheek_raise", "nose_wrinkle", "upper_lid_raise", "inner_brow_raise",
    "outer_brow_raise", "brow_lower", "lip_corner_drop",
]
STYLE_FIELDS = ["skin_hue", "hair_color", "lip_color", "background", "render_mode"]
from faceview.web.data import REF_RENDERS  # noqa: E402  (stems of the reference renders)


def demo_texts() -> list[str]:
    texts = ["Hello world.", "Mm, interesting, let me actually think about that."]
    texts += list(SEED_PROMPTS[:4])
    return texts


def params_dict(p: FaceParams) -> dict:
    return {k: getattr(p, k) for k in PARAM_FIELDS}


def speech_section() -> dict:
    eng = SpeechEngine()
    out = []
    for i, text in enumerate(demo_texts()):
        speed = 1.0 if i % 2 == 0 else 1.3
        ph = eng.text_to_phonemes(text)
        tl = eng.generate_au_sequence(text, speed=speed)
        dur = tl[-1].end_time if tl else 0.0
        times = [round(-0.05 + k * 0.0137, 6) for k in range(int((dur + 0.2) / 0.0137) + 1)]
        out.append({
            "text": text, "speed": speed, "phonemes": ph,
            "timeline": [[t.viseme, t.start_time, t.end_time] for t in tl],
            "samples": [{"t": t, "blend": viseme_blend_at(tl, t), "tongue": tongue_pose_at(tl, t)} for t in times],
        })
    return {"texts": out}


def expression_section() -> dict:
    out = {}
    for name in expression_names():
        s = apply_expression(FaceState(), name)
        s.blink_amount = 0.25
        out[name] = params_dict(face_state_to_params(s))
    return out


def avatar_section() -> dict:
    av = TalkingAvatar(seed=7, emotion="happy", persona="claude")
    frames = []
    text = SEED_PROMPTS[0]
    for i in range(300):
        t = i / 30.0
        if i == 30:
            av.say(text, speed=1.0)
            av._utterance.start_t = t  # the GUI uses wall-clock time; pin it for a reproducible run
        if i == 200:
            av.set_emotion("surprised")
        p = av.tick(t)
        d = params_dict(p)
        d["blink"] = av.state.blink_amount
        d["tongue"] = list(p._talking_tongue) if p._talking_tongue else None
        frames.append(d)
    return {"seed": 7, "emotion": "happy", "persona": "claude", "say_at_frame": 30, "text": text,
            "surprised_at_frame": 200, "fps": 30, "frames": frames}


def mt_section() -> dict:
    r = random.Random(7)
    a = [r.random() for _ in range(5)]
    r2 = random.Random(123456789)
    b = [r2.uniform(2.5, 5.0) for _ in range(5)]
    return {"seed7_random": a, "seed123456789_uniform_2.5_5": b}


def persona_section() -> dict:
    out = {}
    for name in list_personas():
        p = apply_persona(FaceParams(), load_persona(name))
        out[name] = {k: getattr(p, k) for k in STYLE_FIELDS}
    return out


def make_reference() -> dict:
    return {
        "speech": speech_section(),
        "expressions": expression_section(),
        "avatar": avatar_section(),
        "mt": mt_section(),
        "personas": persona_section(),
        "characters": sorted(load_character_registry()),
    }


def ref_params(stem: str) -> FaceParams:
    from faceview.vision.visemes import viseme_au_targets
    s = FaceState()
    if stem.startswith("speaking_"):
        for au, v in viseme_au_targets(stem.split("_", 1)[1]).items():
            s.set(au, v)
    else:
        apply_expression(s, stem)
    return apply_persona(face_state_to_params(s), load_persona("default"))


def make_colors() -> dict:
    from PySide6.QtGui import QColor
    from faceview.vision.sim_face_parts import skin_color
    rgb = lambda c: [c.red(), c.green(), c.blue()]  # noqa: E731
    skins = [{"hue": h, "lightness": l, "rgb": rgb(skin_color(h, lightness=l))}
             for h in (0, 18, 24, 26, 28, 32, 200, 359.5) for l in (0.62, 0.78)]
    ops = []
    for hexc in ("#2c1810", "#a44a4a", "#c2604a", "#0c0f14", "#1a120c", "#fafafa", "#3a2418"):
        for f in (110, 115, 120, 125, 130, 135, 140):
            c = QColor(hexc)
            ops.append({"hex": hexc, "factor": f, "darker": rgb(c.darker(f)), "lighter": rgb(c.lighter(f))})
    for s in skins:
        c = QColor(*s["rgb"])
        for f in (110, 115, 125, 135):
            ops.append({"rgb": s["rgb"], "factor": f, "darker": rgb(c.darker(f)), "lighter": rgb(c.lighter(f))})
    return {"skins": skins, "ops": ops}


def make_renders() -> list[str]:
    from PIL import Image
    from faceview.vision.sim_face import render_face
    REF_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for stem in REF_RENDERS:
        bgr = render_face(ref_params(stem), size=(320, 240))
        Image.fromarray(bgr[:, :, ::-1].copy()).save(REF_DIR / f"{stem}.png", optimize=True)
        made.append(stem)
    return made


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "face_reference.json").write_text(json.dumps(make_reference(), indent=None, separators=(",", ":")))
    print("wrote", FIX / "face_reference.json")
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("PySide6 not installed: skipped face_colors.json and reference renders")
        return
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])  # noqa: F841
    (FIX / "face_colors.json").write_text(json.dumps(make_colors(), separators=(",", ":")))
    print("wrote", FIX / "face_colors.json", "and", len(make_renders()), "renders in", REF_DIR)


if __name__ == "__main__":
    main()
