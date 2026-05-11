"""Render an effect at peak with a persona applied — apples-to-apples
match for the GUI rendering."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PIL import Image
from tools.extreme_pose_relabel import _make_neutral_params
from faceview.vision.ict_face import render_face_ict
from faceview.vision.effects_pre import HANDLERS
from faceview.vision.personas import load_persona, apply_persona


def render(gender, effect, persona, size=(360, 640)):
    p = _make_neutral_params(gender)
    apply_persona(p, load_persona(persona))
    if effect != "neutral":
        HANDLERS[effect](p, 0.5, 1.0)
    return render_face_ict(p, size=tuple(size))


if __name__ == "__main__":
    effects = ["neutral", "arms_up", "arms_crossed", "clap",
                "salute", "wave_left", "kick_left", "stretch_up"]
    persona_male = "ict_male"
    persona_female = "ict_female"
    for g, persona in [("male", persona_male), ("female", persona_female)]:
        for e in effects:
            rgb = render(g, e, persona)
            Image.fromarray(rgb).save(f"/tmp/persona_{g}_{e}.png")
            print(f"{g}/{e} done")
