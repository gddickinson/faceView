"""Avatar effects library — registry + dispatch.

Two stages of effect:

- ``PreFX`` modifies :class:`FaceParams` before the renderer runs.
  Used for shape morphs (eyes huge, head squish, mouth O, pupils
  large), pose perturbations, and AU drives that interact with
  the rest of the animation pipeline. See
  :mod:`faceview.vision.effects_pre` for implementations.

- ``PostFX`` modifies the rendered BGR image after the renderer.
  Used for pixel effects: lighting flashes, scanlines, tinting,
  smoke, comic shock-lines, anatomy flashes, glitch, hologram
  interference. See :mod:`faceview.vision.effects_post`.

Each effect is identified by a string name. A category groups
related effects in the GUI panel. Effects are *active* for a
duration; while active they receive a normalised time `u ∈ [0, 1]`
and an `intensity ∈ [0, 1]` chosen at trigger time.

The :class:`EffectsRuntime` in ``effects_runtime.py`` tracks
active instances + their schedules.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from faceview.vision import effects_post, effects_pre


class Stage(str, Enum):
    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class EffectSpec:
    name: str
    category: str
    stage: Stage
    label: str
    description: str
    default_duration: float = 1.5
    default_intensity: float = 1.0


# Re-export handler maps so callers (runtime, tests) can dispatch.
PRE_HANDLERS: dict[str, Callable] = effects_pre.HANDLERS
POST_HANDLERS: dict[str, Callable] = effects_post.HANDLERS


REGISTRY: list[EffectSpec] = [
    # Warps (PreFX)
    EffectSpec("eyes_huge", "warp", Stage.PRE, "Eyes huge",
                 "Widen eyes for surprise", 1.2),
    EffectSpec("eyes_closed", "warp", Stage.PRE, "Eyes closed",
                 "Squeeze eyes shut", 1.0),
    EffectSpec("mouth_o", "warp", Stage.PRE, "Mouth O",
                 "Round 'O' mouth (gasp)", 1.2),
    EffectSpec("mouth_grin", "warp", Stage.PRE, "Wide grin", "", 1.2),
    EffectSpec("mouth_pout", "warp", Stage.PRE, "Pout",
                 "Lower lip out, mild sulk", 1.5),
    EffectSpec("mouth_sad", "warp", Stage.PRE, "Sad mouth",
                 "Corners drop", 1.5),
    EffectSpec("lips_pursed", "warp", Stage.PRE, "Lips pursed",
                 "Small purse / kiss-shape", 1.2),
    EffectSpec("lips_tight", "warp", Stage.PRE, "Lips tight",
                 "Drawn tight, narrowed", 1.2),
    EffectSpec("lips_pressed", "warp", Stage.PRE, "Lips pressed",
                 "Flat pressed lips (worry)", 1.2),
    EffectSpec("mouth_smirk", "warp", Stage.PRE, "Smirk",
                 "Asymmetric grin", 1.2),
    EffectSpec("mouth_snarl", "warp", Stage.PRE, "Snarl",
                 "Upper lip up + nose wrinkle", 1.2),
    EffectSpec("brow_furrow", "warp", Stage.PRE, "Brow furrow", "", 1.0),
    EffectSpec("brow_arch_one", "warp", Stage.PRE, "Brow arch (one)",
                 "Single-brow raise", 1.2),
    EffectSpec("brow_high", "warp", Stage.PRE, "Brows high",
                 "Both brows raised", 1.2),
    EffectSpec("squint", "warp", Stage.PRE, "Squint",
                 "Suspicion / focus", 1.5),
    EffectSpec("surprise_combo", "warp", Stage.PRE, "Surprise (combo)",
                 "Eyes huge + brows up + jaw drop", 1.2),
    EffectSpec("disgust_combo", "warp", Stage.PRE, "Disgust (combo)",
                 "Snarl + brow down + squint", 1.5),
    # Pre + Post hybrid — opens jaw via PreFX, draws wagging tongue
    # via PostFX (since ICT has no tongue blendshape).
    EffectSpec("tongue_out", "warp", Stage.PRE, "Tongue out 👅",
                 "Stick tongue out (PreFX opens jaw, PostFX draws tongue)",
                 1.6),
    EffectSpec("head_shake", "warp", Stage.PRE, "Head shake (no)", "", 1.5),
    EffectSpec("head_nod", "warp", Stage.PRE, "Head nod (yes)", "", 1.5),
    EffectSpec("head_recoil", "warp", Stage.PRE, "Head recoil (shock)", "", 0.8),

    # Lighting
    EffectSpec("red_flash", "lighting", Stage.POST, "Red flash", "Anger", 0.8),
    EffectSpec("blue_flash", "lighting", Stage.POST, "Blue flash", "Cold/sad", 0.8),
    EffectSpec("green_flash", "lighting", Stage.POST, "Green flash", "Sick", 0.8),
    EffectSpec("color_pulse", "lighting", Stage.POST, "Cyan pulse", "", 1.2),
    EffectSpec("strobe", "lighting", Stage.POST, "Strobe", "", 1.0),
    EffectSpec("neon_flicker", "lighting", Stage.POST, "Neon flicker", "", 1.5),
    EffectSpec("fade_to_black", "lighting", Stage.POST, "Fade to black", "", 1.5),
    EffectSpec("halo_burst", "lighting", Stage.POST, "Halo burst", "", 1.0),
    EffectSpec("vignette", "lighting", Stage.POST, "Dark vignette", "", 1.2),

    # Sci-fi
    EffectSpec("scanlines", "scifi", Stage.POST, "CRT scanlines", "", 2.0),
    EffectSpec("pixelate", "scifi", Stage.POST, "Pixelate", "", 1.5),
    EffectSpec("glitch", "scifi", Stage.POST, "Datamosh glitch", "", 1.0),
    EffectSpec("hologram", "scifi", Stage.POST, "Hologram", "", 2.0),
    EffectSpec("vaporwave", "scifi", Stage.POST, "Vaporwave grid", "", 2.0),
    EffectSpec("invert_colors", "scifi", Stage.POST, "Invert colours", "", 1.0),
    EffectSpec("chromatic_aberration", "scifi", Stage.POST,
                 "Chromatic aberration", "", 1.5),

    # Smoke / particles
    EffectSpec("smoke_rise", "smoke", Stage.POST, "Smoke rise", "", 2.5),
    EffectSpec("sparkle_burst", "smoke", Stage.POST, "Sparkle burst",
                 "Anime sparkles", 1.0),
    EffectSpec("electric_arcs", "smoke", Stage.POST, "Electric arcs", "", 0.8),

    # Anatomy flashes
    EffectSpec("skull_flash", "anatomy", Stage.POST, "Skull flash",
                 "Bone show through skin", 0.6),
    EffectSpec("brain_flash", "anatomy", Stage.POST, "Brain flash",
                 "Pink brain overlay", 0.8),
    EffectSpec("xray_flash", "anatomy", Stage.POST, "X-ray flash",
                 "Photo negative flash", 0.5),
    EffectSpec("vein_show", "anatomy", Stage.POST, "Vein show",
                 "Forehead veins (anger)", 1.5),

    # Comic
    EffectSpec("shock_lines", "comic", Stage.POST, "Shock lines",
                 "Radial burst", 0.7),
    EffectSpec("sweat_drop", "comic", Stage.POST, "Sweat drop", "", 1.5),
    EffectSpec("heart_eyes", "comic", Stage.POST, "Heart eyes",
                 "Floating hearts", 2.0),
    EffectSpec("exclamation", "comic", Stage.POST, "!! mark", "", 1.2),
    EffectSpec("question_mark", "comic", Stage.POST, "?? mark", "", 1.2),

    # Emotional
    EffectSpec("tears", "emotional", Stage.POST, "Tears", "", 2.5),
    EffectSpec("anger_steam", "emotional", Stage.POST, "Anger steam", "", 1.5),
    EffectSpec("blush_extreme", "emotional", Stage.POST, "Blush", "", 1.8),
    EffectSpec("dark_pupils", "emotional", Stage.POST, "Dark pupils",
                 "Sinister black pupils", 1.0),
    EffectSpec("grayscale", "emotional", Stage.POST, "Grayscale (despair)", "", 2.0),
]


def categories() -> list[str]:
    seen: list[str] = []
    for e in REGISTRY:
        if e.category not in seen:
            seen.append(e.category)
    return seen


def specs_by_category() -> dict[str, list[EffectSpec]]:
    out: dict[str, list[EffectSpec]] = {}
    for e in REGISTRY:
        out.setdefault(e.category, []).append(e)
    return out


def get_spec(name: str) -> Optional[EffectSpec]:
    for e in REGISTRY:
        if e.name == name:
            return e
    return None
