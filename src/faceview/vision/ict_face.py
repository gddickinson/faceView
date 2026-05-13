"""ICT-FaceKit blendshape-driven head renderer.

Loads the pre-computed ICT-FaceKit npz (`assets/data/ict/face_kit.npz`)
shipped with the project, applies a dict of blendshape coefficients
(ARKit-named) as vertex displacements on the neutral mesh, and
renders through moderngl when available (fast) or falls back to
QPainter Z-sort (slow but functional).

The npz contains:
- ``vertices``  (N, 3) — neutral positions
- ``triangles`` (M, 3) — index buffer
- ``deltas``    (B, N, 3) — per-blendshape vertex offsets
- ``names``     (B,)     — ARKit-aligned name for each blendshape

The model is from USC ICT, MIT-licensed, designed to ship with the
ARKit 52 blendshape vocabulary so external face-tracking systems
(MediaPipe FaceLandmarker, iOS Face ID) can drive it directly.

Render mode: ``ict_face_3d``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from faceview.assets import assets_dir
from faceview.core.errors import MissingDependency


def _data_path() -> Path:
    return assets_dir() / "data" / "ict" / "face_kit.npz"


@dataclass
class ICTModel:
    vertices: np.ndarray       # (N, 3) float32 — neutral positions
    triangles: np.ndarray      # (M, 3) int32
    deltas: np.ndarray         # (B, N, 3) float32
    names: list[str]           # (B,)
    name_to_idx: dict[str, int]
    tri_materials: np.ndarray  # (M,) int32 material-table index per tri
    materials: list[str]       # material names from the OBJ usemtl tags


# ── Per-material skin palette ──────────────────────────────────────
# Hand-tuned base colors for the 12 ICT material regions. Skin gets
# warm flesh tone; teeth ivory; sclera bright; iris dark amber; lips
# slightly redder than face. Tongue/gums dusky red.

def _hsv_to_rgb(h_deg: float, s: float, v: float) -> tuple[float, float, float]:
    """HSV (hue in degrees) → RGB tuple in [0, 1]."""
    import colorsys
    return colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)


def _hex_to_rgb_f(c: str) -> tuple[float, float, float]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return (int(s[0:2], 16) / 255.0,
                int(s[2:4], 16) / 255.0,
                int(s[4:6], 16) / 255.0)
    except (ValueError, IndexError):
        return 0.5, 0.5, 0.5


# ── Sci-fi palette presets ──────────────────────────────────────
# Each preset overrides the natural HSV-derived skin palette with a
# stylised colour scheme. Styles also control shader uniforms
# (ambient / specular / SSS) via _shader_overrides_for_style.

_SCIFI_PALETTES: dict[str, dict[str, tuple[float, float, float]]] = {
    "neon": {
        "M_Face":         (0.95, 0.18, 0.65),     # hot magenta
        "M_BackHead":     (0.55, 0.12, 0.50),
        "M_GumsTongue":   (1.00, 0.30, 0.65),
        "M_Teeth":        (0.95, 0.95, 0.95),
        "M_ScleraLeft":   (0.10, 1.00, 0.95),     # glowing cyan
        "M_ScleraRight":  (0.10, 1.00, 0.95),
        "M_IrisLeft":     (1.00, 0.95, 0.05),     # neon yellow
        "M_IrisRight":    (1.00, 0.95, 0.05),
        "M_LacrimalFluid":(0.30, 1.00, 0.95),
        "M_EyeBlend":     (0.85, 0.18, 0.60),
        "M_EyeOcclusion": (0.20, 0.05, 0.30),
        "M_EyeLashes":    (0.10, 0.95, 1.00),     # cyan lashes
        "M_HairCap":      (0.20, 1.00, 0.80),     # electric green
    },
    "transparent": {
        "M_Face":         (0.55, 0.78, 0.92),     # ghost pale-blue
        "M_BackHead":     (0.40, 0.62, 0.78),
        "M_GumsTongue":   (0.50, 0.70, 0.85),
        "M_Teeth":        (0.85, 0.92, 0.98),
        "M_ScleraLeft":   (0.75, 0.95, 1.00),
        "M_ScleraRight":  (0.75, 0.95, 1.00),
        "M_IrisLeft":     (0.40, 0.85, 0.95),     # icy blue
        "M_IrisRight":    (0.40, 0.85, 0.95),
        "M_LacrimalFluid":(0.85, 0.95, 1.00),
        "M_EyeBlend":     (0.50, 0.75, 0.90),
        "M_EyeOcclusion": (0.20, 0.35, 0.50),
        "M_EyeLashes":    (0.30, 0.50, 0.70),
        "M_HairCap":      (0.55, 0.78, 0.92),
    },
    "cyberpunk": {
        "M_Face":         (0.30, 0.55, 0.62),     # cool teal skin
        "M_BackHead":     (0.20, 0.40, 0.50),
        "M_GumsTongue":   (0.85, 0.10, 0.45),
        "M_Teeth":        (0.95, 0.90, 0.80),
        "M_ScleraLeft":   (0.92, 0.95, 0.95),
        "M_ScleraRight":  (0.92, 0.95, 0.95),
        "M_IrisLeft":     (1.00, 0.20, 0.50),     # hot pink iris
        "M_IrisRight":    (1.00, 0.20, 0.50),
        "M_LacrimalFluid":(0.80, 0.90, 0.95),
        "M_EyeBlend":     (0.20, 0.40, 0.50),
        "M_EyeOcclusion": (0.05, 0.08, 0.15),
        "M_EyeLashes":    (0.02, 0.05, 0.10),
        "M_HairCap":      (0.85, 0.10, 0.60),     # magenta hair
    },
    "xray": {
        "M_Face":         (0.22, 0.45, 0.55),     # dim cyan-bone
        "M_BackHead":     (0.15, 0.32, 0.42),
        "M_GumsTongue":   (0.18, 0.40, 0.50),
        "M_Teeth":        (0.95, 0.98, 1.00),     # bright bone
        "M_ScleraLeft":   (0.95, 0.98, 1.00),
        "M_ScleraRight":  (0.95, 0.98, 1.00),
        "M_IrisLeft":     (0.30, 0.95, 1.00),     # glowing cyan
        "M_IrisRight":    (0.30, 0.95, 1.00),
        "M_LacrimalFluid":(0.85, 0.95, 1.00),
        "M_EyeBlend":     (0.20, 0.40, 0.50),
        "M_EyeOcclusion": (0.04, 0.08, 0.12),
        "M_EyeLashes":    (0.04, 0.08, 0.12),
        "M_HairCap":      (0.18, 0.32, 0.42),
    },
}


def _shader_overrides_for_style(style: str) -> dict:
    """Per-style shader uniform overrides (ambient / specular / SSS).

    Returns a partial dict that gets merged into the default values
    in ``_ICTRenderer.render``.
    """
    if style == "neon":
        # High emission — push ambient up so colours appear self-lit.
        return {"ambient": 0.85, "specular": 0.40, "shininess": 8.0,
                "sss_tint": (1.0, 0.05, 0.7)}
    if style == "transparent":
        # Soft, even lighting + cool blue SSS terminator.
        return {"ambient": 0.60, "specular": 0.10, "shininess": 6.0,
                "sss_tint": (0.55, 0.85, 1.0)}
    if style == "cyberpunk":
        # Strong rim + dark base, tech-noir feel.
        return {"ambient": 0.20, "specular": 0.60, "shininess": 32.0,
                "sss_tint": (1.0, 0.15, 0.55)}
    if style == "xray":
        # High specular + bone-cyan SSS — bones glow through.
        return {"ambient": 0.25, "specular": 0.50, "shininess": 16.0,
                "sss_tint": (0.40, 0.90, 1.00)}
    return {}


def _xray_mood_offset(params) -> tuple[float, float, float]:
    """Mood-driven RGB delta added on top of the xray skin palette.

    Reads live AU values off ``params`` and produces a small RGB
    shift that's mixed into the xray skin / back-of-head colours
    only — iris / teeth / sclera stay constant so the glow eyes and
    bone-white teeth read consistently.

    Mood mapping:
      * AU12 (smile)        → warm green-cyan tint (lift)
      * AU4  (brow lower)   → red shift (anger)
      * AU15 (corner drop)  → cool blue shift (sad)
      * AU5  (lid raise)    → pale boost (fear)
      * AU25 (jaw open)     → hot magenta core (open mouth → glow)
    """
    smile = max(0.0, float(getattr(params, "smile", 0.0)))
    brow_low = float(getattr(params, "brow_lower", 0.0))
    drop = float(getattr(params, "lip_corner_drop", 0.0))
    lid_raise = float(getattr(params, "upper_lid_raise", 0.0))
    jaw = float(getattr(params, "jaw_open", 0.0))
    inner_brow = float(getattr(params, "inner_brow_raise", 0.0))

    # Per-mood RGB delta in [0..1] colour space; small magnitudes so
    # the xray base reads as the dominant tone.
    happy = (-0.05, 0.18, 0.10)            # green-cyan lift
    angry = (0.30, -0.10, -0.05)            # red shift
    sad   = (-0.05, -0.05, 0.18)            # cool blue
    fear  = (0.10, 0.10, 0.10)              # pale (and inner brow raise sensitises it)
    open_ = (0.20, -0.05, 0.10)             # magenta-hot core when mouth opens

    sad_w = max(drop, inner_brow * 0.6)
    fear_w = lid_raise * 0.7
    dr = (smile * happy[0] + brow_low * angry[0] + sad_w * sad[0]
          + fear_w * fear[0] + jaw * open_[0])
    dg = (smile * happy[1] + brow_low * angry[1] + sad_w * sad[1]
          + fear_w * fear[1] + jaw * open_[1])
    db = (smile * happy[2] + brow_low * angry[2] + sad_w * sad[2]
          + fear_w * fear[2] + jaw * open_[2])
    return float(dr), float(dg), float(db)


def _material_palette(params) -> dict[str, tuple[float, float, float]]:
    """Build a per-material colour palette driven by the persona.

    Skin / eyelids derive from ``persona.skin_hue`` via HSV; iris
    from ``persona.eye_color`` (default brown); lips from
    ``persona.lip_color``; everything else fixed (teeth, sclera,
    lashes etc.).

    When ``persona.style`` is not ``"natural"`` we replace the
    palette wholesale with a sci-fi preset — neon / transparent /
    cyberpunk / xray. The ``xray`` preset is further modulated by
    live AU mood (red for anger, cool for sad, pale for fear, etc.).
    """
    style = getattr(params, "_persona_style", "natural")
    if style in _SCIFI_PALETTES:
        base = dict(_SCIFI_PALETTES[style])
        if style == "xray":
            dr, dg, db = _xray_mood_offset(params)
            for mat in ("M_Face", "M_BackHead"):
                r, g, b = base[mat]
                base[mat] = (
                    float(np.clip(r + dr, 0.0, 1.0)),
                    float(np.clip(g + dg, 0.0, 1.0)),
                    float(np.clip(b + db, 0.0, 1.0)),
                )
        # Slider overrides — apply HSV hue/sat/value shifts to skin
        # materials so the user can recolour the xray (or any sci-fi)
        # palette live without dropping out of style.
        slider_hue = getattr(params, "_slider_skin_hue", None)
        slider_sat = getattr(params, "_slider_skin_sat", None)
        slider_val = getattr(params, "_slider_skin_val", None)
        if slider_hue is not None or slider_sat is not None \
                or slider_val is not None:
            import colorsys
            for mat in ("M_Face", "M_BackHead", "M_HairCap", "M_GumsTongue"):
                if mat not in base:
                    continue
                r, g, b = base[mat]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                if slider_hue is not None:
                    h = (float(slider_hue) % 360.0) / 360.0
                if slider_sat is not None:
                    s = float(np.clip(s * (float(slider_sat) / 0.32),
                                          0.0, 1.0))
                if slider_val is not None:
                    v = float(np.clip(v * (float(slider_val) / 0.86),
                                          0.0, 1.0))
                r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
                base[mat] = (r2, g2, b2)
        # Eye glow colour override — applies to iris materials in
        # all sci-fi styles so the user can swap glow colour live.
        eye_hex = getattr(params, "_persona_eye_color", None)
        if eye_hex and style != "natural":
            iris_rgb = _hex_to_rgb_f(eye_hex)
            base["M_IrisLeft"] = iris_rgb
            base["M_IrisRight"] = iris_rgb
        return base

    sat = float(getattr(params, "_persona_skin_sat", 0.32))
    val = float(getattr(params, "_persona_skin_val", 0.86))
    hue = float(getattr(params, "skin_hue", 28.0))
    skin = _hsv_to_rgb(hue, sat, val)
    skin_dark = _hsv_to_rgb(hue, sat * 1.15, val * 0.86)
    eyelid = _hsv_to_rgb(hue, sat * 0.95, val * 0.94)
    lip = _hex_to_rgb_f(getattr(params, "lip_color", "#a44a4a"))
    iris = _hex_to_rgb_f(getattr(params, "_persona_eye_color", "#5a3818"))

    return {
        "M_Face":         skin,
        "M_BackHead":     skin_dark,
        "M_GumsTongue":   (lip[0] * 0.65, lip[1] * 0.45, lip[2] * 0.45),
        "M_Teeth":        (0.96, 0.93, 0.86),
        "M_ScleraLeft":   (0.96, 0.93, 0.86),
        "M_ScleraRight":  (0.96, 0.93, 0.86),
        "M_IrisLeft":     iris,
        "M_IrisRight":    iris,
        "M_LacrimalFluid":(0.94, 0.88, 0.82),
        "M_EyeBlend":     eyelid,
        "M_EyeOcclusion": (0.36, 0.24, 0.22),
        "M_EyeLashes":    (0.08, 0.06, 0.05),
        "M_HairCap":      _hex_to_rgb_f(
            getattr(params, "hair_color", "#2a1808")),
    }


# Subsurface scattering colour — softer warm tint, scaled down so it
# doesn't overwhelm the base skin.
_SSS_TINT = (0.62, 0.36, 0.30)


# Per-material emissive base (0..1). Drives the glowing-eye effect
# in sci-fi modes. Iris / sclera / lacrimal pop; teeth glow softly
# (xray bone-bright); rest are 0. Modulated per-frame by
# ``u_emit_pulse`` — natural style sets it to 0 → no glow.
_MATERIAL_EMISSIVE: dict[str, float] = {
    "M_IrisLeft":     1.0,
    "M_IrisRight":    1.0,
    "M_ScleraLeft":   0.55,
    "M_ScleraRight":  0.55,
    "M_LacrimalFluid":0.65,
    "M_Teeth":        0.30,
    "M_GumsTongue":   0.15,
}


# Per-style emissive pulse parameters. (base, amp, hz) — output
# pulse is base + amp * sin(2π hz t). Natural mode is (0, 0, 0).
_STYLE_PULSE: dict[str, tuple[float, float, float]] = {
    "neon":        (2.20, 0.50, 0.6),
    "transparent": (1.20, 0.30, 0.4),
    "cyberpunk":   (1.80, 0.50, 0.7),
    "xray":        (2.40, 0.70, 0.5),
}


def _emit_pulse_for(style: str, scale: float = 1.0) -> float:
    """Time-varying emissive pulse magnitude for the given style.

    Reads ``time.monotonic`` so the pulse advances frame-to-frame
    without needing the avatar tick to plumb a phase argument.
    ``scale`` multiplies the final amplitude — used by the live
    Eye-glow-strength slider.
    """
    if style not in _STYLE_PULSE:
        return 0.0
    base, amp, hz = _STYLE_PULSE[style]
    import math, time
    return float((base + amp * math.sin(2.0 * math.pi * hz * time.monotonic()))
                 * scale)


# Per-material specular intensity (A43). Eyes get high mirror-like
# specular for the wet-eye look; teeth a moderate gloss; skin
# subtle; lashes/inner-mouth near zero.
_MATERIAL_SPECULAR: dict[str, float] = {
    "M_Face":         0.30,    # subtle skin sheen
    "M_BackHead":     0.20,
    "M_GumsTongue":   0.55,    # wet inner mouth
    "M_Teeth":        0.65,    # enamel gloss
    "M_ScleraLeft":   0.95,    # wet eye
    "M_ScleraRight":  0.95,
    "M_IrisLeft":     0.90,    # iris under tear film
    "M_IrisRight":    0.90,
    "M_LacrimalFluid":1.00,    # pure liquid
    "M_EyeBlend":     0.40,    # eyelid skin
    "M_EyeOcclusion": 0.10,    # dark inner socket
    "M_EyeLashes":    0.05,    # matte
}


@lru_cache(maxsize=1)
def load_ict_model() -> ICTModel:
    path = _data_path()
    if not path.exists():
        raise MissingDependency(
            "ict-facekit data", "gpu",
            hint=(
                "Generate with `git clone "
                "https://github.com/USC-ICT/ICT-FaceKit /tmp/ICT-FaceKit && "
                "python -m tools.build_ict_blendshapes /tmp/ICT-FaceKit`."
            ),
        )
    data = np.load(path)
    names = data["names"].tolist()
    materials = data["materials"].tolist() if "materials" in data.files else []
    tri_mats = (data["tri_materials"].astype(np.int32)
                if "tri_materials" in data.files
                else np.zeros(len(data["triangles"]), dtype=np.int32))
    return ICTModel(
        vertices=data["vertices"].astype(np.float32),
        triangles=data["triangles"].astype(np.int32),
        deltas=data["deltas"].astype(np.float32),
        names=names,
        name_to_idx={n: i for i, n in enumerate(names)},
        tri_materials=tri_mats,
        materials=materials,
    )


# ── ARKit name ↔ ICT name bridge ─────────────────────────────────


_ARKIT_TO_ICT: dict[str, str] = {
    # ARKit uses camelCase (e.g. "browDownLeft"); ICT uses
    # underscore + L/R suffix (e.g. "browDown_L"). Most map 1:1
    # by lower-casing and replacing the L/R suffix.
    "browDownLeft": "browDown_L",
    "browDownRight": "browDown_R",
    "browInnerUp": "browInnerUp_L",   # ICT splits but only L exists in some sets
    "browOuterUpLeft": "browOuterUp_L",
    "browOuterUpRight": "browOuterUp_R",
    "cheekPuff": "cheekPuff_L",        # likewise
    "cheekSquintLeft": "cheekSquint_L",
    "cheekSquintRight": "cheekSquint_R",
    "eyeBlinkLeft": "eyeBlink_L",
    "eyeBlinkRight": "eyeBlink_R",
    "eyeLookDownLeft": "eyeLookDown_L",
    "eyeLookDownRight": "eyeLookDown_R",
    "eyeLookInLeft": "eyeLookIn_L",
    "eyeLookInRight": "eyeLookIn_R",
    "eyeLookOutLeft": "eyeLookOut_L",
    "eyeLookOutRight": "eyeLookOut_R",
    "eyeLookUpLeft": "eyeLookUp_L",
    "eyeLookUpRight": "eyeLookUp_R",
    "eyeSquintLeft": "eyeSquint_L",
    "eyeSquintRight": "eyeSquint_R",
    "eyeWideLeft": "eyeWide_L",
    "eyeWideRight": "eyeWide_R",
    "jawForward": "jawForward",
    "jawLeft": "jawLeft",
    "jawOpen": "jawOpen",
    "jawRight": "jawRight",
    "mouthClose": "mouthClose",
    "mouthDimpleLeft": "mouthDimple_L",
    "mouthDimpleRight": "mouthDimple_R",
    "mouthFrownLeft": "mouthFrown_L",
    "mouthFrownRight": "mouthFrown_R",
    "mouthFunnel": "mouthFunnel",
    "mouthLeft": "mouthLeft",
    "mouthLowerDownLeft": "mouthLowerDown_L",
    "mouthLowerDownRight": "mouthLowerDown_R",
    "mouthPressLeft": "mouthPress_L",
    "mouthPressRight": "mouthPress_R",
    "mouthPucker": "mouthPucker",
    "mouthRight": "mouthRight",
    "mouthRollLower": "mouthRollLower",
    "mouthRollUpper": "mouthRollUpper",
    "mouthShrugLower": "mouthShrugLower",
    "mouthShrugUpper": "mouthShrugUpper",
    "mouthSmileLeft": "mouthSmile_L",
    "mouthSmileRight": "mouthSmile_R",
    "mouthStretchLeft": "mouthStretch_L",
    "mouthStretchRight": "mouthStretch_R",
    "mouthUpperUpLeft": "mouthUpperUp_L",
    "mouthUpperUpRight": "mouthUpperUp_R",
    "noseSneerLeft": "noseSneer_L",
    "noseSneerRight": "noseSneer_R",
    "tongueOut": "tongueOut",
}


def _apply_camera_orbit(
    verts: np.ndarray, yaw: float, pitch: float,
    pivot: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate the whole scene around the mesh centroid.

    ``pivot`` overrides the centroid when transforming auxiliary
    meshes (tongue, hair) — pass the ICT mesh's centroid so the
    auxiliaries orbit around the head rather than around themselves.
    """
    if pivot is None:
        pivot = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    R = Ry @ Rx
    return ((verts - pivot) @ R.T + pivot).astype(np.float32)


def _apply_limb_rotation(
    verts: np.ndarray,
    pivot: np.ndarray,
    yaw: float, pitch: float, roll: float,
    weight: np.ndarray,
) -> np.ndarray:
    """Rotate verts around ``pivot`` weighted by per-vertex
    ``weight`` in [0, 1]. weight=1 fully rotates, weight=0 stays
    put, intermediate values blend — so the limb-to-torso boundary
    can have a smooth fall-off and triangles don't rip.
    """
    if (abs(yaw) < 1e-3 and abs(pitch) < 1e-3 and abs(roll) < 1e-3):
        return verts
    if not (weight > 1e-3).any():
        return verts
    cy_ = float(np.cos(yaw)); sy_ = float(np.sin(yaw))
    cp_ = float(np.cos(pitch)); sp_ = float(np.sin(pitch))
    cr_ = float(np.cos(roll)); sr_ = float(np.sin(roll))
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]],
                    dtype=np.float32)
    R = Ry @ Rx @ Rz
    diff = verts - pivot
    rotated = (diff @ R.T) + pivot
    w = weight[:, None].astype(np.float32)
    return (verts * (1.0 - w) + rotated * w).astype(np.float32)


def _arm_weight(verts: np.ndarray, side: int, shoulder_y: float,
                 wrist_floor_y: float, x_inner: float,
                 x_fade: float = 2.0,
                 y_fade_top: float = 1.5,
                 y_fade_bot: float = 1.5,
                 shoulder_offset: float = 2.0) -> np.ndarray:
    """Soft skinning weight for an arm segment.

    Tight defaults so arm rotations don't pick up torso pixels:
    sharp X transition at the arm/torso boundary (``x_fade=2``),
    and the top fade starts BELOW the shoulder line by
    ``shoulder_offset`` so the shoulder cap itself stays attached
    to the torso, not the arm.
    """
    sx = side * verts[:, 0]
    y = verts[:, 1]
    # X side mask — sharp transition at x_inner (arm/torso boundary).
    wx = np.clip((sx - x_inner) / max(1e-6, x_fade), 0.0, 1.0)
    wx = wx * wx * (3.0 - 2.0 * wx)
    # Y top: fade 0 above (shoulder_y - shoulder_offset) so arm
    # rotation doesn't lift the shoulder cap.
    top_edge = shoulder_y - shoulder_offset
    wy_top = 1.0 - np.clip((y - top_edge) / max(1e-6, y_fade_top), 0.0, 1.0)
    wy_top = wy_top * wy_top * (3.0 - 2.0 * wy_top)
    wy_bot = np.clip((y - wrist_floor_y) / max(1e-6, y_fade_bot), 0.0, 1.0)
    wy_bot = wy_bot * wy_bot * (3.0 - 2.0 * wy_bot)
    return (wx * wy_top * wy_bot).astype(np.float32)


def _leg_weight(verts: np.ndarray, side: int, hip_y: float,
                 ankle_floor_y: float, x_inner: float = 1.5,
                 x_fade: float = 1.5,
                 y_fade_top: float = 1.5,
                 y_fade_bot: float = 1.5,
                 hip_offset: float = 2.0) -> np.ndarray:
    """Soft skinning weight for a leg segment. Tight masks so
    the rotation doesn't pick up torso/hip pixels."""
    sx = side * verts[:, 0]
    y = verts[:, 1]
    # X mask: must be on this side of midline by at least x_inner.
    wx = np.clip((sx - x_inner) / max(1e-6, x_fade), 0.0, 1.0)
    wx = wx * wx * (3.0 - 2.0 * wx)
    # Y top: below hip by hip_offset so pelvis stays put.
    top_edge = hip_y - hip_offset
    wy_top = 1.0 - np.clip((y - top_edge) / max(1e-6, y_fade_top), 0.0, 1.0)
    wy_top = wy_top * wy_top * (3.0 - 2.0 * wy_top)
    wy_bot = np.clip((y - ankle_floor_y) / max(1e-6, y_fade_bot), 0.0, 1.0)
    wy_bot = wy_bot * wy_bot * (3.0 - 2.0 * wy_bot)
    return (wx * wy_top * wy_bot).astype(np.float32)


def _apply_body_rig(
    body_verts: np.ndarray, params,
    chin_y: float, head_h: float,
    parts: np.ndarray | None = None,
) -> np.ndarray:
    """Apply all per-joint limb rotations to body verts using
    soft skinning weights (smooth fall-off at limb boundaries).

    Joints applied in parent→child order so child motion compounds
    with parent. Each joint reads its rotation triple from ``params``
    via ``getattr`` with a 0 default — unset joints are no-ops.
    """
    if len(body_verts) == 0:
        return body_verts
    HH = head_h
    shoulder_y = chin_y - HH * 0.50
    elbow_y    = chin_y - HH * 1.85
    wrist_y    = chin_y - HH * 3.20
    hip_y      = chin_y - HH * 3.00
    knee_y     = chin_y - HH * 4.50
    ankle_y    = chin_y - HH * 6.20
    arm_x      = HH * 0.70
    leg_x      = HH * 0.30
    # Tight thresholds: arm verts have |X| > ~0.62*head_h; legs
    # have |X| > ~0.10*head_h (just past midline). Verts between
    # are TORSO and stay put during arm/leg rotations.
    arm_inner  = HH * 0.62
    leg_inner  = HH * 0.10
    body_floor = chin_y - HH * 7.0  # below feet

    out = body_verts
    # Body-part hard masks (1.0 only where vert is in the right
    # region, 0.0 elsewhere). Multiplied into soft skinning weights
    # so e.g. an arm rotation can NEVER move a torso vert even if
    # the soft weight's smoothstep would otherwise include it.
    if parts is not None:
        from faceview.vision.body_3d import (
            BP_LEFT_ARM, BP_RIGHT_ARM, BP_LEFT_LEG, BP_RIGHT_LEG,
        )
        mask_l_arm = (parts == BP_LEFT_ARM).astype(np.float32)
        mask_r_arm = (parts == BP_RIGHT_ARM).astype(np.float32)
        mask_l_leg = (parts == BP_LEFT_LEG).astype(np.float32)
        mask_r_leg = (parts == BP_RIGHT_LEG).astype(np.float32)
    else:
        # Fallback: no classification, rely entirely on soft weights.
        ones = np.ones(len(out), dtype=np.float32)
        mask_l_arm = mask_r_arm = mask_l_leg = mask_r_leg = ones
    # ── LEFT ARM ─────────────────────────────────────────────────
    w = _arm_weight(out, side=+1, shoulder_y=shoulder_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_l_arm
    out = _apply_limb_rotation(
        out, np.array([arm_x, shoulder_y, 0], dtype=np.float32),
        float(getattr(params, "l_shoulder_yaw", 0.0)),
        float(getattr(params, "l_shoulder_pitch", 0.0)),
        float(getattr(params, "l_shoulder_roll", 0.0)),
        w,
    )
    w = _arm_weight(out, side=+1, shoulder_y=elbow_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_l_arm
    out = _apply_limb_rotation(
        out, np.array([arm_x, elbow_y, 0], dtype=np.float32),
        float(getattr(params, "l_elbow_yaw", 0.0)),
        float(getattr(params, "l_elbow_pitch", 0.0)),
        float(getattr(params, "l_elbow_roll", 0.0)),
        w,
    )
    w = _arm_weight(out, side=+1, shoulder_y=wrist_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_l_arm
    out = _apply_limb_rotation(
        out, np.array([arm_x, wrist_y, 0], dtype=np.float32),
        float(getattr(params, "l_wrist_yaw", 0.0)),
        float(getattr(params, "l_wrist_pitch", 0.0)),
        float(getattr(params, "l_wrist_roll", 0.0)),
        w,
    )
    # ── RIGHT ARM ────────────────────────────────────────────────
    w = _arm_weight(out, side=-1, shoulder_y=shoulder_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_r_arm
    out = _apply_limb_rotation(
        out, np.array([-arm_x, shoulder_y, 0], dtype=np.float32),
        float(getattr(params, "r_shoulder_yaw", 0.0)),
        float(getattr(params, "r_shoulder_pitch", 0.0)),
        float(getattr(params, "r_shoulder_roll", 0.0)),
        w,
    )
    w = _arm_weight(out, side=-1, shoulder_y=elbow_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_r_arm
    out = _apply_limb_rotation(
        out, np.array([-arm_x, elbow_y, 0], dtype=np.float32),
        float(getattr(params, "r_elbow_yaw", 0.0)),
        float(getattr(params, "r_elbow_pitch", 0.0)),
        float(getattr(params, "r_elbow_roll", 0.0)),
        w,
    )
    w = _arm_weight(out, side=-1, shoulder_y=wrist_y,
                       wrist_floor_y=wrist_y - 6, x_inner=arm_inner)
    w = w * mask_r_arm
    out = _apply_limb_rotation(
        out, np.array([-arm_x, wrist_y, 0], dtype=np.float32),
        float(getattr(params, "r_wrist_yaw", 0.0)),
        float(getattr(params, "r_wrist_pitch", 0.0)),
        float(getattr(params, "r_wrist_roll", 0.0)),
        w,
    )
    # ── LEFT LEG ─────────────────────────────────────────────────
    w = _leg_weight(out, side=+1, hip_y=hip_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_l_leg
    out = _apply_limb_rotation(
        out, np.array([leg_x, hip_y, 0], dtype=np.float32),
        float(getattr(params, "l_hip_yaw", 0.0)),
        float(getattr(params, "l_hip_pitch", 0.0)),
        float(getattr(params, "l_hip_roll", 0.0)),
        w,
    )
    w = _leg_weight(out, side=+1, hip_y=knee_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_l_leg
    out = _apply_limb_rotation(
        out, np.array([leg_x, knee_y, 0], dtype=np.float32),
        float(getattr(params, "l_knee_yaw", 0.0)),
        float(getattr(params, "l_knee_pitch", 0.0)),
        float(getattr(params, "l_knee_roll", 0.0)),
        w,
    )
    w = _leg_weight(out, side=+1, hip_y=ankle_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_l_leg
    out = _apply_limb_rotation(
        out, np.array([leg_x, ankle_y, 0], dtype=np.float32),
        float(getattr(params, "l_ankle_yaw", 0.0)),
        float(getattr(params, "l_ankle_pitch", 0.0)),
        float(getattr(params, "l_ankle_roll", 0.0)),
        w,
    )
    # ── RIGHT LEG ────────────────────────────────────────────────
    w = _leg_weight(out, side=-1, hip_y=hip_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_r_leg
    out = _apply_limb_rotation(
        out, np.array([-leg_x, hip_y, 0], dtype=np.float32),
        float(getattr(params, "r_hip_yaw", 0.0)),
        float(getattr(params, "r_hip_pitch", 0.0)),
        float(getattr(params, "r_hip_roll", 0.0)),
        w,
    )
    w = _leg_weight(out, side=-1, hip_y=knee_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_r_leg
    out = _apply_limb_rotation(
        out, np.array([-leg_x, knee_y, 0], dtype=np.float32),
        float(getattr(params, "r_knee_yaw", 0.0)),
        float(getattr(params, "r_knee_pitch", 0.0)),
        float(getattr(params, "r_knee_roll", 0.0)),
        w,
    )
    w = _leg_weight(out, side=-1, hip_y=ankle_y, ankle_floor_y=body_floor,
                       x_inner=leg_inner)
    w = w * mask_r_leg
    out = _apply_limb_rotation(
        out, np.array([-leg_x, ankle_y, 0], dtype=np.float32),
        float(getattr(params, "r_ankle_yaw", 0.0)),
        float(getattr(params, "r_ankle_pitch", 0.0)),
        float(getattr(params, "r_ankle_roll", 0.0)),
        w,
    )
    return out


# Cervical vertebra rotation distribution from faceforge.
# Each entry is the CUMULATIVE rotation fraction at that vertebra
# level — Atlas (C1) carries 100 % of head rotation, T1 (top of
# thoracic spine) carries 0 %. Rotation accumulates from T1 up so
# the visible skin bends smoothly across all 7 cervical joints
# instead of stretching at a single pivot.
# Pitch + roll fractions are larger than yaw fractions because the
# cervical spine is more flexible in flexion/extension than rotation.
VERTEBRA_FRACTIONS_PITCH: tuple[float, ...] = (
    1.00, 0.98, 0.85, 0.55, 0.25, 0.10, 0.04, 0.015, 0.005,
    0.002, 0.0, 0.0,
)  # above-C1, C1, C2, C3, C4, C5, C6, C7, T1, T2, T3, T4
# Bend CONCENTRATED at the top of the neck (C1-C3) — close to the
# skull, where the head pivots in real anatomy. Mid-neck (C4-C5)
# carries less; lower neck (C6-T1) almost stationary so the base
# stays anchored to the body. Skin stretches across the C2-C4 band.
VERTEBRA_FRACTIONS_YAW: tuple[float, ...] = (
    1.00, 0.98, 0.55, 0.30, 0.15, 0.08, 0.03, 0.01, 0.003,
    0.001, 0.0, 0.0,
)

# Optional alternate cascade profiles selected via FACEVIEW_NOD_MODE.
# Each mode tweaks where the bend lives and whether the cascade
# result is post-anchored back to rest below a Y threshold.
_NOD_MODES: dict[str, dict] = {
    "current": dict(  # legacy — bend leaks into mid-neck (0.10 at C5)
        pitch=VERTEBRA_FRACTIONS_PITCH,
        yaw=VERTEBRA_FRACTIONS_YAW,
        fade=1.5,
        anchor_y_norm=None,
    ),
    "sharper": dict(
        # Concentrate bend at C1-C3 only. Verts below C4 stay put.
        pitch=(1.00, 0.95, 0.65, 0.20, 0.05, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.40, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0,
        anchor_y_norm=None,
    ),
    "spine_ripple": dict(
        # Sharp at top + tiny ripple through thoracic spine for
        # "flex passed down the spine" without visible base motion.
        pitch=(1.00, 0.95, 0.65, 0.25, 0.08, 0.02, 0.01, 0.008,
                 0.006, 0.004, 0.002, 0.0),
        yaw=(1.00, 0.95, 0.40, 0.12, 0.03, 0.01, 0.005, 0.003,
                 0.002, 0.001, 0.0, 0.0),
        fade=1.0,
        anchor_y_norm=None,
    ),
    "anchored": dict(
        # Legacy fractions + hard snap-to-rest below mid-neck.
        # Anything below y_norm = -0.30 (mid-neck level) reverts to
        # rest position regardless of what the cascade computed.
        pitch=VERTEBRA_FRACTIONS_PITCH,
        yaw=VERTEBRA_FRACTIONS_YAW,
        fade=1.5,
        anchor_y_norm=-0.30,
    ),
    "sharp_anchored": dict(
        # Sharper profile + anchor — belt-and-braces.
        pitch=(1.00, 0.95, 0.65, 0.20, 0.05, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.40, 0.10, 0.02, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0,
        anchor_y_norm=-0.25,
    ),
    "flex_anchored": dict(
        # Big bend distributed across C1-C4 for visibly curving
        # neck, plus a hard anchor at y_norm=-0.30 to keep the base
        # stationary. User feedback driver: "neck not flexing enough
        # and base still moving too much".
        # Deltas: skull-C1 0.05, C1-C2 0.30, C2-C3 0.35, C3-C4 0.22,
        # C4-C5 0.08, below = 0. Sum = 1.00.
        pitch=(1.00, 0.95, 0.65, 0.30, 0.08, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.95, 0.45, 0.20, 0.04, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.5,
        anchor_y_norm=-0.30,
    ),
    # ------------------------------------------------------------------
    # Anatomical-pivot family: move pivot_z BACKWARD (negative Z, into
    # the back of the neck) so the chin's front-to-back sweep arc is
    # large. Combined with smoothly-distributed cumulative rotation so
    # the neck visibly CURVES rather than blocking with the skull.
    #
    # pivot_z_offset is added to the per-disc pivot's Z and is in
    # head_h units (i.e. -0.15 ≈ 3.1 ICT units back of centerline).
    # ------------------------------------------------------------------
    "curve_back_pivot": dict(
        # Smooth gradient C7 (=0) up to chin (=1), pivot pushed
        # 0.20 head_h back into the neck. Tight anchor at -0.25.
        pitch=(1.00, 0.92, 0.80, 0.62, 0.40, 0.22, 0.08, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.92, 0.70, 0.45, 0.25, 0.12, 0.04, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0,
        anchor_y_norm=-0.25,
        anchor_fade_band=0.10,
        pivot_z_offset=-0.20,
    ),
    "low_pivot_block": dict(
        # Rotation CONCENTRATED at C4-C6 disc band, pivot back AND
        # low. Chin sweeps the WIDEST front-to-back arc. Tight
        # anchor at -0.22 with narrow fade band so the base locks.
        pitch=(1.00, 0.99, 0.97, 0.93, 0.80, 0.50, 0.15, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.99, 0.95, 0.85, 0.65, 0.35, 0.10, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=0.8,
        anchor_y_norm=-0.22,
        anchor_fade_band=0.08,
        pivot_z_offset=-0.25,
    ),
    "neck_curve_strict": dict(
        # Maximum-curvature distributed cascade with strict anchor
        # at -0.22 and pivot 0.18 back.
        pitch=(1.00, 0.88, 0.72, 0.52, 0.30, 0.12, 0.03, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        yaw=(1.00, 0.88, 0.62, 0.38, 0.18, 0.06, 0.02, 0.0, 0.0,
                 0.0, 0.0, 0.0),
        fade=1.0,
        anchor_y_norm=-0.22,
        anchor_fade_band=0.10,
        pivot_z_offset=-0.18,
    ),
    # ------------------------------------------------------------------
    # Cranium-only modes: SINGLE rotation around an ear-level pivot
    # with everything below the ear (chin / jaw / neck / body) snapped
    # to rest. Driven by the user's spec: "no cyan at the back of the
    # head or red at the front of the head, below the bottom of the
    # ear, only above the ear".
    #
    # When `single_pivot_y_norm` is set, the cascade is REPLACED by
    # one rotation around (pivot_z_offset, single_pivot_y_norm *
    # head_h + chin_y). The cumul/yaw/fade arrays are ignored. The
    # anchor still runs to clamp anything below `anchor_y_norm`.
    # ------------------------------------------------------------------
    "cranium_only": dict(
        # Pivot at ear bottom (y_norm = +0.30) and back of skull.
        # Snap below-ear verts to rest. Result: only the cranium
        # above the ear rotates; the face below the ear (jaw, chin,
        # neck, body) is held in place.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=+0.28,
        anchor_fade_band=0.04,
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.30,
    ),
    "cranium_high_pivot": dict(
        # Pivot higher up (atlanto-occipital level, y_norm = +0.40).
        # Smaller cranium rotates; lower face stays still.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=+0.38,
        anchor_fade_band=0.04,
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.40,
    ),
    "cranium_soft_seam": dict(
        # Same ear-level pivot but wider blend band so the seam at
        # the ear doesn't show a sharp crease.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=+0.20,
        anchor_fade_band=0.10,
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.30,
    ),
    # ------------------------------------------------------------------
    # Whole-head block + neck stretch: the entire head (skull, jaw,
    # face, chin) rotates as ONE rigid block around an ear-level
    # pivot, and the NECK smoothly stretches/compresses to absorb
    # the motion. Body below the neck base stays still.
    #
    # User spec: "the entire skull, the jaw and the entire head has
    # to move as one. The neck is the region that has to stretch and
    # compress - not the face and head".
    # ------------------------------------------------------------------
    "head_block_neck_stretch": dict(
        # Entire HEAD rotates RIGIDLY around the ear pivot. The
        # rigid zone extends from y_norm ≥ -0.10 (covers the whole
        # head + lower jaw + submental area) up to the crown. Verts
        # in the throat/neck (y_norm -0.30 to -0.10) smoothstep
        # blend = neck stretches. Body below -0.30 stays at rest.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=-0.30,         # bottom of fade = neck base
        anchor_fade_band=0.20,       # top of fade at -0.10 (jaw)
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.30,
    ),
    "head_block_short_neck": dict(
        # Same rigid head boundary but narrower stretch zone — neck
        # deforms over a smaller band.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=-0.22,
        anchor_fade_band=0.12,
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.30,
    ),
    "head_block_long_neck": dict(
        # Long stretch zone reaches into the upper torso — gentlest
        # neck deformation. Useful for the cleanest "neck flexes"
        # appearance at the cost of some upper-clavicle motion.
        pitch=(1.0,) * 12, yaw=(1.0,) * 12,
        fade=0.5,
        anchor_y_norm=-0.50,
        anchor_fade_band=0.40,
        pivot_z_offset=-0.20,
        single_pivot_y_norm=+0.30,
    ),
}


def _resolve_nod_mode():
    """Return (pitch, yaw, fade, anchor_y_norm, anchor_fade_band,
    pivot_z_offset, single_pivot_y_norm) for the active mode.

    If `single_pivot_y_norm` is set, the cascade is REPLACED by a
    single rotation around that Y (head_h units relative to chin)
    plus the pivot_z_offset.
    """
    import os as _os
    name = (_os.environ.get("FACEVIEW_NOD_MODE",
                              "head_block_neck_stretch").strip()
            or "head_block_neck_stretch")
    cfg = _NOD_MODES.get(name, _NOD_MODES["head_block_neck_stretch"])
    return (cfg["pitch"], cfg["yaw"], cfg["fade"],
            cfg["anchor_y_norm"],
            cfg.get("anchor_fade_band", 0.15),
            cfg.get("pivot_z_offset", 0.0),
            cfg.get("single_pivot_y_norm", None))
VERTEBRA_Y_FRACS: tuple[float, ...] = (
    +0.05,  # above C1 (skull)
     0.00,  # C1 atlas
    -0.07,  # C2
    -0.13,  # C3
    -0.20,  # C4 (≈ body's morph top)
    -0.27,  # C5
    -0.33,  # C6
    -0.40,  # C7
    -0.50,  # T1
    -0.62,  # T2 (helper, in upper torso)
    -0.75,  # T3 (helper, in mid-chest)
    -0.90,  # T4 (helper, near sternum)
)


def _vertebra_weight_curve(verts_y: np.ndarray, chin_y: float,
                              head_h: float, axis: str) -> np.ndarray:
    """Per-vertex rotation fraction along the cervical spine.

    Linearly interpolates faceforge's vertebra rotation fractions
    across our vertex Y positions. Verts above C1 get fraction 1.0
    (full head rotation). Verts at or below T1 get 0. In between,
    smooth piecewise-linear blend that matches anatomical
    distribution.
    """
    fractions = (VERTEBRA_FRACTIONS_PITCH if axis != "yaw"
                    else VERTEBRA_FRACTIONS_YAW)
    # Vertebra Y positions in absolute ICT frame
    vert_ys = np.array([chin_y + f * head_h for f in VERTEBRA_Y_FRACS],
                          dtype=np.float32)
    fracs = np.array(fractions, dtype=np.float32)
    # np.interp expects xp in increasing order — reverse since we go top→bottom
    xp = vert_ys[::-1]
    fp = fracs[::-1]
    return np.interp(verts_y, xp, fp).astype(np.float32)


def _apply_cervical_cascade(
    verts: np.ndarray, yaw: float, pitch: float, roll: float,
    chin_y: float, head_h: float, pivot_z: float = 0.0,
) -> np.ndarray:
    """Apply head rotation as 7 sequential cervical-disc rotations.

    Each cervical intervertebral disc is its OWN pivot — when the
    head rotates, every disc contributes a small fraction of the
    total angle. The chin (above C1) accumulates all 7, mid-neck
    accumulates the bottom 4, T1-level verts accumulate 0. Because
    each rotation is around a different pivot, the path traced by
    the chin is a chain of small arcs — the neck VISIBLY CURVES,
    not just rotates as a single rigid block.

    Mirrors faceforge's cervical-vertebra distribution but as a
    sequence of rigid rotations on flat vertex arrays instead of
    scene-graph quaternions.
    """
    if (abs(yaw) < 1e-3 and abs(pitch) < 1e-3 and abs(roll) < 1e-3):
        return verts

    (pitch_fracs, yaw_fracs, fade, anchor_y_norm,
     anchor_fade_band, pivot_z_offset,
     single_pivot_y_norm) = _resolve_nod_mode()
    rest_verts = verts  # keep original for optional post-anchor blend
    out = verts.copy()
    # Anatomical-pivot offset: shift the per-disc pivot back into the
    # neck (negative Z). pivot_z_offset is in head_h units.
    pivot_z_eff = float(pivot_z) + float(pivot_z_offset) * float(head_h)

    # SINGLE-PIVOT path: cranium-only modes do one rotation around
    # the specified Y and skip the per-disc cascade entirely. The
    # anchor below still runs to clamp lower regions to rest.
    if single_pivot_y_norm is not None:
        pivot_y_abs = chin_y + float(single_pivot_y_norm) * head_h
        cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
        cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
        cr_, sr_ = float(np.cos(roll)), float(np.sin(roll))
        Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                        dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                        dtype=np.float32)
        Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]],
                        dtype=np.float32)
        R = Ry @ Rx @ Rz
        pivot = np.array([0.0, pivot_y_abs, pivot_z_eff],
                          dtype=np.float32)
        diff = out - pivot
        out = ((diff @ R.T) + pivot).astype(np.float32)
        # Fall through to the anchor block below.
        if anchor_y_norm is not None:
            fade_band_norm = float(anchor_fade_band)
            anchor_y_abs = chin_y + anchor_y_norm * head_h
            fade_abs = fade_band_norm * head_h
            y = rest_verts[:, 1]
            t_anchor = np.clip(
                (y - anchor_y_abs) / max(1e-6, fade_abs), 0.0, 1.0)
            w_keep = (t_anchor * t_anchor *
                        (3.0 - 2.0 * t_anchor)).astype(np.float32)
            out = (rest_verts * (1.0 - w_keep[:, None])
                      + out * w_keep[:, None]).astype(np.float32)
        return out

    vert_ys = [chin_y + f * head_h for f in VERTEBRA_Y_FRACS]
    pitch_cumul = list(pitch_fracs)
    yaw_cumul = list(yaw_fracs)

    n = len(vert_ys)
    # Apply BOTTOM disc first (T1-C7), TOP disc last (C1-aboveC1).
    # Each disc's rotation is the DELTA between its two cumulative
    # fractions.
    for i in range(n - 2, -1, -1):
        disc_y = (vert_ys[i] + vert_ys[i + 1]) / 2.0
        d_pitch = pitch_cumul[i] - pitch_cumul[i + 1]
        d_yaw = yaw_cumul[i] - yaw_cumul[i + 1]
        d_roll = pitch_cumul[i] - pitch_cumul[i + 1]  # roll = pitch curve
        if abs(d_pitch) < 1e-4 and abs(d_yaw) < 1e-4 and abs(d_roll) < 1e-4:
            continue

        ry = float(yaw * d_yaw)
        rp = float(pitch * d_pitch)
        rr = float(roll * d_roll)
        cy_, sy_ = float(np.cos(ry)), float(np.sin(ry))
        cp_, sp_ = float(np.cos(rp)), float(np.sin(rp))
        cr_, sr_ = float(np.cos(rr)), float(np.sin(rr))
        Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                        dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                        dtype=np.float32)
        Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]],
                        dtype=np.float32)
        R = Ry @ Rx @ Rz

        # Smoothstep weight: full ABOVE the disc, 0 BELOW. Per-mode
        # fade band width controls how sharply each disc's rotation
        # decays through neighbouring verts.
        y = out[:, 1]
        t = np.clip((y - (disc_y - fade)) / max(1e-6, fade),
                       0.0, 1.0)
        w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

        pivot = np.array([0.0, disc_y, pivot_z_eff], dtype=np.float32)
        diff = out - pivot
        rotated = (diff @ R.T) + pivot
        out = (out * (1.0 - w[:, None])
                  + rotated * w[:, None]).astype(np.float32)

    # Optional post-anchor: below `anchor_y_norm * head_h + chin_y`,
    # blend results back toward rest position so the neck base and
    # upper torso stay anchored regardless of cumulative leak from
    # the cascade. Smoothstep band sits ABOVE the threshold so the
    # transition is smooth and not a sharp seam.
    if anchor_y_norm is not None:
        fade_band_norm = float(anchor_fade_band)
        anchor_y_abs = chin_y + anchor_y_norm * head_h
        fade_abs = fade_band_norm * head_h
        y = rest_verts[:, 1]
        t_anchor = np.clip(
            (y - anchor_y_abs) / max(1e-6, fade_abs), 0.0, 1.0)
        w_keep = (t_anchor * t_anchor *
                    (3.0 - 2.0 * t_anchor)).astype(np.float32)
        out = (rest_verts * (1.0 - w_keep[:, None])
                  + out * w_keep[:, None]).astype(np.float32)

    return out


def _apply_neck_rotation(
    verts: np.ndarray, yaw: float, pitch: float, roll: float = 0.0,
    *, ref_verts: np.ndarray | None = None,
    y_neck_abs: float | None = None,
    y_head_abs: float | None = None,
    tracking_amp: float = 0.0,
    tracking_decay: float = 4.0,
    pivot_z: float = 0.0,
    cervical_chin_y: float | None = None,
    cervical_head_h: float | None = None,
) -> np.ndarray:
    """Per-vertex Y-weighted skinning around the neck pivot.

    Above ``y_head`` get full Ry @ Rx @ Rz rotation (yaw / pitch /
    roll). Between ``y_neck`` (pivot) and ``y_head`` get smoothstep
    transition. Below the pivot, weight is 0.

    Default thresholds are derived from the ref-mesh y-range
    (28 % / 38 % of span). Callers can pass ``y_neck_abs`` /
    ``y_head_abs`` to use absolute Y values instead — used to
    extend the rotation band down through the body's torso so the
    head + neck + upper body move together as one unit.
    """
    if abs(yaw) < 1e-3 and abs(pitch) < 1e-3 and abs(roll) < 1e-3:
        return verts

    if y_neck_abs is not None and y_head_abs is not None:
        y_neck = float(y_neck_abs)
        y_head = float(y_head_abs)
    else:
        src = ref_verts if ref_verts is not None else verts
        y_min = float(src[:, 1].min())
        y_max = float(src[:, 1].max())
        span = max(1e-6, y_max - y_min)
        y_neck = y_min + span * 0.30
        y_head = y_min + span * 0.42
    band_h = max(1e-6, y_head - y_neck)

    y = verts[:, 1]
    # Cervical vertebra fraction curve takes priority — distributes
    # rotation across 7 cervical joints (faceforge-style) so the
    # neck bends smoothly without single-pivot skin stretch.
    if cervical_chin_y is not None and cervical_head_h is not None:
        axis_for_curve = "pitch" if abs(pitch) >= abs(yaw) else "yaw"
        w = _vertebra_weight_curve(
            y, cervical_chin_y, cervical_head_h, axis=axis_for_curve)
    else:
        t_raw = (y - y_neck) / band_h
        t_above = np.clip(t_raw, 0.0, 1.0)
        ss = t_above * t_above * (3.0 - 2.0 * t_above)
        if tracking_amp > 0.0:
            w_above = float(tracking_amp) + (1.0 - float(tracking_amp)) * ss
            dist_below = np.maximum(0.0, -t_raw * band_h)
            decay_scale = max(1e-6, float(tracking_decay))
            w_below = float(tracking_amp) * np.exp(-dist_below / decay_scale)
            w = np.where(t_raw >= 0.0, w_above, w_below).astype(np.float32)
        else:
            w = ss.astype(np.float32)

    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    cr_, sr_ = float(np.cos(roll)), float(np.sin(roll))
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    Rz = np.array([[cr_, -sr_, 0], [sr_, cr_, 0], [0, 0, 1]],
                    dtype=np.float32)
    R = Ry @ Rx @ Rz

    pivot = np.array([0.0, y_neck, float(pivot_z)], dtype=np.float32)
    diff = verts - pivot
    rotated = (diff @ R.T) + pivot
    return (verts * (1.0 - w[:, None])
              + rotated * w[:, None]).astype(np.float32)


def _project_features(
    model, verts: np.ndarray,
    yaw: float, pitch: float, size: tuple[int, int],
    bbox_verts: np.ndarray | None = None,
    scale_multiplier: float = 1.0,
    focus_y: float = 0.0,
) -> dict[str, tuple[float, float]]:
    """Project key ICT feature centroids to pixel space.

    Mirrors the renderer's exact MVP (centre + scale + flip + ry @ rx
    + aspect-correct X), then maps each named feature's vertex
    centroid to (x, y) pixel coordinates. Used by PostFX overlays
    (tears, blush, heart-eyes, sweat-drops) so they land on the
    actual face features even after pose/morph changes.

    Feature points sourced from material-tagged vertices (iris L/R)
    and from blendshape-affected vertices (cheek, brow, forehead,
    mouth corners, chin).

    ``bbox_verts`` lets callers override the bbox used to compute
    centre + fit-scale (e.g. when a body mesh is being rendered the
    bbox spans head + body, so feature anchors must use the same
    bbox or they fall outside the rendered head). ``scale_multiplier``
    further scales (used for camera zoom).
    """
    w, h = size
    aspect = float(h) / float(w) if w > 0 else 1.0

    bbox_src = bbox_verts if bbox_verts is not None else verts
    vmin = bbox_src.min(axis=0)
    vmax = bbox_src.max(axis=0)
    centre = (vmin + vmax) / 2.0
    if abs(focus_y) > 1e-3:
        y_span = float(vmax[1] - vmin[1])
        centre = centre + np.array([0.0, float(focus_y) * y_span * 0.5, 0.0],
                                       dtype=centre.dtype)
    span = float(np.linalg.norm(vmax - vmin))
    scale = (1.6 / max(span, 1e-6)) * float(scale_multiplier)

    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    flip = np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    R = ry @ rx @ flip
    S = np.diag([scale * aspect, scale, scale]).astype(np.float32)

    def project(p3: np.ndarray) -> tuple[float, float]:
        v = R @ (S @ (p3 - centre))
        x_pix = (v[0] + 1.0) / 2.0 * w
        y_pix = (1.0 - (v[1] + 1.0) / 2.0) * h
        return float(x_pix), float(y_pix)

    out: dict[str, tuple[float, float]] = {}
    # Eye centres via M_IrisLeft / M_IrisRight materials.
    iris_l_idx = next((i for i, n in enumerate(model.materials)
                          if n == "M_IrisLeft"), -1)
    iris_r_idx = next((i for i, n in enumerate(model.materials)
                          if n == "M_IrisRight"), -1)
    iris_v: dict[str, list[int]] = {"L": [], "R": []}
    for ti, mi in enumerate(model.tri_materials):
        tag = "L" if mi == iris_l_idx else "R" if mi == iris_r_idx else None
        if tag is not None:
            for v in model.triangles[ti]:
                iris_v[tag].append(int(v))
    if iris_v["L"]:
        out["eye_L"] = project(verts[np.unique(iris_v["L"])].mean(axis=0))
    if iris_v["R"]:
        out["eye_R"] = project(verts[np.unique(iris_v["R"])].mean(axis=0))

    # Helper: top-K verts by blendshape delta magnitude → centroid
    # in deformed space.
    def from_blendshape(name: str, top_k: int = 60) -> tuple[float, float] | None:
        idx = model.name_to_idx.get(name)
        if idx is None:
            return None
        mags = np.linalg.norm(model.deltas[idx], axis=1)
        top = np.argsort(-mags)[:top_k]
        return project(verts[top].mean(axis=0))

    brow_l = from_blendshape("browOuterUp_L")
    brow_r = from_blendshape("browOuterUp_R")
    if brow_l:
        out["brow_L"] = brow_l
    if brow_r:
        out["brow_R"] = brow_r

    fh = from_blendshape("browInnerUp_L", top_k=80)
    if fh:
        # Move ~40 px above the inner-brow centroid for "above forehead".
        out["forehead"] = (fh[0], max(0.0, fh[1] - 40.0))

    # Mouth centroid + corners — must be computed BEFORE cheek anchor
    # since cheek is derived geometrically from eye+mouth_corner.
    mouth_c = from_blendshape("mouthClose", top_k=80)
    if mouth_c:
        out["mouth"] = mouth_c
    mc_l = from_blendshape("mouthSmile_L")
    mc_r = from_blendshape("mouthSmile_R")
    if mc_l:
        out["mouth_corner_L"] = mc_l
    if mc_r:
        out["mouth_corner_R"] = mc_r

    # Cheek apple — derive geometrically. The apple sits ~30 % down
    # from the eye toward the mouth corner. ICT blendshape regions
    # for cheekPuff/Squint cluster around the mouth corners
    # themselves, which is too low for blush placement.
    def _lerp(p1, p2, t):
        return (p1[0] * (1 - t) + p2[0] * t,
                p1[1] * (1 - t) + p2[1] * t)
    if "eye_L" in out and "mouth_corner_L" in out:
        out["cheek_L"] = _lerp(out["eye_L"], out["mouth_corner_L"], 0.30)
    if "eye_R" in out and "mouth_corner_R" in out:
        out["cheek_R"] = _lerp(out["eye_R"], out["mouth_corner_R"], 0.30)

    chin = from_blendshape("jawOpen", top_k=80)
    if chin:
        out["chin"] = chin

    # Head pixel-size summary so PostFX overlays (blush, !? marks,
    # sweat drops, anger steam, sparkles) can scale their effect
    # sizes to the rendered head — fixed pixel sizes look too big
    # when body-mode shrinks the head in the frame.
    if "chin" in out and "forehead" in out:
        out["_head_height_px"] = float(
            abs(out["chin"][1] - out["forehead"][1]))
    if "eye_L" in out and "eye_R" in out:
        dx = out["eye_R"][0] - out["eye_L"][0]
        dy = out["eye_R"][1] - out["eye_L"][1]
        out["_eye_distance_px"] = float((dx * dx + dy * dy) ** 0.5)

    return out


def apply_blendshapes(
    model: ICTModel,
    arkit_coefs: dict[str, float],
    base: np.ndarray | None = None,
    vert_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return deformed vertex positions given ARKit-named coefficients.

    ``base`` overrides the starting positions (default: model neutral).
    ``vert_mask`` (shape (N,)) scales each vertex's accumulated delta
    so deformations can be confined to a region — used to keep ICT's
    bust / lower neck still during speech blendshapes so the body
    avatar's clavicle line doesn't break through.
    """
    out = (base.copy() if base is not None else model.vertices.copy())
    for arkit_name, value in arkit_coefs.items():
        if value == 0:
            continue
        ict_name = _ARKIT_TO_ICT.get(arkit_name, arkit_name)
        idx = model.name_to_idx.get(ict_name)
        if idx is None:
            continue
        delta = model.deltas[idx] * float(value)
        if vert_mask is not None:
            delta = delta * vert_mask[:, None]
        out += delta
    return out


_LAST_SEPARATION_LOG_T: float = 0.0


def _check_head_body_separation(
    head_verts: np.ndarray, body_verts: np.ndarray | None,
    threshold: float = 12.0, throttle_s: float = 0.5,
) -> dict | None:
    """Diagnostic — measures the gap between ICT's lower jaw / neck
    region and the body mesh, then logs a warning if it exceeds
    ``threshold`` ICT units (≈ 1/5 of a head height — the natural
    chin/neck overlap distance).

    Specifically: takes the ICT verts in a band just below the chin
    (the lower-jaw / throat ring), finds the closest body vert for
    each, and reports the *median* nearest-neighbour distance — a
    robust gap metric that's stable across pose changes.

    Throttled so logs don't flood at frame rate.
    """
    if body_verts is None or len(body_verts) == 0:
        return None
    chin_y = float(head_verts[ICT_CHIN_VERT_IDX_NUM, 1])
    # Lower-jaw / throat band: chin level down to chin - 4 units.
    band_mask = ((head_verts[:, 1] >= chin_y - 4.0)
                   & (head_verts[:, 1] <= chin_y + 0.5))
    if not band_mask.any():
        return None
    jaw_pts = head_verts[band_mask]
    # For efficiency, sub-sample to ~200 points.
    if len(jaw_pts) > 200:
        idx = np.linspace(0, len(jaw_pts) - 1, 200).astype(np.int32)
        jaw_pts = jaw_pts[idx]
    # Nearest-body distance for each jaw point (broadcast).
    # body_verts: (M, 3); jaw_pts: (K, 3). dists: (K, M)
    dists = np.linalg.norm(
        jaw_pts[:, None, :] - body_verts[None, :, :], axis=2)
    nearest = dists.min(axis=1)  # closest body vert for each jaw pt
    median_gap = float(np.median(nearest))
    max_gap = float(nearest.max())
    info = {
        "median_gap": median_gap,
        "max_gap": max_gap,
        "threshold": float(threshold),
    }
    if median_gap > threshold:
        import time as _time
        global _LAST_SEPARATION_LOG_T
        now = _time.monotonic()
        if now - _LAST_SEPARATION_LOG_T > throttle_s:
            _LAST_SEPARATION_LOG_T = now
            print(
                f"[separation] median jaw-to-body gap = {median_gap:.2f} "
                f"ICT units (threshold {threshold:.1f}, max {max_gap:.2f})"
            )
        return info
    return None


@lru_cache(maxsize=1)
def _bust_isolation_mask() -> np.ndarray:
    """Per-vertex deformation weight: 1.0 above chin, fades to 0
    around the lower neck, 0 in the bust. Multiplied into ARKit
    blendshape deltas so jaw / mouth / brow expressions don't ripple
    down into the body's clavicle region.

    Cached on the neutral mesh (lru_cache) — values are stable
    across frames since the mask is built from neutral vertex Ys.
    """
    model = load_ict_model()
    y = model.vertices[:, 1]
    # Chin tip ≈ -6.47, lower neck ≈ -8.5. Mask 1 above chin,
    # smoothstep down to 0 by lower neck.
    chin_y = float(y[ICT_CHIN_VERT_IDX_NUM])
    fade_top = chin_y       # 100 % above this
    fade_bot = chin_y - 2.5  # 0 % below this
    band = max(1e-6, fade_top - fade_bot)
    t = np.clip((y - fade_bot) / band, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


# ICT-FaceKit chin tip vertex (peak-displacement vert of jawOpen).
ICT_CHIN_VERT_IDX_NUM: int = 964


# ── Renderer ──────────────────────────────────────────────────────


def render_face_ict(
    params,
    size: tuple[int, int] = (480, 480),
) -> np.ndarray:
    """Render the ICT face, deformed by params' AU values translated to ARKit."""
    from faceview.vision.anatomy import face_params_to_au_values
    from faceview.vision.arkit_blendshapes import au_to_arkit_values

    model = load_ict_model()
    au_values = face_params_to_au_values(params)
    arkit_coefs = au_to_arkit_values(au_values)

    # Layer identity (PCA) coefficients on top of the expression
    # coefficients — identity reshapes the underlying head, expression
    # animates it. ``identity_weights`` lives on FaceParams (set by
    # apply_persona) and is keyed by ICT identity names like
    # ``identity001``. Skip non-float entries (e.g. ``mh_target``
    # which is meant for the makehuman mode).
    raw_iw = getattr(params, "identity_weights", {}) or {}
    identity_w = {k: float(v) for k, v in raw_iw.items()
                   if isinstance(v, (int, float))}
    # Direct blendshape pass-through — slider sliders + PreFX warps
    # may set named ICT blendshapes (PupilDilate_L/R, jawForward,
    # mouthLeft, mouthRight, mouthFunnel, mouthClose, cheekPuff_L/R)
    # that don't fit the 12-AU vocabulary. These merge in last so
    # they can override AU-derived values when user wants explicit
    # control.
    direct = getattr(params, "direct_blendshapes", None) or {}
    direct_clean = {k: float(v) for k, v in direct.items()
                       if isinstance(v, (int, float))}
    show_body = bool(getattr(params, "_show_body", False))
    if show_body:
        # ICT strip — kink-fix approaches selected via env var
        # FACEVIEW_KINK_FIX. See `tools/eval_kink_fixes.py` for
        # what each approach does. ``default`` keeps ICT down to
        # body_top - 1 (1 unit overlap) — the prior behaviour.
        from faceview.vision.body_3d import ICT_CHIN_VERT_IDX as _CCI
        chin_y_for_strip = float(model.vertices[_CCI, 1])
        head_h_for_strip = float(model.vertices[:, 1].max() - chin_y_for_strip)
        body_top_y = chin_y_for_strip - head_h_for_strip * 0.20
        approach = os.environ.get("FACEVIEW_KINK_FIX", "default")
        if approach == "no_overlap":
            strip_y = body_top_y  # ICT stops exactly at body's top
        elif approach == "deep_overlap":
            strip_y = body_top_y - 4.0  # ICT extends 4 units below
        elif approach == "below_chin":
            strip_y = chin_y_for_strip - 1.0  # body provides neck
        elif approach == "no_strip":
            strip_y = float("-inf")  # ICT keeps full mesh
        elif approach == "deep_jaw":
            strip_y = chin_y_for_strip - 3.0  # ICT keeps chin + jaw + ½ neck
        else:  # "default"
            strip_y = body_top_y - 1.0
        tri_cy = model.vertices[model.triangles].mean(axis=1)[:, 1]
        head_only_tris = model.triangles[tri_cy > strip_y]
    else:
        head_only_tris = model.triangles

    # Apply identity blends first (no mask — full mesh shapes by
    # identity, including bust width). Then apply expression /
    # direct-override blends. When the body avatar is shown those
    # last two are confined to the head + lower-jaw region so the
    # bust / lower neck stays at its identity-shaped position and
    # doesn't ripple-deform through the body's clavicle skin during
    # speech / effects.
    verts = apply_blendshapes(model, identity_w)
    expr_coefs = {**arkit_coefs, **direct_clean}
    if show_body:
        verts = apply_blendshapes(model, expr_coefs, base=verts,
                                     vert_mask=_bust_isolation_mask())
    else:
        verts = apply_blendshapes(model, expr_coefs, base=verts)

    # Two-tier rig: BODY rotation (whole upper body around hip) +
    # HEAD rotation (head around neck base). Effects pick the tier
    # they want — head_shake / head_tilt route to head; body_lean /
    # body_bow / body_twist route to body; head_nod and head_recoil
    # use both for natural full-figure motion.
    head_yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    head_pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    head_roll = float(getattr(params, "roll", 0.0)) * 0.6
    body_yaw = float(getattr(params, "body_yaw", 0.0))
    body_pitch = float(getattr(params, "body_pitch", 0.0))
    body_roll = float(getattr(params, "body_roll", 0.0))
    cam_yaw = float(getattr(params, "_camera_yaw", 0.0))
    cam_pitch = float(getattr(params, "_camera_pitch", 0.0))

    # Joint Y positions in ICT frame (derived from the live deformed
    # mesh so identity blends are honoured).
    chin_y = float(verts[ICT_CHIN_VERT_IDX_NUM, 1])
    crown_y = float(verts[:, 1].max())
    head_h_ict = max(1.0, crown_y - chin_y)
    if show_body:
        # Body tier — pivot at hip (3 head-heights below chin per
        # Vitruvian canon), full rotation above the WAIST line so
        # head, shoulders, chest, navel all rotate as a single
        # block (they all need the same weight to stay attached).
        # Smoothstep band sits across the waist; legs stay planted.
        body_hip_y = chin_y - head_h_ict * 3.0
        body_chest_y = chin_y - head_h_ict * 2.5
    else:
        body_hip_y = None
        body_chest_y = None
    # Head tier uses CERVICAL VERTEBRA cascade (faceforge-style):
    # rotation distributed across 7 cervical levels so the neck
    # bends naturally, not pivots at a single joint. y_neck/y_head
    # are kept for fallback only (when cervical params not passed).
    head_neck_y = chin_y - head_h_ict * 0.40   # T1 line
    head_jaw_y  = chin_y + head_h_ict * 0.05   # above C1
    head_pivot_z = 0.0  # spine z (so shake amplitude is full)

    # Save the un-rotated deformed mesh for any extras that need to
    # be built in the head's LOCAL frame (e.g. the tongue, whose
    # back-of-mouth offset is along the head's local -Z, not world).
    pre_rotation_verts = verts.copy()

    # Anatomical head rotation: per-vertex Y-weighted skinning so
    # the head rotates around the neck base while the bust stays
    # mostly in place. We bake the rotation into the vertices on the
    # CPU and then render with zero global rotation.
    # Apply BODY rotation first — bends the whole upper body around
    # the hip joint. Head verts (above neck) inherit this rotation
    # because they're above the body band's full-rotation threshold.
    if show_body and (abs(body_yaw) > 1e-3 or abs(body_pitch) > 1e-3
                       or abs(body_roll) > 1e-3):
        verts = _apply_neck_rotation(
            verts, body_yaw, body_pitch, body_roll,
            y_neck_abs=body_hip_y, y_head_abs=body_chest_y,
        )
    # Apply HEAD rotation as a SEQUENCE of 7 cervical-disc
    # rotations (faceforge-style true cascade). Each disc has its
    # OWN pivot, so the chin's path is a chain of small arcs and
    # the neck VISIBLY CURVES.
    verts = _apply_cervical_cascade(
        verts, head_yaw, head_pitch, head_roll,
        chin_y=chin_y, head_h=head_h_ict, pivot_z=head_pivot_z,
    )

    # Camera orbit — rotates the WHOLE scene (head + bust together)
    # around the mesh centroid at fixed distance. Slider-driven
    # via params._camera_yaw/_pitch. Applied AFTER neck skinning so
    # the head's natural pose stacks with the orbit view.
    if abs(cam_yaw) > 1e-3 or abs(cam_pitch) > 1e-3:
        verts = _apply_camera_orbit(verts, cam_yaw, cam_pitch)

    # Procedural 3D extras — hair + tongue. Both are appended to
    # the ICT vertex stream so they get the same MVP (neck rotation
    # + camera orbit) and Phong shader as the head.
    extras = []
    hair_style = getattr(params, "_slider_hair_style", "none")
    hair_color = getattr(params, "_slider_hair_color", "#3a2418")
    if hair_style and hair_style != "none":
        try:
            from faceview.vision.hair_3d import gen_hair_mesh
            hm = gen_hair_mesh(hair_style, verts, hair_color)
            if hm is not None:
                extras.append(hm)
        except Exception:
            pass
    if getattr(params, "_show_body", False):
        try:
            from faceview.vision.body_3d import gen_body_mesh
            from faceview.vision.hair_3d import HairMesh
            # Match body skin colour to the head's back/side material
            # so the chin/clavicle join blends instead of showing a
            # hard colour seam. M_BackHead is what ICT uses for the
            # bust/neck region, so the body picks up the same tint.
            palette = _material_palette(params)
            bh_rgb = palette.get("M_BackHead", (0.35, 0.55, 0.65))
            body_color_hex = "#{:02x}{:02x}{:02x}".format(
                int(np.clip(bh_rgb[0], 0, 1) * 255),
                int(np.clip(bh_rgb[1], 0, 1) * 255),
                int(np.clip(bh_rgb[2], 0, 1) * 255),
            )
            bm = gen_body_mesh(
                pre_rotation_verts,
                morph=float(getattr(params, "_body_morph", 0.0)),
                color_hex=body_color_hex,
            )
            if bm is not None:
                # Apply the SAME Y-weighted neck rotation that the
                # head uses, with ref_verts=ICT mesh so both meshes
                # share identical pivot + smoothstep band. The body's
                # upper region (clavicle / shoulder caps morphed
                # toward ICT's neck) sits in the rotation band, so
                # those verts track the head's nod / shake / recoil
                # — the lower torso stays put. This stops the head
                # from detaching from the body during head movements.
                ict_centre = (pre_rotation_verts.min(axis=0)
                                  + pre_rotation_verts.max(axis=0)) / 2.0
                # Per-joint LIMB rig — shoulders, elbows, wrists,
                # hips, knees, ankles. Applied to body's neutral
                # pose first so subsequent body/head tier rotations
                # carry the limbs along correctly. Hard-masked by
                # per-vertex body-part labels so e.g. an arm
                # rotation can never pick up torso/leg verts.
                if os.environ.get("FACEVIEW_DEBUG_PARTS", "").strip() in (
                        "1", "true", "yes"):
                    _verts_before_rig = bm.verts.copy()
                # Prefer the painted-label-driven v2 rig when fine
                # labels are available (uses bone-hierarchy hard
                # masks + 3D joint pivots from limb_landmarks).
                # Fall back to the legacy heuristic rig if not.
                _use_v2 = False
                if os.environ.get("FACEVIEW_RIG_V1", "").strip().lower() not in (
                        "1", "true", "yes", "on"):
                    from faceview.vision.body_3d import (
                        classify_body_parts_fine as _cbpf,
                    )
                    _fine = _cbpf(bm.verts, chin_y=chin_y,
                                     head_h=head_h_ict)
                    if _fine is not None and len(_fine) == len(bm.verts):
                        from faceview.vision.body_rig import (
                            build_rig_state, apply_body_rig_v2,
                            filter_phantom_triangles,
                            filter_empirical_bad_triangles,
                        )
                        _morph = float(getattr(params, "_body_morph", 0.0))
                        # Pass 1: anatomical-pair filter — strips
                        # cross-region phantoms (hand↔thigh bridges
                        # etc.).
                        from faceview.vision.body_rig import (
                            _build_adjacency as _ba,
                            _smooth_labels_mode as _sl,
                            _apply_manual_overrides as _amo,
                        )
                        _adj = _ba(bm.tris, len(bm.verts))
                        _smoothed = _sl(_fine.copy(), _adj, n_iters=2)
                        # Apply manual overrides BEFORE the phantom-
                        # triangle filter. Otherwise a vert that's
                        # overridden to a different anatomical region
                        # leaves its old bridge triangles in the mesh,
                        # which stretch into dark slivers when the
                        # neighbouring limb rotates.
                        _smoothed = _amo(_smoothed, body_morph=_morph)
                        _fine = _smoothed.copy()
                        _kept_tris, _removed_mask = \
                            filter_phantom_triangles(
                                bm.tris, _smoothed)

                        def _apply_tri_mask(removed):
                            nonlocal bm
                            if not removed.any():
                                return
                            bm.tris = bm.tris[~removed]
                            for _attr in ("tri_colors",
                                            "tri_specular",
                                            "tri_emissive"):
                                _arr = getattr(bm, _attr, None)
                                if _arr is not None and len(_arr) == \
                                        len(removed):
                                    setattr(bm, _attr,
                                              _arr[~removed])

                        _apply_tri_mask(_removed_mask)
                        _rig = build_rig_state(
                            bm.verts, bm.tris, _fine,
                            body_morph=_morph)
                        if _rig is not None:
                            # Pass 2: empirical bad-triangle filter —
                            # runs trial rotations on each joint and
                            # strips any triangle whose longest edge
                            # grows >3× under any test rotation.
                            # Catches stretches the anatomical filter
                            # missed (e.g. mis-classified verts on
                            # either side of a real seam).
                            _kept2, _bad2 = filter_empirical_bad_triangles(
                                bm.verts, bm.tris, _fine,
                                pivots=_rig.pivots,
                                masks=_rig.weights,
                                edge_grow_max=3.0)
                            if _bad2.any():
                                _apply_tri_mask(_bad2)
                                # Rebuild rig with the cleaned mesh
                                # so seam_indices etc. are accurate.
                                _rig = build_rig_state(
                                    bm.verts, bm.tris, _fine,
                                    body_morph=_morph)
                            bm.verts[:] = apply_body_rig_v2(
                                bm.verts, params, _rig)
                            _use_v2 = True
                if not _use_v2:
                    bm.verts[:] = _apply_body_rig(
                        bm.verts, params, chin_y, head_h_ict,
                        parts=getattr(bm, "parts", None),
                    )
                if os.environ.get("FACEVIEW_DEBUG_PARTS", "").strip() in (
                        "1", "true", "yes"):
                    from faceview.vision.body_3d import (
                        part_movement_summary as _pms,
                    )
                    if getattr(bm, "parts", None) is not None:
                        report = _pms(
                            _verts_before_rig, bm.verts, bm.parts,
                        )
                        # Log only parts that moved > 0.01 ICT units —
                        # a static body shouldn't have any movement.
                        movers = [
                            f"{name}:mean={d['mean']:.3f}"
                            for name, d in report.items()
                            if d["mean"] > 0.01
                        ]
                        if movers:
                            print("[parts] limb rig:", " ".join(movers))
                # Body rotation tier: bends body around the hip.
                if abs(body_yaw) > 1e-3 or abs(body_pitch) > 1e-3 \
                        or abs(body_roll) > 1e-3:
                    bm.verts[:] = _apply_neck_rotation(
                        bm.verts, body_yaw, body_pitch, body_roll,
                        ref_verts=pre_rotation_verts,
                        y_neck_abs=body_hip_y,
                        y_head_abs=body_chest_y,
                    )
                # Head rotation tier — multi-pivot cervical cascade
                # (faceforge-style). Body's morphed neck verts get
                # the upper-cascade fractions, torso gets ~0.
                _debug_parts = os.environ.get(
                    "FACEVIEW_DEBUG_PARTS", "").strip() in (
                    "1", "true", "yes")
                if _debug_parts:
                    _vbefore = bm.verts.copy()
                bm.verts[:] = _apply_cervical_cascade(
                    bm.verts, head_yaw, head_pitch, head_roll,
                    chin_y=chin_y, head_h=head_h_ict,
                    pivot_z=head_pivot_z,
                )
                if _debug_parts and getattr(bm, "parts", None) is not None:
                    from faceview.vision.body_3d import (
                        part_movement_summary as _pms,
                    )
                    report = _pms(_vbefore, bm.verts, bm.parts)
                    movers = [
                        f"{name}:mean={d['mean']:.3f}"
                        for name, d in report.items()
                        if d["mean"] > 0.01
                    ]
                    if movers:
                        print("[parts] cervical:", " ".join(movers))
                if abs(cam_yaw) > 1e-3 or abs(cam_pitch) > 1e-3:
                    bm.verts[:] = _apply_camera_orbit(
                        bm.verts, cam_yaw, cam_pitch, pivot=ict_centre,
                    )
                # Debug separation check — gated by FACEVIEW_DEBUG_SEP=1.
                # Logs a warning when chin drifts > 5 ICT units from
                # the nearest body vertex (a sensible "head detached"
                # threshold). Throttled so logs don't flood.
                if os.environ.get("FACEVIEW_DEBUG_SEP", "").strip() in (
                        "1", "true", "yes"):
                    _check_head_body_separation(verts, bm.verts)
                extras.append(HairMesh(
                    verts=bm.verts, tris=bm.tris, colors=bm.colors,
                    specular=bm.specular, emissive=bm.emissive,
                ))
        except Exception as e:
            print(f"body err: {e}")

    if getattr(params, "_show_tongue", False):
        try:
            from faceview.vision.hair_3d import gen_tongue_mesh
            # Slider-driven shape; PreFX warp falls back to a default
            # extension via _tongue_protrusion when no slider set.
            extend = float(getattr(params, "_tongue_extend",
                                       getattr(params, "_tongue_protrusion",
                                                 0.4) * 2 - 1.0))
            # Build tongue in the un-rotated frame (back-of-mouth is
            # along local -Z), then apply the same neck + camera
            # transforms so the tongue follows the head.
            tm = gen_tongue_mesh(
                pre_rotation_verts, model,
                color_hex="#b04050",
                extend=extend,
                lateral=float(getattr(params, "_tongue_lateral", 0.0)),
                vertical=float(getattr(params, "_tongue_vertical", 0.0)),
                curl=float(getattr(params, "_tongue_curl", 0.0)),
                taper=float(getattr(params, "_tongue_taper", 0.4)),
                jaw_open=float(getattr(params, "jaw_open", 0.0)),
            )
            if tm is not None:
                # Apply same rotations as ICT, using ICT's bbox /
                # centroid so the tongue tracks the head.
                ict_centre = (pre_rotation_verts.min(axis=0)
                                  + pre_rotation_verts.max(axis=0)) / 2.0
                if abs(body_yaw) > 1e-3 or abs(body_pitch) > 1e-3 \
                        or abs(body_roll) > 1e-3:
                    tm.verts[:] = _apply_neck_rotation(
                        tm.verts, body_yaw, body_pitch, body_roll,
                        ref_verts=pre_rotation_verts,
                        y_neck_abs=body_hip_y,
                        y_head_abs=body_chest_y,
                    )
                tm.verts[:] = _apply_neck_rotation(
                    tm.verts, head_yaw, head_pitch, head_roll,
                    ref_verts=pre_rotation_verts,
                    y_neck_abs=head_neck_y,
                    y_head_abs=head_jaw_y,
                )
                if abs(cam_yaw) > 1e-3 or abs(cam_pitch) > 1e-3:
                    tm.verts[:] = _apply_camera_orbit(
                        tm.verts, cam_yaw, cam_pitch, pivot=ict_centre,
                    )
                extras.append(tm)
        except Exception:
            pass

    # Project feature pixel positions for PostFX (tears, blush, heart
    # eyes, sweat drops, !? marks) AFTER extras are built. The body
    # mesh, when present, expands the rendered bbox so a head-only
    # bbox would put feature anchors in the wrong place. Use the same
    # combined bbox + zoom multiplier the renderer is about to use.
    cam_zoom = float(getattr(params, "_camera_zoom", 1.0) or 1.0)
    cam_focus_y = float(getattr(params, "_camera_focus_y", 0.0) or 0.0)
    big_extra = next((e.verts for e in extras if len(e.verts) > 1000), None)
    bbox_verts = (np.vstack([verts, big_extra])
                    if big_extra is not None else verts)
    try:
        feat = _project_features(
            model, verts, 0.0, 0.0, size,
            bbox_verts=bbox_verts, scale_multiplier=cam_zoom,
            focus_y=cam_focus_y,
        )
        # BP3D's gpu renderer pre-rotates with X-mirror, so its yaw
        # direction is opposite ICT's — negate yaw for anatomy
        # overlays. Pitch stays the same (no Y/Z mirror).
        feat["_yaw"] = float(-(yaw + cam_yaw))
        feat["_pitch"] = float(pitch + cam_pitch)
        params._feature_pixels = feat
    except Exception:
        params._feature_pixels = {}

    if extras:
        # Combine all extras into one stream of verts/tris.
        all_v = np.vstack([e.verts for e in extras])
        offsets = np.cumsum([0] + [len(e.verts) for e in extras[:-1]])
        all_t = np.vstack([e.tris + off for e, off in zip(extras, offsets)])
        all_c = np.vstack([e.colors for e in extras])
        all_s = np.concatenate([e.specular for e in extras])
        all_e_emit = np.concatenate([e.emissive for e in extras])
        bgr = _render_via_moderngl(
            verts, head_only_tris, size, 0.0, 0.0, params,
            extra_verts=all_v, extra_tris=all_t,
            extra_colors=all_c, extra_spec=all_s,
            extra_emit=all_e_emit,
        )
    else:
        bgr = _render_via_moderngl(verts, head_only_tris, size,
                                      0.0, 0.0, params)

    # Sci-fi bloom — extract bright pixels and add a blurred halo back
    # over the original. Reads as glowing eyes / hot teeth / etc.
    style = getattr(params, "_persona_style", "natural")
    if style != "natural":
        amp_override = getattr(params, "_slider_bloom_amp", None)
        bgr = _apply_bloom(bgr, style, amp_override=amp_override)

    # 2D hair overlay (off by default — the procedural overlay is
    # rough; the bald ICT head reads cleaner). Set
    # ``params._enable_hair = True`` if you want to opt in.
    if getattr(params, "_enable_hair", False):
        bgr = _composite_hair_overlay(bgr, params, yaw)
    return bgr


def _apply_bloom(bgr: np.ndarray, style: str,
                  amp_override: float | None = None) -> np.ndarray:
    """Cheap bloom — Gaussian blur of bright pixels mixed back in.

    Pulls a high-pass mask above ``threshold``, blurs it large, and
    additively blends. Per-style amplitude tunes the strength;
    ``amp_override`` (0..1) lets the live Bloom-strength slider
    replace the per-style default.
    """
    import cv2
    if amp_override is not None:
        amp = float(max(0.0, amp_override))
    else:
        amp = {"neon": 0.35, "transparent": 0.20,
               "cyberpunk": 0.30, "xray": 0.45}.get(style, 0.0)
    if amp <= 0:
        return bgr
    threshold = 180  # luminance cutoff (0..255)
    # Build a mask of bright pixels.
    luma = bgr.max(axis=2)
    mask = (luma > threshold).astype(np.uint8)
    if not mask.any():
        return bgr
    bright = bgr * mask[:, :, None]
    # Wide Gaussian halo.
    blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=8.0, sigmaY=8.0)
    out = np.clip(bgr.astype(np.float32) + blurred.astype(np.float32) * amp,
                   0.0, 255.0).astype(np.uint8)
    return out


def _composite_hair_overlay(bgr: np.ndarray, params, yaw: float) -> np.ndarray:
    """Draw a smooth 2D hair cap on top of the rendered ICT head.

    Approach: detect the head silhouette via the background mask,
    smooth the upper outline with a moving average, and paint an
    opaque cap. Cheap CPU work, ~2 ms.
    """
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
    )

    h, w, _ = bgr.shape
    rgb = bgr[:, :, ::-1].copy()

    # Foreground mask via background detection.
    bg_mask = (rgb.sum(axis=2) < 60)
    fg_mask = ~bg_mask
    cols_with_face = np.any(fg_mask, axis=0)
    if not np.any(cols_with_face):
        return bgr

    # Top silhouette per column.
    top_y = np.full(w, h, dtype=np.int32)
    has = cols_with_face
    if has.any():
        idxs = np.argmax(fg_mask, axis=0)
        top_y[has] = idxs[has]

    fg_xs = np.where(cols_with_face)[0]
    left, right = int(fg_xs[0]), int(fg_xs[-1])
    head_w = right - left
    if head_w < 20:
        return bgr

    # Smooth the silhouette with a running average so the hair cap
    # follows a gentle curve instead of jagged pixel edges.
    smooth = top_y.copy().astype(np.float32)
    win = max(3, head_w // 25)
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(smooth, kernel, mode="same")

    # Hair cap extends from forehead up to ~head_w * 0.5 above.
    cap_pad = int(0.42 * head_w)
    fringe_drop = int(0.08 * head_w)

    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    p = QPainter(qimg)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    hair_hex = getattr(params, "hair_color", "#2a1808")
    hair = QColor(hair_hex)
    hair_dark = hair.darker(130)
    hair_light = hair.lighter(135)

    # Build the cap path: trace the smoothed top silhouette
    # (with a small downward bias for fringe) then arc up over the head.
    cap_path = QPainterPath()
    centre_x = (left + right) / 2
    cap_top_y = max(0, float(np.min(smooth[left:right + 1])) - cap_pad)

    # Start from left edge, slightly below the forehead silhouette.
    cap_path.moveTo(QPointF(left - 4,
                              float(smooth[left]) + fringe_drop))
    # Walk along the silhouette top with a fringe drop.
    step = max(1, head_w // 40)
    for x in range(left, right + 1, step):
        sy = float(smooth[x])
        # Fringe pulls down toward the brow on the lower part.
        f = abs(x - centre_x) / max(1, head_w / 2)
        drop = fringe_drop * (1 - f * 0.6)
        cap_path.lineTo(QPointF(x, sy + drop))
    cap_path.lineTo(QPointF(right + 4,
                              float(smooth[right]) + fringe_drop))
    # Up over the right side.
    cap_path.cubicTo(
        QPointF(right + 8, cap_top_y + 0.5 * (smooth[right] - cap_top_y)),
        QPointF(centre_x + 0.4 * head_w, cap_top_y),
        QPointF(centre_x, cap_top_y),
    )
    cap_path.cubicTo(
        QPointF(centre_x - 0.4 * head_w, cap_top_y),
        QPointF(left - 8, cap_top_y + 0.5 * (smooth[left] - cap_top_y)),
        QPointF(left - 4, float(smooth[left]) + fringe_drop),
    )
    cap_path.closeSubpath()

    # Fill cap with a vertical gradient (darker at top, lighter at fringe).
    grad = QLinearGradient(QPointF(centre_x, cap_top_y),
                            QPointF(centre_x, float(smooth[int(centre_x)]) + fringe_drop))
    grad.setColorAt(0.0, hair_dark)
    grad.setColorAt(0.6, hair)
    grad.setColorAt(1.0, hair)
    p.setBrush(QBrush(grad))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPath(cap_path)

    # A small fringe across the forehead — slightly lower-saturation
    # band that breaks the hard edge.
    fringe_path = QPainterPath()
    fringe_l = left + 0.10 * head_w
    fringe_r = right - 0.10 * head_w
    fringe_path.moveTo(QPointF(fringe_l,
                                  float(smooth[int(fringe_l)]) + fringe_drop * 0.4))
    fringe_path.cubicTo(
        QPointF(fringe_l + 0.15 * head_w,
                 float(smooth[int(fringe_l + 0.15 * head_w)]) + fringe_drop * 1.2),
        QPointF(fringe_r - 0.15 * head_w,
                 float(smooth[int(fringe_r - 0.15 * head_w)]) + fringe_drop * 0.6),
        QPointF(fringe_r,
                 float(smooth[int(fringe_r)]) + fringe_drop * 0.3),
    )
    fringe_path.lineTo(QPointF(fringe_r, float(smooth[int(fringe_r)]) - 4))
    fringe_path.cubicTo(
        QPointF(fringe_r - 0.15 * head_w,
                 float(smooth[int(fringe_r - 0.15 * head_w)]) - 4),
        QPointF(fringe_l + 0.15 * head_w,
                 float(smooth[int(fringe_l + 0.15 * head_w)]) - 4),
        QPointF(fringe_l, float(smooth[int(fringe_l)]) - 4),
    )
    fringe_path.closeSubpath()
    p.setBrush(QBrush(hair))
    p.drawPath(fringe_path)

    # Soft strand highlights (subtle).
    pen = QPen(hair_light, max(1.0, head_w * 0.004))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    n_strands = 5
    for i in range(n_strands):
        f = (i + 0.5) / n_strands
        x0 = float(left + f * head_w)
        y0 = float(smooth[int(x0)]) - cap_pad * 0.65
        x1 = x0 + head_w * 0.07
        y1 = y0 - cap_pad * 0.18
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    p.end()

    out = qimg.convertToFormat(QImage.Format.Format_RGB888)
    ptr = out.constBits()
    if ptr is None:
        return bgr
    arr = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return arr[:, :, ::-1].copy()


def _render_via_moderngl(
    verts: np.ndarray,
    triangles: np.ndarray,
    size: tuple[int, int],
    yaw: float, pitch: float,
    params,
    *,
    extra_verts: np.ndarray | None = None,
    extra_tris: np.ndarray | None = None,
    extra_colors: np.ndarray | None = None,
    extra_spec: np.ndarray | None = None,
    extra_emit: np.ndarray | None = None,
) -> np.ndarray:
    """Render through moderngl with a Phong shader. GPU-only path.

    ``extra_*`` arrays append a secondary mesh (e.g. procedural 3D
    hair) to the ICT vertex stream so it gets the same MVP and
    shader treatment. extra_tris should already be offset by the
    ICT vertex count or by 0 — we apply the offset here.
    """
    try:
        import moderngl
    except ImportError as exc:
        raise MissingDependency("moderngl", "gpu") from exc

    rend = _ensure_renderer()

    # Combine ICT mesh with extras.
    ict_n = len(verts)
    ict_colors = _per_vertex_colors_for(params).astype(np.float32)
    ict_spec = _per_vertex_specular().astype(np.float32)
    ict_emit = _per_vertex_emissive().astype(np.float32)
    if extra_verts is not None and len(extra_verts) > 0:
        all_verts = np.vstack([verts, extra_verts]).astype(np.float32)
        all_tris = np.vstack([
            triangles, extra_tris.astype(np.int32) + ict_n,
        ]).astype(np.int32)
        all_colors = np.vstack([ict_colors, extra_colors.astype(np.float32)])
        all_spec = np.concatenate([ict_spec, extra_spec.astype(np.float32)])
        all_emit = np.concatenate([ict_emit, extra_emit.astype(np.float32)])
    else:
        all_verts, all_tris = verts.astype(np.float32), triangles
        all_colors, all_spec, all_emit = ict_colors, ict_spec, ict_emit

    # Per-vertex normals (averaged from incident triangles).
    v0 = all_verts[all_tris[:, 0]]
    v1 = all_verts[all_tris[:, 1]]
    v2 = all_verts[all_tris[:, 2]]
    tri_norms = np.cross(v1 - v0, v2 - v0)
    tri_norms /= np.maximum(np.linalg.norm(tri_norms, axis=1, keepdims=True), 1e-9)
    vert_norms = np.zeros_like(all_verts)
    np.add.at(vert_norms, all_tris[:, 0], tri_norms)
    np.add.at(vert_norms, all_tris[:, 1], tri_norms)
    np.add.at(vert_norms, all_tris[:, 2], tri_norms)
    vert_norms /= np.maximum(np.linalg.norm(vert_norms, axis=1, keepdims=True), 1e-9)

    # Centre + scale to fit. When a large extra (body) is present
    # we use the combined bbox so the whole avatar fits in frame;
    # for small extras (hair, tongue) we keep the ICT bbox so the
    # head fills the frame normally.
    if extra_verts is not None and len(extra_verts) > 1000:
        # Probably a body mesh — fit to combined bbox.
        bbox_src = np.vstack([verts, extra_verts])
    else:
        bbox_src = verts
    vmin = bbox_src.min(axis=0)
    vmax = bbox_src.max(axis=0)
    centre = (vmin + vmax) / 2
    # Vertical-focus offset — shifts the framing centre up (+) or
    # down (-) by this fraction of the bbox Y span. Lets the user
    # snap to head-only framing while body is shown.
    focus_y = float(getattr(params, "_camera_focus_y", 0.0) or 0.0)
    if abs(focus_y) > 1e-3:
        y_span = float(vmax[1] - vmin[1])
        centre = centre + np.array([0.0, focus_y * y_span * 0.5, 0.0],
                                       dtype=centre.dtype)
    span = float(np.linalg.norm(vmax - vmin))
    scale = 1.6 / max(span, 1e-6)
    # Slider-driven zoom multiplies the fit-to-frame scale.
    cam_zoom = float(getattr(params, "_camera_zoom", 1.0) or 1.0)
    scale *= max(0.1, cam_zoom)

    style = getattr(params, "_persona_style", "natural")
    rend._style_uniforms = _shader_overrides_for_style(style)
    pulse_scale = float(getattr(params, "_slider_emit_pulse_scale", 1.0) or 1.0)
    rend._emit_pulse = _emit_pulse_for(style, scale=pulse_scale)
    return rend.render(
        verts=all_verts,
        normals=vert_norms.astype(np.float32),
        triangles=all_tris.astype(np.uint32),
        vert_colors=all_colors,
        vert_spec=all_spec,
        vert_emit=all_emit,
        centre=centre.astype(np.float32),
        scale=float(scale),
        yaw=yaw, pitch=pitch,
        size=size,
        bg=_hex_to_rgb(getattr(params, "background", "#0a0d12")),
    )


@lru_cache(maxsize=1)
def _vertex_regions() -> dict[str, np.ndarray]:
    """Derive anatomical region masks from blendshape delta magnitudes.

    Each ARKit blendshape is named after the facial action it
    drives (``mouthSmile_L``, ``browInnerUp_R``, ``cheekPuff_L``).
    The vertices it moves are anatomically members of that region.

    But blendshape deformations ripple across the face, so a naive
    "non-zero" mask gives ~half the mesh. We aggregate by region
    (sum of delta magnitudes across all blendshapes in that region)
    then classify each vertex by its dominant region — the one that
    moves it the most. That gives clean anatomical region masks
    that match what the artist actually rigged.
    """
    m = load_ict_model()
    n_verts = len(m.vertices)

    # Per-region blendshape WHITELIST. We pick blendshapes whose
    # primary deformation is the named region — broader prefixes
    # (e.g. "mouth*") catch corner-pull / dimple shapes that drag
    # the whole lower face, smearing the lip colour onto the chin.
    region_blendshapes = {
        # Vermillion border (the actual lip surface): these shapes
        # roll, shrug, funnel, pucker, close — all on the lip skin.
        "lips": [
            "mouthClose",
            "mouthRollLower", "mouthRollUpper",
            "mouthShrugLower", "mouthShrugUpper",
            "mouthFunnel", "mouthPucker",
        ],
        # Brow ridge / hair line.
        "brow": [
            "browDown_L", "browDown_R",
            "browInnerUp_L", "browInnerUp_R",
            "browOuterUp_L", "browOuterUp_R",
        ],
        # Eyelid skin.
        "eyelid": [
            "eyeBlink_L", "eyeBlink_R",
            "eyeSquint_L", "eyeSquint_R",
            "eyeWide_L", "eyeWide_R",
        ],
        # Cheek apples (puff + raiser only — squint pulls eyelid).
        "cheek": [
            "cheekPuff_L", "cheekPuff_R",
            "cheekRaiser_L", "cheekRaiser_R",
        ],
        "nose": ["noseSneer_L", "noseSneer_R"],
        "jaw":  ["jawOpen"],
    }

    # Max |delta| per region.
    region_mag: dict[str, np.ndarray] = {}
    for region, blendshapes in region_blendshapes.items():
        agg = np.zeros(n_verts, dtype=np.float32)
        for name in blendshapes:
            idx = m.name_to_idx.get(name)
            if idx is None:
                continue
            agg = np.maximum(agg, np.linalg.norm(m.deltas[idx], axis=1))
        region_mag[region] = agg

    # For each vertex, pick the region with the highest magnitude.
    # Vertices below ALL region thresholds get classified as "skin".
    region_keys = list(region_mag.keys())
    stack = np.stack([region_mag[k] for k in region_keys], axis=0)  # (R, N)
    max_region = np.argmax(stack, axis=0)
    max_mag = np.max(stack, axis=0)

    # Vertices need a minimum motion magnitude to be considered IN a
    # region; otherwise they're plain skin.
    threshold = 0.0015
    in_any = max_mag > threshold

    out: dict[str, np.ndarray] = {}
    for i, key in enumerate(region_keys):
        out[key] = (max_region == i) & in_any
    out["skin"] = ~in_any
    return out


def _per_vertex_colors_for(params) -> np.ndarray:
    """Return (N, 3) RGB colors keyed off the OBJ's material tags +
    persona palette + blendshape-derived anatomical regions.

    Region detection uses ICT blendshape deltas (winner-take-all per
    vertex) so colours land in anatomically correct places — lips
    on lip vertices, eyebrows on the brow ridge, cheeks on cheek
    apples, etc. Hair cap stays on the Y heuristic since ICT has
    no hair blendshapes.
    """
    palette = _material_palette(params)
    m = load_ict_model()
    verts = m.vertices                       # (N, 3) float32
    n_verts = len(verts)
    rng = np.random.default_rng(seed=42)

    # Default: per-material colour (lookup by triangle material).
    # We assign each vertex the colour of the most-frequent material
    # of its incident triangles.
    fallback = palette["M_Face"]
    vert_mat = np.zeros(n_verts, dtype=np.int32) - 1
    for ti, mi in enumerate(m.tri_materials):
        for v in m.triangles[ti]:
            if vert_mat[v] < 0:
                vert_mat[v] = int(mi)

    mat_array = np.array([
        palette.get(name, fallback) for name in m.materials
    ], dtype=np.float32)            # (M, 3)
    mat_array = np.vstack([mat_array, np.array(fallback, dtype=np.float32)])
    vert_mat_safe = np.where(vert_mat >= 0, vert_mat, len(m.materials))
    colors = mat_array[vert_mat_safe].copy()  # (N, 3) float32

    # Mesh bbox for normalised positions.
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_span = y_max - y_min
    z_mid = (z_min + z_max) / 2

    on_face = np.array([
        m.materials[vm] in ("M_Face", "M_BackHead")
        if 0 <= vm < len(m.materials) else False for vm in vert_mat_safe
    ])

    hair_color = np.array(palette["M_HairCap"], dtype=np.float32)
    lip_color = np.array(
        _hex_to_rgb_f(getattr(params, "lip_color", "#a44a4a")),
        dtype=np.float32,
    )

    # Sci-fi styles use their own palette wholesale — skip the
    # lip/brow/cheek post-processing which is meant for natural skin.
    style = getattr(params, "_persona_style", "natural")
    if style != "natural":
        # Xray is hairless — bare skull look avoids the uncanny
        # valley and matches the medical-glow aesthetic.
        if style != "xray":
            hair_y = verts[:, 1].max() - (verts[:, 1].max() - verts[:, 1].min()) * 0.32
            hair_mask_sf = on_face & (verts[:, 1] > hair_y)
            if hair_mask_sf.any():
                colors[hair_mask_sf] = hair_color
        return np.clip(colors, 0.0, 1.0).astype(np.float32)

    # ── Hair cap (Y heuristic — no hair blendshapes in ICT) ──
    # Top of head, fading at the forehead.
    hair_y = y_max - y_span * 0.32
    hair_fade_y = hair_y - y_span * 0.05
    hair_mask = on_face & (verts[:, 1] > hair_y)
    fade_mask = (on_face & (verts[:, 1] > hair_fade_y)
                 & (verts[:, 1] <= hair_y))
    hair_noise = 1.0 + rng.standard_normal(n_verts).astype(np.float32) * 0.06
    hair_noise = np.clip(hair_noise, 0.85, 1.15)
    colors[hair_mask] = hair_color * hair_noise[hair_mask, None]
    if fade_mask.any():
        t = ((verts[fade_mask, 1] - hair_fade_y)
              / max(1e-6, hair_y - hair_fade_y))[:, None]
        colors[fade_mask] = colors[fade_mask] * (1 - t) + hair_color * t

    # ── Anatomical regions from blendshape deltas ──
    regions = _vertex_regions()

    # Brows — coloured with hair_color, blended so it reads as hair
    # over skin (the brow region from blendshapes IS the eyebrow
    # ridge / brow hair area). Don't apply where hair cap already painted.
    brow_mask = regions["brow"] & on_face & ~hair_mask
    if brow_mask.any():
        brow_noise = 1.0 + rng.standard_normal(n_verts).astype(np.float32) * 0.05
        brow_noise = np.clip(brow_noise, 0.88, 1.12)
        colors[brow_mask] = (colors[brow_mask] * 0.30
                              + hair_color * 0.70 * brow_noise[brow_mask, None])

    # Lips — winner-take-all from "mouth*" blendshapes. Filter out
    # vertices that are visibly NOT face material (teeth, gums, tongue).
    lips_mask = (regions["lips"] & on_face & ~hair_mask)
    if lips_mask.any():
        # Lips read as redder skin — 60/40 blend.
        colors[lips_mask] = colors[lips_mask] * 0.40 + lip_color * 0.60

    # Cheek blush — subtle pink at cheek apple area.
    cheek_mask = regions["cheek"] & on_face & ~hair_mask & ~lips_mask & ~brow_mask
    if cheek_mask.any():
        blush = np.array([0.95, 0.62, 0.55], dtype=np.float32)
        colors[cheek_mask] = colors[cheek_mask] * 0.90 + blush * 0.10

    # ── Subtle global skin noise ──
    if on_face.any():
        skin_noise = 1.0 + rng.standard_normal(n_verts).astype(np.float32) * 0.012
        skin_noise = np.clip(skin_noise, 0.97, 1.03)
        colors[on_face] *= skin_noise[on_face, None]

    return np.clip(colors, 0.0, 1.0).astype(np.float32)


@lru_cache(maxsize=1)
def _per_vertex_colors() -> np.ndarray:
    """Compatibility shim for the default neutral palette."""
    class _DefaultParams:
        skin_hue = 28.0
        lip_color = "#a44a4a"
        hair_color = "#2a1808"
        _persona_eye_color = "#5a3818"
        _persona_skin_sat = 0.32
        _persona_skin_val = 0.86
    return _per_vertex_colors_for(_DefaultParams())


@lru_cache(maxsize=1)
def _per_vertex_emissive() -> np.ndarray:
    """Per-vertex emissive intensity, baked from material map.

    Always passed into the GL pipeline; per-frame ``u_emit_pulse``
    scales it (0 in natural mode → invisible)."""
    m = load_ict_model()
    n_verts = len(m.vertices)
    accum = np.zeros(n_verts, dtype=np.float64)
    counts = np.zeros(n_verts, dtype=np.int32)
    for ti, mat_idx in enumerate(m.tri_materials):
        e = (_MATERIAL_EMISSIVE.get(m.materials[mat_idx], 0.0)
             if 0 <= mat_idx < len(m.materials) else 0.0)
        for v in m.triangles[ti]:
            accum[v] += e
            counts[v] += 1
    counts = np.maximum(counts, 1)
    return (accum / counts).astype(np.float32)


@lru_cache(maxsize=1)
def _per_vertex_specular() -> np.ndarray:
    """Per-vertex specular intensity (A43): eyes wet, skin matte."""
    m = load_ict_model()
    n_verts = len(m.vertices)
    accum = np.zeros(n_verts, dtype=np.float64)
    counts = np.zeros(n_verts, dtype=np.int32)
    fallback = _MATERIAL_SPECULAR.get("M_Face", 0.30)
    for ti, mat_idx in enumerate(m.tri_materials):
        s = (_MATERIAL_SPECULAR.get(m.materials[mat_idx], fallback)
             if 0 <= mat_idx < len(m.materials) else fallback)
        for v in m.triangles[ti]:
            accum[v] += s
            counts[v] += 1
    counts = np.maximum(counts, 1)
    return (accum / counts).astype(np.float32)


_VERT_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat3 u_norm_mat;
in vec3 in_pos;
in vec3 in_norm;
in vec3 in_color;
in float in_spec;
in float in_emit;
out vec3 v_norm;
out vec3 v_color;
out float v_spec;
out float v_emit;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_norm = u_norm_mat * in_norm;
    v_color = in_color;
    v_spec = in_spec;
    v_emit = in_emit;
}
"""

# Subsurface-scattering Phong skin shader.
# Components:
#   1. Wrap diffuse — soft falloff (Lambert × 0.5 + 0.5) instead of hard.
#   2. Subsurface tint — warm flesh colour bleeds through when light
#      hits the back of the surface, simulating light penetration.
#   3. Fresnel rim — thin areas (ear edge, nostril, neck) glow warm.
#   4. Sky-tinted ambient — cool from below, warm from above for
#      natural environment lighting.
#   5. Dual-lobe specular — broad soft highlight + tight glint, the
#      MetaHuman trick for skin sheen.
# All cheap per-pixel — fits in our 88-fps budget.
_FRAG_SHADER = """
#version 330
uniform vec3 u_light_dir;
uniform vec3 u_sss_tint;
uniform float u_ambient;
uniform float u_specular;
uniform float u_shininess;
uniform float u_emit_pulse;
in vec3 v_norm;
in vec3 v_color;
in float v_spec;
in float v_emit;
out vec4 frag;
void main() {
    vec3 n = normalize(v_norm);
    vec3 l = normalize(-u_light_dir);
    vec3 view = vec3(0, 0, 1);

    // Wrap diffuse — soft Lambert.
    float ndl = dot(n, l);
    float wrap = clamp(ndl * 0.5 + 0.5, 0.0, 1.0);

    // Subsurface tint: only visible at the *transition* between lit
    // and shadow (terminator), not on fully back-lit surfaces. Use a
    // narrow band around ndl=0 to localise the bleed.
    float back_lit = clamp(-ndl, 0.0, 1.0);
    float terminator = smoothstep(0.0, 0.5, back_lit) *
                        (1.0 - smoothstep(0.5, 1.0, back_lit));
    vec3 sss = u_sss_tint * terminator * 0.35;

    // Sky-tinted ambient: warmer from above (n.y > 0), cooler below.
    float sky_amount = clamp(n.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 sky_warm = vec3(1.05, 0.95, 0.85);
    vec3 sky_cool = vec3(0.80, 0.85, 1.00);
    vec3 ambient_tint = mix(sky_cool, sky_warm, sky_amount);

    // Dual-lobe specular: broad soft + narrow glint, scaled by the
    // per-vertex material intensity (eyes get a wet glint, skin is
    // subtle, lashes are matte).
    vec3 half_v = normalize(l + view);
    float ndh = max(0.0, dot(n, half_v));
    float spec_broad = pow(ndh, max(2.0, u_shininess * 0.30)) * 0.20;
    float spec_tight = pow(ndh, u_shininess) * u_specular;
    float spec = (spec_broad + spec_tight) * v_spec;

    // Fresnel rim — visible at grazing angles for thin features.
    // Tighter falloff than SSS since this is meant to be subtle.
    float fresnel = pow(1.0 - max(0.0, dot(n, view)), 5.0) * 0.20;
    vec3 rim = u_sss_tint * fresnel;

    vec3 ambient = u_ambient * ambient_tint * v_color;
    vec3 diffuse = (1.0 - u_ambient) * wrap * v_color;
    vec3 col = ambient + diffuse + sss + spec + rim;

    // Emissive — eyes / teeth glow in sci-fi modes. Pulse scalar is
    // 0 in natural mode so this term vanishes there. Adding the
    // emissive bypasses Lambertian shading so iris reads as a
    // self-lit panel rather than a recessed wet surface.
    col += v_emit * u_emit_pulse * v_color;

    frag = vec4(col, 1.0);
}
"""


class _ICTRenderer:
    def __init__(self) -> None:
        import moderngl
        self.mgl = moderngl
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=_VERT_SHADER,
                                       fragment_shader=_FRAG_SHADER)
        self._fbo = None
        self._fbo_size: tuple[int, int] | None = None
        self._style_uniforms: dict = {}
        self._emit_pulse: float = 0.0

    def _ensure_fbo(self, w: int, h: int) -> None:
        if self._fbo_size == (w, h) and self._fbo is not None:
            return
        if self._fbo is not None:
            self._fbo.release()
        col = self.ctx.texture((w, h), 4)
        depth = self.ctx.depth_renderbuffer((w, h))
        self._fbo = self.ctx.framebuffer(color_attachments=[col],
                                           depth_attachment=depth)
        self._fbo_size = (w, h)
        self._color = col
        self._depth = depth

    def render(
        self,
        verts: np.ndarray,
        normals: np.ndarray,
        triangles: np.ndarray,
        vert_colors: np.ndarray,
        vert_spec: np.ndarray,
        vert_emit: np.ndarray,
        centre: np.ndarray,
        scale: float,
        yaw: float,
        pitch: float,
        size: tuple[int, int],
        bg: tuple[int, int, int],
    ) -> np.ndarray:
        w, h = size
        self._ensure_fbo(w, h)
        self._fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

        cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
        cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
        ry = np.array([[cy_, 0, sy_, 0], [0, 1, 0, 0],
                        [-sy_, 0, cy_, 0], [0, 0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0, 0], [0, cp_, -sp_, 0],
                        [0, sp_, cp_, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = -centre
        # Aspect-correct the scale: viewport maps NDC X→[0,w] and
        # Y→[0,h]. With uniform scale, a non-square framebuffer
        # (e.g. 640×480) stretches X by w/h. Compensate by shrinking
        # the X axis on the model side.
        aspect = float(h) / float(w) if w > 0 else 1.0
        S = np.diag([scale * aspect, scale, scale, 1.0]).astype(np.float32)
        # ICT mesh has +Y up (head at +Y) and +Z back. We need to
        # flip Y for screen and flip Z so the face points -Z (toward
        # camera).
        flip = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        model = ry @ rx @ flip @ S @ T
        norm_mat = model[:3, :3]
        # Inverse-transpose for normals (orthogonal so just transpose)
        # — but for orthonormal rotation the rotation IS its own
        # inverse-transpose, so we can use it directly.

        self.prog["u_mvp"].write(model.T.tobytes())
        self.prog["u_norm_mat"].write(norm_mat.T.tobytes())
        self.prog["u_light_dir"].value = (-0.4, -0.3, -0.7)
        # Resolve per-style overrides (passed in via render kwargs).
        self.prog["u_sss_tint"].value = self._style_uniforms.get(
            "sss_tint", _SSS_TINT)
        self.prog["u_ambient"].value = self._style_uniforms.get("ambient", 0.32)
        self.prog["u_specular"].value = self._style_uniforms.get("specular", 0.30)
        self.prog["u_shininess"].value = self._style_uniforms.get("shininess", 22.0)
        self.prog["u_emit_pulse"].value = float(self._emit_pulse)

        vbo = self.ctx.buffer(verts.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())
        cbo = self.ctx.buffer(vert_colors.tobytes())
        sbo = self.ctx.buffer(vert_spec.tobytes())
        ebo = self.ctx.buffer(vert_emit.tobytes())
        ibo = self.ctx.buffer(triangles.tobytes())
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, "3f", "in_pos"),
            (nbo, "3f", "in_norm"),
            (cbo, "3f", "in_color"),
            (sbo, "1f", "in_spec"),
            (ebo, "1f", "in_emit"),
        ], ibo)
        vao.render(self.mgl.TRIANGLES)
        vao.release()
        vbo.release()
        nbo.release()
        cbo.release()
        sbo.release()
        ebo.release()
        ibo.release()

        data = self._fbo.read(components=3)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        arr = np.flipud(arr).copy()
        return arr[:, :, ::-1].copy()  # RGB → BGR


@lru_cache(maxsize=1)
def _ensure_renderer() -> _ICTRenderer:
    return _ICTRenderer()


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
