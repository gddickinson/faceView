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


def _apply_neck_rotation(
    verts: np.ndarray, yaw: float, pitch: float,
) -> np.ndarray:
    """Per-vertex Y-weighted skinning around the neck pivot.

    Vertices in the head region get the full ry @ rx rotation;
    vertices in the bust / shoulders below the neck stay put. The
    transition is a smoothstep over a thin band. Result: when
    ``yaw`` / ``pitch`` are non-zero, only the head turns — the
    bust/collar reads as a stationary platform the head moves on.
    """
    if abs(yaw) < 1e-3 and abs(pitch) < 1e-3:
        return verts

    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    span = max(1e-6, y_max - y_min)
    # Neck base ≈ 30 % up from the bottom of the bust; transition
    # band runs to ≈ 50 % (just below the chin).
    y_neck = y_min + span * 0.30
    y_head = y_min + span * 0.55

    y = verts[:, 1]
    t = np.clip((y - y_neck) / max(1e-6, y_head - y_neck), 0.0, 1.0)
    w = (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep

    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)
    R = Ry @ Rx

    pivot = np.array([0.0, y_neck, 0.0], dtype=np.float32)
    diff = verts - pivot
    rotated = (diff @ R.T) + pivot
    return (verts * (1.0 - w[:, None])
              + rotated * w[:, None]).astype(np.float32)


def _project_features(
    model, verts: np.ndarray,
    yaw: float, pitch: float, size: tuple[int, int],
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
    """
    w, h = size
    aspect = float(h) / float(w) if w > 0 else 1.0

    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    centre = (vmin + vmax) / 2.0
    span = float(np.linalg.norm(vmax - vmin))
    scale = 1.6 / max(span, 1e-6)

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

    cheek_l = from_blendshape("cheekPuff_L")
    cheek_r = from_blendshape("cheekPuff_R")
    if cheek_l:
        out["cheek_L"] = cheek_l
    if cheek_r:
        out["cheek_R"] = cheek_r

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

    # Mouth centroid + corners.
    mouth_c = from_blendshape("mouthClose", top_k=80)
    if mouth_c:
        out["mouth"] = mouth_c
    mc_l = from_blendshape("mouthSmile_L")
    mc_r = from_blendshape("mouthSmile_R")
    if mc_l:
        out["mouth_corner_L"] = mc_l
    if mc_r:
        out["mouth_corner_R"] = mc_r

    chin = from_blendshape("jawOpen", top_k=80)
    if chin:
        out["chin"] = chin

    return out


def apply_blendshapes(
    model: ICTModel,
    arkit_coefs: dict[str, float],
) -> np.ndarray:
    """Return deformed vertex positions given ARKit-named coefficients."""
    out = model.vertices.copy()
    for arkit_name, value in arkit_coefs.items():
        if value == 0:
            continue
        ict_name = _ARKIT_TO_ICT.get(arkit_name, arkit_name)
        idx = model.name_to_idx.get(ict_name)
        if idx is None:
            continue
        out += float(value) * model.deltas[idx]
    return out


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
    full_coefs = {**arkit_coefs, **identity_w, **direct_clean}
    verts = apply_blendshapes(model, full_coefs)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4

    # Anatomical head rotation: per-vertex Y-weighted skinning so
    # the head rotates around the neck base while the bust stays
    # mostly in place. We bake the rotation into the vertices on the
    # CPU and then render with zero global rotation.
    verts = _apply_neck_rotation(verts, yaw, pitch)

    # Stash feature pixel positions on params so PostFX (tears, blush,
    # heart eyes, sweat drops, !? marks) can anchor on the actual eye
    # / cheek / forehead / mouth-corner positions. Verts already have
    # the neck rotation baked in, so project at zero global yaw/pitch.
    try:
        params._feature_pixels = _project_features(
            model, verts, 0.0, 0.0, size,
        )
    except Exception:
        params._feature_pixels = {}

    bgr = _render_via_moderngl(verts, model.triangles, size,
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
) -> np.ndarray:
    """Render through moderngl with a Phong shader. GPU-only path."""
    try:
        import moderngl
    except ImportError as exc:
        raise MissingDependency("moderngl", "gpu") from exc

    # Cache the renderer in a module global to keep the GL context alive.
    rend = _ensure_renderer()

    # Compute per-vertex normals (averaged from incident triangles).
    v0 = verts[triangles[:, 0]]
    v1 = verts[triangles[:, 1]]
    v2 = verts[triangles[:, 2]]
    tri_norms = np.cross(v1 - v0, v2 - v0)
    tri_norms /= np.maximum(np.linalg.norm(tri_norms, axis=1, keepdims=True), 1e-9)
    vert_norms = np.zeros_like(verts)
    np.add.at(vert_norms, triangles[:, 0], tri_norms)
    np.add.at(vert_norms, triangles[:, 1], tri_norms)
    np.add.at(vert_norms, triangles[:, 2], tri_norms)
    vert_norms /= np.maximum(np.linalg.norm(vert_norms, axis=1, keepdims=True), 1e-9)

    # Centre + scale to fit.
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    centre = (vmin + vmax) / 2
    span = float(np.linalg.norm(vmax - vmin))
    scale = 1.6 / max(span, 1e-6)

    style = getattr(params, "_persona_style", "natural")
    rend._style_uniforms = _shader_overrides_for_style(style)
    pulse_scale = float(getattr(params, "_slider_emit_pulse_scale", 1.0) or 1.0)
    rend._emit_pulse = _emit_pulse_for(style, scale=pulse_scale)
    return rend.render(
        verts=verts.astype(np.float32),
        normals=vert_norms.astype(np.float32),
        triangles=triangles.astype(np.uint32),
        vert_colors=_per_vertex_colors_for(params).astype(np.float32),
        vert_spec=_per_vertex_specular().astype(np.float32),
        vert_emit=_per_vertex_emissive().astype(np.float32),
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
