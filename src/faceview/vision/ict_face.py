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
        if style == "xray":
            base = dict(_SCIFI_PALETTES["xray"])
            dr, dg, db = _xray_mood_offset(params)
            for mat in ("M_Face", "M_BackHead"):
                r, g, b = base[mat]
                base[mat] = (
                    float(np.clip(r + dr, 0.0, 1.0)),
                    float(np.clip(g + dg, 0.0, 1.0)),
                    float(np.clip(b + db, 0.0, 1.0)),
                )
            return base
        return _SCIFI_PALETTES[style]

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


def _emit_pulse_for(style: str) -> float:
    """Time-varying emissive pulse magnitude for the given style.

    Reads ``time.monotonic`` so the pulse advances frame-to-frame
    without needing the avatar tick to plumb a phase argument.
    """
    if style not in _STYLE_PULSE:
        return 0.0
    base, amp, hz = _STYLE_PULSE[style]
    import math, time
    return float(base + amp * math.sin(2.0 * math.pi * hz * time.monotonic()))


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
    # Jelly composite mode: anatomy underlay + translucent xray skin.
    # Diverts before the standard GL path because it composes two
    # renderers; falls back to plain xray if BP3D meshes aren't built.
    if getattr(params, "_persona_style", "natural") == "jelly":
        return _render_jelly_composite(params, size)

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
    full_coefs = {**arkit_coefs, **identity_w}
    verts = apply_blendshapes(model, full_coefs)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4

    bgr = _render_via_moderngl(verts, model.triangles, size,
                                  yaw, pitch, params)

    # Sci-fi bloom — extract bright pixels and add a blurred halo back
    # over the original. Reads as glowing eyes / hot teeth / etc.
    style = getattr(params, "_persona_style", "natural")
    if style != "natural":
        bgr = _apply_bloom(bgr, style)

    # 2D hair overlay (off by default — the procedural overlay is
    # rough; the bald ICT head reads cleaner). Set
    # ``params._enable_hair = True`` if you want to opt in.
    if getattr(params, "_enable_hair", False):
        bgr = _composite_hair_overlay(bgr, params, yaw)
    return bgr


# Tokens identifying neck / throat / suprahyoid muscles in the BP3D
# muscle catalog. We exclude these from the jelly-mode anatomy so
# the BP3D layer only fills the head (skull + face muscles + eyes
# + ears + nose) — matching ICT's head silhouette.
_NECK_MUSCLE_TOKENS = (
    "Cap.",          # Longus Capitis / Rectus Cap. / Obliquus Cap.
    "Colli",         # Longus Colli
    "Sterno",        # Sternocleidomastoid / Sternohyoid / Sternothyroid
    "Thyro",         # Thyrohyoid
    "Hyoid",         # Stylohyoid / Mylohyoid / Geniohyoid / Sternohyoid (suffix)
    "Scalene",
    "Levator Scap",  # Levator Scapulae
    "Omohyoid",
    "Platysma",
    "Digastric",
)


def _is_neck_muscle(name: str) -> bool:
    return any(tok in name for tok in _NECK_MUSCLE_TOKENS)


def _blendshape_anchor(model, verts: np.ndarray, name: str,
                          top_k: int = 60) -> np.ndarray | None:
    """Centroid of the top-K vertices most affected by a blendshape.

    Blendshape names directly label anatomical regions (mouthSmile_L
    moves the left lip corner verts, browOuterUp_R moves the right
    outer brow verts). The centroid of the K verts with the largest
    delta magnitude for that blendshape is a stable landmark for
    that region.
    """
    idx = model.name_to_idx.get(name)
    if idx is None:
        return None
    mags = np.linalg.norm(model.deltas[idx], axis=1)
    top = np.argsort(-mags)[:top_k]
    return verts[top].mean(axis=0)


def _ict_feature_points_3d_for(
    identity_weights: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """ICT extended-landmark feature anchors in the model's local frame.

    18 anchors covering the head silhouette (crown, temple L/R,
    chin), the brow line (inner/outer L/R), the mid-face (cheeks
    L/R, nose tip), the eye line (eye L/R), the mouth line (mouth
    centre, mouth corners L/R), and the jaw line (jaw L/R).
    Computed from the *deformed* mesh so per-persona identity
    weights re-anchor the alignment correctly.
    """
    m = load_ict_model()
    regions = _vertex_regions()
    eyelid = regions["eyelid"]
    lips = regions["lips"]
    if not eyelid.any() or not lips.any():
        return {}

    iw = {k: float(v) for k, v in (identity_weights or {}).items()
          if isinstance(v, (int, float))}
    verts = apply_blendshapes(m, iw) if iw else m.vertices

    out: dict[str, np.ndarray] = {}

    # --- eye line ---
    eyelid_verts = verts[eyelid]
    left_eye = eyelid_verts[eyelid_verts[:, 0] > 0]
    right_eye = eyelid_verts[eyelid_verts[:, 0] < 0]
    if not len(left_eye) or not len(right_eye):
        return {}
    out["eye_L"] = left_eye.mean(axis=0)
    out["eye_R"] = right_eye.mean(axis=0)

    # --- brow line (blendshape-driven) ---
    for ict_name, key in [
        ("browInnerUp_L", "brow_inner_L"),
        ("browInnerUp_R", "brow_inner_R"),
        ("browOuterUp_L", "brow_outer_L"),
        ("browOuterUp_R", "brow_outer_R"),
    ]:
        c = _blendshape_anchor(m, verts, ict_name)
        if c is not None:
            out[key] = c

    # --- cheeks ---
    for ict_name, key in [
        ("cheekPuff_L", "cheek_L"),
        ("cheekPuff_R", "cheek_R"),
    ]:
        c = _blendshape_anchor(m, verts, ict_name)
        if c is not None:
            out[key] = c

    # --- nose tip (sneer pulls the alar/tip) ---
    for ict_name, key in [
        ("noseSneer_L", "nose_L"),
        ("noseSneer_R", "nose_R"),
    ]:
        c = _blendshape_anchor(m, verts, ict_name)
        if c is not None:
            out[key] = c

    # --- mouth line ---
    lips_verts = verts[lips]
    out["mouth"] = lips_verts.mean(axis=0)
    for ict_name, key in [
        ("mouthSmile_L", "mouth_corner_L"),
        ("mouthSmile_R", "mouth_corner_R"),
    ]:
        c = _blendshape_anchor(m, verts, ict_name)
        if c is not None:
            out[key] = c

    # --- jaw / chin / temple / crown / ear — silhouette anchors via
    # M_Face vertex extrema, NOT centroids. They pin the actual
    # outline so TPS warp can fit the BP3D anatomy inside the ICT
    # head shape.
    face_mat_idx = next(
        (i for i, n in enumerate(m.materials) if n == "M_Face"), -1,
    )
    backhead_mat_idx = next(
        (i for i, n in enumerate(m.materials) if n == "M_BackHead"), -1,
    )
    if face_mat_idx >= 0:
        face_v_mask = np.zeros(len(verts), dtype=bool)
        for ti, mi in enumerate(m.tri_materials):
            if mi == face_mat_idx or mi == backhead_mat_idx:
                for v in m.triangles[ti]:
                    face_v_mask[v] = True
        face_v = verts[face_v_mask]
        mouth_y = float(out["mouth"][1])
        eye_y = float((out["eye_L"][1] + out["eye_R"][1]) / 2.0)

        # Jaw: jawbone-line anchors (already exist as masseter ones
        # in BP3D — here we anchor on the lower-face surface verts).
        below_mouth = face_v[face_v[:, 1] < mouth_y]
        if len(below_mouth) > 50:
            jaw_l = below_mouth[below_mouth[:, 0] > 5]
            jaw_r = below_mouth[below_mouth[:, 0] < -5]
            if len(jaw_l):
                out["jaw_L"] = jaw_l.mean(axis=0)
            if len(jaw_r):
                out["jaw_R"] = jaw_r.mean(axis=0)

            # Chin: lowest 1 % of midline face verts → the chin tip.
            chin_band = below_mouth[np.abs(below_mouth[:, 0]) < 10]
            if len(chin_band):
                y_thr = np.percentile(chin_band[:, 1], 5)
                tip = chin_band[chin_band[:, 1] < y_thr]
                if len(tip):
                    out["chin"] = tip.mean(axis=0)

        # Temple: lateral-most face verts at the eye y level (not the
        # ear — closer to the brow ridge). Wider band for robustness.
        eye_band = face_v[
            (face_v[:, 1] > eye_y - 8) & (face_v[:, 1] < eye_y + 25)
        ]
        if len(eye_band) > 50:
            x_thr_L = np.percentile(eye_band[:, 0], 97)
            x_thr_R = np.percentile(eye_band[:, 0], 3)
            t_l = eye_band[eye_band[:, 0] > x_thr_L]
            t_r = eye_band[eye_band[:, 0] < x_thr_R]
            if len(t_l):
                out["temple_L"] = t_l.mean(axis=0)
            if len(t_r):
                out["temple_R"] = t_r.mean(axis=0)

        # Crown: highest 0.5 % of face / back-head Y values — the
        # actual top of the head, not the forehead.
        if len(face_v):
            y_thr = np.percentile(face_v[:, 1], 99.5)
            crown = face_v[face_v[:, 1] > y_thr]
            if len(crown):
                out["crown"] = crown.mean(axis=0)

        # Ear-line: lateral-most verts a bit *below* the eye level
        # (where ears actually sit). Mirror BP3D's ear anchor.
        ear_band = face_v[
            (face_v[:, 1] > eye_y - 30) & (face_v[:, 1] < eye_y - 5)
        ]
        if len(ear_band) > 30:
            x_thr_L = np.percentile(ear_band[:, 0], 97)
            x_thr_R = np.percentile(ear_band[:, 0], 3)
            e_l = ear_band[ear_band[:, 0] > x_thr_L]
            e_r = ear_band[ear_band[:, 0] < x_thr_R]
            if len(e_l):
                out["ear_L"] = e_l.mean(axis=0)
            if len(e_r):
                out["ear_R"] = e_r.mean(axis=0)

    return {k: v.astype(np.float32) for k, v in out.items()}


@lru_cache(maxsize=1)
def _ict_feature_points_3d() -> dict[str, np.ndarray]:
    """Neutral-mesh anchors (compat shim for callers without identity)."""
    return _ict_feature_points_3d_for(None)


@lru_cache(maxsize=1)
def _bp3d_feature_points_3d() -> dict[str, np.ndarray] | None:
    """BP3D extended-landmark feature anchors in BP3D's local frame.

    Each landmark is the centroid of an FMA-keyed mesh (or pair of
    meshes for L/R-symmetric structures). Returns None if any mesh
    is missing on disk.
    """
    try:
        from faceview.vision.anatomy_meshes import (
            list_available_meshes, load_mesh,
        )
    except Exception:
        return None

    # NOTE: BP3D's gpu_renderer pre-rotation (ry180) mirrors X in
    # screen space — a muscle anatomically on subject's left side
    # ends up on screen-LEFT after rendering. ICT, with no mirror,
    # puts subject's-left features on screen-RIGHT. To make the
    # similarity-transform fit work without needing a reflection
    # (which estimateAffinePartial2D can't produce), we label BP3D
    # anchors by SCREEN position: "eye_L" = the eye that lands on
    # the LEFT half of the rendered frame = anatomically subject's
    # right (FMA*_R). The names are chosen to match the ICT
    # screen-position convention, not the underlying anatomy.
    fma_single: dict[str, str] = {
        # Eye line — screen labels.
        "eye_L":          "FMA46782",  # screen-L = anat. R (Orbic. Oculi Orb. R)
        "eye_R":          "FMA46783",  # screen-R = anat. L
        # Brow line.
        "brow_inner_L":   "FMA46796",  # Corrugator Sup. R (anat.)
        "brow_inner_R":   "FMA46797",
        "brow_outer_L":   "FMA46759",  # Frontalis R (anat.)
        "brow_outer_R":   "FMA46760",
        # Cheeks (Zygomatic Maj.).
        "cheek_L":        "FMA46812",  # anat. R
        "cheek_R":        "FMA46813",
        # Nose alae (Lev. Labii Alae).
        "nose_L":         "FMA46803",  # anat. R
        "nose_R":         "FMA46804",
        # Mouth line.
        "mouth":          "FMA46841",  # Orbicularis Oris (midline)
        "mouth_corner_L": "FMA46823",  # Lev. Anguli Oris R (anat.)
        "mouth_corner_R": "FMA46824",
        # Jaw (Masseter Sup.).
        "jaw_L":          "FMA49001",  # anat. R
        "jaw_R":          "FMA49002",
        # Temple (Temporalis).
        "temple_L":       "FMA49007",  # anat. R
        "temple_R":       "FMA49008",
    }
    avail = set(list_available_meshes())
    out: dict[str, np.ndarray] = {}
    for key, fma in fma_single.items():
        if fma not in avail:
            continue
        out[key] = load_mesh(fma).vertices.mean(axis=0).astype(np.float32)

    # Silhouette anchors via vertex *extrema*, not centroids — these
    # pin the actual outline, not interior centres.
    # Crown: topmost (max-y) vertex of the Frontal Bone (FMA52734).
    if "FMA52734" in avail:
        v = load_mesh("FMA52734").vertices
        top = v[v[:, 1] > np.percentile(v[:, 1], 99)]
        out["crown"] = top.mean(axis=0).astype(np.float32)
    # Chin: bottom-most vertex of the Mandible (FMA52748).
    if "FMA52748" in avail:
        v = load_mesh("FMA52748").vertices
        bot = v[v[:, 1] < np.percentile(v[:, 1], 1)]
        # Centred on x=0 (on-midline chin tip).
        bot = bot[np.abs(bot[:, 0]) < 15]
        if len(bot):
            out["chin"] = bot.mean(axis=0).astype(np.float32)
    # Override temple_L / R with the lateral-most extents of the
    # Temporalis muscles so they land on the actual head silhouette
    # rather than the muscle interior. NOTE: gpu_renderer's ry180
    # flips X — anatomical subject's-right (raw -X) ends up at
    # screen-LEFT (low pixel x). With our screen-position label
    # convention, "temple_L" = screen-LEFT = subject's right
    # = lateral-most negative raw X of Temporalis R.
    if "FMA49007" in avail:                                # Temporalis R (anat.)
        v = load_mesh("FMA49007").vertices
        side = v[v[:, 0] < np.percentile(v[:, 0], 5)]      # most-neg raw X
        out["temple_L"] = side.mean(axis=0).astype(np.float32)
    if "FMA49008" in avail:                                # Temporalis L (anat.)
        v = load_mesh("FMA49008").vertices
        side = v[v[:, 0] > np.percentile(v[:, 0], 95)]     # most-pos raw X
        out["temple_R"] = side.mean(axis=0).astype(np.float32)
    # Side-of-head anchors via the (combined L+R) Ear mesh. Same
    # screen-position logic — most-negative raw X = screen LEFT.
    if "FMA52780" in avail:
        v = load_mesh("FMA52780").vertices
        screen_left = v[v[:, 0] < np.percentile(v[:, 0], 5)]
        screen_right = v[v[:, 0] > np.percentile(v[:, 0], 95)]
        if len(screen_left) and len(screen_right):
            out["ear_L"] = screen_left.mean(axis=0).astype(np.float32)
            out["ear_R"] = screen_right.mean(axis=0).astype(np.float32)

    if "eye_L" not in out or "mouth" not in out:
        return None
    return out


def _project_ict_to_pixel(
    points_3d: dict[str, np.ndarray],
    yaw: float, pitch: float, size: tuple[int, int],
    identity_weights: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """Project ICT model-space points through the same MVP the ICT
    renderer uses, returning pixel coordinates.

    When ``identity_weights`` are non-empty the renderer normalises
    on the *deformed* bbox — we mirror that here so anchor pixels
    actually land where the renderer drew them.
    """
    m = load_ict_model()
    iw = {k: float(v) for k, v in (identity_weights or {}).items()
          if isinstance(v, (int, float))}
    verts = apply_blendshapes(m, iw) if iw else m.vertices
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
    M = ry @ rx @ flip

    w, h = size
    out: dict[str, tuple[float, float]] = {}
    for name, p in points_3d.items():
        v = (p - centre) * scale
        v = M @ v
        # Orthographic to NDC then to pixel. Frag y was flipped via
        # np.flipud after read → so screen y matches our negated y.
        # We undo the flipud here: y_pix = (1 - (v.y + 1)/2) * h
        x_pix = (v[0] + 1.0) / 2.0 * w
        y_pix = (1.0 - (v[1] + 1.0) / 2.0) * h
        out[name] = (float(x_pix), float(y_pix))
    return out


def _project_bp3d_to_pixel(
    points_3d: dict[str, np.ndarray],
    specs: list, yaw: float, pitch: float, size: tuple[int, int],
) -> dict[str, tuple[float, float]]:
    """Project BP3D model-space points through the gpu_renderer MVP."""
    from faceview.vision.anatomy_meshes import load_mesh

    rx0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    ry180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    R_pre = ry180 @ rx0  # BP3D → screen reorientation (gpu_renderer)

    bone_specs = [s for s in specs if s.category == "bone"]
    ref = bone_specs if bone_specs else specs
    all_verts: list[np.ndarray] = []
    for s in ref:
        m = load_mesh(s.fma)
        all_verts.append(m.vertices @ R_pre.T)
    verts_all = np.vstack(all_verts)
    vmin = verts_all.min(axis=0)
    vmax = verts_all.max(axis=0)
    centre = (vmin + vmax) / 2.0
    span = float(np.linalg.norm(vmax - vmin))
    scale = 1.7 / max(span, 1e-6)

    cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
    cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
    ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]],
                    dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cp_, -sp_], [0, sp_, cp_]],
                    dtype=np.float32)

    w, h = size
    out: dict[str, tuple[float, float]] = {}
    for name, p in points_3d.items():
        v = R_pre @ p           # apply BP3D-screen pre-rotation
        v = (v - centre) * scale
        v = ry @ rx @ v
        x_pix = (v[0] + 1.0) / 2.0 * w
        y_pix = (1.0 - (v[1] + 1.0) / 2.0) * h
        out[name] = (float(x_pix), float(y_pix))
    return out


def _tps_warp(
    img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
    output_shape: tuple[int, int], regularisation: float = 0.0,
    border_value: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Thin-plate-spline warp via cv2.remap.

    Builds the inverse map (output→input) by solving the TPS linear
    system for an N-anchor non-rigid warp, then evaluating it on
    every output pixel. ``regularisation`` adds λI to the kernel
    matrix — small λ stiffens the warp toward affine, large λ → no
    bending (pure affine fit).

    src_pts, dst_pts: (N, 2) arrays of corresponding 2D points.
    output_shape: (H, W) of output image.
    Returns: warped (H, W, 3) BGR image.
    """
    import cv2

    h, w = output_shape
    n = len(src_pts)
    if n < 3 or len(src_pts) != len(dst_pts):
        return img

    # K[i, j] = U(||dst[i] - dst[j]||) — RBF on the *target* points
    # because we're learning the inverse map (output → input).
    diff = dst_pts[:, None, :] - dst_pts[None, :, :]
    R2 = (diff * diff).sum(axis=2).astype(np.float64)
    eps = 1e-12
    K = R2 * np.log(R2 + eps)
    P = np.hstack([np.ones((n, 1)), dst_pts]).astype(np.float64)
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K + regularisation * np.eye(n)
    L[:n, n:] = P
    L[n:, :n] = P.T
    Y = np.zeros((n + 3, 2), dtype=np.float64)
    Y[:n] = src_pts
    try:
        coefs = np.linalg.solve(L, Y)
    except np.linalg.LinAlgError:
        return img
    rbf_w = coefs[:n]    # (n, 2)
    aff = coefs[n:]      # (3, 2) — [const, x, y]

    # Pixel grid in output (dst) space.
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack(
        [xx.ravel().astype(np.float64), yy.ravel().astype(np.float64)],
        axis=1,
    )                                  # (H*W, 2)
    g_diff = grid[:, None, :] - dst_pts[None, :, :].astype(np.float64)
    g_R2 = (g_diff * g_diff).sum(axis=2)
    U = g_R2 * np.log(g_R2 + eps)        # (H*W, n)
    Pg = np.hstack([np.ones((h * w, 1), dtype=np.float64), grid])  # (H*W, 3)
    map_xy = Pg @ aff + U @ rbf_w        # (H*W, 2) — source positions

    map_x = map_xy[:, 0].reshape(h, w).astype(np.float32)
    map_y = map_xy[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in border_value),
    )


def _align_anatomy_to_ict(
    anatomy: np.ndarray, bg_rgb: tuple[int, int, int],
    specs: list, yaw: float, pitch: float, size: tuple[int, int],
    identity_weights: dict[str, float] | None = None,
    return_pixels: bool = False,
):
    """Multi-landmark rigid (similarity) warp.

    Projects ICT + BP3D anchor sets to pixel space through each
    renderer's MVP (using keys present in both dicts), then fits an
    LSQ similarity transform — uniform scale + rotation +
    translation — over all matched anchors. Preserves BP3D mesh
    proportions; eyes/mouth/jaw line up as closely as a rigid fit
    allows without distorting individual structures.

    Returns the warped image. If ``return_pixels`` is True, also
    returns ``(ict_pix, bp3_pix, used_keys)`` for the assessment
    tool to overlay anchor markers.
    """
    import cv2

    w, h = size
    bg_arr = np.array(bg_rgb, dtype=np.float32)

    ict_3d = _ict_feature_points_3d_for(identity_weights)
    bp3_3d = _bp3d_feature_points_3d()
    if not ict_3d or not bp3_3d:
        return (anatomy, ({}, {}, [])) if return_pixels else anatomy

    ict_pix = _project_ict_to_pixel(ict_3d, yaw, pitch, size,
                                       identity_weights=identity_weights)
    bp3_pix = _project_bp3d_to_pixel(bp3_3d, specs, yaw, pitch, size)
    used = [k for k in ict_pix if k in bp3_pix]
    if len(used) < 5:
        return (anatomy, (ict_pix, bp3_pix, used)) if return_pixels else anatomy

    src = np.array([bp3_pix[k] for k in used], dtype=np.float32)
    dst = np.array([ict_pix[k] for k in used], dtype=np.float32)

    # Rigid SIMILARITY transform (uniform scale + rotation +
    # translation, no shear). LSQ-fit over all 19 anchors via
    # estimateAffinePartial2D — stable and preserves BP3D mesh
    # proportions. (Tried TPS non-rigid: introduces visible mesh
    # distortion that doesn't read as "anatomy".)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(src[:3], dst[:3])
    warped = cv2.warpAffine(
        anatomy, M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(c) for c in bg_arr),
    )
    if return_pixels:
        return warped, (ict_pix, bp3_pix, used)
    return warped


def _render_jelly_composite(params, size: tuple[int, int]) -> np.ndarray:
    """Anatomy-underlay jelly-person mode.

    Renders BP3D head anatomy (skull / muscles / eyes / cartilages)
    at the same yaw/pitch as the ICT face, then alpha-blends an xray
    ICT skin over it. Skin pixels go translucent so the anatomy
    shows through; eye / teeth glows stay opaque so they read as
    self-lit through the skin.

    The BP3D and ICT meshes have different topologies (no vertex
    correspondence) so only **head pose and jaw open** sync — smile,
    pucker, and other ICT blendshape deformations don't propagate
    onto the BP3D anatomy. For a translucent xray look that's
    acceptable; the deeper anatomy is mostly bone (rigid) and the
    mismatch is hidden behind the ICT lip surface.
    """
    import cv2

    # First render the ICT face with style temporarily set to xray
    # so the palette + shaders + pulses all do the right thing.
    style_in = getattr(params, "_persona_style", "natural")
    params._persona_style = "xray"
    try:
        ict_bgr = render_face_ict(params, size)
    finally:
        params._persona_style = style_in

    # Render the BP3D anatomy underlay through a GpuRenderer that
    # *shares the ICT moderngl context*. moderngl 5.x exposes no
    # context-switch API, so two standalone contexts in the same
    # thread can't alternate — sharing the context keeps both
    # renderers usable in one composite.
    try:
        from faceview.vision.anatomy_meshes import (
            list_available_meshes, meshes_available,
        )
        if not meshes_available():
            return ict_bgr
        from faceview.vision.anatomy_catalog import specs_for_layer_set
    except Exception:
        return ict_bgr

    avail = set(list_available_meshes())
    raw_specs = [s for s in specs_for_layer_set("features") if s.fma in avail]
    # Restrict to head-only meshes — drop cervical vertebrae and the
    # neck-muscle group so the BP3D anatomy doesn't extend below the
    # ICT chin/collar line.
    specs = [s for s in raw_specs if s.category != "vertebra"
             and not _is_neck_muscle(s.name)]
    if not specs:
        return ict_bgr

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    bg_rgb = _hex_to_rgb(getattr(params, "background", "#000810"))
    try:
        anatomy_bgr = _shared_anatomy_renderer().render(
            specs, size, yaw=yaw, pitch=pitch, bg=bg_rgb,
        )
    except Exception:
        return ict_bgr

    # Feature-anchor warp: project ICT (deformed by identity PCA)
    # and BP3D eye-L/R + mouth + jaw-L/R centroids to pixel space
    # through each renderer's MVP, then 5-point LSQ-affine warp BP3D
    # so all anchors overlay ICT's. Identity-aware so young /
    # elder personas get correctly-scaled anatomy.
    iw_in = getattr(params, "identity_weights", None) or {}
    iw = {k: float(v) for k, v in iw_in.items()
          if isinstance(v, (int, float))}
    anatomy_bgr = _align_anatomy_to_ict(
        anatomy_bgr, bg_rgb, specs, yaw, pitch, size,
        identity_weights=iw,
    )

    # Cool-tint and brighten the anatomy so it reads as part of the
    # xray aesthetic instead of warm bone — crush warm channels hard
    # and lift overall brightness so muscles/skull pop through skin.
    anatomy_f = anatomy_bgr.astype(np.float32)
    cool_tint = np.array([1.80, 1.40, 0.35], dtype=np.float32)  # BGR
    anatomy_bgr = np.clip(anatomy_f * cool_tint + 12.0,
                            0, 255).astype(np.uint8)

    # Build per-pixel alpha for the ICT layer. Pixels much brighter
    # than background are face/eyes/teeth → high alpha (opaque).
    # Mid-tone skin → low alpha (anatomy shows through).
    import cv2
    luma = ict_bgr.max(axis=2).astype(np.float32)
    bg_ref = max(np.array(bg_rgb).max(), 8.0)
    face_mask_hard = (luma > bg_ref + 12).astype(np.uint8)
    # Soft silhouette mask of the ICT head — used to clip the BP3D
    # anatomy so it only shows inside the ICT outline (BP3D extends
    # further down into the neck/shoulders).
    silhouette = cv2.GaussianBlur(face_mask_hard.astype(np.float32),
                                    (0, 0), sigmaX=4.0, sigmaY=4.0)
    silhouette = np.clip(silhouette, 0.0, 1.0)

    # Skin α scales smoothly with luminance — bright glow ~0.85,
    # plain skin ~0.20. Lower baseline lets the anatomy show
    # clearly through neutral skin tones too.
    alpha = np.clip(0.05 + (luma / 255.0) * 0.85, 0.0, 0.92)
    alpha = alpha * face_mask_hard.astype(np.float32)
    # Knock down bright-but-not-glowing pixels (face highlights from
    # SSS terminator) so we don't get a thick opaque cyan band on the
    # forehead/cheeks. Glow regions (~250+) stay opaque.
    bright_skin = (luma > 130) & (luma < 230)
    alpha[bright_skin] *= 0.35

    # Clip anatomy to the ICT silhouette: outside the silhouette
    # show plain background; inside, show anatomy.
    bg_color = np.array([bg_rgb[2], bg_rgb[1], bg_rgb[0]],
                          dtype=np.float32)  # bg_rgb is RGB; cv2 is BGR
    s3 = silhouette[:, :, None]
    anatomy_clipped = (anatomy_bgr.astype(np.float32) * s3
                       + bg_color * (1.0 - s3))

    a3 = alpha[:, :, None]
    composite = ict_bgr.astype(np.float32) * a3 \
        + anatomy_clipped * (1.0 - a3)
    out = np.clip(composite, 0.0, 255.0).astype(np.uint8)

    # Re-apply a softer bloom over the composite so the eye glow
    # halos through the anatomy underlay.
    return _apply_bloom(out, "xray")


def _apply_bloom(bgr: np.ndarray, style: str) -> np.ndarray:
    """Cheap bloom — Gaussian blur of bright pixels mixed back in.

    Pulls a high-pass mask above ``threshold``, blurs it large, and
    additively blends. Per-style amplitude tunes the strength.
    Total cost ≈ 3 ms at 320×320 (cv2 GaussianBlur).
    """
    import cv2  # already imported at top in test paths
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
    rend._emit_pulse = _emit_pulse_for(style)
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
        S = np.eye(4, dtype=np.float32) * scale
        S[3, 3] = 1.0
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


@lru_cache(maxsize=1)
def _shared_anatomy_renderer():
    """BP3D anatomy GpuRenderer sharing the ICT moderngl context.

    Built lazily on first jelly-composite call. Sharing the context
    avoids the moderngl-5 multi-context limitation (no context-
    switching API → only one standalone context can be active per
    thread → second renderer's draws turn into black frames).
    """
    from faceview.vision.gpu_renderer import _GpuRenderer
    ict = _ensure_renderer()
    return _GpuRenderer(ctx=ict.ctx)


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
