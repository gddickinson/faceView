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

_MATERIAL_COLORS: dict[str, tuple[float, float, float]] = {
    "M_Face":         (0.95, 0.82, 0.74),    # natural skin (less pink)
    "M_BackHead":     (0.88, 0.76, 0.68),
    "M_GumsTongue":   (0.62, 0.34, 0.34),    # dusky pink-red
    "M_Teeth":        (0.96, 0.93, 0.86),    # ivory
    "M_ScleraLeft":   (0.97, 0.94, 0.88),    # bright sclera
    "M_ScleraRight":  (0.97, 0.94, 0.88),
    "M_IrisLeft":     (0.30, 0.45, 0.55),    # blue-grey iris (visible)
    "M_IrisRight":    (0.30, 0.45, 0.55),
    "M_LacrimalFluid":(0.94, 0.88, 0.82),    # tear duct
    "M_EyeBlend":     (0.92, 0.78, 0.70),    # eyelid skin (matches face)
    "M_EyeOcclusion": (0.45, 0.30, 0.25),    # softer inner socket
    "M_EyeLashes":    (0.10, 0.07, 0.05),
}

# Subsurface scattering colour — warm flesh tint, less aggressive red.
_SSS_TINT = (0.78, 0.46, 0.40)


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
    from faceview.vision.anatomy import face_params_to_au_values
    from faceview.vision.arkit_blendshapes import au_to_arkit_values

    model = load_ict_model()
    au_values = face_params_to_au_values(params)
    arkit_coefs = au_to_arkit_values(au_values)

    # Layer identity (PCA) coefficients on top of the expression
    # coefficients — identity reshapes the underlying head, expression
    # animates it. ``identity_weights`` lives on FaceParams (set by
    # apply_persona) and is keyed by ICT identity names like
    # ``identity001``.
    identity_w = dict(getattr(params, "identity_weights", {}) or {})
    full_coefs = {**arkit_coefs, **identity_w}
    verts = apply_blendshapes(model, full_coefs)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4

    bgr = _render_via_moderngl(verts, model.triangles, size,
                                  yaw, pitch, params)

    # 2D hair overlay so the head doesn't read as bald. The mesh
    # itself doesn't include hair, so we draw a simple cap through
    # QPainter on top of the rendered output. Persona.hair_color
    # drives the colour.
    if not getattr(params, "_no_hair", False):
        bgr = _composite_hair_overlay(bgr, params, yaw)
    return bgr


def _composite_hair_overlay(bgr: np.ndarray, params, yaw: float) -> np.ndarray:
    """Draw a simple hair cap on top of the rendered ICT head.

    Approach: detect the head silhouette top via background mask,
    then paint an opaque hair cap above the brow line with strand
    highlights. Cheap CPU work, ~1 ms.
    """
    from PIL import Image
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QPainter, QPainterPath, QPen,
    )

    h, w, _ = bgr.shape
    rgb = bgr[:, :, ::-1].copy()
    img = Image.fromarray(rgb)

    # Find the top-most non-background pixel per column to follow the
    # head silhouette.
    bg_mask = (rgb.sum(axis=2) < 60)   # near-black background
    fg_mask = ~bg_mask
    cols_with_face = np.any(fg_mask, axis=0)
    top_y = np.full(w, -1, dtype=np.int32)
    if np.any(cols_with_face):
        for x in range(w):
            if cols_with_face[x]:
                top_y[x] = np.argmax(fg_mask[:, x])

    # Find left/right extents (head width).
    fg_xs = np.where(cols_with_face)[0]
    if len(fg_xs) < 4:
        return bgr
    left, right = int(fg_xs[0]), int(fg_xs[-1])

    # Convert PIL → QImage for QPainter drawing.
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    p = QPainter(qimg)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    hair_hex = getattr(params, "hair_color", "#2a1808")
    hair = QColor(hair_hex)
    hair_light = hair.lighter(140)

    # Build a hair cap from the top silhouette: above each top_y
    # pixel, pad upward by a fraction of head height so the cap
    # extends above the skull.
    head_h = max(1, int((right - left) * 1.2))
    cap_pad = int(0.08 * head_h)

    cap_path = QPainterPath()
    started = False
    # Trace top silhouette left → right, slightly above the head.
    for x in range(left, right + 1):
        ty = top_y[x]
        if ty < 0:
            continue
        y = max(0, ty - 4)
        if not started:
            cap_path.moveTo(QPointF(x, y))
            started = True
        else:
            cap_path.lineTo(QPointF(x, y))
    # Close above the head.
    cap_path.lineTo(QPointF(right, max(0, top_y[right] - cap_pad)))
    cap_path.lineTo(QPointF((left + right) // 2,
                              max(0, int(top_y[(left + right) // 2]) - 2 * cap_pad)))
    cap_path.lineTo(QPointF(left, max(0, top_y[left] - cap_pad)))
    cap_path.closeSubpath()

    p.setBrush(QBrush(hair))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPath(cap_path)

    # Side fringe — darker streaks down past temples.
    fringe = QPainterPath()
    fringe.moveTo(QPointF(left + 0.05 * (right - left),
                            top_y[left] if top_y[left] >= 0 else 0))
    fringe.cubicTo(
        QPointF(left + 0.20 * (right - left), top_y[left + (right - left) // 4] + 4),
        QPointF(left + 0.45 * (right - left), top_y[(left + right) // 2] - 2),
        QPointF(left + 0.55 * (right - left), top_y[(left + right) // 2] + 4),
    )
    fringe.cubicTo(
        QPointF(left + 0.45 * (right - left), top_y[(left + right) // 2] - 8),
        QPointF(left + 0.20 * (right - left), top_y[left + (right - left) // 4] - 8),
        QPointF(left + 0.05 * (right - left),
                 (top_y[left] if top_y[left] >= 0 else 0) - 4),
    )
    fringe.closeSubpath()
    p.drawPath(fringe)

    # Strand highlights.
    pen = QPen(hair_light, 1.4)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    n_strands = 7
    for i in range(n_strands):
        f = (i + 0.5) / n_strands
        x0 = int(left + f * (right - left))
        y0 = max(0, top_y[x0] - 4)
        x1 = x0 + 14
        y1 = y0 - 16
        p.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    p.end()

    # QImage → numpy → BGR
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

    return rend.render(
        verts=verts.astype(np.float32),
        normals=vert_norms.astype(np.float32),
        triangles=triangles.astype(np.uint32),
        vert_colors=_per_vertex_colors().astype(np.float32),
        vert_spec=_per_vertex_specular().astype(np.float32),
        centre=centre.astype(np.float32),
        scale=float(scale),
        yaw=yaw, pitch=pitch,
        size=size,
        bg=_hex_to_rgb(getattr(params, "background", "#0a0d12")),
    )


@lru_cache(maxsize=1)
def _per_vertex_colors() -> np.ndarray:
    """Return (N, 3) RGB float colors keyed off the OBJ's material tags.

    Vertex color = mean of the colors of every triangle that
    references it. Vertices shared by face and non-face materials
    end up blended, which gives a natural transition.
    """
    m = load_ict_model()
    n_verts = len(m.vertices)
    accum = np.zeros((n_verts, 3), dtype=np.float64)
    counts = np.zeros(n_verts, dtype=np.int32)
    fallback = _MATERIAL_COLORS.get("M_Face", (0.85, 0.7, 0.6))
    for ti, mat_idx in enumerate(m.tri_materials):
        color = (_MATERIAL_COLORS.get(m.materials[mat_idx], fallback)
                 if 0 <= mat_idx < len(m.materials) else fallback)
        for v in m.triangles[ti]:
            accum[v] += color
            counts[v] += 1
    counts = np.maximum(counts, 1)
    return (accum / counts[:, None]).astype(np.float32)


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
out vec3 v_norm;
out vec3 v_color;
out float v_spec;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_norm = u_norm_mat * in_norm;
    v_color = in_color;
    v_spec = in_spec;
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
in vec3 v_norm;
in vec3 v_color;
in float v_spec;
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
        self.prog["u_sss_tint"].value = _SSS_TINT
        self.prog["u_ambient"].value = 0.32
        self.prog["u_specular"].value = 0.30
        self.prog["u_shininess"].value = 22.0

        vbo = self.ctx.buffer(verts.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())
        cbo = self.ctx.buffer(vert_colors.tobytes())
        sbo = self.ctx.buffer(vert_spec.tobytes())
        ibo = self.ctx.buffer(triangles.tobytes())
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, "3f", "in_pos"),
            (nbo, "3f", "in_norm"),
            (cbo, "3f", "in_color"),
            (sbo, "1f", "in_spec"),
        ], ibo)
        vao.render(self.mgl.TRIANGLES)
        vao.release()
        vbo.release()
        nbo.release()
        cbo.release()
        sbo.release()
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
