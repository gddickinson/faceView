"""GPU-accelerated photo-anatomical renderer (Apple Metal via OpenGL 4.1).

Replaces the CPU rasteriser in :mod:`faceview.vision.anatomy_meshes` for
the ``faceforge_3d_gpu`` render mode. Loads the same BP3D STL meshes
(via the catalog), uploads vertices + normals + per-mesh material into
VBOs once, and renders Phong-shaded triangles in a single GL call per
mesh through an offscreen FBO.

Why moderngl: it's a thin Python wrapper over OpenGL 3.3+, ships
standalone offscreen contexts, and on Apple Silicon goes through the
Metal compatibility layer so we get GPU acceleration without writing
Metal shaders directly.

Falls back gracefully when:
- ``moderngl`` isn't installed (raise :class:`MissingDependency` with
  install hint).
- ``faceforge_3d_gpu`` is requested but the BP3D meshes aren't copied
  yet (raise the standard mesh-missing error).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.vision.anatomy_catalog import specs_for_layer_set
from faceview.vision.anatomy_meshes import (
    list_available_meshes,
    load_mesh,
    mesh_dir,
    meshes_available,
)


VERT_SHADER = """
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_norm;
out vec3 v_norm;
out vec3 v_world;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_world = (u_model * vec4(in_pos, 1.0)).xyz;
    v_norm = mat3(u_model) * in_norm;
}
"""

FRAG_SHADER = """
#version 330
uniform vec3 u_color;
uniform float u_opacity;
uniform float u_shininess;
uniform vec3 u_light_dir;
uniform float u_ambient;
uniform float u_specular;

in vec3 v_norm;
in vec3 v_world;
out vec4 frag;

void main() {
    vec3 n = normalize(v_norm);
    vec3 l = normalize(-u_light_dir);
    float diff = abs(dot(n, l));        // double-sided
    vec3 view = vec3(0.0, 0.0, 1.0);
    vec3 half_v = normalize(l + view);
    float spec = pow(max(0.0, abs(dot(n, half_v))), u_shininess) * u_specular;
    float shade = clamp(u_ambient + diff * (1.0 - u_ambient) + spec, 0.0, 1.6);
    frag = vec4(u_color * shade, u_opacity);
}
"""


@dataclass
class _GpuMesh:
    vbo: object         # moderngl.Buffer
    nbo: object
    n_verts: int
    color: tuple[float, float, float]
    opacity: float
    shininess: float
    draw_order: int


class _GpuRenderer:
    """Persistent moderngl context + cached mesh VBOs."""

    def __init__(self) -> None:
        try:
            import moderngl
        except ImportError as exc:
            raise MissingDependency(
                "moderngl",
                install_hint="pip install moderngl",
            ) from exc
        self._mgl = moderngl
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.prog = self.ctx.program(vertex_shader=VERT_SHADER,
                                       fragment_shader=FRAG_SHADER)
        self._meshes: dict[str, _GpuMesh] = {}
        self._fbo_size: tuple[int, int] | None = None
        self._fbo = None
        self._color_tex = None
        self._depth_buf = None

    def _ensure_fbo(self, w: int, h: int) -> None:
        if self._fbo_size == (w, h) and self._fbo is not None:
            return
        if self._fbo is not None:
            self._fbo.release()
            self._color_tex.release()
            self._depth_buf.release()
        self._color_tex = self.ctx.texture((w, h), 4)
        self._depth_buf = self.ctx.depth_renderbuffer((w, h))
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self._color_tex],
            depth_attachment=self._depth_buf,
        )
        self._fbo_size = (w, h)

    def _upload_mesh(self, fma: str, spec) -> _GpuMesh:
        if fma in self._meshes:
            return self._meshes[fma]
        m = load_mesh(fma)
        # Apply BP3D→screen reorientation on CPU once.
        rx0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
        ry180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        R = (ry180 @ rx0).astype(np.float32)
        verts = (m.vertices @ R.T).astype(np.float32)
        # Per-vertex normals (from per-tri normals replicated).
        per_vert_normals = np.empty_like(verts)
        for ti in range(len(m.triangles)):
            i0, i1, i2 = m.triangles[ti]
            n_ = (m.normals[ti] @ R.T).astype(np.float32)
            per_vert_normals[i0] = n_
            per_vert_normals[i1] = n_
            per_vert_normals[i2] = n_
        # Index buffer expanded so we can use a single draw_arrays.
        flat_v = verts[m.triangles.reshape(-1)]
        flat_n = per_vert_normals[m.triangles.reshape(-1)]
        vbo = self.ctx.buffer(flat_v.tobytes())
        nbo = self.ctx.buffer(flat_n.tobytes())
        gpu_m = _GpuMesh(
            vbo=vbo, nbo=nbo, n_verts=len(flat_v),
            color=(spec.color[0] / 255.0, spec.color[1] / 255.0,
                    spec.color[2] / 255.0),
            opacity=spec.opacity, shininess=max(1.0, spec.shininess),
            draw_order=spec.draw_order,
        )
        self._meshes[fma] = gpu_m
        return gpu_m

    def render(
        self,
        specs: list,
        size: tuple[int, int],
        *,
        yaw: float,
        pitch: float,
        bg: tuple[int, int, int],
    ) -> np.ndarray:
        w, h = size
        self._ensure_fbo(w, h)
        self._fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

        # Upload all meshes (cached after first call).
        gpu_meshes = [(self._upload_mesh(s.fma, s), s) for s in specs]

        # Choose scale from bone bbox to keep face filling the frame.
        bone_specs = [s for s in specs if s.category == "bone"]
        if bone_specs:
            ref = bone_specs
        else:
            ref = specs
        # Compute bbox on CPU from cached vertex data — cheap.
        all_verts: list[np.ndarray] = []
        for s in ref:
            m = load_mesh(s.fma)
            rx0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
            ry180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
            R = ry180 @ rx0
            all_verts.append(m.vertices @ R.T)
        verts_all = np.vstack(all_verts)
        vmin = verts_all.min(axis=0)
        vmax = verts_all.max(axis=0)
        centre = (vmin + vmax) / 2.0
        span = float(np.linalg.norm(vmax - vmin))
        scale = 1.7 / max(span, 1e-6)

        # Build MVP. Orthographic projection so output matches CPU look.
        cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
        cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
        ry = np.array([[cy_, 0, sy_, 0], [0, 1, 0, 0],
                        [-sy_, 0, cy_, 0], [0, 0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0, 0], [0, cp_, -sp_, 0],
                        [0, sp_, cp_, 0], [0, 0, 0, 1]], dtype=np.float32)
        T_centre = np.eye(4, dtype=np.float32)
        T_centre[:3, 3] = -centre
        S = np.eye(4, dtype=np.float32) * scale
        S[3, 3] = 1.0
        # Orthographic: world coords already in [-1, 1] after S.
        # No flipY in the matrix — np.flipud after fbo.read() handles
        # moderngl's bottom-up convention.
        model = ry @ rx @ S @ T_centre
        mvp = model

        # Sort by draw_order then back-to-front.
        gpu_meshes.sort(key=lambda x: (x[1].draw_order, 0))

        light_dir = np.array([-0.4, 0.5, 0.7], dtype=np.float32)

        self.prog["u_mvp"].write(mvp.T.tobytes())
        self.prog["u_model"].write(model.T.tobytes())
        self.prog["u_light_dir"].value = tuple(light_dir.tolist())
        self.prog["u_ambient"].value = 0.30
        self.prog["u_specular"].value = 0.35

        for gm, spec in gpu_meshes:
            self.prog["u_color"].value = gm.color
            self.prog["u_opacity"].value = gm.opacity
            self.prog["u_shininess"].value = gm.shininess
            vao = self.ctx.vertex_array(self.prog, [
                (gm.vbo, "3f", "in_pos"),
                (gm.nbo, "3f", "in_norm"),
            ])
            vao.render(self._mgl.TRIANGLES, vertices=gm.n_verts)
            vao.release()

        # Read back as RGBA -> BGR.
        data = self._fbo.read(components=3)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        # moderngl reads bottom-up; flip vertically.
        arr = np.flipud(arr).copy()
        return arr[:, :, ::-1].copy()  # RGB → BGR


@lru_cache(maxsize=1)
def _get_renderer() -> _GpuRenderer:
    return _GpuRenderer()


def gpu_available() -> bool:
    try:
        import moderngl  # noqa: F401
        return True
    except ImportError:
        return False


def render_face_faceforge_gpu(
    params,
    size: tuple[int, int] = (640, 480),
    *,
    layer_set: str = "lifelike",
) -> np.ndarray:
    if not meshes_available():
        raise MissingDependency(
            "BodyParts3D STL meshes",
            install_hint=(
                f"Copy STLs into {mesh_dir()} via "
                "`python -m tools.copy_anatomy_meshes /path/to/bodyparts3D/stl`."
            ),
        )

    avail = set(list_available_meshes())
    specs = [s for s in specs_for_layer_set(layer_set) if s.fma in avail]
    if not specs:
        raise MissingDependency(
            "BodyParts3D STL meshes",
            install_hint="No STLs found for layer_set; run copy_anatomy_meshes.",
        )

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    bg = _hex_to_rgb(getattr(params, "background", "#0a0d12"))

    return _get_renderer().render(specs, size, yaw=yaw, pitch=pitch, bg=bg)


def _hex_to_rgb(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16
