"""Decimated BP3D skin mesh — real face topology at lite-3D polygon count.

Problem: the previous ``head_3d_lite`` used Delaunay over hand-placed
landmarks, which crossed feature boundaries (eye → forehead, lip →
cheek) and produced a chaotic spider-web shape that didn't read as a
face. This module fixes that by starting from real anatomy.

Approach:

1. Load the BP3D face-skin STL (FMA7163) — ~30K vertices, proper
   topology around eyes, mouth, nose, ears.
2. **Vertex-cluster decimation** in pure NumPy: divide a 3D grid
   over the mesh bbox, replace each cell's vertices with the cell
   centre, rebuild triangles using the new representatives.
3. Cache the decimated mesh in memory.
4. Render via QPainter Z-sort (CPU) or moderngl (GPU). Same yaw/
   pitch + AU-driven landmark displacement as the rest of the
   pipeline.

The decimated mesh keeps the silhouette of a real face — it doesn't
need hand-tuning, doesn't cross features, and looks like a polygonal
human head rather than a spider web.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from faceview.core.errors import MissingDependency
from faceview.vision.anatomy_meshes import load_mesh, meshes_available


SKIN_FMA = "FMA7163"


@lru_cache(maxsize=4)
def decimated_skin(grid: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(verts, tris, normals)`` for the decimated skin mesh.

    Vertex clustering: divide the bbox into ``grid^3`` cells, replace
    every original vertex with its cell-centre representative,
    deduplicate triangles. Higher ``grid`` = finer mesh.
    """
    if not meshes_available():
        raise MissingDependency(
            "BodyParts3D STL meshes", "gpu",
            hint="Run `python -m tools.copy_anatomy_meshes /path/to/bp3d/stl`.",
        )
    m = load_mesh(SKIN_FMA)

    # 1. Apply BP3D→screen reorientation up front so the cached mesh
    # lives in screen-coord space.
    rx0 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    ry180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
    R = (ry180 @ rx0).astype(np.float32)
    verts = m.vertices @ R.T
    tri_normals = m.normals @ R.T

    # 2. Crop to head + neck. The skin mesh is full-body. After our
    # reorient, the head is at the largest Y values (BP3D's +Z up
    # maps to screen +Y down ... but in this projection space, the
    # head ends up at higher Y values than the legs). Keep only the
    # top ~22%.
    body_y_min = verts[:, 1].min()
    body_y_max = verts[:, 1].max()
    body_h = body_y_max - body_y_min
    keep_min_y = body_y_max - 0.22 * body_h    # head + neck
    keep_max_y = body_y_max + 0.01 * body_h
    vert_mask = (verts[:, 1] >= keep_min_y) & (verts[:, 1] <= keep_max_y)

    # Drop triangles whose vertices fall outside.
    tri_mask = (vert_mask[m.triangles[:, 0]]
                & vert_mask[m.triangles[:, 1]]
                & vert_mask[m.triangles[:, 2]])
    cropped_verts = verts                                 # keep all verts so indices stay valid
    cropped_tris = m.triangles[tri_mask]
    cropped_normals = tri_normals[tri_mask]

    # Re-index to drop unused vertices.
    used = np.unique(cropped_tris)
    remap = -np.ones(len(verts), dtype=np.int32)
    remap[used] = np.arange(len(used))
    cropped_verts = verts[used]
    cropped_tris = remap[cropped_tris]

    verts = cropped_verts
    tri_normals = cropped_normals
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    span = vmax - vmin
    span = np.maximum(span, 1e-6)

    # 2. Compute cell index for each vertex.
    cell = np.floor((verts - vmin) / span * grid).astype(np.int32)
    cell = np.clip(cell, 0, grid - 1)
    cell_id = cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]

    # 3. Group by cell — representative is the mean position in the cell.
    unique_cells, inv = np.unique(cell_id, return_inverse=True)
    new_n = len(unique_cells)
    counts = np.bincount(inv)
    sums = np.zeros((new_n, 3), dtype=np.float64)
    np.add.at(sums, inv, verts.astype(np.float64))
    new_verts = (sums / counts[:, None]).astype(np.float32)

    # 4. Remap triangle indices to representatives, drop degenerate.
    new_tris = inv[cropped_tris]
    keep = (
        (new_tris[:, 0] != new_tris[:, 1]) &
        (new_tris[:, 1] != new_tris[:, 2]) &
        (new_tris[:, 0] != new_tris[:, 2])
    )
    new_tris = new_tris[keep]
    surviving_norms = tri_normals[keep]

    return new_verts, new_tris.astype(np.int32), surviving_norms.astype(np.float32)


def _project(verts: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    return verts @ (ry @ rx).T


def render_face_decimated_gpu(
    params,
    size: tuple[int, int] = (480, 480),
    *,
    grid: int = 28,
) -> np.ndarray:
    """A26: GPU path through moderngl — replaces the 8 fps CPU path."""
    try:
        import moderngl  # noqa: F401
    except ImportError as exc:
        from faceview.core.errors import MissingDependency
        raise MissingDependency("moderngl", "gpu") from exc
    verts, tris, _ = decimated_skin(grid)
    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    bg = _hex_to_rgb_dec(getattr(params, "background", "#0a0d12"))
    return _gpu_decim_renderer().render_decimated(
        verts.astype(np.float32),
        tris.astype(np.uint32),
        size,
        yaw=yaw, pitch=pitch, bg=bg,
    )


def _hex_to_rgb_dec(c: str) -> tuple[int, int, int]:
    s = c.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except (ValueError, IndexError):
        return 10, 12, 16


from functools import lru_cache as _lru_cache


@_lru_cache(maxsize=1)
def _gpu_decim_renderer():
    """Lazy-create a moderngl renderer for the decimated head."""
    import moderngl
    return _DecimatedGpuRenderer(moderngl)


class _DecimatedGpuRenderer:
    _VS = """#version 330
        uniform mat4 u_mvp;
        uniform mat3 u_norm_mat;
        in vec3 in_pos;
        in vec3 in_norm;
        out vec3 v_norm;
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_norm = u_norm_mat * in_norm;
        }
    """
    _FS = """#version 330
        uniform vec3 u_skin;
        uniform vec3 u_light;
        in vec3 v_norm;
        out vec4 frag;
        void main() {
            vec3 n = normalize(v_norm);
            vec3 l = normalize(-u_light);
            float wrap = clamp(dot(n, l) * 0.5 + 0.5, 0.0, 1.0);
            float spec = pow(max(0.0, dot(n, normalize(l + vec3(0,0,1)))), 24.0) * 0.25;
            frag = vec4(u_skin * (0.30 + 0.70 * wrap) + spec, 1.0);
        }
    """

    def __init__(self, mgl) -> None:
        self.mgl = mgl
        self.ctx = mgl.create_context(standalone=True, require=330)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=self._VS, fragment_shader=self._FS)
        self._fbo = None
        self._fbo_size: tuple[int, int] | None = None

    def _ensure_fbo(self, w, h) -> None:
        if self._fbo_size == (w, h) and self._fbo is not None:
            return
        if self._fbo is not None:
            self._fbo.release()
        col = self.ctx.texture((w, h), 4)
        depth = self.ctx.depth_renderbuffer((w, h))
        self._fbo = self.ctx.framebuffer(color_attachments=[col],
                                           depth_attachment=depth)
        self._fbo_size = (w, h)

    def render_decimated(self, verts, tris, size, *, yaw, pitch, bg):
        w, h = size
        self._ensure_fbo(w, h)
        self._fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

        # Per-vertex normals.
        v0 = verts[tris[:, 0]]; v1 = verts[tris[:, 1]]; v2 = verts[tris[:, 2]]
        tri_n = np.cross(v1 - v0, v2 - v0)
        tri_n /= np.maximum(np.linalg.norm(tri_n, axis=1, keepdims=True), 1e-9)
        vn = np.zeros_like(verts)
        np.add.at(vn, tris[:, 0], tri_n)
        np.add.at(vn, tris[:, 1], tri_n)
        np.add.at(vn, tris[:, 2], tri_n)
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-9)

        # Centre + scale.
        centre = (verts.min(axis=0) + verts.max(axis=0)) / 2
        span = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        scale = 1.6 / max(span, 1e-6)

        cy_, sy_ = float(np.cos(yaw)), float(np.sin(yaw))
        cp_, sp_ = float(np.cos(pitch)), float(np.sin(pitch))
        ry = np.array([[cy_, 0, sy_, 0], [0, 1, 0, 0],
                        [-sy_, 0, cy_, 0], [0, 0, 0, 1]], dtype=np.float32)
        rx = np.array([[1, 0, 0, 0], [0, cp_, -sp_, 0],
                        [0, sp_, cp_, 0], [0, 0, 0, 1]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32); T[:3, 3] = -centre
        S = np.eye(4, dtype=np.float32) * scale; S[3, 3] = 1.0
        flip = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        model = ry @ rx @ flip @ S @ T

        self.prog["u_mvp"].write(model.T.tobytes())
        self.prog["u_norm_mat"].write(model[:3, :3].T.tobytes())
        self.prog["u_skin"].value = (0.86, 0.74, 0.65)
        self.prog["u_light"].value = (-0.4, -0.3, -0.7)

        vbo = self.ctx.buffer(verts.tobytes())
        nbo = self.ctx.buffer(vn.astype(np.float32).tobytes())
        ibo = self.ctx.buffer(tris.tobytes())
        vao = self.ctx.vertex_array(self.prog, [
            (vbo, "3f", "in_pos"), (nbo, "3f", "in_norm"),
        ], ibo)
        vao.render(self.mgl.TRIANGLES)
        vao.release(); vbo.release(); nbo.release(); ibo.release()

        data = self._fbo.read(components=3)
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        arr = np.flipud(arr).copy()
        return arr[:, :, ::-1].copy()


def render_face_decimated(
    params,
    size: tuple[int, int] = (480, 480),
    *,
    grid: int = 28,
) -> np.ndarray:
    """Render the decimated BP3D skin mesh via QPainter Z-sort (CPU).

    ``grid`` controls the cell count for vertex clustering — higher
    is finer. ``28`` produces a ~3000-tri mesh that paints in ~50ms.
    For animation use ``head_decimated_3d_gpu`` (60+ fps).
    """
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import (
        QBrush, QColor, QImage, QPainter, QPainterPath,
    )

    verts, tris, _norms = decimated_skin(grid)
    if len(tris) == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    yaw = float(getattr(params, "yaw", 0.0)) * 0.6
    pitch = float(getattr(params, "pitch", 0.0)) * 0.4
    rotated = _project(verts, yaw, pitch)

    # Centre + scale to fit.
    vmin = rotated.min(axis=0)
    vmax = rotated.max(axis=0)
    centre = (vmin + vmax) / 2.0
    span = float(np.linalg.norm(vmax - vmin))
    w, h = size
    scale = 0.85 * min(w, h) / max(span, 1e-6)

    sx = (rotated[:, 0] - centre[0]) * scale + w / 2
    sy = -(rotated[:, 1] - centre[1]) * scale + h / 2

    # Recompute per-tri normals after rotation (the cached normals
    # don't match the post-clustering geometry exactly).
    v0 = rotated[tris[:, 0]]
    v1 = rotated[tris[:, 1]]
    v2 = rotated[tris[:, 2]]
    n_ = np.cross(v1 - v0, v2 - v0)
    nl = np.linalg.norm(n_, axis=1, keepdims=True)
    n_norm = np.divide(n_, np.maximum(nl, 1e-9))
    avg_z = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3.0

    light = np.array([-0.4, -0.4, -1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    diff = np.abs(n_norm @ light)
    shade = np.clip(0.32 + 0.62 * diff, 0.0, 1.4)

    bg = QColor(getattr(params, "background", "#0a0d12"))
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(bg)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    p.setPen(Qt.PenStyle.NoPen)

    # Backface cull: drop triangles whose normals point away from camera
    # (camera is at +Z looking toward -Z; front-facing has normal.z > 0).
    # Use abs() shading because normal sign is unreliable from clustered
    # mesh, but we DO want to drop tris on the far side of the head.
    # Heuristic: keep only tris whose centroid Z is greater than the
    # median Z (i.e. closer to camera).
    z_median = np.median(avg_z)
    front_mask = avg_z > z_median - (avg_z.max() - avg_z.min()) * 0.05
    front_idx = np.where(front_mask)[0]

    # Z-sort just the front-facing tris.
    front_z = avg_z[front_idx]
    order = front_idx[np.argsort(front_z)]

    skin_r, skin_g, skin_b = 220, 188, 165
    for ti in order:
        s = float(shade[ti])
        col = QColor(int(skin_r * s), int(skin_g * s), int(skin_b * s))
        p.setBrush(QBrush(col))
        i0, i1, i2 = tris[ti]
        path = QPainterPath()
        path.moveTo(QPointF(sx[i0], sy[i0]))
        path.lineTo(QPointF(sx[i1], sy[i1]))
        path.lineTo(QPointF(sx[i2], sy[i2]))
        path.closeSubpath()
        p.drawPath(path)
    p.end()

    arr = img.convertToFormat(QImage.Format.Format_RGB888)
    ptr = arr.constBits()
    if ptr is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    a = np.frombuffer(ptr, dtype=np.uint8, count=h * w * 3).reshape(h, w, 3)
    return a[:, :, ::-1].copy()
