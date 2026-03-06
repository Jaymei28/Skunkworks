"""
engine/gl_offscreen_renderer.py
================================
Headless OpenGL renderer -- same shaders as the live viewport.

Creates a QOffscreenSurface + QOpenGLContext (no window required).
Call init_gl() once from the thread that will do rendering, then
call render_frame() for each image.
"""

import math
import ctypes
import os
import numpy as np

from PySide6.QtGui      import QOffscreenSurface, QSurfaceFormat, QOpenGLContext
from PySide6.QtOpenGL   import (QOpenGLShader, QOpenGLShaderProgram,
                                QOpenGLFramebufferObject,
                                QOpenGLFramebufferObjectFormat)
from PySide6.QtCore     import QSize

# Import shader GLSL source strings and math helpers from viewport.
# These are all module-level names in app/panels/viewport.py.
from app.panels.viewport import (
    _VERT_SRC, _FRAG_SRC,
    _OCEAN_VERT, _OCEAN_FRAG,
    _SKY_VERT, _SKY_FRAG,
    _make_sphere, _make_cube_verts,
    _mat4_perspective, _lookat, _mat4_identity,
    _mat4_translate, _mat4_scale, _mat4_mul,
    _mat4_rot_x, _mat4_rot_y, _mat4_rot_z,
    _norm3, _cross3, _dot3, _mat4_inverse, _as_c_floats,
    _mat3_inverse_transpose,
)

from OpenGL.GL import (
    glEnable, glDisable, glDepthFunc, glClear, glClearColor,
    glViewport, glGenVertexArrays, glBindVertexArray,
    glGenBuffers, glBindBuffer, glBufferData, glVertexAttribPointer,
    glEnableVertexAttribArray, glDrawElements, glDrawArrays,
    glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glGenerateMipmap, glActiveTexture,
    glGetError, glUniform1f, glUniform1i, glUniform3f,
    glUniformMatrix3fv, glUniformMatrix4fv,
    glDepthMask, glBlendFunc,
    GL_DEPTH_TEST, GL_LESS, GL_LEQUAL,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
    GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER,
    GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
    GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_UNSIGNED_INT,
    GL_TEXTURE_2D, GL_RGB,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR, GL_CLAMP_TO_EDGE,
    GL_TEXTURE0, GL_TEXTURE1,
    GL_UNSIGNED_BYTE, GL_RGBA,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
    glReadPixels,
    glDeleteTextures, glDeleteVertexArrays, glDeleteBuffers,
    GL_CULL_FACE, GL_BACK, glCullFace, GL_FRONT_AND_BACK, GL_FILL,
    glPolygonMode,
)


# Vertex stride: pos(3) + normal(3) + uv(2) = 8 floats x 4 bytes
_STRIDE = 8 * 4


def _interleave(verts, norms, uvs):
    """Build an interleaved (N, 8) float32 array [pos|nrm|uv]."""
    n = len(verts)
    if len(norms) != n:
        norms = np.zeros((n, 3), dtype=np.float32)
    if len(uvs) != n:
        uvs = np.zeros((n, 2), dtype=np.float32)
    buf = np.zeros((n, 8), dtype=np.float32)
    buf[:, 0:3] = verts
    buf[:, 3:6] = norms
    buf[:, 6:8] = uvs
    return buf


class GLOffscreenRenderer:
    """
    Headless OpenGL 3.3 renderer using the same shaders as GL3DPreview.
    Must call init_gl() from the thread that will do rendering.
    """

    def __init__(self, width=512, height=512):
        self.width  = width
        self.height = height

        # QOffscreenSurface must be created on the main (GUI) thread.
        fmt = QSurfaceFormat.defaultFormat()
        self._surface = QOffscreenSurface()
        self._surface.setFormat(fmt)
        self._surface.create()

        self._ctx         = None
        self._fbo         = None
        self._pbr_prog    = None
        self._ocean_prog  = None
        self._sky_prog    = None

        self._sphere_vao = self._sphere_vbo = self._sphere_ebo = None
        self._sphere_idx_count = 0
        self._sky_vao = self._sky_vbo = self._sky_ebo = None
        self._sky_idx_count = 0
        self._ocean_vao = self._ocean_vbo = self._ocean_ebo = None
        self._ocean_idx_count = 0

        self._env_tex    = None
        # mesh_path -> {vao, vbo, ebo, idx_count, local_aabb, alb_tex}
        self._mesh_gpu   = {}

    # ------------------------------------------------------------------
    # Init (call from rendering thread)
    # ------------------------------------------------------------------

    def init_gl(self):
        """Create context + FBO + shaders. Call once from the rendering thread."""
        self._ctx = QOpenGLContext()
        self._ctx.setFormat(self._surface.format())
        if not self._ctx.create():
            raise RuntimeError("GLOffscreenRenderer: failed to create GL context")
        if not self._ctx.makeCurrent(self._surface):
            raise RuntimeError("GLOffscreenRenderer: failed to make context current")

        fbo_fmt = QOpenGLFramebufferObjectFormat()
        fbo_fmt.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
        fbo_fmt.setSamples(0)
        self._fbo = QOpenGLFramebufferObject(QSize(self.width, self.height), fbo_fmt)

        self._pbr_prog   = self._compile(_VERT_SRC,   _FRAG_SRC,   "PBR")
        self._ocean_prog = self._compile(_OCEAN_VERT, _OCEAN_FRAG, "Ocean")
        self._sky_prog   = self._compile(_SKY_VERT,   _SKY_FRAG,   "Sky")

        self._upload_sphere()
        self._upload_sky()
        self._upload_ocean()

        # Stub 1x1 black env texture so the sampler uniform is always bound
        self._env_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._env_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, b'\x00\x00\x00')
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)

        self._ctx.doneCurrent()

    # ------------------------------------------------------------------
    # Asset loading
    # ------------------------------------------------------------------

    def load_hdri(self, path):
        if not path or not os.path.exists(path):
            return
        self._ctx.makeCurrent(self._surface)
        try:
            import cv2
            img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
            if img is None:
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.flipud(img)
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            if self._env_tex:
                glDeleteTextures(1, [self._env_tex])
            self._env_tex = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, self._env_tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)
        except Exception as e:
            print("[GLOffscreen] HDRI load failed:", e)
        finally:
            self._ctx.doneCurrent()

    def load_mesh_from_path(self, mesh_path, tex_albedo=""):
        """Load a mesh file into a GPU VAO. Skips if already loaded."""
        if mesh_path in self._mesh_gpu:
            return
        from app.engine.mesh_loader import load_mesh
        try:
            lm = load_mesh(mesh_path)
        except Exception as e:
            print("[GLOffscreen] Mesh load failed:", e)
            return

        # Normalize to ~2-unit bounding box
        verts = lm.vertices.copy()
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        verts -= (mn + mx) / 2.0
        span = float((mx - mn).max()) or 1.0
        verts *= (2.0 / span)
        local_aabb = (verts.min(axis=0), verts.max(axis=0))

        buf = _interleave(verts, lm.normals, lm.uvs)
        idx = lm.indices

        self._ctx.makeCurrent(self._surface)
        vao = int(glGenVertexArrays(1))
        glBindVertexArray(vao)
        vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, buf.nbytes, buf.flatten().tobytes(), GL_STATIC_DRAW)
        ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx.tobytes(), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, _STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, _STRIDE, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, _STRIDE, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)

        # Texture
        tex_path = tex_albedo or lm.tex_albedo
        alb_tex = self._upload_texture(tex_path) if tex_path else None
        self._ctx.doneCurrent()

        self._mesh_gpu[mesh_path] = {
            "vao": vao, "vbo": vbo, "ebo": ebo,
            "idx_count": len(idx),
            "local_aabb": local_aabb,
            "alb_tex": alb_tex,
        }

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_frame(self, cfg, azim, elev, dist, t=0.0):
        """
        Render one frame.
        Returns (rgb_np, bboxes) where:
          rgb_np  -- (H, W, 3) uint8 numpy array
          bboxes  -- list of (class_name, x_c, y_c, w, h) YOLO format [0,1]
        """
        self._ctx.makeCurrent(self._surface)
        self._fbo.bind()

        glViewport(0, 0, self.width, self.height)
        glClearColor(0.12, 0.13, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Camera matrices
        aspect = self.width / max(self.height, 1)
        fov    = getattr(cfg, 'fov_y', 60.0)
        proj   = _mat4_perspective(fov, aspect, 0.1, 5000.0)

        r  = math.radians(elev)
        a  = math.radians(azim)
        ex = dist * math.cos(r) * math.sin(a)
        ey = dist * math.sin(r)
        ez = dist * math.cos(r) * math.cos(a)
        eye = [ex, ey, ez]
        view = _lookat(eye, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])

        # Sky / HDRI background
        if self._sky_prog and self._sky_vao and self._env_tex:
            glDepthMask(GL_FALSE)
            self._sky_prog.bind()
            vr = list(view)
            vr[12] = vr[13] = vr[14] = 0.0
            self._umat4(self._sky_prog, "uViewRot", vr)
            self._umat4(self._sky_prog, "uProj", proj)
            self._uf(self._sky_prog, "uEnvStrength", 1.0)
            self._uf(self._sky_prog, "uHdriRotation", 0.0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._env_tex)
            self._ui(self._sky_prog, "uEnvMap", 0)
            glBindVertexArray(self._sky_vao)
            glDrawElements(GL_TRIANGLES, self._sky_idx_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            self._sky_prog.release()
            glDepthMask(GL_TRUE)

        glDepthFunc(GL_LESS)

        # PBR meshes
        sun_dir = [0.3, 0.8, 0.6]
        bboxes  = []
        if self._pbr_prog:
            self._pbr_prog.bind()
            self._umat4(self._pbr_prog, "uView", view)
            self._umat4(self._pbr_prog, "uProj", proj)
            self._uv3(self._pbr_prog,  "uCamPos",         *eye)
            self._uv3(self._pbr_prog,  "uLightDir",       *sun_dir)
            self._uf(self._pbr_prog,   "uLightIntensity", 3.5)
            self._uf(self._pbr_prog,   "uEnvStrength",    1.0)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._env_tex)
            self._ui(self._pbr_prog, "uEnvMap", 0)

            for obj in cfg.scene_objects:
                if not getattr(obj, 'visible', True):
                    continue
                path = obj.config.mesh_path
                if path not in self._mesh_gpu:
                    continue
                gd = self._mesh_gpu[path]

                sc = getattr(obj, 'scale', 1.0)
                model = _mat4_mul(
                    _mat4_translate(
                        getattr(obj, 'pos_x', 0.0),
                        getattr(obj, 'pos_y', 0.0),
                        getattr(obj, 'pos_z', 0.0)),
                    _mat4_mul(
                        _mat4_rot_y(getattr(obj, 'rot_y', 0.0)),
                        _mat4_mul(
                            _mat4_rot_x(getattr(obj, 'rot_x', 0.0)),
                            _mat4_mul(
                                _mat4_rot_z(getattr(obj, 'rot_z', 0.0)),
                                _mat4_scale(sc)))))
                self._umat4(self._pbr_prog, "uModel", model)
                nm = _mat3_inverse_transpose(model)
                glUniformMatrix3fv(
                    self._pbr_prog.uniformLocation("uNormalMat"),
                    1, GL_FALSE, _as_c_floats(nm))

                self._uv3(self._pbr_prog, "uAlbedo",    0.8, 0.8, 0.8)
                self._uf(self._pbr_prog,  "uMetallic",  getattr(obj, 'metallic',  0.0))
                self._uf(self._pbr_prog,  "uRoughness", getattr(obj, 'roughness', 0.5))

                alb = gd.get("alb_tex")
                if alb:
                    glActiveTexture(GL_TEXTURE1)
                    glBindTexture(GL_TEXTURE_2D, alb)
                    self._ui(self._pbr_prog, "uAlbedoMap",    1)
                    self._ui(self._pbr_prog, "uHasAlbedoMap", 1)
                else:
                    self._ui(self._pbr_prog, "uHasAlbedoMap", 0)
                self._ui(self._pbr_prog, "uHasNormalMap", 0)

                glBindVertexArray(gd["vao"])
                glDrawElements(GL_TRIANGLES, gd["idx_count"], GL_UNSIGNED_INT, None)
                glBindVertexArray(0)

                bb = self._project_bbox(gd["local_aabb"], model, view, proj, sc)
                if bb is not None:
                    label = getattr(obj, 'label', None) or obj.config.class_name
                    bboxes.append((label, *bb))

            self._pbr_prog.release()

        # Ocean
        if self._ocean_prog and self._ocean_vao and cfg.ocean.enabled:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(GL_FALSE)
            o = cfg.ocean
            self._ocean_prog.bind()
            self._umat4(self._ocean_prog, "uProj", proj)
            self._umat4(self._ocean_prog, "uView", view)
            self._uv3(self._ocean_prog, "uCamPos",          *eye)
            self._uf(self._ocean_prog,  "uTime",            t * o.time_multiplier)
            self._uf(self._ocean_prog,  "uLevel",           o.level)
            self._uf(self._ocean_prog,  "uWindSpeed",       o.wind_speed)
            self._uf(self._ocean_prog,  "uWindDirection",   o.wind_direction)
            self._uf(self._ocean_prog,  "uChoppiness",      o.choppiness)
            self._uf(self._ocean_prog,  "uChaos",           o.chaos)
            self._uf(self._ocean_prog,  "uStorm",           o.storm_intensity)
            self._uv3(self._ocean_prog, "uRefractionColor", *o.refraction_color)
            self._uv3(self._ocean_prog, "uScatteringColor", *o.scattering_color)
            self._uf(self._ocean_prog,  "uAbsorptionDistance", o.absorption_distance)
            self._uf(self._ocean_prog,  "uAmbientScattering",  o.ambient_scattering)
            self._uf(self._ocean_prog,  "uHeightScattering",   o.height_scattering)
            self._uf(self._ocean_prog,  "uSmoothness",      o.smoothness)
            self._uf(self._ocean_prog,  "uTransparency",    o.transparency)
            self._uf(self._ocean_prog,  "uEnvStrength",     o.reflection)
            self._ui(self._ocean_prog,  "uFoamEnabled",     1 if o.foam_enabled else 0)
            self._uf(self._ocean_prog,  "uFoamAmount",      o.foam_amount)
            self._uv3(self._ocean_prog, "uLightDir",        *sun_dir)
            self._uf(self._ocean_prog,  "uLightIntensity",  3.5)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._env_tex)
            self._ui(self._ocean_prog, "uEnvMap", 0)
            # Check if this is the real HDRI or the 1x1 black stub
            # (Stub is created in init_gl)
            if self._env_tex and self._env_tex > 0:
                # We can't easily distinguish between stub and real tex ID here without 
                # a boolean, so we assume if load_hdri was called it's real.
                # For safety, let's just always use the same logical as viewport.
                self._ui(self._ocean_prog, "uHasEnvMap", 1) 
            else:
                self._ui(self._ocean_prog, "uHasEnvMap", 0)
            glBindVertexArray(self._ocean_vao)
            glDrawElements(GL_TRIANGLES, self._ocean_idx_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
            self._ocean_prog.release()
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        # Read back pixels
        raw = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        arr = np.flipud(arr).copy()   # GL origin is bottom-left

        QOpenGLFramebufferObject.bindDefault()
        self._ctx.doneCurrent()
        return arr, bboxes

    def cleanup(self):
        if not self._ctx:
            return
        self._ctx.makeCurrent(self._surface)
        if self._env_tex:
            glDeleteTextures(1, [self._env_tex])
        for gd in self._mesh_gpu.values():
            glDeleteVertexArrays(1, [gd["vao"]])
            glDeleteBuffers(1, [gd["vbo"]])
            glDeleteBuffers(1, [gd["ebo"]])
            if gd.get("alb_tex"):
                glDeleteTextures(1, [gd["alb_tex"]])
        self._ctx.doneCurrent()
        self._ctx = None

    # ------------------------------------------------------------------
    # Geometry upload helpers
    # ------------------------------------------------------------------

    def _upload_sphere(self):
        verts, idx = _make_sphere(24, 24)
        self._sphere_idx_count = len(idx)
        self._sphere_vao, self._sphere_vbo, self._sphere_ebo = \
            self._upload_vao(verts.tobytes(), idx.tobytes(), _STRIDE)

    def _upload_sky(self):
        verts, idx = _make_cube_verts()
        self._sky_idx_count = len(idx)
        self._sky_vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._sky_vao)
        self._sky_vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._sky_vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts.tobytes(), GL_STATIC_DRAW)
        self._sky_ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._sky_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx.tobytes(), GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _upload_ocean(self, size=250, res=256):
        half = size / 2.0
        step = size / res
        verts, indices = [], []
        for r in range(res + 1):
            for c in range(res + 1):
                x = -half + c * step
                z = -half + r * step
                verts.extend([x, 0.0, z, float(c) / res, float(r) / res])
        for r in range(res):
            for c in range(res):
                tl = r * (res + 1) + c
                indices.extend([
                    tl, tl + res + 1, tl + 1,
                    tl + 1, tl + res + 1, tl + res + 2])
        v = np.array(verts, dtype=np.float32)
        i = np.array(indices, dtype=np.uint32)
        self._ocean_idx_count = len(i)
        self._ocean_vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._ocean_vao)
        self._ocean_vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._ocean_vbo)
        glBufferData(GL_ARRAY_BUFFER, v.nbytes, v.tobytes(), GL_STATIC_DRAW)
        self._ocean_ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ocean_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, i.nbytes, i.tobytes(), GL_STATIC_DRAW)
        stride5 = 5 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride5, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride5, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _upload_vao(self, vb, ib, stride):
        vao = int(glGenVertexArrays(1))
        glBindVertexArray(vao)
        vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, len(vb), vb, GL_STATIC_DRAW)
        ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(ib), ib, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glBindVertexArray(0)
        return vao, vbo, ebo

    def _upload_texture(self, path):
        if not path or not os.path.exists(path):
            return None
        try:
            import cv2
            img = cv2.imread(path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.flipud(img)
            h, w = img.shape[:2]
            tex = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, img.tobytes())
            glGenerateMipmap(GL_TEXTURE_2D)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glBindTexture(GL_TEXTURE_2D, 0)
            return tex
        except Exception as e:
            print("[GLOffscreen] Texture upload failed:", e)
            return None

    # ------------------------------------------------------------------
    # Shader helpers
    # ------------------------------------------------------------------

    def _compile(self, vert, frag, name):
        prog = QOpenGLShaderProgram()
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   vert):
            print("[GLOffscreen]", name, "vert:", prog.log())
            return None
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag):
            print("[GLOffscreen]", name, "frag:", prog.log())
            return None
        if not prog.link():
            print("[GLOffscreen]", name, "link:", prog.log())
            return None
        return prog

    def _umat4(self, p, name, m):
        loc = p.uniformLocation(name)
        if loc >= 0:
            glUniformMatrix4fv(loc, 1, GL_FALSE, _as_c_floats(m))

    def _uv3(self, p, name, x, y, z):
        loc = p.uniformLocation(name)
        if loc >= 0:
            glUniform3f(loc, x, y, z)

    def _uf(self, p, name, v):
        loc = p.uniformLocation(name)
        if loc >= 0:
            glUniform1f(loc, v)

    def _ui(self, p, name, v):
        loc = p.uniformLocation(name)
        if loc >= 0:
            glUniform1i(loc, v)

    # ------------------------------------------------------------------
    # 2D bbox projection from 3D AABB
    # ------------------------------------------------------------------

    def _project_bbox(self, local_aabb, model, view, proj, scale):
        """
        Project the 8 corners of the local AABB through MVP.
        Returns (x_c, y_c, w, h) in [0,1] YOLO format, or None if off-screen.
        """
        mn, mx = local_aabb
        corners = []
        for xi in (mn[0], mx[0]):
            for yi in (mn[1], mx[1]):
                for zi in (mn[2], mx[2]):
                    corners.append([xi * scale, yi * scale, zi * scale, 1.0])

        mvp = _mat4_mul(proj, _mat4_mul(view, model))
        m   = np.array(mvp, dtype=np.float64).reshape(4, 4)
        xs, ys = [], []
        for c in corners:
            v    = np.array(c, dtype=np.float64)
            clip = m.T @ v
            if abs(clip[3]) < 1e-6:
                continue
            ndc = clip[:3] / clip[3]
            if not (-1.5 < ndc[0] < 1.5 and -1.5 < ndc[1] < 1.5):
                continue
            xs.append((ndc[0] + 1.0) / 2.0)
            ys.append(1.0 - (ndc[1] + 1.0) / 2.0)

        if len(xs) < 2:
            return None
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        w = x2 - x1
        h = y2 - y1
        if w < 0.001 or h < 0.001:
            return None
        return (x1 + w / 2.0, y1 + h / 2.0, w, h)
