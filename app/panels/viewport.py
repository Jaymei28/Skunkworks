"""
panels/viewport.py
==================
PySide6 QOpenGLWidget — real-time 3D preview with:
  • GPU rendering via OpenGL 3.3 Core
  • PBR shaders (Cook-Torrance BRDF)
  • HDR latlong environment map (equirectangular)
  • Camera orbit with inertia (lerp-based smooth spin after drag release)
  • Mouse drag + scroll zoom
  • Bottom toolbar: Zoom+/-, Reset camera, Wireframe toggle
  • Top toolbar: Play / Pause / Stop (unchanged API for MainWindow)
"""

import math
import time
import os
import sys
import struct
import ctypes

# ─────────────────────────────────────────────────────────────────────────────
# Path Helper: ensure the project root is in sys.path so 'renderer' can be found.
# ─────────────────────────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_this_dir)) # Up to Skunkworks
if _root not in sys.path:
    sys.path.insert(0, _root)

from PySide6.QtWidgets  import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
    QPushButton, QFrame, QScrollArea, QDoubleSpinBox, QColorDialog
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore     import Qt, QTimer, QSize, Signal, QPointF
from PySide6.QtGui      import (
    QColor, QPixmap, QImage, QIcon, QPainter, QSurfaceFormat,
    QFont, QFontMetrics
)
from PySide6.QtOpenGL   import QOpenGLShader, QOpenGLShaderProgram

try:
    from OpenGL.GL import (
        glEnable, glDisable, glDepthFunc, glClear, glClearColor,
        glViewport, glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData, glVertexAttribPointer,
        glEnableVertexAttribArray, glDrawArrays, glDrawElements,
        glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
        glGenerateMipmap, glActiveTexture, glPolygonMode,
        glGetError, glIsVertexArray, glUniform1f, glUniform1i, glUniform3f,
        glUniformMatrix3fv, glUniformMatrix4fv,
        glDepthMask, glLineWidth, glUseProgram,
        glBlendFunc, glScissor,
        GL_NO_ERROR, GL_DEPTH_TEST, GL_LESS, GL_LEQUAL,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
        GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_UNSIGNED_INT, GL_LINE,
        GL_FILL, GL_FRONT_AND_BACK, GL_TEXTURE_2D, GL_RGB, GL_RGB16F,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,
        GL_LINEAR_MIPMAP_LINEAR, GL_CLAMP_TO_EDGE, GL_TEXTURE0,
        GL_UNSIGNED_BYTE, GL_RGBA,
        GL_CULL_FACE, GL_BACK, glCullFace, GL_LINES,
        GL_SCISSOR_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
        glDeleteTextures, glDeleteVertexArrays, glDeleteBuffers,
    )
    import numpy as np
    HAS_OPENGL = True
except Exception:
    HAS_OPENGL = False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: GL surface format applied ONCE at module load time.
# Must run before any QOpenGLWidget is instantiated.
# ─────────────────────────────────────────────────────────────────────────────
_fmt = QSurfaceFormat()
_fmt.setVersion(3, 3)
_fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
_fmt.setDepthBufferSize(24)
_fmt.setSamples(4)
QSurfaceFormat.setDefaultFormat(_fmt)


# ─────────────────────────────────────────────────────────────────────────────
# PBR vertex shader
# ─────────────────────────────────────────────────────────────────────────────
_VERT_SRC = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;
uniform mat3 uNormalMat;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

void main(){
    vec4 world = uModel * vec4(aPos, 1.0);
    vWorldPos  = world.xyz;
    vNormal    = normalize(uNormalMat * aNormal);
    vUV        = aUV;
    gl_Position = uProj * uView * world;
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# PBR fragment shader (Cook-Torrance + HDR environment)
# ─────────────────────────────────────────────────────────────────────────────
_FRAG_SRC = """
#version 330 core
in  vec3 vWorldPos;
in  vec3 vNormal;
in  vec2 vUV;
out vec4 FragColor;

uniform vec3  uCamPos;
uniform sampler2D uEnvMap;    // equirectangular HDR
uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform float uEnvStrength;
uniform vec3  uLightDir;       // world-space directional light (normalised)
uniform float uLightIntensity; // key light radiance scale

const float PI = 3.14159265359;

// ── Equirectangular env-map sample ──────────────────────────────────────────
vec3 sampleEnv(vec3 dir, float lod){
    float phi   = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    vec2  uv    = vec2(phi / (2.0*PI) + 0.5, 1.0 - (theta / PI + 0.5));
    return textureLod(uEnvMap, uv, lod).rgb * uEnvStrength;
}

// ── GGX / Trowbridge-Reitz Normal Distribution ──────────────────────────────
float DistributionGGX(vec3 N, vec3 H, float r){
    float a  = r*r;
    float a2 = a*a;
    float d  = max(dot(N,H), 0.0);
    float d2 = d*d;
    float denom = d2*(a2-1.0)+1.0;
    return a2 / (PI * denom * denom);
}

// ── Schlick-GGX geometry ─────────────────────────────────────────────────────
float GeometrySchlickGGX(float NdV, float r){
    float k = (r+1.0); k = k*k/8.0;
    return NdV / (NdV*(1.0-k)+k);
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float r){
    return GeometrySchlickGGX(max(dot(N,V),0.0),r)
          *GeometrySchlickGGX(max(dot(N,L),0.0),r);
}

// ── Fresnel-Schlick ───────────────────────────────────────────────────────────
vec3 FresnelSchlick(float cosT, vec3 F0){
    return F0 + (1.0-F0)*pow(clamp(1.0-cosT,0.0,1.0),5.0);
}
vec3 FresnelSchlickRoughness(float cosT, vec3 F0, float r){
    return F0 + (max(vec3(1.0-r),F0)-F0)*pow(clamp(1.0-cosT,0.0,1.0),5.0);
}

void main(){
    vec3  N    = normalize(vNormal);
    vec3  V    = normalize(uCamPos - vWorldPos);
    float NdV  = max(dot(N,V),0.0);

    vec3 F0 = mix(vec3(0.04), uAlbedo, uMetallic);

    // ── Key light (controllable direction) ───────────────────────────────────
    vec3  L0   = normalize(uLightDir);
    vec3  H0   = normalize(V + L0);
    vec3  rad0 = vec3(uLightIntensity);

    float NDF  = DistributionGGX(N, H0, uRoughness);
    float G    = GeometrySmith(N, V, L0, uRoughness);
    vec3  F    = FresnelSchlick(max(dot(H0,V),0.0), F0);

    vec3  num  = NDF * G * F;
    float den  = 4.0 * NdV * max(dot(N,L0),0.0) + 0.0001;
    vec3  spec = num / den;

    vec3  kD   = (1.0 - F)*(1.0 - uMetallic);
    vec3  Lo   = (kD * uAlbedo / PI + spec) * rad0 * max(dot(N,L0),0.0);

    // ── IBL ambient (env map) ──────────────────────────────────────────────
    vec3 kS_a  = FresnelSchlickRoughness(NdV, F0, uRoughness);
    vec3 kD_a  = (1.0-kS_a)*(1.0-uMetallic);

    vec3 irrad  = sampleEnv(N, 4.0);           // diffuse: blurry sample
    vec3 diffIBL = kD_a * irrad * uAlbedo;

    vec3 R        = reflect(-V, N);
    float mipLvl  = uRoughness * 6.0;
    vec3 prefilt  = sampleEnv(R, mipLvl);      // specular: glossy sample
    vec3 specIBL  = (kS_a * prefilt);

    vec3 ambient  = diffIBL + specIBL;

    vec3 color = Lo + ambient;

    // ── Tone-map (ACES filmic) ─────────────────────────────────────────────
    color = (color * (2.51*color + 0.03)) / (color * (2.43*color + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    // ── Gamma correct ──────────────────────────────────────────────────────
    color = pow(color, vec3(1.0/2.2));

    FragColor = vec4(color, 1.0);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Skybox (equirectangular) shaders
# ─────────────────────────────────────────────────────────────────────────────
_SKY_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uViewRot;   // view without translation
uniform mat4 uProj;
out vec3 vDir;
void main(){
    vDir        = aPos;
    vec4 clip   = uProj * uViewRot * vec4(aPos, 1.0);
    gl_Position = clip.xyww;  // always at far plane
}
"""
_SKY_FRAG = """
#version 330 core
in  vec3 vDir;
out vec4 FragColor;
uniform sampler2D uEnvMap;
uniform float     uEnvStrength;
uniform float     uHdriRotation;  // radians, rotates HDRI yaw
const float PI = 3.14159265359;
void main(){
    vec3  d   = normalize(vDir);
    float phi = atan(d.z, d.x) + uHdriRotation;
    float th  = asin(clamp(d.y, -1.0, 1.0));
    vec2  uv  = vec2(phi/(2.0*PI)+0.5, 1.0 - (th/PI+0.5));
    vec3  col = texture(uEnvMap, uv).rgb * uEnvStrength;
    // filmic tone-map
    col = (col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14);
    col = clamp(col, 0.0, 1.0);
    col = pow(col, vec3(1.0/2.2));
    FragColor = vec4(col, 1.0);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Gizmo shaders (flat colour, no lighting)
# ─────────────────────────────────────────────────────────────────────────────
_GIZMO_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;
uniform mat4 uView;
uniform mat4 uProj;
out vec3 vColor;
void main(){
    vColor = aColor;
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
"""
_GIZMO_FRAG = """
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main(){
    FragColor = vec4(vColor, 1.0);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Infinite Grid Shader
# ─────────────────────────────────────────────────────────────────────────────
_GRID_VERT = """
#version 330 core
layout(location=0) in vec3 aPos; // NDC [-1, 1]
out vec2 vNDC;
void main(){
    vNDC = aPos.xy;
    gl_Position = vec4(aPos, 1.0);
}
"""
_GRID_FRAG = """
#version 330 core
in vec2 vNDC;
out vec4 FragColor;

uniform mat4 uInvViewProj;
uniform vec3 uCamPos;

float grid(vec2 st, float res){
    vec2 p = st / res;
    vec2 fw = fwidth(p);
    // Anti-aliased line of ~1 pixel width
    vec2 aLine = smoothstep(fw * 1.5, fw * 0.5, abs(fract(p - 0.5) - 0.5));
    float g = max(aLine.x, aLine.y);
    return g;
}

void main(){
    // Unproject NDC to world space to find ray direction
    vec4 farPosNDC = vec4(vNDC, 1.0, 1.0);
    vec4 farPosWorld = uInvViewProj * farPosNDC;
    farPosWorld /= farPosWorld.w;
    
    vec3 rayOrigin = uCamPos;
    vec3 rayDir = normalize(farPosWorld.xyz - rayOrigin);
    
    // Intersect ray with Y=0 plane
    float t = -rayOrigin.y / rayDir.y;
    
    if (t < 0.0 || abs(rayDir.y) < 0.0001) {
        discard;
    }
    
    vec3 worldPos = rayOrigin + t * rayDir;
    float dist = t;
    
    // Fading based on distance
    float fade = exp(-dist * 0.01); 
    if (fade < 0.001) discard;

    float m1 = grid(worldPos.xz, 1.0);
    float m2 = grid(worldPos.xz, 0.1) * 0.5;
    
    float g = max(m1, m2);
    vec3 color = vec3(0.4) * g;
    
    // Highlight axes
    if(abs(worldPos.x) < 0.03) color = mix(color, vec3(0.8, 0.2, 0.2), 0.9);
    if(abs(worldPos.z) < 0.03) color = mix(color, vec3(0.2, 0.2, 0.8), 0.9);
    
    FragColor = vec4(color * fade, g * fade * 0.5);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers (pure Python, no numpy dep at module level)
# ─────────────────────────────────────────────────────────────────────────────
def _mat4_identity():
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

def _mat4_translate(x, y, z):
    m = _mat4_identity()
    m[12], m[13], m[14] = x, y, z
    return m

def _mat4_mul(a, b):
    import numpy as np
    # Treat inputs as Column-Major lists, return a Column-Major list
    # MatMemory(A*B) = (A*B)^T = B^T * A^T = MatMemory(B) * MatMemory(A)
    # Using np.array().reshape(4,4) on a CM list gives the TRANSPOSE (Row-Major view).
    A_mat = np.array(a).reshape(4,4)
    B_mat = np.array(b).reshape(4,4)
    res = B_mat @ A_mat
    return res.flatten().tolist()

def _mat4_perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    nf = 1.0 / (near - far)
    return [
        f/aspect, 0,  0,              0,
        0,        f,  0,              0,
        0,        0, (far+near)*nf,  -1,
        0,        0, 2*far*near*nf,   0,
    ]

def _lookat(eye, center, up):
    f = _norm3([center[i]-eye[i] for i in range(3)])
    r = _norm3(_cross3(f, up))
    u = _cross3(r, f)
    return [
         r[0],  u[0], -f[0], 0,
         r[1],  u[1], -f[1], 0,
         r[2],  u[2], -f[2], 0,
        -_dot3(r,eye), -_dot3(u,eye), _dot3(f,eye), 1,
    ]

def _mat3_from_mat4(m):
    return [m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]]

def _mat3_inverse_transpose(m4):
    # For uniform-scaled models, mat3 of the model is fine
    return _mat3_from_mat4(m4)

def _norm3(v):
    l = math.sqrt(sum(x*x for x in v))
    return [x/(l+1e-9) for x in v]

def _cross3(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def _dot3(a, b):
    return sum(a[i]*b[i] for i in range(3))

def _mat4_vec4_mul(m, v):
    return [
        m[0]*v[0] + m[4]*v[1] + m[8]*v[2] + m[12]*v[3],
        m[1]*v[0] + m[5]*v[1] + m[9]*v[2] + m[13]*v[3],
        m[2]*v[0] + m[6]*v[1] + m[10]*v[2] + m[14]*v[3],
        m[3]*v[0] + m[7]*v[1] + m[11]*v[2] + m[15]*v[3]
    ]

def _mat4_inverse(m):
    import numpy as np
    # MatMemory(Inv M) = (Inv M)^T = Inv (M^T) = Inv (MatMemory M)
    try:
        mat = np.array(m).reshape(4,4)
        inv = np.linalg.inv(mat)
        return inv.flatten().tolist()
    except:
        return _mat4_identity()

def _mat4_scale(s):
    m = _mat4_identity()
    m[0]=s; m[5]=s; m[10]=s
    return m

def _mat4_rot_x(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = _mat4_identity()
    m[5]=c; m[6]=s; m[9]=-s; m[10]=c
    return m

def _mat4_rot_y(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = _mat4_identity()
    m[0]=c; m[2]=-s; m[8]=s; m[10]=c
    return m

def _mat4_rot_z(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    m = _mat4_identity()
    m[0]=c; m[1]=s; m[4]=-s; m[5]=c
    return m

def _mat4_translate(tx, ty, tz):
    m = _mat4_identity()
    m[12]=tx; m[13]=ty; m[14]=tz
    return m


def _as_c_floats(lst):
    count = len(lst)
    # Using the * operator on ctypes types is standard, but we'll try to be more explicit
    # for the linter.
    _type = ctypes.c_float * count
    return _type(*lst)


# ─────────────────────────────────────────────────────────────────────────────
# UV-sphere geometry
# ─────────────────────────────────────────────────────────────────────────────
def _make_sphere(rings=32, sectors=32):
    verts, tris = [], []
    for r in range(rings+1):
        phi = math.pi/2 - math.pi * r / rings
        y   = math.sin(phi)
        for s in range(sectors+1):
            theta = 2*math.pi * s / sectors
            x = math.cos(phi)*math.cos(theta)
            z = math.cos(phi)*math.sin(theta)
            # pos, normal (same for unit sphere), uv
            verts += [x,y,z, x,y,z, s/sectors, r/rings]
    for r in range(rings):
        for s in range(sectors):
            a = r*(sectors+1)+s
            b = a+(sectors+1)
            tris += [a, b, a+1, b, b+1, a+1]
    import numpy as np
    return (np.array(verts, dtype=np.float32),
            np.array(tris,  dtype=np.uint32))

def _make_cube_verts():
    """Unit cube (pos only) for skybox."""
    v = [
        -1,-1,-1,  1,-1,-1,  1, 1,-1, -1, 1,-1,
        -1,-1, 1,  1,-1, 1,  1, 1, 1, -1, 1, 1,
    ]
    idx = [
        0,1,2,2,3,0,  4,5,6,6,7,4,
        0,4,7,7,3,0,  1,5,6,6,2,1,
        3,2,6,6,7,3,  0,1,5,5,4,0,
    ]
    import numpy as np
    return (np.array(v,   dtype=np.float32),
            np.array(idx, dtype=np.uint32))


# ─────────────────────────────────────────────────────────────────────────────
# The OpenGL preview widget
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Transparent 2D overlay widget (lives on top of GL3DPreview, no GL context)
# ─────────────────────────────────────────────────────────────────────────────
class _OverlayWidget(QWidget):
    """Transparent child widget for drawing 2D images on top of a QOpenGLWidget.
    
    Rationale: mixing QPainter inside paintGL() on a Core Profile QOpenGLWidget
    causes GL_INVALID_OPERATION because Qt puts the context in a transitional
    state when constructing QPainter.  The solution is to keep paintGL() as
    *pure* OpenGL and draw all 2D content in a separate, transparent child widget
    whose paintEvent uses a standard Qt 2D painter — no GL context involvement.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._pixmap: QPixmap | None = None

    def set_pixmap(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        if not self._pixmap or self._pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        scaled = self._pixmap.scaled(
            W, H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        rect = scaled.rect()
        rect.moveCenter(self.rect().center())
        painter.drawPixmap(rect, self._pixmap)
        painter.end()


class GL3DPreview(QOpenGLWidget):
    """OpenGL 3.3 core preview — PBR + HDR environment."""

    def __init__(self, parent=None):
        # Format already applied at module level (see _fmt above).
        super().__init__(parent)

        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        # Transparent 2D overlay (lives above the GL surface, has no GL context)
        # Created after super().__init__ so the parent QOpenGLWidget is valid
        self._overlay = _OverlayWidget(self)
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()

        # Camera state
        self._azim   = 45.0
        self._elev   = 25.0
        self._dist   = 3.0
        self._target = [0.0, 0.0, 0.0]

        # Inertia
        self._v_azim = 0.0   # angular velocity (deg/frame)
        self._v_elev = 0.0
        self._dragging = False        # left-drag = camera orbit
        self._panning  = False        # middle-drag = camera pan
        self._walking  = False        # shift+middle-drag = walk
        self._last_pos = QPointF()
        self._last_drag_delta = QPointF()

        # Flythrough mode (RMB held)
        self._flying = False
        self._fly_keys = set()         # currently held keys: 'w','a','s','d','q','e'
        self._fly_speed = 0.05         # base movement speed per tick
        self._fly_speed_boost = 3.0    # multiplier when Shift held
        self._fly_shift = False        # Shift key state

        # Callback list: called with (x, y) whenever object is moved via drag
        self._obj_offset_listeners = []    # list of callables(x, y)

        # Display state
        self._wireframe  = False
        self._env_strength = 1.0
        self._hdri_rotation = 0.0        # degrees
        self._albedo     = [0.9, 0.35, 0.15]   # default: rusty orange
        self._metallic   = 0.1
        self._roughness  = 0.45
        # Light
        self._light_azim  = 45.0          # degrees, 0=+Z, CCW
        self._light_elev  = 60.0          # degrees above horizon
        self._light_intensity = 2.8       # PBR radiance scale
        # Object transform
        self._obj_offset  = [0.0, 0.0, 0.0]  # world-space XYZ offset
        self._obj_scale   = 1.0               # uniform scale multiplier
        # Mesh / HDRI paths
        self._mesh_verts = None
        self._mesh_idx   = None
        self._hdri_path: str = ""
        self._pending_mesh_path: str = ""   # queued before GL ready
        self._pending_hdri_path: str  = ""   # queued before GL ready
        # Dirty flags — GPU work deferred to next paintGL frame
        self._env_needs_reload  = False
        self._mesh_needs_upload = False
        self._preview_pixmap: QPixmap | None = None

        # GL objects (initialised in initializeGL)
        self._prog = self._sky_prog = None
        self._sphere_vao = self._sphere_vbo = self._sphere_ebo = None
        self._sky_vao = self._sky_vbo = self._sky_ebo = None
        self._env_tex = None
        self._sphere_idx_count = 0
        self._sky_idx_count    = 0
        self._gl_ready = False

        # 60 fps inertia / animation timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)
        
        # Grid/Axis resources
        self._grid_vao = self._grid_vbo = None
        self._grid_count = 0

        # ── Gizmo state ──────────────────────────────────────────────────────
        self._gizmo_mode = 'translate'        # 'translate', 'rotate', 'scale'
        self._gizmo_hover_axis = None         # None, 'x', 'y', 'z'
        self._gizmo_active_axis = None        # axis being dragged
        self._gizmo_drag_start_pos = QPointF()
        self._gizmo_drag_start_offset = [0.0, 0.0, 0.0]
        self._gizmo_drag_start_scale = 1.0
        self._gizmo_drag_start_angle = 0.0    # for rotate mode
        self._obj_rotation = [0.0, 0.0, 0.0]  # Euler angles (degrees) X, Y, Z
        # Gizmo GL resources
        self._gizmo_prog = None
        self._gizmo_vao = self._gizmo_vbo = None

        # Object selection state
        self._obj_selected = False

        self.setMouseTracking(True)  # needed for hover detection

    # ── Camera helpers ────────────────────────────────────────────────────────

    def _camera_pos(self):
        """Return the camera's world-space position (target + orbit offset)."""
        az  = math.radians(self._azim)
        el  = math.radians(self._elev)
        d   = self._dist
        x   = self._target[0] + d * math.cos(el) * math.sin(az)
        y   = self._target[1] + d * math.sin(el)
        z   = self._target[2] + d * math.cos(el) * math.cos(az)
        return [x, y, z]

    def _camera_up(self):
        """Return up vector that avoids gimbal lock at top/bottom views."""
        el = math.radians(self._elev)
        # At near-vertical angles, use Z as the up vector
        if abs(self._elev - 90) < 1.0 or abs(self._elev + 90) < 1.0:
            # Looking straight down (elev=90) → up = +Z
            # Looking straight up  (elev=-90) → up = -Z
            return [0, 0, 1] if self._elev > 0 else [0, 0, -1]
        return [0, -1, 0] if math.cos(el) < 0 else [0, 1, 0]

    def set_azim_elev(self, azim, elev):
        self._azim  = azim
        self._elev  = elev    # no clamp — full 360° orbit
        self.update()

    def reset_camera(self):
        self._azim   = 45.0
        self._elev   = 25.0
        self._dist   = 3.0
        self._v_azim = 0.0
        self._v_elev = 0.0
        self.update()

    def set_wireframe(self, on: bool):
        self._wireframe = on
        self.update()

    # ── Scene inspector setters ─────────────────────────────────────────

    def set_albedo(self, r: float, g: float, b: float):
        self._albedo = [r, g, b]
        self.update()

    def set_material(self, metallic: float, roughness: float):
        self._metallic  = metallic
        self._roughness = roughness
        self.update()

    def set_light(self, azim: float, elev: float, intensity: float):
        self._light_azim      = azim
        self._light_elev      = elev
        self._light_intensity = intensity
        self.update()

    def set_env_strength(self, v: float):
        self._env_strength = v
        self.update()

    def set_hdri_rotation(self, deg: float):
        self._hdri_rotation = deg
        self.update()

    def set_obj_offset(self, x: float, y: float):
        self._obj_offset = [x, y, 0.0]
        self.update()

    def set_obj_scale(self, s: float):
        self._obj_scale = max(0.001, s)
        self.update()

    def get_scene_state(self) -> dict:
        """Return the current scene configuration for the PyTorch3D preview worker."""
        laz = math.radians(self._light_azim)
        lel = math.radians(self._light_elev)
        lx  =  math.cos(lel) * math.sin(laz)
        ly  =  math.sin(lel)
        lz  =  math.cos(lel) * math.cos(laz)
        return {
            "dist":            self._dist,
            "elev":            self._elev,
            "azim":            self._azim,
            "obj_offset":      list(self._obj_offset),
            "obj_scale":       self._obj_scale,
            "albedo":          list(self._albedo),
            "metallic":        self._metallic,
            "roughness":       self._roughness,
            "light_dir":       [lx, ly, lz],
            "light_intensity": self._light_intensity,
            "env_strength":    self._env_strength,
            "hdri_rotation":   self._hdri_rotation,
        }

    def set_hdri(self, path: str):
        """Set HDRI path. GPU upload is deferred to the next paintGL frame."""
        self._hdri_path = path or ""
        if not self._gl_ready:
            # Store for flush in initializeGL — always mark reload needed
            self._pending_hdri_path = self._hdri_path
            self._env_needs_reload = True   # initializeGL will honour this
            return
        self._env_needs_reload = True
        self.update()

    def set_preview_image(self, qimage: QImage):
        """Display a QImage from the PyTorch3D worker over the GL viewport."""
        self._preview_pixmap = QPixmap.fromImage(qimage)
        self._overlay.set_pixmap(self._preview_pixmap)

    def load_obj_mesh(self, path: str):
        """Set mesh path. GPU upload is deferred to the next paintGL frame."""
        if not HAS_OPENGL or not path or not os.path.exists(path):
            return
        self._pending_mesh_path = path
        if not self._gl_ready:
            return   # initializeGL will flush _pending_mesh_path
        self._mesh_needs_upload = True
        self.update()

    def _do_upload_mesh(self, path: str):
        """
        Internal: load + upload one mesh from disk.  Called only from paintGL
        so the GL context is always active.  Never call directly from UI code.
        """
        from renderer.loader import MeshLoader
        import numpy as np

        mesh = MeshLoader.load(path, device="cpu")
        v = mesh.verts_padded().cpu().numpy()[0].astype(np.float32)
        f = mesh.faces_padded().cpu().numpy()[0].astype(np.uint32)

        # Normalise to fit camera orbit (dist=3.0 → 1.5-unit diameter)
        v_min, v_max = v.min(axis=0), v.max(axis=0)
        v -= (v_min + v_max) / 2.0
        v *= 1.5 / max(float(np.linalg.norm(v_max - v_min)), 0.001)

        try:
            n = mesh.verts_normals_padded().cpu().numpy()[0].astype(np.float32)
        except Exception:
            n = None

        self._upload_custom_mesh(v, f, n)

    # ── Phase 5 — Safe GPU teardown + reinit ───────────────────────────────

    def reset_gl(self):
        """
        Delete all GPU resources and reinitialise from scratch.  Safe to call
        after context loss (e.g., on some drivers when the window is minimised).
        NOT meant to be called from paintGL.
        """
        if not self._gl_ready:
            return
        self.makeCurrent()   
        try:
            from OpenGL.GL import glDeleteTextures, glDeleteVertexArrays, glDeleteBuffers
            tex = self._env_tex
            if tex is not None:
                glDeleteTextures(1, [int(tex)])
                self._env_tex = None
            
            vaos = [int(v) for v in [self._sphere_vao, self._sky_vao, self._grid_vao] if v is not None]
            if vaos:
                glDeleteVertexArrays(len(vaos), vaos)
                
            bufs = [int(b) for b in [self._sphere_vbo, self._sphere_ebo, self._sky_vbo, self._sky_ebo, self._grid_vbo] if b is not None]
            if bufs:
                glDeleteBuffers(len(bufs), bufs)
        except Exception as exc:
            print(f"[GL] reset_gl cleanup error: {exc}")
            
        self._sphere_vao = self._sky_vao = self._grid_vao = None
        self._sphere_vbo = self._sphere_ebo = None
        self._sky_vbo    = self._sky_ebo    = None
        self._grid_vbo   = None
        self._env_tex    = None
        self._prog = self._sky_prog = None
        self._gl_ready = False
        if self._hdri_path:
            self._pending_hdri_path = self._hdri_path
        # Note: Do NOT call initializeGL() manually. It will leak if re-called.
        # Instead, just trigger update and let Qt handle it.
        self.update()

    # ── Mouse / scroll ────────────────────────────────────────────────────────

    def add_obj_offset_listener(self, fn):
        """Register a callback(x, y) called whenever the object is dragged."""
        self._obj_offset_listeners.append(fn)

    def _emit_obj_offset(self):
        x, y = self._obj_offset[0], self._obj_offset[1]
        for fn in self._obj_offset_listeners:
            try:
                fn(x, y)
            except Exception:
                pass

    def mousePressEvent(self, event):
        pos = event.position()
        # Always clear the static preview overlay on any click
        if hasattr(self, '_overlay'):
            self._overlay.clear()
        # Notify parent to stop idle rendering
        parent = self.parentWidget()
        if hasattr(parent, "_on_gl_drag_start"):
            parent._on_gl_drag_start()

        if event.button() == Qt.MouseButton.RightButton:
            # RMB = Enter flythrough mode
            self._flying = True
            self._fly_keys.clear()
            self._fly_shift = False
            self._last_pos = pos
            self.setCursor(Qt.CursorShape.BlankCursor)
            self.setFocus()  # ensure we get key events
            self.grabKeyboard()  # capture all keys during fly mode
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._orientation_gizmo_hit_test(pos):
                return

            # In View mode, always do camera orbit (no gizmo)
            if self._gizmo_mode == 'view':
                self._dragging = True
                self._last_pos = pos
                self._last_drag_delta = QPointF(0, 0)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                return

            # Other modes: try gizmo first (only if object is selected)
            if self._obj_selected:
                axis = self._gizmo_hit_test(pos.x(), pos.y())
                if axis:
                    self._gizmo_active_axis = axis
                    self._gizmo_drag_start_pos = pos
                    self._gizmo_drag_start_offset = list(self._obj_offset)
                    self._gizmo_drag_start_scale = self._obj_scale
                    self._gizmo_drag_start_angle = 0.0
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    if hasattr(self, '_overlay'):
                        self._overlay.clear()
                    return

            # Try clicking on the object to select it
            if self._hit_test_object(pos.x(), pos.y()):
                self._obj_selected = True
                self.update()
                return

            # Clicked empty space — deselect
            if self._obj_selected:
                self._obj_selected = False
                self.update()
                return

            # Camera orbit (nothing hit)
            self._dragging = True
            self._last_pos = pos
            self._last_drag_delta = QPointF(0, 0)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            parent = self.parentWidget()
            if hasattr(parent, "_on_gl_drag_start"):
                parent._on_gl_drag_start()
        elif event.button() == Qt.MouseButton.MiddleButton:
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                # Shift + MMB = Walk (move in look direction)
                self._walking = True
                self._last_pos = pos
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            else:
                # MMB = Pan (Unity-style)
                self._panning = True
                self._last_pos = pos
                self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            # Exit flythrough mode
            self._flying = False
            self._fly_keys.clear()
            self._fly_shift = False
            self.releaseKeyboard()
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._gizmo_active_axis:
                self._gizmo_active_axis = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                self._emit_obj_offset()
            elif self._dragging:
                self._dragging = False
                self._v_azim =  self._last_drag_delta.x() * 0.5
                self._v_elev = -self._last_drag_delta.y() * 0.5
                self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._walking = False
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        # Restart idle timer for PyTorch3D renders
        parent = self.parentWidget()
        if hasattr(parent, "_on_gl_drag_end"):
            parent._on_gl_drag_end()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._flying:
            # FPS-style look-around: keep camera position fixed, rotate view
            delta = pos - self._last_pos
            self._last_pos = pos

            # Save current camera world position before changing angles
            cam_world = self._camera_pos()

            # Rotate view direction (inverted for orbit model → FPS feel)
            self._azim -= delta.x() * 0.25
            self._elev += delta.y() * 0.25
            self._elev = max(-89.0, min(89.0, self._elev))

            # Recompute target so camera stays at the same world position
            # cam_world = target + offset  →  target = cam_world - offset
            az  = math.radians(self._azim)
            el  = math.radians(self._elev)
            d   = self._dist
            self._target[0] = cam_world[0] - d * math.cos(el) * math.sin(az)
            self._target[1] = cam_world[1] - d * math.sin(el)
            self._target[2] = cam_world[2] - d * math.cos(el) * math.cos(az)
            self.update()
        elif getattr(self, '_walking', False):
            # Shift+MMB walk: mouse-up/down = move up/down, left/right = strafe
            delta = pos - self._last_pos
            self._last_pos = pos
            speed = self._dist * 0.003
            az = math.radians(self._azim)
            # Right vector (horizontal strafe)
            r_x =  math.cos(az)
            r_z = -math.sin(az)
            # Mouse-X → strafe left/right, Mouse-Y → move up/down (world Y)
            move_right = delta.x() * speed
            move_up = -delta.y() * speed
            self._target[0] += move_right * r_x
            self._target[1] += move_up
            self._target[2] += move_right * r_z
            self.update()
        elif self._gizmo_active_axis:
            # Constrained drag along the active axis
            self._gizmo_apply_drag(pos)
            self.update()
        elif self._dragging:
            delta = pos - self._last_pos
            self._last_drag_delta = delta
            self._last_pos = pos
            self._azim += delta.x() * 0.4
            self._elev -= delta.y() * 0.4
            self.update()
        elif getattr(self, '_panning', False):
            delta = pos - self._last_pos
            self._last_pos = pos
            speed = self._dist * 0.002
            # Compute proper camera right & up from azimuth + elevation
            az = math.radians(self._azim)
            el = math.radians(self._elev)
            # Camera right vector (perpendicular to view in horizontal plane)
            r_x =  math.cos(az)
            r_y =  0.0
            r_z = -math.sin(az)
            # Camera up vector (accounts for elevation tilt)
            u_x = -math.sin(az) * math.sin(el)
            u_y =  math.cos(el)
            u_z = -math.cos(az) * math.sin(el)
            # Move target along camera right (horizontal) and up (vertical)
            dx = -delta.x() * speed
            dy =  delta.y() * speed
            self._target[0] += dx * r_x + dy * u_x
            self._target[1] += dx * r_y + dy * u_y
            self._target[2] += dx * r_z + dy * u_z
            self.update()
        else:
            # Hover detection for gizmo
            axis = self._gizmo_hit_test(pos.x(), pos.y())
            if axis != self._gizmo_hover_axis:
                self._gizmo_hover_axis = axis
                self.setCursor(Qt.CursorShape.SizeAllCursor if axis else Qt.CursorShape.ArrowCursor)
                self.update()

    def wheelEvent(self, event):
        self._dist = max(0.5, min(20.0, self._dist - event.angleDelta().y() * 0.005))
        self.update()

    def keyPressEvent(self, event):
        key = event.key()

        # In fly mode, capture movement keys instead of gizmo shortcuts
        if self._flying:
            fly_key_map = {
                Qt.Key.Key_W: 'w',
                Qt.Key.Key_S: 's',
                Qt.Key.Key_A: 'a',
                Qt.Key.Key_D: 'd',
                Qt.Key.Key_Q: 'q',
                Qt.Key.Key_E: 'e',
            }
            if key in fly_key_map:
                self._fly_keys.add(fly_key_map[key])
                return
            if key == Qt.Key.Key_Shift:
                self._fly_shift = True
                return
            # Let other keys pass through
            super().keyPressEvent(event)
            return

        mode_map = {
            Qt.Key.Key_Q: 'view',
            Qt.Key.Key_W: 'translate',
            Qt.Key.Key_E: 'rotate',
            Qt.Key.Key_R: 'scale',
            Qt.Key.Key_T: 'rect',
            Qt.Key.Key_Y: 'transform',
        }
        if key in mode_map:
            mode = mode_map[key]
            self._gizmo_mode = mode
            self.update()
            # Sync toolbar if it exists
            parent = self.parentWidget()
            if parent and hasattr(parent, 'parentWidget'):
                vp = parent.parentWidget()
                if vp and hasattr(vp, '_scene_toolbar'):
                    vp._scene_toolbar.set_mode(mode)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        key = event.key()
        if self._flying:
            fly_key_map = {
                Qt.Key.Key_W: 'w',
                Qt.Key.Key_S: 's',
                Qt.Key.Key_A: 'a',
                Qt.Key.Key_D: 'd',
                Qt.Key.Key_Q: 'q',
                Qt.Key.Key_E: 'e',
            }
            if key in fly_key_map:
                self._fly_keys.discard(fly_key_map[key])
                return
            if key == Qt.Key.Key_Shift:
                self._fly_shift = False
                return
        super().keyReleaseEvent(event)

    # ── Gizmo helpers ────────────────────────────────────────────────────────

    def _world_to_screen(self, wp, mvp, W, H):
        """Project a 3D world point to 2D screen coords using the MVP matrix."""
        x, y, z = wp
        # MVP is column-major flat list (16 floats)
        cx = mvp[0]*x + mvp[4]*y + mvp[8]*z  + mvp[12]
        cy = mvp[1]*x + mvp[5]*y + mvp[9]*z  + mvp[13]
        cz = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14]
        cw = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15]
        if abs(cw) < 1e-6:
            return None
        ndx, ndy = cx / cw, cy / cw
        sx = (ndx * 0.5 + 0.5) * W
        sy = (1.0 - (ndy * 0.5 + 0.5)) * H
        return (sx, sy)

    def _gizmo_axis_endpoints(self):
        """Return world-space start/end for each gizmo axis."""
        ox, oy, oz = self._obj_offset
        length = 0.8  # fixed arm length matching _draw_gizmo
        return {
            'x': ([ox, oy, oz], [ox + length, oy, oz]),
            'y': ([ox, oy, oz], [ox, oy + length, oz]),
            'z': ([ox, oy, oz], [ox, oy, oz + length]),
        }

    def _hit_test_object(self, mx, my):
        """Test if mouse click hits the object (ray-sphere intersection)."""
        if not self._gl_ready:
            return False
        W, H = self.width(), self.height()
        if W < 1 or H < 1:
            return False

        aspect = W / max(H, 1)
        proj = _mat4_perspective(60.0, aspect, 0.1, 5000.0)
        eye  = self._camera_pos()
        up   = self._camera_up()
        view = _lookat(eye, self._target, up)

        # Convert mouse position to NDC
        ndc_x = (2.0 * mx / W) - 1.0
        ndc_y = 1.0 - (2.0 * my / H)

        # Unproject near and far points
        vp = _mat4_mul(proj, view)
        inv_vp = _mat4_inverse(vp)

        near_ndc = [ndc_x, ndc_y, -1.0, 1.0]
        far_ndc  = [ndc_x, ndc_y,  1.0, 1.0]

        near_world = _mat4_vec4_mul(inv_vp, near_ndc)
        far_world  = _mat4_vec4_mul(inv_vp, far_ndc)

        if abs(near_world[3]) < 1e-9 or abs(far_world[3]) < 1e-9:
            return False

        near_w = [near_world[i] / near_world[3] for i in range(3)]
        far_w  = [far_world[i]  / far_world[3]  for i in range(3)]

        # Ray direction
        ray_dir = [far_w[i] - near_w[i] for i in range(3)]
        ray_len = math.sqrt(sum(d*d for d in ray_dir))
        if ray_len < 1e-9:
            return False
        ray_dir = [d / ray_len for d in ray_dir]

        # Sphere center = object offset, radius = object scale
        center = self._obj_offset
        radius = self._obj_scale

        # Ray-sphere intersection: |O + tD - C|^2 = r^2
        oc = [near_w[i] - center[i] for i in range(3)]
        a = sum(ray_dir[i]**2 for i in range(3))
        b = 2.0 * sum(oc[i] * ray_dir[i] for i in range(3))
        c = sum(oc[i]**2 for i in range(3)) - radius**2
        disc = b*b - 4*a*c

        return disc >= 0

    def _gizmo_hit_test(self, mx, my):
        """Return 'x'/'y'/'z'/'free' if mouse is near a gizmo axis, else None."""
        if not self._gl_ready:
            return None
        W, H = self.width(), self.height()
        if W < 1 or H < 1:
            return None

        aspect = W / max(H, 1)
        proj = _mat4_perspective(60.0, aspect, 0.1, 500.0)
        eye  = self._camera_pos()
        up   = self._camera_up()
        view = _lookat(eye, self._target, up)
        mvp = self._compute_mvp(proj, view)

        # Use ring-based hit test for rotate mode
        if self._gizmo_mode == 'rotate':
            return self._gizmo_ring_hit_test(mx, my, mvp, W, H)

        # Check center box first (free movement handle)
        if self._gizmo_mode == 'translate':
            ox, oy, oz = self._obj_offset
            center_2d = self._world_to_screen([ox, oy, oz], mvp, W, H)
            if center_2d:
                dist_to_center = math.hypot(mx - center_2d[0], my - center_2d[1])
                if dist_to_center < 18.0:  # pixel radius for center box
                    return 'free'

        # Line-segment hit test for translate/scale/etc.
        endpoints = self._gizmo_axis_endpoints()
        threshold = 15.0  # pixels

        best_axis = None
        best_dist = threshold

        for axis_name, (start, end) in endpoints.items():
            s2d = self._world_to_screen(start, mvp, W, H)
            e2d = self._world_to_screen(end, mvp, W, H)
            if s2d is None or e2d is None:
                continue
            d = self._point_to_segment_dist(mx, my, s2d, e2d)
            if d < best_dist:
                best_dist = d
                best_axis = axis_name
        return best_axis

    def _gizmo_ring_hit_test(self, mx, my, mvp, W, H):
        """Hit-test the rotation gizmo rings (circles in YZ, XZ, XY planes)."""
        ox, oy, oz = self._obj_offset
        radius = 0.6  # must match _gizmo_rotate_verts
        segments = 64
        threshold = 12.0  # pixels

        # Define the 3 rings: each is a list of (axis_name, point_generator)
        rings = {
            'x': [],  # YZ plane (red)
            'y': [],  # XZ plane (green)
            'z': [],  # XY plane (blue)
        }

        for i in range(segments + 1):
            a = 2.0 * math.pi * i / segments
            c, s = math.cos(a), math.sin(a)
            # Red ring in YZ plane
            rings['x'].append([ox, oy + c * radius, oz + s * radius])
            # Green ring in XZ plane
            rings['y'].append([ox + c * radius, oy, oz + s * radius])
            # Blue ring in XY plane
            rings['z'].append([ox + c * radius, oy + s * radius, oz])

        best_axis = None
        best_dist = threshold

        for axis_name, points in rings.items():
            for i in range(len(points) - 1):
                s2d = self._world_to_screen(points[i], mvp, W, H)
                e2d = self._world_to_screen(points[i + 1], mvp, W, H)
                if s2d is None or e2d is None:
                    continue
                d = self._point_to_segment_dist(mx, my, s2d, e2d)
                if d < best_dist:
                    best_dist = d
                    best_axis = axis_name

        return best_axis

    def _compute_mvp(self, proj, view):
        """Compute MVP = proj * view (column-major OpenGL convention)."""
        # Both proj and view are 4x4 column-major flat lists
        # C[col][row] = sum_k A[k][row] * B[col][k]
        # In flat column-major: index(row, col) = col*4 + row
        result = [0.0] * 16
        for col in range(4):
            for row in range(4):
                s = 0.0
                for k in range(4):
                    s += proj[k * 4 + row] * view[col * 4 + k]
                result[col * 4 + row] = s
        return result

    def _point_to_segment_dist(self, px, py, s, e):
        """Distance from point (px,py) to line segment s→e in 2D."""
        sx, sy = s
        ex, ey = e
        dx, dy = ex - sx, ey - sy
        lenSq = dx * dx + dy * dy
        if lenSq < 1e-6:
            return math.hypot(px - sx, py - sy)
        t = max(0, min(1, ((px - sx) * dx + (py - sy) * dy) / lenSq))
        cx, cy = sx + t * dx, sy + t * dy
        return math.hypot(px - cx, py - cy)

    def _gizmo_apply_drag(self, pos):
        """Apply gizmo drag based on mode and active axis."""
        axis = self._gizmo_active_axis
        if not axis:
            return

        dx = pos.x() - self._gizmo_drag_start_pos.x()
        dy = pos.y() - self._gizmo_drag_start_pos.y()
        speed = self._dist * 0.002

        if self._gizmo_mode == 'translate':
            if axis == 'free':
                # Free movement on camera plane
                az = math.radians(self._azim)
                el = math.radians(self._elev)
                # Camera right vector
                cr_x =  math.cos(az)
                cr_y =  0.0
                cr_z = -math.sin(az)
                # Camera up vector (accounts for elevation)
                cu_x = -math.sin(az) * math.sin(el)
                cu_y =  math.cos(el)
                cu_z = -math.cos(az) * math.sin(el)

                self._obj_offset = list(self._gizmo_drag_start_offset)
                self._obj_offset[0] += (dx * cr_x + (-dy) * cu_x) * speed
                self._obj_offset[1] += (dx * cr_y + (-dy) * cu_y) * speed
                self._obj_offset[2] += (dx * cr_z + (-dy) * cu_z) * speed
            else:
                # Project screen delta onto the gizmo axis direction
                W, H = self.width(), self.height()
                aspect = W / max(H, 1)
                proj = _mat4_perspective(60.0, aspect, 0.1, 500.0)
                eye  = self._camera_pos()
                up   = self._camera_up()
                view = _lookat(eye, self._target, up)
                mvp  = self._compute_mvp(proj, view)

                ox, oy, oz = self._gizmo_drag_start_offset
                endpoints = self._gizmo_axis_endpoints()
                start_world, end_world = endpoints[axis]

                s2d = self._world_to_screen([ox, oy, oz], mvp, W, H)
                e2d = self._world_to_screen(end_world, mvp, W, H)
                if s2d and e2d:
                    # Screen-space axis direction
                    adx = e2d[0] - s2d[0]
                    ady = e2d[1] - s2d[1]
                    alen = math.hypot(adx, ady)
                    if alen > 1e-3:
                        adx /= alen
                        ady /= alen
                        # Project mouse delta onto screen axis
                        proj_len = dx * adx + dy * ady
                        world_delta = proj_len * speed

                        self._obj_offset = list(self._gizmo_drag_start_offset)
                        if axis == 'x':
                            self._obj_offset[0] += world_delta
                        elif axis == 'y':
                            self._obj_offset[1] += world_delta
                        elif axis == 'z':
                            self._obj_offset[2] += world_delta

        elif self._gizmo_mode == 'scale':
            # Scale: drag right = bigger, drag left = smaller
            scale_delta = dx * 0.01
            self._obj_scale = max(0.01, self._gizmo_drag_start_scale + scale_delta)

        elif self._gizmo_mode == 'rotate':
            # Rotate around the selected axis: use screen X as angle
            angle = dx * 0.5  # degrees
            idx = {'x': 0, 'y': 1, 'z': 2}[axis]
            self._obj_rotation[idx] = self._gizmo_drag_start_angle + angle


    # ── Inertia tick ─────────────────────────────────────────────────────────

    def _tick(self):
        # Don't update if not visible or if currently dragging (reduces redundant calls)
        if not self.isVisible() or self._dragging:
            return

        # ── Flythrough movement ─────────────────────────────────────────
        if self._flying and self._fly_keys:
            speed = self._fly_speed
            if self._fly_shift:
                speed *= self._fly_speed_boost

            az = math.radians(self._azim)
            el = math.radians(self._elev)

            # Camera forward vector (into the screen)
            fwd_x = -math.cos(el) * math.sin(az)
            fwd_y = -math.sin(el)
            fwd_z = -math.cos(el) * math.cos(az)
            # Camera right vector
            right_x =  math.cos(az)
            right_y =  0.0
            right_z = -math.sin(az)
            # World up
            up_x, up_y, up_z = 0.0, 1.0, 0.0

            dx, dy, dz = 0.0, 0.0, 0.0
            if 'w' in self._fly_keys:
                dx += fwd_x; dy += fwd_y; dz += fwd_z
            if 's' in self._fly_keys:
                dx -= fwd_x; dy -= fwd_y; dz -= fwd_z
            if 'd' in self._fly_keys:
                dx += right_x; dy += right_y; dz += right_z
            if 'a' in self._fly_keys:
                dx -= right_x; dy -= right_y; dz -= right_z
            if 'e' in self._fly_keys:
                dx += up_x; dy += up_y; dz += up_z
            if 'q' in self._fly_keys:
                dx -= up_x; dy -= up_y; dz -= up_z

            # Normalize and apply speed
            length = math.sqrt(dx*dx + dy*dy + dz*dz)
            if length > 0.001:
                dx /= length; dy /= length; dz /= length
                self._target[0] += dx * speed
                self._target[1] += dy * speed
                self._target[2] += dz * speed
                self.update()
            return

        if abs(self._v_azim) > 0.01 or abs(self._v_elev) > 0.01:
            self._azim  += self._v_azim
            self._elev  += self._v_elev    # no clamp — full 360° orbit
            self._v_azim *= 0.92
            self._v_elev *= 0.92
            self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the transparent overlay geometry in sync with our size
        self._overlay.setGeometry(self.rect())
        self._overlay.raise_()  # ensure it stays on top

    # ── OpenGL lifecycle ──────────────────────────────────────────────────────

    def initializeGL(self):
        if not HAS_OPENGL:
            return
        try:
            # ── Permanent GL state (set once) ───────────────────────────
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glClearColor(0.05, 0.07, 0.09, 1.0)

            # ── One-time GPU resource creation ──────────────────────────
            self._build_pbr_program()
            self._build_sky_program()
            self._build_gizmo_program()
            self._upload_sphere()
            self._upload_skybox()
            self._upload_grid()
            self._upload_gizmo()
            
            # Clear any errors during init
            while glGetError() != GL_NO_ERROR: pass
            
            self._gl_ready = True

            if self._pending_hdri_path:
                self._hdri_path = self._pending_hdri_path
                self._pending_hdri_path = ""
            
            self._env_needs_reload = True
        except Exception as e:
            print(f"[GL] initializeGL error: {e}")
            self._gl_ready = False

    def paintGL(self):
        if not HAS_OPENGL or not self._gl_ready:
            return
        if not self.context() or not self.isValid():
            return

        try:
            while glGetError() != GL_NO_ERROR: 
                pass

            # ── Flush deferred GPU resources ────
            if self._env_needs_reload:
                self._load_env_texture()
                self._env_needs_reload = False

            if self._mesh_needs_upload and self._pending_mesh_path:
                try:
                    self._do_upload_mesh(self._pending_mesh_path)
                except Exception as exc:
                    print(f"[GL] Mesh upload failed: {exc}")
                self._pending_mesh_path = ""
                self._mesh_needs_upload = False

            # ── Pure rendering ────────────────────────────────────
            W, H = self.width(), self.height()
            glDisable(GL_SCISSOR_TEST) # Safeguard against orientation gizmo state leakage
            glClearColor(0.22, 0.22, 0.22, 1.0)  # Dark Unity-style background
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            aspect = W / max(H, 1)
            proj   = _mat4_perspective(60.0, aspect, 0.1, 5000.0)
            eye    = self._camera_pos()
            up     = self._camera_up()
            view   = _lookat(eye, self._target, up)

            # Cache ALL resources once at the top — never re-assign below
            env_tex  = self._env_tex
            sky_prog = self._sky_prog
            pbr_prog = self._prog
            grid_vao = self._grid_vao
            sky_vao  = self._sky_vao
            mesh_vao = self._sphere_vao

            # Helper to validate VAOs
            def is_valid_vao(vid):
                if vid is None or vid <= 0: return False
                try: return bool(glIsVertexArray(int(vid)))
                except: return False

            # ── Rendering Pipeline ───────────────────────────────────────────────────

            # ── Skybox ────────────────────────────────────────────────────────────────
            if env_tex is not None and sky_vao is not None and sky_prog is not None \
                    and not self._wireframe:
                v_id = int(sky_vao)
                if is_valid_vao(v_id):
                    glDepthFunc(GL_LEQUAL)
                    glDepthMask(GL_FALSE)
                    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
                    sky_prog.bind()
                    # MatMemory(PureRot) = PureRot^T
                    view_rot = list(view)
                    view_rot[12] = view_rot[13] = view_rot[14] = 0.0
                    self._set_mat4(sky_prog, "uViewRot", view_rot)
                    self._set_mat4(sky_prog, "uProj",    proj)
                    self._set_f   (sky_prog, "uEnvStrength",  self._env_strength)
                    self._set_f   (sky_prog, "uHdriRotation", math.radians(self._hdri_rotation))
                    try:
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, int(env_tex))
                        self._set_i(sky_prog, "uEnvMap", 0)
                        while glGetError() != GL_NO_ERROR: pass
                        glBindVertexArray(v_id)
                        glDrawElements(GL_TRIANGLES, self._sky_idx_count, GL_UNSIGNED_INT, None)
                    except Exception as e:
                        print(f"[GL] Skybox draw error: {e}")
                    finally:
                        glBindVertexArray(0)
                        sky_prog.release()
                        glDepthMask(GL_TRUE)
                        glDepthFunc(GL_LESS)

            # ── Grid floor ─────────────────────────────────────────────────────
            if getattr(self, '_gizmo_prog', None) is not None:
                self._draw_grid_floor(proj, view)

            # ── PBR mesh ──────────────────────────────────────────────────────
            if pbr_prog is not None and mesh_vao is not None:
                v_id = int(mesh_vao)
                if is_valid_vao(v_id):
                    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self._wireframe else GL_FILL)
                    pbr_prog.bind()
                    s  = self._obj_scale
                    ox, oy, oz = self._obj_offset
                    rx, ry, rz = self._obj_rotation
                    model = _mat4_translate(ox, oy, oz)
                    model = _mat4_mul(model, _mat4_rot_z(rz))
                    model = _mat4_mul(model, _mat4_rot_y(ry))
                    model = _mat4_mul(model, _mat4_rot_x(rx))
                    model = _mat4_mul(model, _mat4_scale(s))
                    normal_m = _mat3_inverse_transpose(model)
                    laz, lel = math.radians(self._light_azim), math.radians(self._light_elev)
                    lx, ly, lz = math.cos(lel)*math.sin(laz), math.sin(lel), math.cos(lel)*math.cos(laz)
                    self._set_mat4(pbr_prog, "uModel",     model)
                    self._set_mat4(pbr_prog, "uView",      view)
                    self._set_mat4(pbr_prog, "uProj",      proj)
                    self._set_mat3(pbr_prog, "uNormalMat", normal_m)
                    self._set_v3  (pbr_prog, "uCamPos",    eye[0], eye[1], eye[2])
                    self._set_v3  (pbr_prog, "uAlbedo",    self._albedo[0], self._albedo[1], self._albedo[2])
                    self._set_f   (pbr_prog, "uMetallic",      self._metallic)
                    self._set_f   (pbr_prog, "uRoughness",     self._roughness)
                    self._set_f   (pbr_prog, "uEnvStrength",   self._env_strength)
                    self._set_v3  (pbr_prog, "uLightDir",      lx, ly, lz)
                    self._set_f   (pbr_prog, "uLightIntensity",self._light_intensity)
                    if env_tex is not None:
                        try:
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D, int(env_tex))
                            self._set_i(pbr_prog, "uEnvMap", 0)
                        except: pass
                    try:
                        while glGetError() != GL_NO_ERROR: pass
                        glBindVertexArray(v_id)
                        glDrawElements(GL_TRIANGLES, self._sphere_idx_count, GL_UNSIGNED_INT, None)
                    finally:
                        glBindVertexArray(0)
                        pbr_prog.release()

            # ── Transform Gizmo (only when object is selected)
            if getattr(self, '_gizmo_prog', None) is not None and self._obj_selected:
                self._draw_gizmo(proj, view)

            # ── Orientation Gizmo (LAST — uses QPainter which breaks GL state)
            if getattr(self, '_gizmo_prog', None) is not None:
                self._draw_orientation_gizmo(proj, view)


        except Exception as e:
            print(f"[GL] paintGL crash: {e}")
            import traceback; traceback.print_exc()
        finally:
            glDisable(GL_SCISSOR_TEST)
            # Ensure viewport is reset to full widget size for any QPainter overlays
            W_phys, H_phys = int(self.width() * self.devicePixelRatio()), int(self.height() * self.devicePixelRatio())
            glViewport(0, 0, W_phys, H_phys)

    def _paint_fallback(self):
        """Show PyTorch3D preview image or idle message if no OpenGL."""
        pass   # handled by overlay in parent widget

    # ── GL resource builders ──────────────────────────────────────────────────

    def _build_gizmo_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog is not None:
            ok_v = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _GIZMO_VERT)
            if not ok_v:
                print(f"[GL] Gizmo vertex shader compile failed: {prog.log()}")
                return
            ok_f = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _GIZMO_FRAG)
            if not ok_f:
                print(f"[GL] Gizmo fragment shader compile failed: {prog.log()}")
                return
            if not prog.link():
                print(f"[GL] Gizmo shader link failed: {prog.log()}")
                return
            print("[GL] Gizmo shader built OK")
            self._gizmo_prog = prog

    def _upload_gizmo(self):
        """Create a simple VAO/VBO for the gizmo axis lines."""
        import numpy as np
        # 6 vertices: 2 per axis (start + end), each with pos(3) + color(3)
        # We'll update this buffer each frame in _draw_gizmo
        data = np.zeros(6 * 6, dtype=np.float32)
        self._gizmo_vao = int(glGenVertexArrays(1))
        self._gizmo_vbo = int(glGenBuffers(1))
        glBindVertexArray(int(self._gizmo_vao))
        glBindBuffer(GL_ARRAY_BUFFER, int(self._gizmo_vbo))
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        stride = 6 * 4  # 6 floats * 4 bytes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _draw_grid_floor(self, proj, view):
        """Draw a simple geometry-based grid on the Y=0 plane using the gizmo shader."""
        import numpy as np
        from OpenGL.GL import GL_DYNAMIC_DRAW

        if self._gizmo_prog is None:
            return

        # Lazy-init dedicated grid VAO/VBO (separate from gizmo)
        if not hasattr(self, '_grid_line_vao') or self._grid_line_vao is None:
            self._grid_line_vao = int(glGenVertexArrays(1))
            self._grid_line_vbo = int(glGenBuffers(1))
            glBindVertexArray(self._grid_line_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self._grid_line_vbo)
            stride = 6 * 4  # 6 floats * 4 bytes
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glBindVertexArray(0)
            self._grid_line_count = 0

        # Only regenerate vertices once (or if not yet generated)
        if self._grid_line_count == 0:
            verts = []
            grid_size = 20       # half-extent: grid spans -20..+20
            spacing = 1.0        # 1 unit between major lines
            gray = [0.35, 0.35, 0.35]   # minor grid color
            red  = [0.6, 0.15, 0.15]    # X axis color
            blue = [0.15, 0.15, 0.6]    # Z axis color

            for i in range(-grid_size, grid_size + 1):
                x = i * spacing
                # Line along Z (constant X)
                col = red if i == 0 else gray
                verts += [x, 0, -grid_size * spacing] + col
                verts += [x, 0,  grid_size * spacing] + col
                # Line along X (constant Z)
                col = blue if i == 0 else gray
                verts += [-grid_size * spacing, 0, x] + col
                verts += [ grid_size * spacing, 0, x] + col

            self._grid_line_count = len(verts) // 6
            data = np.array(verts, dtype=np.float32)

            glBindBuffer(GL_ARRAY_BUFFER, self._grid_line_vbo)
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)

        # Draw
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._gizmo_prog.bind()
        self._set_mat4(self._gizmo_prog, "uView", view)
        self._set_mat4(self._gizmo_prog, "uProj", proj)

        glBindVertexArray(self._grid_line_vao)
        glDrawArrays(GL_LINES, 0, self._grid_line_count)
        glBindVertexArray(0)
        self._gizmo_prog.release()
        glDisable(GL_BLEND)

    def _draw_gizmo(self, proj, view):
        """Draw mode-specific gizmo overlay."""
        if self._gizmo_mode == 'view':
            return  # No gizmo in view mode

        import numpy as np
        ox, oy, oz = self._obj_offset

        # Extract camera right & up from the view matrix (column-major)
        # view is a 16-element list in column-major order
        # Row 0 of view = right vector, Row 1 = up vector, Row 2 = -forward
        cam_right = [view[0], view[4], view[8]]
        cam_up    = [view[1], view[5], view[9]]

        # Collect line + triangle vertices from mode-specific generators
        line_verts, tri_verts = [], []
        if self._gizmo_mode == 'translate':
            line_verts, tri_verts = self._gizmo_translate_verts(ox, oy, oz, cam_right, cam_up)
        elif self._gizmo_mode == 'rotate':
            line_verts, tri_verts = self._gizmo_rotate_verts(ox, oy, oz)
        elif self._gizmo_mode == 'scale':
            line_verts, tri_verts = self._gizmo_scale_verts(ox, oy, oz, cam_right, cam_up)
        elif self._gizmo_mode == 'rect':
            line_verts, tri_verts = self._gizmo_rect_verts(ox, oy, oz, cam_right, cam_up)
        elif self._gizmo_mode == 'transform':
            line_verts, tri_verts = self._gizmo_transform_verts(ox, oy, oz, cam_right, cam_up)
        else:
            return

        n_lines = len(line_verts) // 6   # 6 floats per vertex (pos3 + col3)
        n_tris  = len(tri_verts)  // 6

        if n_lines == 0 and n_tris == 0:
            return

        all_verts = line_verts + tri_verts
        data = np.array(all_verts, dtype=np.float32)

        # Upload
        glBindBuffer(GL_ARRAY_BUFFER, int(self._gizmo_vbo))
        from OpenGL.GL import GL_DYNAMIC_DRAW
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        # Draw
        glDisable(GL_DEPTH_TEST)
        self._gizmo_prog.bind()
        self._set_mat4(self._gizmo_prog, "uView", view)
        self._set_mat4(self._gizmo_prog, "uProj", proj)

        while glGetError() != GL_NO_ERROR: pass
        glBindVertexArray(int(self._gizmo_vao))
        if n_lines > 0:
            glDrawArrays(GL_LINES, 0, n_lines)
        if n_tris > 0:
            glDrawArrays(GL_TRIANGLES, n_lines, n_tris)
        glBindVertexArray(0)
        self._gizmo_prog.release()
        glEnable(GL_DEPTH_TEST)

    def _draw_orientation_gizmo(self, proj, view):
        """Draw a Unity-style orientation axis helper in the top-right corner."""
        import numpy as np
        ratio = max(1.0, self.devicePixelRatio())
        # Use logical pixels for high-level layout, converted to physical for GL
        logical_size = 100
        logical_margin = 10
        W_log, H_log = self.width(), self.height()
        
        # Physical coordinates for GL
        W_phys, H_phys = int(W_log * ratio), int(H_log * ratio)
        size_phys = int(logical_size * ratio)
        margin_phys = int(logical_margin * ratio)
        
        # Set a small viewport and scissor in the corner (physical pixels)
        glEnable(GL_SCISSOR_TEST)
        # Note: glViewport/glScissor Y=0 is BOTTOM
        glScissor(W_phys - size_phys - margin_phys, H_phys - size_phys - margin_phys, size_phys, size_phys)
        glViewport(W_phys - size_phys - margin_phys, H_phys - size_phys - margin_phys, size_phys, size_phys)
        
        glClear(GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST) 
        glDepthFunc(GL_LEQUAL)
        # glLineWidth(2.0 * ratio)  # Often unsupported in Core Profile, causing GLError

        # We use a fixed perspective projection for the gizmo
        gizmo_proj = _mat4_perspective(45.0, 1.0, 0.1, 10.0)
        # Use camera rotation but move eye back along fixed Z
        # Row 0,1,2 of view matrix are the basis vectors
        gizmo_view = list(view)
        # Remove translation
        gizmo_view[12] = 0.0
        gizmo_view[13] = 0.0
        gizmo_view[14] = 0.0
        # Orbit camera: eye is at (0,0,3) in view space
        # Translate the WHOLE view matrix by -3 on Z
        gizmo_view = _mat4_mul(_mat4_translate(0, 0, -5), gizmo_view)

        # Axes: X=Red, Y=Green, Z=Blue
        # Use thicker lines by drawing quads instead of GL_LINES
        arm = 1.0
        line_verts = [
            0,0,0, 0.7,0.2,0.2,  arm,0,0, 0.9,0.2,0.2, # X
            0,0,0, 0.2,0.7,0.2,  0,arm,0, 0.2,0.9,0.2, # Y
            0,0,0, 0.2,0.2,0.7,  0,0,arm, 0.2,0.2,0.9, # Z
            # Back-facing axis stubs (dashed gray)
            0,0,0, 0.35,0.35,0.35,  -arm*0.5,0,0, 0.35,0.35,0.35,  # -X
            0,0,0, 0.35,0.35,0.35,  0,-arm*0.5,0, 0.35,0.35,0.35,  # -Y
            0,0,0, 0.35,0.35,0.35,  0,0,-arm*0.5, 0.35,0.35,0.35,  # -Z
        ]
        
        # Basis for billboard cones
        cam_r = [gizmo_view[0], gizmo_view[4], gizmo_view[8]]
        cam_u = [gizmo_view[1], gizmo_view[5], gizmo_view[9]]
        
        tri_verts = []
        # Front-facing colored cones (larger)
        tri_verts += self._billboard_tri(arm, 0, 0, [1,0,0], cam_r, cam_u, 0.22, [0.9,0.2,0.2])
        tri_verts += self._billboard_tri(0, arm, 0, [0,1,0], cam_r, cam_u, 0.22, [0.2,0.9,0.2])
        tri_verts += self._billboard_tri(0, 0, arm, [0,0,1], cam_r, cam_u, 0.22, [0.2,0.2,0.9])

        # Back-facing gray circles (smaller)
        gray = [0.45, 0.45, 0.45]
        tri_verts += self._billboard_tri(-arm*0.5, 0, 0, [-1,0,0], cam_r, cam_u, 0.1, gray)
        tri_verts += self._billboard_tri(0, -arm*0.5, 0, [0,-1,0], cam_r, cam_u, 0.1, gray)
        tri_verts += self._billboard_tri(0, 0, -arm*0.5, [0,0,-1], cam_r, cam_u, 0.1, gray)

        # Center sphere (small white/gray circle)
        cs = 0.12
        cc = [0.5, 0.5, 0.5]
        # Build a camera-facing polygon (8-segment circle approximation)
        import math as _m
        n_seg = 8
        for i in range(n_seg):
            a0 = 2.0 * _m.pi * i / n_seg
            a1 = 2.0 * _m.pi * (i + 1) / n_seg
            p0 = [cs * (_m.cos(a0) * cam_r[k] + _m.sin(a0) * cam_u[k]) for k in range(3)]
            p1 = [cs * (_m.cos(a1) * cam_r[k] + _m.sin(a1) * cam_u[k]) for k in range(3)]
            tri_verts += [0,0,0, *cc,  p0[0],p0[1],p0[2], *cc,  p1[0],p1[1],p1[2], *cc]

        all_verts = line_verts + tri_verts
        data = np.array(all_verts, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, int(self._gizmo_vbo))
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_DYNAMIC_DRAW)

        self._gizmo_prog.bind()
        self._set_mat4(self._gizmo_prog, "uView", gizmo_view)
        self._set_mat4(self._gizmo_prog, "uProj", gizmo_proj)
        
        glBindVertexArray(int(self._gizmo_vao))
        glDrawArrays(GL_LINES, 0, len(line_verts)//6)
        glDrawArrays(GL_TRIANGLES, len(line_verts)//6, len(tri_verts)//6)
        
        glBindVertexArray(0)
        self._gizmo_prog.release()
        
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, W_phys, H_phys) # Reset viewport

        # ── Axis Labels (Logical Coords for QPainter) ─────────────────────
        rect = [W_log - logical_size - logical_margin, 0, logical_size, logical_size] 
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        def get_screen(p):
            clip = _mat4_vec4_mul(_mat4_mul(gizmo_proj, gizmo_view), [p[0], p[1], p[2], 1.0])
            if abs(clip[3]) < 0.0001: return None
            ndc = [clip[0]/clip[3], clip[1]/clip[3]]
            return ( (ndc[0] * 0.5 + 0.5) * logical_size + rect[0], 
                     ( (1.0 - (ndc[1] * 0.5 + 0.5)) * logical_size ) + rect[1] )

        painter.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        labels = [([1.35, 0, 0], "X", QColor(255, 80, 80)), 
                  ([0, 1.35, 0], "Y", QColor(80, 255, 80)), 
                  ([0, 0, 1.35], "Z", QColor(80, 80, 255))]
        
        for pos_w, txt, col in labels:
            sc = get_screen(pos_w)
            if sc:
                painter.setPen(col)
                painter.drawText(int(sc[0]-5), int(sc[1]+5), txt)
        
        # View label at bottom of gizmo (Unity-style "< Persp")
        painter.setPen(QColor(180, 180, 180))
        painter.setFont(QFont("Inter", 8))
        view_name = self._get_view_name()
        painter.drawText(rect[0], rect[1] + logical_size - 5, logical_size, 20, Qt.AlignmentFlag.AlignCenter, view_name)
        painter.end()

    def _get_view_name(self):
        """Return human name for current camera orientation."""
        a, e = self._azim % 360, self._elev
        if abs(e - 90) < 5: return "Top"      # elev ≈ +90 → camera above → Top
        if abs(e + 90) < 5: return "Bottom"   # elev ≈ -90 → camera below → Bottom
        if abs(e) < 5:
            if abs(a) < 5:    return "Front"
            if abs(a-180) < 5: return "Back"
            if abs(a-90) < 5:  return "Left"
            if abs(a+90) < 5:  return "Right"
        return "Perspective"

    def _orientation_gizmo_hit_test(self, pos):
        """Check if the mouse click is within the orientation gizmo and handle snaps."""
        W_log, H_log = self.width(), self.height()
        size_log = 100
        margin_log = 10
        # Gizmo is at top-right (Qt Y=0 is top)
        rect_x = W_log - size_log - margin_log
        rect_y = 0
        
        if not (rect_x <= pos.x() <= rect_x + size_log and 
                rect_y <= pos.y() <= rect_y + size_log + 20): # +20 for bottom label
            return False

        # If inside the gizmo area, determine which axis was clicked
        # We can do this by projecting the 3 axes to screen space within the tiny viewport
        import numpy as np
        gizmo_proj = _mat4_perspective(45.0, 1.0, 0.1, 10.0)
        gizmo_view = list(self._camera_view())
        gizmo_view[12] = 0.0
        gizmo_view[13] = 0.0
        gizmo_view[14] = 0.0
        gizmo_view = _mat4_mul(_mat4_translate(0, 0, -5), gizmo_view)
        
        def project(p):
            # p is [x,y,z]
            clip = _mat4_vec4_mul(_mat4_mul(gizmo_proj, gizmo_view), [p[0], p[1], p[2], 1.0])
            if abs(clip[3]) < 0.0001: return None
            ndc = [clip[0]/clip[3], clip[1]/clip[3]]
            # Map NDC (-1..1) to local gizmo square (0..size) using logic matching _draw
            return [ (ndc[0] * 0.5 + 0.5) * size_log + rect_x, 
                     ((1.0 - (ndc[1] * 0.5 + 0.5)) * size_log) + rect_y ]

        # Test +X, -X, +Y, -Y, +Z, -Z
        axes = [
            ([1, 0, 0], "right"),  ([-1, 0, 0], "left"),
            ([0, 1, 0], "top"),    ([0, -1, 0], "bottom"),
            ([0, 0, 1], "front"),  ([0, 0, -1], "back")
        ]
        
        best_axis = None
        min_dist = 15.0 # pixels threshold
        
        for world_pos, name in axes:
            screen = project(world_pos)
            if screen:
                d = math.hypot(pos.x() - screen[0], pos.y() - screen[1])
                if d < min_dist:
                    min_dist = d
                    best_axis = name
        
        if best_axis:
            self._snap_to_axis(best_axis)
            return True
        
        return True # Intercepted click even if no axis hit

    def _snap_to_axis(self, axis_name):
        """Snap camera to a cardinal direction."""
        if axis_name == "top":    self._azim, self._elev =   0,  89.9
        elif axis_name == "bottom": self._azim, self._elev = 0, -89.9
        elif axis_name == "front":  self._azim, self._elev = 0, 0
        elif axis_name == "back":   self._azim, self._elev = 180, 0
        elif axis_name == "right":  self._azim, self._elev = -90, 0
        elif axis_name == "left":   self._azim, self._elev = 90, 0
        
        # Reset velocity
        self._v_azim = 0
        self._v_elev = 0
        self.update()

    def _camera_view(self):
        """Return the current camera view matrix."""
        eye = self._camera_pos()
        up = self._camera_up()
        return _lookat(eye, self._target, up)

    # ── Axis colour helper ─────────────────────────────────────────────────
    def _axis_col(self, axis, base):
        if self._gizmo_active_axis == axis:
            return [1.0, 1.0, 0.3]
        if self._gizmo_hover_axis == axis:
            return [1.0, 1.0, 1.0]
        return list(base)

    # ── Billboard triangle helper ──────────────────────────────────────────
    @staticmethod
    def _billboard_tri(cx, cy, cz, axis_dir, cam_r, cam_u, size, col):
        """Create a camera-facing triangle pointing along axis_dir at (cx,cy,cz)."""
        # Compute perpendicular to axis in camera plane
        # Cross axis_dir with cam_forward to get a screen-space perp
        import math
        ax, ay, az = axis_dir
        # Use cam_right and cam_up to build two perp offsets
        # Project cam_r and cam_u onto the plane perpendicular to axis
        # Simple approach: just use cam_up and cam_right scaled
        p1 = [cx - cam_r[i]*size + ax*size*1.5 for i in range(3)]
        p2 = [cx + cam_r[i]*size + ax*size*1.5 for i in range(3)]
        p3 = [cx + ax*size*3 for i in range(3)]
        # Better: compute the triangle tip and two base corners
        tip = [cx + ax*size*2.5, cy + ay*size*2.5, cz + az*size*2.5]
        b1 = [cx - cam_r[0]*size - cam_u[0]*size*0.3,
              cy - cam_r[1]*size - cam_u[1]*size*0.3,
              cz - cam_r[2]*size - cam_u[2]*size*0.3]
        b2 = [cx + cam_r[0]*size + cam_u[0]*size*0.3,
              cy + cam_r[1]*size + cam_u[1]*size*0.3,
              cz + cam_r[2]*size + cam_u[2]*size*0.3]
        return [*b1, *col, *tip, *col, *b2, *col]

    # ── Move Gizmo: arrows with camera-facing cone tips ────────────────────
    def _gizmo_translate_verts(self, ox, oy, oz, cam_r, cam_u):
        arm, tip = 0.8, 0.08
        rx = self._axis_col('x', [1.0, 0.15, 0.15])
        gx = self._axis_col('y', [0.15, 1.0, 0.15])
        bx = self._axis_col('z', [0.3, 0.3, 1.0])

        lines = [
            ox,oy,oz, *rx,  ox+arm,oy,oz, *rx,
            ox,oy,oz, *gx,  ox,oy+arm,oz, *gx,
            ox,oy,oz, *bx,  ox,oy,oz+arm, *bx,
        ]
        # Camera-facing arrowhead triangles
        tris = []
        # X arrow
        xT = ox + arm
        tris += [
            xT, oy-cam_u[1]*tip-cam_r[1]*tip, oz-cam_u[2]*tip-cam_r[2]*tip, *rx,
            xT+tip*2, oy, oz, *rx,
            xT, oy+cam_u[1]*tip+cam_r[1]*tip, oz+cam_u[2]*tip+cam_r[2]*tip, *rx,
        ]
        # Y arrow
        yT = oy + arm
        tris += [
            ox-cam_r[0]*tip, yT, oz-cam_r[2]*tip, *gx,
            ox, yT+tip*2, oz, *gx,
            ox+cam_r[0]*tip, yT, oz+cam_r[2]*tip, *gx,
        ]
        # Z arrow
        zT = oz + arm
        tris += [
            ox-cam_r[0]*tip, oy-cam_u[1]*tip, zT, *bx,
            ox, oy, zT+tip*2, *bx,
            ox+cam_r[0]*tip, oy+cam_u[1]*tip, zT, *bx,
        ]

        # Center box (camera-facing quad for free movement)
        bs = 0.07  # box half-size
        wc = [0.9, 0.9, 0.3]  # yellow/white color
        # Two triangles forming a small camera-facing square
        cr, cu = cam_r, cam_u
        # Corner offsets from center (ox, oy, oz)
        org = [ox, oy, oz]
        c00 = [org[i] - cr[i]*bs - cu[i]*bs for i in range(3)]
        c10 = [org[i] + cr[i]*bs - cu[i]*bs for i in range(3)]
        c11 = [org[i] + cr[i]*bs + cu[i]*bs for i in range(3)]
        c01 = [org[i] - cr[i]*bs + cu[i]*bs for i in range(3)]
        tris += [
            *c00, *wc,  *c10, *wc,  *c11, *wc,
            *c00, *wc,  *c11, *wc,  *c01, *wc,
        ]

        return lines, tris

    # ── Rotate Gizmo: 3 axis circles ──────────────────────────────────────
    def _gizmo_rotate_verts(self, ox, oy, oz):
        import math
        radius = 0.6
        segments = 64
        rx = self._axis_col('x', [1.0, 0.15, 0.15])
        gx = self._axis_col('y', [0.15, 1.0, 0.15])
        bx = self._axis_col('z', [0.3, 0.3, 1.0])
        wh = [0.7, 0.7, 0.7]

        lines = []
        for i in range(segments):
            a0 = 2.0 * math.pi * i / segments
            a1 = 2.0 * math.pi * (i + 1) / segments
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            # Red circle in YZ plane
            lines += [ox, oy+c0*radius, oz+s0*radius, *rx,
                       ox, oy+c1*radius, oz+s1*radius, *rx]
            # Green circle in XZ plane
            lines += [ox+c0*radius, oy, oz+s0*radius, *gx,
                       ox+c1*radius, oy, oz+s1*radius, *gx]
            # Blue circle in XY plane
            lines += [ox+c0*radius, oy+s0*radius, oz, *bx,
                       ox+c1*radius, oy+s1*radius, oz, *bx]
            # White outer circle
            r2 = radius * 1.15
            lines += [ox+c0*r2, oy+s0*r2, oz, *wh,
                       ox+c1*r2, oy+s1*r2, oz, *wh]
        return lines, []

    # ── Scale Gizmo: axis lines with camera-facing square tips ────────────
    def _gizmo_scale_verts(self, ox, oy, oz, cam_r, cam_u):
        arm, sq = 0.7, 0.05
        rx = self._axis_col('x', [1.0, 0.15, 0.15])
        gx = self._axis_col('y', [0.15, 1.0, 0.15])
        bx = self._axis_col('z', [0.3, 0.3, 1.0])

        lines = [
            ox,oy,oz, *rx,  ox+arm,oy,oz, *rx,
            ox,oy,oz, *gx,  ox,oy+arm,oz, *gx,
            ox,oy,oz, *bx,  ox,oy,oz+arm, *bx,
        ]

        def _screen_quad(cx, cy, cz, col, s):
            """Camera-facing filled square at (cx,cy,cz)."""
            # Fix: use [cx,cy,cz][j] for each component
            org = [cx, cy, cz]
            c00 = [org[j] - cam_r[j]*s - cam_u[j]*s for j in range(3)]
            c10 = [org[j] + cam_r[j]*s - cam_u[j]*s for j in range(3)]
            c11 = [org[j] + cam_r[j]*s + cam_u[j]*s for j in range(3)]
            c01 = [org[j] - cam_r[j]*s + cam_u[j]*s for j in range(3)]
            return [
                *c00, *col, *c10, *col, *c11, *col,
                *c00, *col, *c11, *col, *c01, *col,
            ]

        tris = []
        tris += _screen_quad(ox+arm, oy, oz, rx, sq)
        tris += _screen_quad(ox, oy+arm, oz, gx, sq)
        tris += _screen_quad(ox, oy, oz+arm, bx, sq)
        # Center cube
        tris += _screen_quad(ox, oy, oz, [0.9, 0.9, 0.9], sq * 1.2)
        return lines, tris

    # ── Rect Gizmo: camera-facing orange bounding rectangle ───────────────
    def _gizmo_rect_verts(self, ox, oy, oz, cam_r, cam_u):
        hw = 0.4  # half-size
        oc = [1.0, 0.6, 0.1]  # orange
        dot = 0.03

        # Rectangle corners in camera-facing plane
        c00 = [ox - cam_r[j]*hw - cam_u[j]*hw for j in range(3)]
        c10 = [ox + cam_r[j]*hw - cam_u[j]*hw for j in range(3)]
        c11 = [ox + cam_r[j]*hw + cam_u[j]*hw for j in range(3)]
        c01 = [ox - cam_r[j]*hw + cam_u[j]*hw for j in range(3)]

        lines = [
            *c00, *oc, *c10, *oc,
            *c10, *oc, *c11, *oc,
            *c11, *oc, *c01, *oc,
            *c01, *oc, *c00, *oc,
        ]

        # Mid-edge points
        m_bot = [(c00[j]+c10[j])/2 for j in range(3)]
        m_right = [(c10[j]+c11[j])/2 for j in range(3)]
        m_top = [(c11[j]+c01[j])/2 for j in range(3)]
        m_left = [(c01[j]+c00[j])/2 for j in range(3)]

        # Corner + edge handle dots
        dc = [0.3, 0.5, 1.0]
        tris = []
        for pt in [c00, c10, c11, c01, m_bot, m_right, m_top, m_left]:
            cx, cy, cz = pt
            d00 = [cx - cam_r[j]*dot - cam_u[j]*dot for j in range(3)]
            d10 = [cx + cam_r[j]*dot - cam_u[j]*dot for j in range(3)]
            d11 = [cx + cam_r[j]*dot + cam_u[j]*dot for j in range(3)]
            d01 = [cx - cam_r[j]*dot + cam_u[j]*dot for j in range(3)]
            tris += [
                *d00, *dc, *d10, *dc, *d11, *dc,
                *d00, *dc, *d11, *dc, *d01, *dc,
            ]
        return lines, tris

    # ── Transform Gizmo: combined arrows + partial arcs ───────────────────
    def _gizmo_transform_verts(self, ox, oy, oz, cam_r, cam_u):
        import math
        arm, tip = 0.6, 0.07
        rx = self._axis_col('x', [1.0, 0.15, 0.15])
        gx = self._axis_col('y', [0.15, 1.0, 0.15])
        bx = self._axis_col('z', [0.3, 0.3, 1.0])

        lines = [
            ox,oy,oz, *rx,  ox+arm,oy,oz, *rx,
            ox,oy,oz, *gx,  ox,oy+arm,oz, *gx,
            ox,oy,oz, *bx,  ox,oy,oz+arm, *bx,
        ]
        # Camera-facing arrow tips
        xT, yT, zT = ox+arm, oy+arm, oz+arm
        tris = [
            xT, oy-cam_u[1]*tip, oz-cam_u[2]*tip, *rx,
            xT+tip*2.5, oy, oz, *rx,
            xT, oy+cam_u[1]*tip, oz+cam_u[2]*tip, *rx,

            ox-cam_r[0]*tip, yT, oz-cam_r[2]*tip, *gx,
            ox, yT+tip*2.5, oz, *gx,
            ox+cam_r[0]*tip, yT, oz+cam_r[2]*tip, *gx,

            ox-cam_r[0]*tip, oy-cam_u[1]*tip, zT, *bx,
            ox, oy, zT+tip*2.5, *bx,
            ox+cam_r[0]*tip, oy+cam_u[1]*tip, zT, *bx,
        ]
        # Partial arcs
        arc_r = 0.35
        arc_segs = 16
        for i in range(arc_segs):
            a0 = (math.pi / 2.0) * i / arc_segs
            a1 = (math.pi / 2.0) * (i + 1) / arc_segs
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            lines += [ox+c0*arc_r,oy+s0*arc_r,oz, *bx,
                       ox+c1*arc_r,oy+s1*arc_r,oz, *bx]
            lines += [ox+c0*arc_r,oy,oz+s0*arc_r, *gx,
                       ox+c1*arc_r,oy,oz+s1*arc_r, *gx]
            lines += [ox,oy+c0*arc_r,oz+s0*arc_r, *rx,
                       ox,oy+c1*arc_r,oz+s1*arc_r, *rx]
        return lines, tris


    def _build_pbr_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog is not None:
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _VERT_SRC)
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _FRAG_SRC)
            if not prog.link():
                print(f"[GL] PBR Shader Link Error: {prog.log()}")
            self._prog = prog

    def _build_sky_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog is not None:
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _SKY_VERT)
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _SKY_FRAG)
            if not prog.link():
                print(f"[GL] Sky Shader Link Error: {prog.log()}")
            self._sky_prog = prog

    def _upload_sphere(self):
        if not HAS_OPENGL:
            return
        import numpy as np
        verts, idx = _make_sphere()
        self._sphere_idx_count = len(idx)

        # PyOpenGL style: gen functions return IDs directly
        # Force to int to avoid numpy type leakage
        self._sphere_vao = int(glGenVertexArrays(1))
        glBindVertexArray(int(self._sphere_vao))

        self._sphere_vbo = int(glGenBuffers(1))
        self._sphere_ebo = int(glGenBuffers(1))

        glBindBuffer(GL_ARRAY_BUFFER, int(self._sphere_vbo))
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)

        stride = 8 * 4   # 8 floats × 4 bytes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(self._sphere_ebo))
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        glBindVertexArray(0)

    def _upload_skybox(self):
        if not HAS_OPENGL:
            return
        verts, idx = _make_cube_verts()
        self._sky_idx_count = len(idx)

        self._sky_vao = int(glGenVertexArrays(1))
        glBindVertexArray(int(self._sky_vao))

        self._sky_vbo = int(glGenBuffers(1))
        self._sky_ebo = int(glGenBuffers(1))

        glBindBuffer(GL_ARRAY_BUFFER, int(self._sky_vbo))
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(self._sky_ebo))
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
        glBindVertexArray(0)

    def _upload_grid(self):
        """Build a simple 3D grid [pos(3)] as a VAO."""
        if not HAS_OPENGL: return
        pts = []
        size, steps = 10, 20
        for i in range(-steps, steps + 1):
            v = (i / steps) * size
            pts += [v, 0, -size,  v, 0, size]
            pts += [-size, 0, v,  size, 0, v]
        # X and Z axes colored if we wanted more shaders, but keep it simple
        import numpy as np
        data = np.array(pts, dtype=np.float32)
        self._grid_count = len(data) // 3
        self._grid_vao = int(glGenVertexArrays(1))
        glBindVertexArray(int(self._grid_vao))
        self._grid_vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, int(self._grid_vbo))
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _upload_custom_mesh(self, verts, faces, normals=None):
        """Upload custom mesh data into interleaved VBO."""
        if not HAS_OPENGL or not self._gl_ready:
            return
        import numpy as np

        # Interleaved [pos(3), norm(3), uv(2)] — default to zero for normals/uvs if missing
        num_v = len(verts)
        data = np.zeros((num_v, 8), dtype=np.float32)
        data[:, 0:3] = verts
        if normals is not None and len(normals) == num_v:
            data[:, 3:6] = normals
        else:
            # Simple fallback normal (up)
            data[:, 4] = 1.0

        self._sphere_idx_count = len(faces.flatten())
        idx_data = faces.astype(np.uint32).flatten()

        from OpenGL.GL import glIsVertexArray
        v_id = int(self._sphere_vao) if self._sphere_vao is not None else -1
        if v_id >= 0 and glIsVertexArray(v_id):
            glBindVertexArray(v_id)
            
            glBindBuffer(GL_ARRAY_BUFFER, int(self._sphere_vbo))
            glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(self._sphere_ebo))
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL_STATIC_DRAW)
            
            glBindVertexArray(0)

    def _load_env_texture(self):
        """Load equirectangular HDRI .hdr / fallback to placeholder."""
        if not HAS_OPENGL or not self._gl_ready:
            return
        try:
            import numpy as np
            arr = None

            if self._hdri_path and os.path.exists(self._hdri_path):
                ext = os.path.splitext(self._hdri_path)[1].lower()
                if ext in (".hdr", ".exr"):
                    try:
                        import cv2
                        arr = cv2.imread(self._hdri_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if arr is not None:
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        print(f"[GL] HDR load failed: {e}")
                elif ext in (".png", ".jpg", ".jpeg"):
                    qi = QImage(self._hdri_path).convertToFormat(QImage.Format.Format_RGB888)
                    arr = (
                        np.frombuffer(qi.bits(), dtype=np.uint8)
                          .reshape(qi.height(), qi.width(), 3)
                          .astype(np.float32) / 255.0
                    )

            if arr is None:
                # Procedural gradient sky (blue top → warm horizon)
                H, W = 256, 512
                arr = np.zeros((H, W, 3), dtype=np.float32)
                for row in range(H):
                    t = row / H
                    arr[row, :, 0] = 0.10 + 0.55 * t
                    arr[row, :, 1] = 0.18 + 0.35 * t
                    arr[row, :, 2] = 0.60 - 0.35 * t

            # Ensure 3-channel float32 contiguous array
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = np.ascontiguousarray(arr.astype(np.float32))
            H, W = arr.shape[:2]

            # Delete old texture if re-loading
            if self._env_tex is not None:
                from OpenGL.GL import glDeleteTextures
                try: glDeleteTextures(1, [int(self._env_tex)])
                except: pass

            self._env_tex = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, int(self._env_tex))
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGB16F, W, H, 0,
                GL_RGB, GL_FLOAT, arr
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glGenerateMipmap(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, 0)

        except Exception as e:
            print(f"[GL] env texture error: {e}")
            import traceback; traceback.print_exc()
            self._env_tex = None

    # ── Uniform helpers ───────────────────────────────────────────────────────
    # We use raw glUniform* calls instead of Qt's setUniformValue because the 
    # latter can sometimes leak error states (INVALID_OPERATION) in Core Profile.

    def _set_mat4(self, prog, name, m):
        loc = prog.uniformLocation(name)
        if loc >= 0:
            c_array_type = ctypes.c_float * 16
            c_array = c_array_type(*m)
            glUniformMatrix4fv(loc, 1, GL_FALSE, c_array)

    def _set_mat3(self, prog, name, m):
        loc = prog.uniformLocation(name)
        if loc >= 0:
            c_array_type = ctypes.c_float * 9
            c_array = c_array_type(*m)
            glUniformMatrix3fv(loc, 1, GL_FALSE, c_array)

    def _set_f(self, prog, name, val):
        loc = prog.uniformLocation(name)
        if loc >= 0:
            glUniform1f(loc, float(val))

    def _set_i(self, prog, name, val):
        loc = prog.uniformLocation(name)
        if loc >= 0:
            glUniform1i(loc, int(val))

    def _set_v3(self, prog, name, x, y, z):
        loc = prog.uniformLocation(name)
        if loc >= 0:
            glUniform3f(loc, float(x), float(y), float(z))


# ─────────────────────────────────────────────────────────────────────────────
# Left-side scene tool bar (Unity-style)
# ─────────────────────────────────────────────────────────────────────────────
class SceneToolBar(QWidget):
    """Vertical toolbar with View/Move/Rotate/Scale/Rect/Transform buttons."""
    tool_changed = Signal(str)   # emits: 'view','translate','rotate','scale','rect','transform'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(34)
        self.setStyleSheet("""
            QWidget { background: #1a1f2e; border-right: 1px solid #252d3d; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 6, 3, 6)
        layout.setSpacing(2)

        # Icon directory
        _icon_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "sources", "icons")

        def _tool_icon(filename, active=False):
            """Load a PNG icon, tint it gray (normal) or cyan (active)."""
            path = os.path.join(_icon_dir, filename)
            raw = QPixmap(path)
            if raw.isNull():
                return QIcon()
            color = QColor(79, 195, 247) if active else QColor(139, 148, 158)
            tinted = QPixmap(raw.size())
            tinted.fill(Qt.GlobalColor.transparent)
            p = QPainter(tinted)
            p.drawPixmap(0, 0, raw)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(tinted.rect(), color)
            p.end()
            return QIcon(tinted)

        # Tool definitions: (icon_file, tooltip, mode_name)
        tools = [
            ("View Tool.png",   "View Tool (Q)",    "view"),
            ("Move Tool.png",   "Move Tool (W)",    "translate"),
            ("Rotate Tool.png", "Rotate Tool (E)",  "rotate"),
            ("Scale Tool.png",  "Scale Tool (R)",   "scale"),
        ]

        self._btns = {}
        self._icons_normal = {}
        self._icons_active = {}

        for icon_file, tip, mode in tools:
            btn = QPushButton()
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(28, 28)
            btn.setIconSize(QSize(18, 18))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent; border: 1px solid transparent;
                    border-radius: 4px; padding: 0px;
                }
                QPushButton:hover {
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(255,255,255,0.1);
                }
                QPushButton:checked {
                    background: #1a2744;
                    border: 1px solid #4fc3f7;
                }
            """)
            icon_n = _tool_icon(icon_file, active=False)
            icon_a = _tool_icon(icon_file, active=True)
            self._icons_normal[mode] = icon_n
            self._icons_active[mode] = icon_a
            btn.setIcon(icon_n)
            btn.clicked.connect(lambda checked, m=mode: self._select(m))
            layout.addWidget(btn)
            self._btns[mode] = btn

        layout.addStretch()
        # Default: move tool
        self._select("translate")

    def _select(self, mode):
        for m, btn in self._btns.items():
            is_active = (m == mode)
            btn.setChecked(is_active)
            # Swap icon tint: cyan when active, gray when not
            if m in self._icons_active:
                btn.setIcon(self._icons_active[m] if is_active else self._icons_normal[m])
        self.tool_changed.emit(mode)

    def set_mode(self, mode):
        """Programmatically select a tool (called by keyboard shortcuts)."""
        if mode in self._btns:
            self._select(mode)


# ─────────────────────────────────────────────────────────────────────────────
# Bottom camera control bar
# ─────────────────────────────────────────────────────────────────────────────
class CameraControlBar(QWidget):
    def __init__(self, gl_widget: GL3DPreview, parent=None):
        super().__init__(parent)
        self._gl = gl_widget
        self.setFixedHeight(32)
        self.setStyleSheet(
            "background-color: #161b22; border-top: 1px solid #21262d;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(4)

        def _btn(label, tip, cb):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setFixedHeight(22)
            b.setFixedWidth(80)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    background: #21262d; border: 1px solid #30363d;
                    border-radius: 4px; color: #8b949e;
                    font-size: 10px; font-weight: 600;
                }
                QPushButton:hover {
                    background: #30363d; color: #cdd6f4;
                    border-color: #4fc3f7;
                }
                QPushButton:checked {
                    background: #1a2744; color: #4fc3f7;
                    border-color: #4fc3f7;
                }
            """)
            b.clicked.connect(cb)
            return b

        layout.addStretch()
        layout.addWidget(_btn("Zoom −",    "Zoom out",       lambda: self._zoom(-1)))
        layout.addWidget(_btn("Zoom +",    "Zoom in",        lambda: self._zoom(+1)))
        layout.addWidget(_btn("⟳  Reset",  "Reset camera",   self._gl.reset_camera))

        self._wf_btn = QPushButton("Wireframe")
        self._wf_btn.setCheckable(True)
        self._wf_btn.setFixedHeight(22)
        self._wf_btn.setFixedWidth(80)
        self._wf_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._wf_btn.setStyleSheet("""
            QPushButton {
                background: #21262d; border: 1px solid #30363d;
                border-radius: 4px; color: #8b949e;
                font-size: 10px; font-weight: 600;
            }
            QPushButton:hover  { background:#30363d; color:#cdd6f4; border-color:#4fc3f7; }
            QPushButton:checked{ background:#1a2744;  color:#4fc3f7; border-color:#4fc3f7; }
        """)
        self._wf_btn.toggled.connect(self._gl.set_wireframe)
        layout.addWidget(self._wf_btn)
        layout.addStretch()

    def _zoom(self, direction):
        self._gl._dist = max(0.5, min(20.0, self._gl._dist - direction * 0.4))
        self._gl.update()


# ─────────────────────────────────────────────────────────────────────────────
# Scene Inspector Panel
# ─────────────────────────────────────────────────────────────────────────────
class SceneInspectorPanel(QWidget):
    """Unity-style scene inspector: Object / Material / Light / Environment."""

    def __init__(self, gl: "GL3DPreview", parent=None):
        super().__init__(parent)
        self._gl = gl
        self.setFixedWidth(220)
        self.setStyleSheet("""
            SceneInspectorPanel {
                background: #161b22;
                border-left: 1px solid #21262d;
            }
            QLabel#section {
                color: #8b949e; font-size: 10px; font-weight: 700;
                letter-spacing: 1px; padding: 4px 8px 2px 8px;
                background: #0d1117;
                border-bottom: 1px solid #21262d;
            }
            QLabel { color: #cdd6f4; font-size: 11px; }
            QDoubleSpinBox, QSpinBox {
                background: #0d1117; color: #cdd6f4;
                border: 1px solid #30363d; border-radius: 3px;
                font-size: 11px; padding: 1px 4px;
                min-width: 60px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #4fc3f7;
            }
            QPushButton#color_btn {
                border: 2px solid #30363d; border-radius: 3px;
                min-height: 18px; max-height: 18px;
            }
            QPushButton#color_btn:hover { border-color: #4fc3f7; }
            QPushButton#reset_btn {
                background: #21262d; color: #8b949e;
                border: 1px solid #30363d; border-radius: 3px;
                font-size: 10px; padding: 2px 8px;
            }
            QPushButton#reset_btn:hover { color: #cdd6f4; border-color: #4fc3f7; }
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        container.setStyleSheet("background: #161b22;")
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # ── helper to build a row ────────────────────────────────────────────
        def _section(title):
            lbl = QLabel(title)
            lbl.setObjectName("section")
            vl.addWidget(lbl)

        def _row(label, widget):
            row = QWidget()
            row.setStyleSheet("background: transparent;")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 3, 8, 3)
            rl.addWidget(QLabel(label))
            rl.addStretch()
            rl.addWidget(widget)
            vl.addWidget(row)
            return widget

        def _dspin(val, lo, hi, step=0.01, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setValue(val)
            s.setFixedWidth(72)
            return s

        # ── Object Transform ────────────────────────────────────────────────
        _section("OBJECT TRANSFORM")
        # GL world scale: mesh=1.5 units, camera dist=3 → offsets in that range
        self._pos_x = _row("Pos X",  _dspin(0.0, -5.0, 5.0, 0.05, 2))
        self._pos_y = _row("Pos Y",  _dspin(0.0, -5.0, 5.0, 0.05, 2))
        self._scale = _row("Scale",  _dspin(1.0,  0.1,  10.0, 0.05, 2))

        reset_xfm = QPushButton("Reset")
        reset_xfm.setObjectName("reset_btn")
        reset_xfm.clicked.connect(self._reset_transform)
        row_r = QWidget()
        row_r.setStyleSheet("background:transparent;")
        rl = QHBoxLayout(row_r)
        rl.setContentsMargins(8,2,8,4)
        rl.addStretch()
        rl.addWidget(reset_xfm)
        vl.addWidget(row_r)

        # ── Material ────────────────────────────────────────────────────────
        _section("MATERIAL")
        self._albedo_btn = QPushButton()
        self._albedo_btn.setObjectName("color_btn")
        self._albedo_btn.setFixedWidth(44)
        self._albedo_color = QColor(230, 89, 38)  # default rusty orange
        self._update_albedo_btn()
        self._albedo_btn.clicked.connect(self._pick_albedo)
        _row("Albedo", self._albedo_btn)
        self._metallic  = _row("Metallic",  _dspin(0.1, 0.0, 1.0, 0.05))
        self._roughness = _row("Roughness", _dspin(0.45, 0.0, 1.0, 0.05))

        # ── Lighting ────────────────────────────────────────────────────────
        _section("LIGHTING")
        self._l_azim  = _row("Azimuth",   _dspin(45.0, 0.0, 360.0, 5.0, 1))
        self._l_elev  = _row("Elevation", _dspin(60.0, -90, 90.0,  5.0, 1))
        self._l_inten = _row("Intensity", _dspin(2.8,  0.0, 10.0,  0.1, 2))

        # ── Environment ─────────────────────────────────────────────────────
        _section("ENVIRONMENT")
        self._env_str = _row("HDRI Strength", _dspin(1.0, 0.0, 10.0, 0.1, 2))
        self._env_rot = _row("HDRI Rotation", _dspin(0.0, 0.0, 360.0, 5.0, 1))

        vl.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QLabel("  ⚙  Scene Inspector")
        hdr.setFixedHeight(28)
        hdr.setStyleSheet(
            "background:#0d1117; color:#8b949e; font-size:11px; font-weight:700;"
            "border-bottom:1px solid #21262d;"
        )
        outer.addWidget(hdr)
        outer.addWidget(scroll)

        # Connect all widgets
        self._pos_x.valueChanged.connect(self._apply_transform)
        self._pos_y.valueChanged.connect(self._apply_transform)
        self._scale.valueChanged.connect(self._apply_transform)
        self._metallic.valueChanged.connect(self._apply_material)
        self._roughness.valueChanged.connect(self._apply_material)
        self._l_azim.valueChanged.connect(self._apply_light)
        self._l_elev.valueChanged.connect(self._apply_light)
        self._l_inten.valueChanged.connect(self._apply_light)
        self._env_str.valueChanged.connect(self._apply_env)
        self._env_rot.valueChanged.connect(self._apply_env)

        # ── Sync Pos X/Y spinboxes when user drags the object in the viewport ──────
        self._gl.add_obj_offset_listener(self._on_drag_offset)

    def _on_drag_offset(self, x: float, y: float):
        """Called by GL when object is dragged — sync spinboxes silently."""
        self._pos_x.blockSignals(True)
        self._pos_y.blockSignals(True)
        self._pos_x.setValue(x)
        self._pos_y.setValue(y)
        self._pos_x.blockSignals(False)
        self._pos_y.blockSignals(False)

    def _update_albedo_btn(self):
        c = self._albedo_color
        self._albedo_btn.setStyleSheet(
            f"QPushButton#color_btn {{ background: rgb({c.red()},{c.green()},{c.blue()}); "
            f"border: 2px solid #30363d; border-radius:3px; "
            f"min-height:18px; max-height:18px; }}"
            f"QPushButton#color_btn:hover {{ border-color: #4fc3f7; }}"
        )

    def _pick_albedo(self):
        c = QColorDialog.getColor(self._albedo_color, self, "Pick Albedo")
        if c.isValid():
            self._clear_overlay()
            self._albedo_color = c
            self._update_albedo_btn()
            self._apply_material()

    def _reset_transform(self):
        self._pos_x.setValue(0.0)
        self._pos_y.setValue(0.0)
        self._scale.setValue(1.0)

    def _clear_overlay(self):
        """Dismiss the static PyTorch3D preview so the live GL is visible."""
        if hasattr(self._gl, '_overlay'):
            self._gl._overlay.clear()

    def _apply_transform(self):
        self._clear_overlay()
        self._gl.set_obj_offset(self._pos_x.value(), self._pos_y.value())
        self._gl.set_obj_scale(self._scale.value())

    def _apply_material(self):
        self._clear_overlay()
        c = self._albedo_color
        self._gl.set_albedo(c.redF(), c.greenF(), c.blueF())
        self._gl.set_material(self._metallic.value(), self._roughness.value())

    def _apply_light(self):
        self._clear_overlay()
        self._gl.set_light(self._l_azim.value(), self._l_elev.value(), self._l_inten.value())

    def _apply_env(self):
        self._clear_overlay()
        self._gl.set_env_strength(self._env_str.value())
        self._gl.set_hdri_rotation(self._env_rot.value())



class ViewportWidget(QWidget):
    """
    Container for the 3D preview. API-compatible with the original QWidget version.
    Adds: QOpenGLWidget PBR/HDR scene, camera inertia, bottom camera controls.

    Signal contract (consumed by MainWindow):
      camera_active  -- emitted the moment the user starts dragging
      camera_idle    -- emitted 400 ms after the camera stops moving,
                        carries (dist_world, elev, azim)
      (camera_changed is kept as legacy for any external listeners)
    """
    camera_changed   = Signal(float, float, float)  # legacy: dist, elev, azim
    camera_active    = Signal()                      # drag started
    camera_idle      = Signal(float, float, float)   # camera settled: dist, elev, azim
    generate_clicked = Signal()
    pause_clicked    = Signal()
    stop_clicked     = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("viewport_container")
        self.setMinimumSize(480, 400)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top toolbar (play/pause/stop) ─────────────────────────────────────
        self.toolbar = QWidget()
        self.toolbar.setFixedHeight(32)
        self.toolbar.setStyleSheet(
            "background-color: #1c2333; border-bottom: 1px solid #252d3d;"
        )
        tb_layout = QHBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(8, 0, 8, 0)
        tb_layout.setSpacing(2)
        tb_layout.addStretch()

        icon_path = os.path.join(os.getcwd(), "sources", "icons")

        def _tinted_icon(filename, color=QColor(200, 210, 230)):
            raw = QPixmap(os.path.join(icon_path, filename))
            if raw.isNull():
                return QIcon()
            tinted = QPixmap(raw.size())
            tinted.fill(Qt.GlobalColor.transparent)
            painter = QPainter(tinted)
            painter.drawPixmap(0, 0, raw)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            painter.fillRect(tinted.rect(), color)
            painter.end()
            return QIcon(tinted)

        self.play_btn  = QPushButton()
        self.pause_btn = QPushButton()
        self.stop_btn  = QPushButton()

        self.play_btn .setIcon(_tinted_icon("play-button.png", QColor(100, 220, 150)))
        self.pause_btn.setIcon(_tinted_icon("pause.png",       QColor(180, 185, 200)))
        self.stop_btn .setIcon(_tinted_icon("stop-button.png", QColor(180, 185, 200)))

        for btn in [self.play_btn, self.pause_btn, self.stop_btn]:
            btn.setFixedSize(26, 24)
            btn.setIconSize(QSize(13, 13))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton { background:transparent; border:1px solid transparent;
                              border-radius:3px; padding:3px; }
                QPushButton:hover   { background:rgba(255,255,255,18);
                                      border:1px solid rgba(255,255,255,12); }
                QPushButton:pressed { background:rgba(0,0,0,30); }
                QPushButton:disabled{ opacity:0.25; }
            """)
            tb_layout.addWidget(btn)

        self.play_btn .setToolTip("Start Generation (Play)")
        self.pause_btn.setToolTip("Pause/Resume")
        self.pause_btn.setEnabled(False)
        self.stop_btn .setToolTip("Stop")
        self.stop_btn .setEnabled(False)

        # ── Inspector toggle button ──────────────────────────────────────────
        self._insp_btn = QPushButton("⚙")
        self._insp_btn.setToolTip("Toggle Scene Inspector")
        self._insp_btn.setFixedSize(26, 24)
        self._insp_btn.setCheckable(True)
        self._insp_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._insp_btn.setStyleSheet("""
            QPushButton { background:transparent; border:1px solid transparent;
                          border-radius:3px; color:#8b949e; font-size:13px; }
            QPushButton:hover   { background:rgba(255,255,255,18); color:#cdd6f4;
                                  border:1px solid rgba(255,255,255,12); }
            QPushButton:checked { background:#1a2744; color:#4fc3f7;
                                  border:1px solid #4fc3f7; }
        """)
        tb_layout.addWidget(self._insp_btn)
        tb_layout.addStretch()
        root.addWidget(self.toolbar)

        # ── GL + Inspector in a horizontal split ──────────────────────────────
        self._gl_insp_row = QWidget()
        gi_layout = QHBoxLayout(self._gl_insp_row)
        gi_layout.setContentsMargins(0, 0, 0, 0)
        gi_layout.setSpacing(0)

        if HAS_OPENGL:
            self._gl = GL3DPreview(self._gl_insp_row)
            # Scene tool toolbar (left side, Unity-style)
            self._scene_toolbar = SceneToolBar(self._gl_insp_row)
            gi_layout.addWidget(self._scene_toolbar)
            gi_layout.addWidget(self._gl, stretch=1)
            # Wire toolbar → GL gizmo mode
            def _on_tool_changed(mode):
                if self._gl:
                    self._gl._gizmo_mode = mode
                    self._gl.update()
            self._scene_toolbar.tool_changed.connect(_on_tool_changed)
            # Scene Inspector
            self._inspector = SceneInspectorPanel(self._gl, self._gl_insp_row)
            self._inspector.setVisible(False)
            gi_layout.addWidget(self._inspector)
            self._insp_btn.toggled.connect(self._inspector.setVisible)
            # Camera idle debounce
            self._idle_timer = QTimer(self)
            self._idle_timer.setSingleShot(True)
            self._idle_timer.setInterval(400)
            self._idle_timer.timeout.connect(self._emit_camera_idle)
        else:
            self._gl = None
            self._inspector = None
            fallback = QLabel("OpenGL not available — install PyOpenGL", self._gl_insp_row)
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color:#8b949e; background:#0d1117;")
            gi_layout.addWidget(fallback, stretch=1)

        root.addWidget(self._gl_insp_row, stretch=1)

        # ── Bottom camera controls ────────────────────────────────────────────
        if self._gl:
            self._cam_bar = CameraControlBar(self._gl, self)
            root.addWidget(self._cam_bar)

        # Signals
        self.play_btn .clicked.connect(self.generate_clicked)
        self.pause_btn.clicked.connect(self.pause_clicked)
        self.stop_btn .clicked.connect(self.stop_clicked)

        # Legacy state (for fallback QImage display)
        self._pixmap: QPixmap | None = None
        self._scene_info: dict = {}
        self._loading: bool = False

    # ── Camera lifecycle callbacks (called by GL3DPreview) ────────────────────

    def _on_gl_drag_start(self):
        """Called by GL3DPreview when a drag starts. Clear preview immediately."""
        self._idle_timer.stop()
        self.camera_active.emit()

    def _on_gl_drag_end(self):
        """Called by GL3DPreview when a drag/scroll ends. Start idle countdown."""
        self._idle_timer.start()  # restarts if already running

    def _emit_camera_idle(self):
        """Fires 400ms after camera stops. MainWindow uses this to start PyTorch3D."""
        if self._gl:
            dist_world = self._gl._dist * 200
            self.camera_idle.emit(dist_world, self._gl._elev, self._gl._azim)

    # ── Public API (unchanged for MainWindow) ─────────────────────────────────

    def set_preview(self, qimage: QImage):
        """Display a rendered frame from the worker."""
        if self._gl:
            self._gl.set_preview_image(qimage)
        else:
            self._pixmap = QPixmap.fromImage(qimage)
            self.update()

    def set_scene_info(self, info: dict):
        self._scene_info = info

    def set_loading(self, loading: bool):
        self._loading = loading

    def clear_preview(self):
        self._pixmap = None
        if self._gl:
            self._gl._preview_pixmap = None
            self._gl._overlay.clear()  # clear the transparent overlay widget

    def set_hdri(self, path: str):
        """Forward HDRI path to GL widget for env-map rendering."""
        if self._gl:
            self._gl.set_hdri(path)

    def load_mesh(self, path: str):
        """Forward mesh path to GL widget."""
        if self._gl:
            self._gl.load_obj_mesh(path)
