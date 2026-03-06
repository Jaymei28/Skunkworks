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

from app.engine.ocean_sim import ocean_physics

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
        glGetError, glIsVertexArray, glIsFramebuffer, glUniform1f, glUniform1i, glUniform3f,
        glUniformMatrix3fv, glUniformMatrix4fv,
        glDepthMask, glLineWidth, glUseProgram,
        glBlendFunc, glScissor,
        glGenFramebuffers, glBindFramebuffer, glFramebufferTexture2D,
        glGenRenderbuffers, glBindRenderbuffer, glRenderbufferStorage,
        glFramebufferRenderbuffer,
        glDeleteFramebuffers, glDeleteRenderbuffers,
        glIsRenderbuffer,
        GL_NO_ERROR, GL_DEPTH_TEST, GL_LESS, GL_LEQUAL,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
        GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW,
        GL_FLOAT, GL_FALSE, GL_TRUE, GL_TRIANGLES, GL_UNSIGNED_INT, GL_LINE,
        GL_FILL, GL_FRONT_AND_BACK, GL_TEXTURE_2D, GL_RGB, GL_RGB16F,
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, 
        GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT,
        glFramebufferRenderbuffer,
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_vao(vao_id):
    """Check if a VAO ID is valid and exists in GL."""
    if not vao_id or vao_id <= 0: return False
    try:
        from OpenGL.GL import glIsVertexArray
        return glIsVertexArray(int(vao_id))
    except Exception:
        return False

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
uniform sampler2D uAlbedoMap;
uniform sampler2D uNormalMap;
uniform bool  uHasAlbedoMap;
uniform bool  uHasNormalMap;

uniform vec3  uAlbedo;
uniform float uMetallic;
uniform float uRoughness;
uniform float uLightIntensity; // key light radiance scale
uniform float uEnvStrength;    // HDRI intensity
uniform vec3  uLightDir;       // sun direction

// --- Weather ---
uniform int   uWeatherType;      // 0:clear, 1:cloudy, 2:rain, 3:stormy, 4:snow, 5:foggy
uniform float uWeatherIntensity;
uniform float uFogDensity;

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
    vec3  V    = normalize(uCamPos - vWorldPos);
    
    // ── Robust Normal Handling ──
    vec3 N_geom = normalize(vNormal);
    
    // Double-sided lighting trick: if the normal points away from the camera, flip it.
    // This solves issues with inverted geometry or viewing faces from behind.
    if (dot(N_geom, V) < 0.0) N_geom = -N_geom;

    vec3 N;
    if(uHasNormalMap){
        // Planar TBN approx
        vec3 Q1 = dFdx(vWorldPos);
        vec3 Q2 = dFdy(vWorldPos);
        vec2 st1 = dFdx(vUV);
        vec2 st2 = dFdy(vUV);
        
        float det = (st1.x * st2.y - st1.y * st2.x);
        if (abs(det) < 0.000001) {
             N = N_geom;
        } else {
            vec3 T = normalize(Q1*st2.t - Q2*st1.t);
            vec3 B = -normalize(cross(N_geom, T));
            mat3 TBN = mat3(T, B, N_geom);
            vec3 tangentNormal = texture(uNormalMap, vUV).rgb * 2.0 - 1.0;
            N = normalize(TBN * tangentNormal);
        }
    } else {
        N = N_geom;
    }

    float NdV  = max(dot(N,V), 0.001);

    vec3 albedo = uAlbedo;
    if(uHasAlbedoMap){
        vec4 texColor = texture(uAlbedoMap, vUV);
        albedo *= texColor.rgb;
        if(texColor.a < 0.05) discard;
    }
    
    albedo = max(albedo, vec3(0.01));

    // --- Weather Surface Fix ---
    float roughness = uRoughness;
    if (uWeatherType == 2 || uWeatherType == 3) {
        float wet = uWeatherIntensity * 0.7;
        albedo *= (1.0 - wet * 0.3);
        roughness = mix(roughness, 0.05, wet);
    }
    if (uWeatherType == 4) {
        float snow = pow(clamp(N.y, 0.0, 1.0), 2.5) * uWeatherIntensity;
        albedo = mix(albedo, vec3(0.92, 0.95, 1.0), snow * 0.9);
        roughness = mix(roughness, 0.95, snow);
    }

    vec3 F0 = mix(vec3(0.04), albedo, uMetallic);

    // ── Key light (controllable direction) ───────────────────────────────────
    vec3  L0   = normalize(uLightDir);
    vec3  H0   = normalize(V + L0);
    vec3  rad0 = vec3(uLightIntensity);

    float NDF  = DistributionGGX(N, H0, roughness);
    float G    = GeometrySmith(N, V, L0, roughness);
    vec3  F    = FresnelSchlick(max(dot(H0,V),0.0), F0);

    vec3  num  = NDF * G * F;
    float den  = 4.0 * NdV * max(dot(N,L0),0.0) + 0.0001;
    vec3  spec = num / den;

    vec3  nD   = (1.0 - F)*(1.0 - uMetallic);
    vec3  Lo   = (nD * albedo / PI + spec) * rad0 * max(dot(N,L0),0.0);

    // ── Indirect Lighting (IBL) ──
    vec3 kS_a  = FresnelSchlickRoughness(NdV, F0, roughness);
    vec3 kD_a  = (1.0 - kS_a) * (1.0 - uMetallic);

    vec3 irrad  = sampleEnv(N, 6.0);
    vec3 diffIBL = kD_a * irrad * albedo;

    vec3 R        = reflect(-V, N);
    vec3 prefilt  = sampleEnv(R, roughness * 7.0);
    vec3 specIBL  = (kS_a * prefilt);

    // --- Hemisphere-style Ambient fallback ---
    // If no HDRI, we still want a "ground/sky" feel so objects are readable.
    vec3 skyColor = vec3(0.18, 0.20, 0.25);
    vec3 groundColor = vec3(0.08, 0.07, 0.05);
    float hemi = N.y * 0.5 + 0.5;
    vec3 hemiAmbient = mix(groundColor, skyColor, hemi) * albedo;
    
    vec3 ambient = (diffIBL + specIBL) + (hemiAmbient * 0.4) + vec3(0.02);

    vec3 color = Lo + ambient;

    // ── Tone-map (ACES filmic) ── Improved saturation handling
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    color = clamp((color*(a*color+b))/(color*(c*color+d)+e), 0.0, 1.0);

    // ── Gamma correct ──────────────────────────────────────────────────────
    color = pow(color, vec3(1.0/2.2));

    // ── Fog / Atmosphere ──
    vec3 fogColor = vec3(0.4, 0.45, 0.5); 
    if (uWeatherType == 2) fogColor = vec3(0.25, 0.3, 0.35); // Rain
    if (uWeatherType == 3) fogColor = vec3(0.08, 0.1, 0.12); // Storm
    if (uWeatherType == 4) fogColor = vec3(0.8, 0.85, 0.9);  // Snow

    float dist = length(uCamPos - vWorldPos);
    float fogFactor = 1.0 - exp(-dist * uFogDensity * (0.8 + uWeatherIntensity * 2.0));
    color = mix(color, fogColor, clamp(fogFactor, 0.0, 1.0));

    if (uWeatherType >= 1) {
        float gray = dot(color, vec3(0.299, 0.587, 0.114));
        color = mix(color, vec3(gray), uWeatherIntensity * 0.3);
    }

    FragColor = vec4(color, 1.0);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Ocean Shaders (Gerstner Waves + Water FX)
# ─────────────────────────────────────────────────────────────────────────────
_OCEAN_COMMON = """
#version 330 core
const float PI = 3.14159265359;
uniform float uTime;
uniform float uRepetitionSize;
uniform float uWindSpeed;
uniform float uWindDirection;
uniform float uChoppiness;
uniform float uWaveAmplitude;
uniform float uBand0Multiplier;
uniform float uBand1Multiplier;
uniform float uLevel;
uniform float uStorm;

vec4 waves[8] = vec4[](
    vec4(1.0, 0.5,  0.50, 1.2), vec4(0.8, 1.0,  0.45, 0.9), 
    vec4(0.2, 1.2,  0.40, 0.7), vec4(-0.5, 0.8, 0.35, 0.5), 
    vec4(0.1, -1.0, 0.30, 0.3), vec4(-0.7, -0.2, 0.25, 0.15), 
    vec4(0.5, 0.5,  0.20, 0.08),vec4(-0.2, 0.8, 0.15, 0.04)  
);

void calculateWave(vec2 basePos, out vec3 displacement, out vec3 tangent, out vec3 binormal, out float height) {
    displacement = vec3(0.0);
    tangent = vec3(1.0, 0.0, 0.0);
    binormal = vec3(0.0, 0.0, 1.0);
    height = 0.0;

    float windAngle = radians(uWindDirection);
    mat2 windRot = mat2(cos(windAngle), -sin(windAngle), sin(windAngle), cos(windAngle));
    
    float globalAmp = (uWindSpeed / 10.0) * uWaveAmplitude;
    float globalChop = uChoppiness * 1.8;

    for(int i=0; i<8; i++){
        vec4 w = waves[i];
        vec2 dir = normalize(windRot * w.xy);
        float bandMul = (i < 4) ? uBand0Multiplier : uBand1Multiplier;
        float wavelength = w.w * uRepetitionSize * 0.2;
        float k = 2.0 * PI / max(wavelength, 0.01);
        float a = (wavelength / 40.0) * globalAmp * bandMul; 
        float q = (w.z * globalChop) / (k * a * 8.0 + 0.001);
        float speed = sqrt(9.81 / k);
        float f = k * (dot(dir, basePos) - speed * uTime);
        float cosF = cos(f); float sinF = sin(f);
        displacement.x += q * a * dir.x * cosF;
        displacement.y += a * sinF;
        displacement.z += q * a * dir.y * cosF;
        height += a * sinF;
        float ksc = k * a * cosF; float kss = k * a * sinF;
        tangent  += vec3(-q * dir.x * dir.x * kss, dir.x * ksc, -q * dir.x * dir.y * kss);
        binormal += vec3(-q * dir.x * dir.y * kss, dir.y * ksc, -q * dir.y * dir.y * kss);
    }
}
"""


_OCEAN_VERT = _OCEAN_COMMON + """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aUV;
out vec3 vWorldPos;
out vec2 vUV;
out float vTotalHeight;
uniform mat4 uProj;
uniform mat4 uView;
void main(){
    vec3 displacement, tangent, binormal;
    float height;
    calculateWave(aPos.xz, displacement, tangent, binormal, height);
    vec3 p = aPos + displacement;
    p.y += uLevel;
    vWorldPos = p;
    vTotalHeight = height;
    vUV = aPos.xz; 
    gl_Position = uProj * uView * vec4(p, 1.0);
}
"""


_OCEAN_FRAG = _OCEAN_COMMON + """
in vec3 vWorldPos;
in vec2 vUV;
in float vTotalHeight;
out vec4 FragColor;
uniform vec3  uCamPos;
uniform vec3  uRefractionColor;
uniform vec3  uScatteringColor;
uniform float uAbsorptionDistance;
uniform float uAmbientScattering;
uniform float uDirectLightTipScattering;
uniform float uDirectLightBodyScattering;
uniform float uSmoothness;
uniform float uTransparency;
uniform sampler2D uEnvMap;
uniform bool  uHasEnvMap;
uniform float uEnvStrength;
uniform vec3  uLightDir;
uniform float uLightIntensity;
uniform float uFogDensity;
uniform bool  uRipplesEnabled;
uniform float uRipplesWindSpeed;
uniform float uRipplesWindDir;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }

vec3 oceanRipples(vec2 uv, float time) {
    vec3 n = vec3(0.0, 1.0, 0.0);
    float freq = 6.0; float weight = 0.4;
    for(int i=0; i<4; i++) {
        vec2 d = vec2(sin(float(i)*1.5), cos(float(i)*1.5));
        float h = sin(dot(uv, d * freq) + time * (1.2 + float(i)*0.1));
        n.xz += d * h * weight * 0.15;
        freq *= 2.0; weight *= 0.5;
    }
    return normalize(n);
}

float DistributionGGX(float NdH, float roughness) {
    float a = roughness * roughness; float a2 = a * a;
    float d = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d + 0.00001);
}
float GeometrySmith(float NdV, float NdL, float roughness) {
    float k = (roughness + 1.0); k = k * k / 8.0;
    return (NdV/(NdV*(1.0-k)+k)) * (NdL/(NdL*(1.0-k)+k));
}
vec3 FresnelSchlick(float cosT, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosT, 0.0, 1.0), 5.0);
}

void main(){
    vec3 displacement, tangent, binormal;
    float height;
    calculateWave(vUV, displacement, tangent, binormal, height);
    vec3 N_geo = normalize(cross(binormal, tangent));
    vec3 N = N_geo;
    if (uRipplesEnabled) {
        vec3 N_ripple = oceanRipples(vWorldPos.xz * 2.5, uTime);
        N = normalize(mix(N_geo, N_ripple, 0.25));
    }
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 L = normalize(uLightDir);
    vec3 H = normalize(V + L);
    float NdV = max(dot(N, V), 0.001);
    float NdL = max(dot(N, L), 0.0);
    float NdH = max(dot(N, H), 0.0);
    float HdV = max(dot(H, V), 0.0);
    float depth = clamp(-(vWorldPos.y - uLevel) / (uAbsorptionDistance + 0.1), 0.0, 1.0);
    vec3 baseColor = mix(uRefractionColor, uScatteringColor * 0.3, 1.0 - exp(-depth * 5.0));
    float crest = clamp(vTotalHeight / (uWaveAmplitude * 2.5 + 0.1), 0.0, 1.0);
    float tipScat = pow(crest, 2.0) * pow(1.0 - NdV, 3.0) * uDirectLightTipScattering;
    vec3 scattering = uScatteringColor * (uAmbientScattering + tipScat + uDirectLightBodyScattering * NdL);
    vec3 colorBody = baseColor + scattering;
    vec3 F0 = vec3(0.03); vec3 F = FresnelSchlick(NdV, F0);
    vec3 reflection = vec3(0.0);
    vec3 R_vec = reflect(-V, N);
    if (uHasEnvMap) {
        float p = atan(R_vec.z, R_vec.x); float t = acos(clamp(R_vec.y, -1.0, 1.0));
        reflection = texture(uEnvMap, vec2(p/(2.0*PI)+0.5, t/PI)).rgb;
    } else {
        reflection = mix(vec3(0.2, 0.4, 0.6), vec3(0.7, 0.9, 1.0), pow(clamp(R_vec.y, 0.0, 1.0), 0.5));
    }
    reflection *= uEnvStrength;
    float rough = max(1.0 - uSmoothness, 0.05);
    float D = DistributionGGX(NdH, rough);
    float G = GeometrySmith(NdV, NdL, rough);
    vec3 spec = (D * G * FresnelSchlick(HdV, F0)) / (4.0 * NdV * NdL + 0.0001);
    vec3 sunSpec = spec * vec3(uLightIntensity * 50.0) * NdL;
    vec3 color = mix(colorBody, reflection, F) + sunSpec;
    color = mix(color, vec3(0.9, 0.95, 1.0), smoothstep(0.7, 1.0, crest) * 0.3); // Foam
    float fog = 1.0 - exp(-length(uCamPos - vWorldPos) * uFogDensity);
    color = mix(color, vec3(0.5, 0.6, 0.7), clamp(fog, 0.0, 1.0));
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
    FragColor = vec4(color, uTransparency);
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

// --- Weather ---
uniform int       uWeatherType;
uniform float     uWeatherIntensity;
uniform float     uThunderFlash;

const float PI = 3.14159265359;
void main(){
    vec3  d   = normalize(vDir);
    float phi = atan(d.z, d.x) + uHdriRotation;
    float th  = asin(clamp(d.y, -1.0, 1.0));
    vec2  uv  = vec2(phi/(2.0*PI)+0.5, 1.0 - (th/PI+0.5));
    vec3  col = texture(uEnvMap, uv).rgb * uEnvStrength;

    // Apply weather darkening/tinting to sky
    if (uWeatherType == 1) col *= mix(1.0, 0.7, uWeatherIntensity); // Cloudy
    if (uWeatherType == 2) col *= mix(1.0, 0.5, uWeatherIntensity); // Rain
    if (uWeatherType == 3) col *= mix(1.0, 0.2, uWeatherIntensity); // Stormy
    if (uWeatherType == 4) col *= mix(1.0, 0.9, uWeatherIntensity); // Snow
    if (uWeatherType == 5) col *= mix(1.0, 0.6, uWeatherIntensity); // Foggy

    // filmic tone-map
    col = (col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14);
    col = clamp(col, 0.0, 1.0);
    col = pow(col, vec3(1.0/2.2));
    // --- Thunder Flash ---
    if (uWeatherType == 3 && uThunderFlash > 0.0) {
        col += vec3(uThunderFlash * 1.5);
    }

    FragColor = vec4(col, 1.0);
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Weather Overlay Shader (Rain / Snow / Thunder)
# ─────────────────────────────────────────────────────────────────────────────
_WEATHER_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec2 vUV;
void main(){
    vUV = aPos.xy * 0.5 + 0.5;
    gl_Position = vec4(aPos, 1.0);
}
"""

_WEATHER_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform float uTime;
uniform int   uWeatherType;
uniform float uWeatherIntensity;
uniform float uAspect;
uniform float uThunderFlash;
uniform vec3  uCamPos;
uniform vec3  uCamRight;
uniform vec3  uCamUp;
uniform vec3  uCamFwd;

float hash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

void main() {
    // Start with a clean transparent color
    vec4 col = vec4(0.0);
    
    // View ray reconstruction
    vec2 p = (vUV * 2.0 - 1.0) * vec2(uAspect, 1.0);
    vec3 rd = normalize(uCamFwd + uCamRight * p.x + uCamUp * p.y);
    float t = uTime;

    // --- Weather Volume Rendering (Planar Ray-Casting) ---
    // This removes the "spinning" distortion seen with spherical shells.
    float cosView = dot(rd, uCamFwd);
    if (cosView > 0.0) {
        // --- Rain (Type 2: rain, Type 3: stormy) ---
        if (uWeatherType == 2 || uWeatherType == 3) {
            float rainTime = t * 6.0; 
            for (int i = 0; i < 5; i++) {
                float layer = float(i) + 1.0;
                float dist = 1.0 + layer * 2.0;
                float planeDist = dist / cosView;
                
                vec3 worldPos = uCamPos + rd * planeDist;
                
                // Lower resolution = bigger drops
                float res = 6.0 + layer * 3.0; 
                vec3 boxPos = worldPos * res;
                boxPos.y += rainTime * (3.0 + layer * 0.5);
                boxPos.x += boxPos.y * 0.035; 
                
                vec3 ip = floor(boxPos);
                vec3 fp = fract(boxPos);
                float h = hash(ip + layer * 19.7);
                
                if (h > 0.98 - uWeatherIntensity * 0.07) {
                    // Rain drop: wider streaks
                    float drop = smoothstep(0.12, 0.0, abs(fp.x - 0.5));
                    drop *= smoothstep(0.15, 0.0, abs(fp.z - 0.5));
                    drop *= smoothstep(0.0, 0.2, fp.y) * smoothstep(1.0, 0.2, fp.y);
                    
                    float alpha = drop * (0.4 + uWeatherIntensity * 0.6);
                    col += vec4(0.85, 0.9, 1.0, alpha * 0.7);
                }
            }
        }

        // --- Snow (Type 4) ---
        if (uWeatherType == 4) {
            float snowTime = t * 1.5;
            for (int i = 0; i < 4; i++) {
                float layer = float(i) + 1.0;
                float dist = 2.0 + layer * 3.5;
                float planeDist = dist / cosView;
                
                vec3 worldPos = uCamPos + rd * planeDist;
                
                float res = 4.0 + layer * 1.2;
                vec3 boxPos = worldPos * res;
                boxPos.y += snowTime * (1.1 - layer * 0.1);
                boxPos.xz += sin(snowTime * 0.5 + worldPos.y * 0.1) * 0.4;
                
                vec3 ip = floor(boxPos);
                vec3 fp = fract(boxPos);
                float h = hash(ip + layer * 37.1);
                
                if (h > 0.96 - uWeatherIntensity * 0.1) {
                    float streak = 0.4 + layer * 0.3;
                    float d = length((fp - 0.5) / vec3(1.0, 1.0 + streak, 1.0));
                    float rSize = 0.06 + 0.1 * hash(ip + 13.3);
                    
                    float glow = exp(-d * 7.0) * 0.5;
                    float core = smoothstep(rSize, rSize * 0.5, d);
                    
                    vec3 snowCol = vec3(1.0);
                    float alpha = (core + glow) * (0.3 + uWeatherIntensity * 0.7);
                    col.rgb += snowCol * alpha;
                    col.a += alpha * 0.4;
                }
            }
        }
    }

    // --- Stormy Lighting (Flash) ---
    if (uWeatherType == 3 && uThunderFlash > 0.01) {
        col += vec4(0.95, 1.0, 1.0, uThunderFlash * 0.5);
    }

    col = clamp(col, 0.0, 1.0);
    FragColor = col;
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

_PP_VERT = """
#version 330 core
layout(location=0) in vec3 aPos;
out vec2 vUV;
void main(){
    vUV = aPos.xy * 0.5 + 0.5;
    gl_Position = vec4(aPos, 1.0);
}
"""

_PP_FRAG = """
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uScreenTex;
uniform bool  uFisheyeEnabled;
uniform float uFisheyeStrength;
uniform float uAspect;

void main(){
    vec2 uv = vUV;
    
    if(uFisheyeEnabled){
        // Normalized coordinates centered at 0 [-1, 1]
        vec2 p = vUV * 2.0 - 1.0;
        
        // Correct for aspect ratio to keep the lens circular
        p.x *= uAspect;
        
        float d = length(p);
        
        // True panoramic fisheye mapping
        // Curvilinear distortion (Equidistant-ish approximation)
        // Strength scales how 'tight' the fisheye wrap is
        float k = uFisheyeStrength * 1.5;
        float nr = atan(d * k) / k;
        
        if (d > 0.001) {
             p = (p / d) * nr;
        }
        
        // Re-apply aspect and bring back to [0, 1]
        p.x /= uAspect;
        uv = p * 0.5 + 0.5;
        
        // Smooth vignette mask based on original radius
        // The vignette should scale with aspect
        float bind = (uAspect > 1.0) ? uAspect : 1.0;
        float vignette = smoothstep(bind * 1.1, bind * 1.0, d);
        
        // Sample screen texture with wrap check
        if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        } else {
            vec3 col = texture(uScreenTex, uv).rgb;
            FragColor = vec4(col * vignette, 1.0);
        }
    } else {
        FragColor = texture(uScreenTex, uv);
    }
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
    camera_active    = Signal()
    object_selected  = Signal(object)
    object_moved     = Signal(object)
    texture_discovered = Signal(str, str) # mesh_path, albedo_path

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
        self._hdri_rotation = 0.0

        # Mesh Storage
        self._mesh_cache = {} # path -> {vao, vbo, ebo, count, albedo, normal}
        self._pending_uploads = [] # list of mesh paths

        self._obj_offset = [0.0, 0.0, 0.0]
        self._obj_rotation = [0.0, 0.0, 0.0]
        self._obj_scale = 1.0
        self._albedo     = [0.9, 0.35, 0.15]   # default: rusty orange
        self._metallic   = 0.1
        self._roughness  = 0.45
        # Light
        self._light_azim  = 45.0          # degrees, 0=+Z, CCW
        self._light_elev  = 60.0          # degrees above horizon
        self._light_intensity = 2.8       # PBR radiance scale
        # Object transform
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
        self._albedo_tex = None # Default texture for fallback sphere

        # Caching
        self._mesh_cache = {}    # mesh_path -> {vao, vbo, ebo, count, bbox}
        self._texture_cache = {} # file_path -> texture_id

        # Scene state reference
        self.cfg = None
        self._selected_obj = None

        # GL objects (initialised in initializeGL)
        self._prog = self._sky_prog = None
        self._sphere_vao = self._sphere_vbo = self._sphere_ebo = None
        self._sky_vao = self._sky_vbo = self._sky_ebo = None
        self._env_tex = None
        self._sphere_idx_count = 0
        self._sky_idx_count    = 0
        self._gl_ready = False

        # Post-Process FBO
        self._pp_fbo = None
        self._pp_tex = None
        self._pp_depth = None
        self._pp_prog = None

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
        # Gizmo GL resources
        self._gizmo_prog = None
        self._gizmo_vao = self._gizmo_vbo = None

        # Object selection state
        self._obj_selected = False

        self.setMouseTracking(True)  # needed for hover detection
        self._discovered_textures = {} # mesh_path -> albedo_path

        # ── Ocean Resources ──────────────────────────────────────────────────
        self._ocean_prog = None
        self._ocean_vao = self._ocean_vbo = self._ocean_ebo = None
        self._ocean_idx_count = 0

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

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # Scale zoom sensitivity by distance (closer = slower)
        speed = 1.15 if delta < 0 else 0.85
        self._dist *= speed
        self._dist = max(0.1, min(self._dist, 4500.0))
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
        """Set mesh path for pre-loading. Actual render loop uses cfg.scene_objects."""
        if not path or not os.path.exists(path):
            return
        if not HAS_OPENGL:
            return

        if path not in self._mesh_cache and path not in self._pending_uploads:
            self._pending_uploads.append(path)
            self._mesh_needs_upload = True
        elif path in self._discovered_textures:
            # Mesh already loaded, but we should still emit discovery if we found one
            self.texture_discovered.emit(path, self._discovered_textures[path])
        self.update()

    def set_scene_config(self, cfg):
        self.cfg = cfg
        self.update()

    def set_selected_object(self, obj):
        self._selected_obj = obj
        self.update()


    def _do_upload_mesh(self, path: str):
        """
        Internal: load + upload one mesh from disk.
        """
        from app.engine.mesh_loader import load_mesh
        import numpy as np

        print(f"[GL] _do_upload_mesh: loading {path!r}")
        try:
            loaded = load_mesh(path)
        except Exception as e:
            print(f"[GL] Failed to load mesh {path}: {e}")
            return

        v_raw = loaded.vertices
        f = loaded.indices
        n_raw = loaded.normals if loaded.normals.size else None
        u = loaded.uvs if loaded.uvs.size else None

        # Normalize: Center and scale the mesh so it's ~1.0 unit tall at origin
        # This solves the "vanishing object" problem for models with huge offsets or tiny scales.
        v = (v_raw - loaded.center) * loaded.scale_hint
        # Normals don't need translation, and scale is uniform so normalize is enough
        n = n_raw # will be normalized in shader too

        # Upload using a dedicated helper that returns mesh resource info
        resource = self._create_mesh_resource(v, f, n, u, loaded.tex_albedo)
        if resource:
            self._mesh_cache[path] = resource
            print(f"[GL] Mesh {os.path.basename(path)} cached: {len(v)} verts, {len(f)//3} tris.")
            if loaded.tex_albedo:
                self._discovered_textures[path] = loaded.tex_albedo
                self.texture_discovered.emit(path, loaded.tex_albedo)

    def _create_mesh_resource(self, vertices, faces, normals, uvs, albedo_path=None):
        """Create OpenGL buffers for a mesh and upload its default albedo."""
        if not HAS_OPENGL:
            return None
        import numpy as np
        from OpenGL.GL import (
            glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer,
            glBufferData, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW,
            glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
            glGenerateMipmap, GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
            GL_LINEAR, GL_CLAMP_TO_EDGE, glActiveTexture, GL_TEXTURE0, GL_TEXTURE1,
            glVertexAttribPointer, glEnableVertexAttribArray, GL_FLOAT, GL_LINEAR_MIPMAP_LINEAR, GL_RGBA
        )
        from PySide6.QtGui import QImage

        vao = int(glGenVertexArrays(1))
        vbo = int(glGenBuffers(1))
        ebo = int(glGenBuffers(1))

        glBindVertexArray(vao)

        # FAST interleaving using numpy (instead of Python loops)
        count = len(vertices)
        # Interleave: Pos(3), Norm(3), UV(2) = 8 floats
        v_arr = np.zeros((count, 8), dtype=np.float32)
        v_arr[:, 0:3] = vertices
        if normals is not None and len(normals) == count:
            v_arr[:, 3:6] = normals
        else:
            # Default normal: Up
            v_arr[:, 4] = 1.0
        
        if uvs is not None and len(uvs) == count:
            v_arr[:, 6:8] = uvs

        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, v_arr.nbytes, v_arr, GL_STATIC_DRAW)

        f_arr = np.array(faces, dtype=np.uint32)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, f_arr.nbytes, f_arr, GL_STATIC_DRAW)

        stride = 8 * 4
        # Positions
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        # Normals
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        # UVs
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))

        glBindVertexArray(0)

        # Handle albedo
        tex_id = None
        if albedo_path and os.path.exists(albedo_path):
            tex_id = self._get_or_create_texture(albedo_path)

        return {
            "vao": vao, "vbo": vbo, "ebo": ebo, 
            "count": len(faces),
            "albedo": tex_id
        }

    def _get_or_create_texture(self, path):
        if not path or not os.path.exists(path):
            return None
        if path in self._texture_cache:
            return self._texture_cache[path]
        
        try:
            from PySide6.QtGui import QImage
            img = QImage(path).convertToFormat(QImage.Format.Format_RGBA8888)
            if img.isNull(): return None

            tid = int(glGenTextures(1))
            glBindTexture(GL_TEXTURE_2D, tid)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits().tobytes())
            glGenerateMipmap(GL_TEXTURE_2D)
            
            self._texture_cache[path] = tid
            return tid
        except Exception as e:
            print(f"[GL] Texture upload failed for {path}: {e}")
            return None

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
            tex_to_delete = []
            if self._env_tex is not None:
                tex_to_delete.append(int(self._env_tex))
                self._env_tex = None
            if self._albedo_tex is not None: # For fallback sphere
                tex_to_delete.append(int(self._albedo_tex))
                self._albedo_tex = None
            for mesh_res in self._mesh_cache.values():
                if mesh_res["albedo"] is not None:
                    tex_to_delete.append(int(mesh_res["albedo"]))
            if tex_to_delete:
                glDeleteTextures(len(tex_to_delete), tex_to_delete)
            
            vaos = [int(v) for v in [self._sphere_vao, self._sky_vao, self._grid_vao] if v is not None]
            for mesh_res in self._mesh_cache.values():
                if mesh_res["vao"] is not None:
                    vaos.append(int(mesh_res["vao"]))
            if vaos:
                glDeleteVertexArrays(len(vaos), vaos)
                
            bufs = [int(b) for b in [self._sphere_vbo, self._sphere_ebo, self._sky_vbo, self._sky_ebo, self._grid_vbo] if b is not None]
            for mesh_res in self._mesh_cache.values():
                if mesh_res["vbo"] is not None:
                    bufs.append(int(mesh_res["vbo"]))
                if mesh_res["ebo"] is not None:
                    bufs.append(int(mesh_res["ebo"]))
            if bufs:
                glDeleteBuffers(len(bufs), bufs)
        except Exception as exc:
            print(f"[GL] reset_gl cleanup error: {exc}")
            
        self._sphere_vao = self._sky_vao = self._grid_vao = None
        self._sphere_vbo = self._sphere_ebo = None
        self._sky_vbo    = self._sky_ebo    = None
        self._grid_vbo   = None
        self._env_tex    = None
        self._mesh_cache = {} # Clear cache
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
        # Notify stop idle rendering
        self.camera_active.emit()


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
                return

            # Other modes: try gizmo first (only if object is selected)
            if self._selected_obj:
                axis = self._gizmo_hit_test(pos.x(), pos.y())
                if axis:
                    self._gizmo_active_axis = axis
                    self._gizmo_drag_start_pos = pos
                    self._gizmo_drag_start_offset = [self._selected_obj.pos_x, self._selected_obj.pos_y, self._selected_obj.pos_z]
                    self._gizmo_drag_start_scale = self._selected_obj.scale
                    self._gizmo_drag_start_angle = 0.0 # Placeholder
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return

            # Try clicking on an object to select it
            hit_obj = self._hit_test_all_objects(pos.x(), pos.y())
            if hit_obj:
                self._selected_obj = hit_obj
                self.object_selected.emit(hit_obj)
                self.update()
                return

            # Clicked empty space — deselect
            if self._selected_obj:
                self._selected_obj = None
                self.object_selected.emit(None)
                self.update()
                return

            # Camera orbit (nothing hit)
            self._dragging = True
            self._last_pos = pos
            self._last_drag_delta = QPointF(0, 0)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.camera_active.emit()

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
            self._azim -= delta.x() * 0.15
            self._elev += delta.y() * 0.15
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
            speed = self._dist * 0.001
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
        elif self._gizmo_active_axis and self._selected_obj:
            delta = pos - self._gizmo_drag_start_pos
            self._on_gizmo_drag(delta.x(), delta.y())
            self.update()
            return
        elif self._dragging:
            delta = pos - self._last_pos
            self._last_drag_delta = delta
            self._last_pos = pos
            self._azim += delta.x() * 0.12 # Reduced from 0.15
            self._elev -= delta.y() * 0.12 # Reduced from 0.15
            self.update()
        elif getattr(self, '_panning', False):
            delta = pos - self._last_pos
            self._last_pos = pos
            # Pan sensitivity scales with distance
            sens = 0.0008 * self._dist
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
        self._dist = max(0.5, min(20.0, self._dist - event.angleDelta().y() * 0.002))
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
        if not self._selected_obj: return {}
        ox, oy, oz = self._selected_obj.pos_x, self._selected_obj.pos_y, self._selected_obj.pos_z
        length = 0.8  # fixed arm length matching _draw_gizmo
        return {
            'x': ([ox, oy, oz], [ox + length, oy, oz]),
            'y': ([ox, oy, oz], [ox, oy + length, oz]),
            'z': ([ox, oy, oz], [ox, oy, oz + length]),
        }

    def _hit_test_all_objects(self, mx, my):
        if not getattr(self, 'cfg', None): return None
        # Returns the closest hit object (if any)
        best_obj = None
        best_dist = 1e9
        for obj in self.cfg.scene_objects:
            if obj.visible and self._hit_test_object(mx, my, obj):
                # Simple distance check from camera
                cam = self._camera_pos()
                dist = math.sqrt((obj.pos_x-cam[0])**2 + (obj.pos_y-cam[1])**2 + (obj.pos_z-cam[2])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_obj = obj
        return best_obj

    def _hit_test_object(self, mx, my, obj):
        if not self._gl_ready or not obj: return False
        W, H = self.width(), self.height()
        if W < 1 or H < 1: return False
        
        aspect = W / max(H, 1)
        fov = self.cfg.fov_y if (self.cfg and hasattr(self.cfg, 'fov_y')) else 60.0
        proj = _mat4_perspective(fov, aspect, 0.1, 5000.0)
        view = _lookat(self._camera_pos(), self._target, self._camera_up())
        inv_vp = _mat4_inverse(_mat4_mul(proj, view))

        ndc_x = (2.0 * mx / W) - 1.0
        ndc_y = 1.0 - (2.0 * my / H)
        
        near_world = _mat4_vec4_mul(inv_vp, [ndc_x, ndc_y, -1.0, 1.0])
        far_world  = _mat4_vec4_mul(inv_vp, [ndc_x, ndc_y,  1.0, 1.0])
        
        if abs(near_world[3]) < 1e-9 or abs(far_world[3]) < 1e-9: return False
        
        near_w = [near_world[i]/near_world[3] for i in range(3)]
        far_w  = [far_world[i]/far_world[3] for i in range(3)]
        
        ray_dir = [far_w[i]-near_w[i] for i in range(3)]
        ray_len = math.sqrt(sum(d*d for d in ray_dir))
        if ray_len < 1e-9: return False
        ray_dir = [d/ray_len for d in ray_dir]
        
        center = [obj.pos_x, obj.pos_y, obj.pos_z]
        radius = obj.scale * 1.5
        
        L = [center[i] - near_w[i] for i in range(3)]
        tca = sum(L[i] * ray_dir[i] for i in range(3))
        if tca < 0: return False
        d2 = sum(L[i]*L[i] for i in range(3)) - tca*tca
        return d2 < (radius*radius)




    def _gizmo_hit_test(self, mx, my):
        """Return 'x'/'y'/'z'/'free' if mouse is near a gizmo axis, else None."""
        if not self._gl_ready or not self._selected_obj:
            return None
        W, H = self.width(), self.height()
        if W < 1 or H < 1:
            return None

        aspect = W / max(H, 1)
        fov = self.cfg.fov_y if (self.cfg and hasattr(self.cfg, 'fov_y')) else 60.0
        proj = _mat4_perspective(fov, aspect, 0.1, 500.0)
        eye  = self._camera_pos()
        up   = self._camera_up()
        view = _lookat(eye, self._target, up)
        mvp = self._compute_mvp(proj, view)

        # Use ring-based hit test for rotate mode
        if self._gizmo_mode == 'rotate':
            return self._gizmo_ring_hit_test(mx, my, mvp, W, H)

        # Check center box first (free movement handle)
        if self._gizmo_mode == 'translate':
            center_2d = self._world_to_screen([self._selected_obj.pos_x, self._selected_obj.pos_y, self._selected_obj.pos_z], mvp, W, H)
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
        if not self._selected_obj: return None
        ox, oy, oz = self._selected_obj.pos_x, self._selected_obj.pos_y, self._selected_obj.pos_z
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


    def _on_gizmo_drag(self, dx, dy):
        axis = self._gizmo_active_axis
        if not axis or not self._selected_obj:
            return

        if self._gizmo_mode == 'translate':
            W, H = self.width(), self.height()
            aspect = W / max(H, 1)
            proj = _mat4_perspective(60.0, aspect, 0.1, 500.0)
            view = _lookat(self._camera_pos(), self._target, self._camera_up())
            mvp  = self._compute_mvp(proj, view)
            
            p0 = self._world_to_screen(self._gizmo_drag_start_offset, mvp, W, H)
            if not p0: return
            
            # Simple projected dragging
            sens = 0.05
            if axis == 'x':
                self._selected_obj.pos_x = self._gizmo_drag_start_offset[0] + dx * sens
            elif axis == 'y':
                self._selected_obj.pos_y = self._gizmo_drag_start_offset[1] - dy * sens
            elif axis == 'z':
                self._selected_obj.pos_z = self._gizmo_drag_start_offset[2] + dx * sens
            elif axis == 'free':
                self._selected_obj.pos_x = self._gizmo_drag_start_offset[0] + dx * sens
                self._selected_obj.pos_y = self._gizmo_drag_start_offset[1] - dy * sens

        elif self._gizmo_mode == 'scale':
            s = self._gizmo_drag_start_scale + (dx - dy) * 0.01
            self._selected_obj.scale = max(0.01, s)

        elif self._gizmo_mode == 'rotate':
            angle = dx * 0.5
            # Simplified: rotate around Y for now
            self._selected_obj.rot_y += angle # simplified for this turn
        
        # Notify transform change
        self.object_moved.emit(self._selected_obj)



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

        # ── Ocean Physics (Buoyancy & Tilting) ─────────────────────────────
        if self.cfg and getattr(self.cfg, 'ocean', None) and self.cfg.ocean.enabled:
            o = self.cfg.ocean
            # Use same relative time for CPU physics to stay in sync
            t = (time.time() % 3600.0)
            for obj in self.cfg.scene_objects:
                if getattr(obj, 'floating', False) and getattr(obj, 'visible', False):
                    try:
                        px = getattr(obj, 'pos_x', 0.0)
                        pz = getattr(obj, 'pos_z', 0.0)
                        
                        # Get wave height for bobbing
                        bob_intensity = getattr(obj, 'float_bob', 1.0)
                        h = ocean_physics.get_wave_height(
                            px, pz, t,
                            wind_speed=o.wind_speed,
                            wind_direction=o.wind_direction,
                            choppiness=o.choppiness,
                            wave_amplitude=o.wave_amplitude,
                            repetition_size=o.repetition_size,
                            band0_mul=o.band0_multiplier,
                            band1_mul=o.band1_multiplier,
                            chaos=o.chaos,
                            storm_intensity=o.storm_intensity
                        ) * bob_intensity
                        
                        target_y = o.level + h + getattr(obj, 'buoyancy_offset', 0.0)
                        
                        # Smooth bobbing
                        alpha = o.buoyancy * 0.15
                        obj.pos_y = obj.pos_y * (1.0 - alpha) + target_y * alpha

                        # Get surface normal for tilting
                        n = ocean_physics.get_surface_normal(
                            px, pz, t,
                            wind_speed=o.wind_speed,
                            wind_direction=o.wind_direction,
                            choppiness=o.choppiness,
                            wave_amplitude=o.wave_amplitude,
                            repetition_size=o.repetition_size,
                            band0_mul=o.band0_multiplier,
                            band1_mul=o.band1_multiplier
                        )
                        
                        # Apply tilt multiplier by lerping normal towards UP
                        tilt_strength = getattr(obj, 'float_tilt', 1.0)
                        # n starts as [nx, ny, nz]. We want to blend towards [0, 1, 0]
                        nx = n[0] * tilt_strength
                        ny = n[1] # Keep ny or boost it? Keep simple for now.
                        nz = n[2] * tilt_strength
                        # Re-normalize isn't strictly needed for atan2 but helps consistency
                        
                        # Set Pitch & Roll based on (potentially flattened) normal
                        obj.rot_x = math.degrees(math.atan2(nz, ny))
                        obj.rot_z = math.degrees(math.atan2(-nx, ny))
                        
                    except Exception as e:
                        print(f"[Physics] Buoyancy error: {e}")
            
            # Continuous update for waves
            self.update()
        
        # --- Weather Auto-refresh ---
        # Ensure rain/snow falls even if ocean is off and camera is static.
        if self.cfg and self.cfg.weather:
            wt = self.cfg.weather.type
            if wt in ["rain", "stormy", "snow"]:
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
            glDisable(GL_CULL_FACE)
            glClearColor(0.05, 0.07, 0.09, 1.0)

            # ── One-time GPU resource creation ──────────────────────────
            self._build_pbr_program()
            self._build_sky_program()
            self._build_gizmo_program()
            self._build_ocean_program()
            self._build_weather_program()
            self._build_pp_program()
            self._upload_sphere()
            self._upload_skybox()
            self._upload_grid()
            self._upload_gizmo()
            self._upload_ocean()
            self._upload_weather()
            
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

    def resizeGL(self, w, h):
        if not self._gl_ready: return
        self._setup_pp_fbo(w, h)

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

            if self._mesh_needs_upload and self._pending_uploads:
                to_upload = list(self._pending_uploads)
                self._pending_uploads.clear()
                for path in to_upload:
                    try:
                        self._do_upload_mesh(path)
                    except Exception as exc:
                        print(f"[GL] Mesh upload failed for {path}: {exc}")
                self._mesh_needs_upload = False

            # ── Pure rendering ────────────────────────────────────
            W, H = self.width(), self.height()

            # --- Reset GL State ...
            glDisable(GL_SCISSOR_TEST) 
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glDisable(GL_CULL_FACE)
            glDisable(GL_BLEND)
            
            # ── Capture to FBO ────────────────────────────────────
            use_fbo = False
            if self._gl_ready:
                fe_requested = self.cfg.post_process.fisheye_enabled if (self.cfg and self.cfg.post_process) else False
                
                # Try FBO if requested or if we already have it
                if fe_requested or (self._pp_fbo and glIsFramebuffer(self._pp_fbo)):
                    if not self._pp_fbo or not glIsFramebuffer(self._pp_fbo):
                        self._setup_pp_fbo(W, H)
                    
                    if self._pp_fbo and glIsFramebuffer(self._pp_fbo):
                        glBindFramebuffer(GL_FRAMEBUFFER, self._pp_fbo)
                        use_fbo = True
                
                if not use_fbo:
                    glBindFramebuffer(GL_FRAMEBUFFER, self.defaultFramebufferObject())
            
            glClearColor(0.22, 0.22, 0.22, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # ── Pure rendering ────────────────────────────────────
            W, H = self.width(), self.height()
            
            # --- Lightning / Thunder logic ---
            t_now = time.time()
            thunder_flash = 0.0
            if self.cfg and self.cfg.weather and self.cfg.weather.type == "stormy":
                # Random flash every 3-10 seconds
                if not hasattr(self, '_next_thunder'): self._next_thunder = t_now + 2.0
                if not hasattr(self, '_thunder_end'): self._thunder_end = 0.0
                
                if t_now > self._next_thunder:
                    import random
                    dur = random.uniform(0.1, 0.4)
                    self._thunder_end = t_now + dur
                    self._next_thunder = t_now + random.uniform(3.0, 10.0)
                
                if t_now < self._thunder_end:
                    # Pulsing flash effect
                    thunder_flash = math.sin((self._thunder_end - t_now) * 20.0) * 0.5 + 0.5
                    thunder_flash *= self.cfg.weather.intensity

            # Compute weather-based lighting
            def _pmix(a, b, t): return a * (1.0 - t) + b * t
            base_light = 3.5
            if self.cfg and self.cfg.weather:
                wt = self.cfg.weather.type
                wi = self.cfg.weather.intensity
                if wt == "cloudy": base_light *= _pmix(1.0, 0.6, wi)
                elif wt == "rain": base_light *= _pmix(1.0, 0.4, wi)
                elif wt == "stormy": base_light *= _pmix(1.0, 0.2, wi)
                elif wt == "snow": base_light *= _pmix(1.0, 0.8, wi)
                elif wt == "foggy": base_light *= _pmix(1.0, 0.5, wi)
            
            # Add lightning flash to base light
            base_light += thunder_flash * 6.0
            
            # --- Reset GL State ...
            glDisable(GL_SCISSOR_TEST) 
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            glDisable(GL_CULL_FACE)
            glDisable(GL_BLEND)
            glClearColor(0.22, 0.22, 0.22, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            aspect = W / max(H, 1)
            # Base FOV from config
            base_fov = self.cfg.fov_y if (self.cfg and hasattr(self.cfg, 'fov_y')) else 60.0
            
            # If fisheye is enabled, we MUST render a wider field of view into the FBO
            # so that the distortion shader has more content to "wrap" in.
            fe_active = self.cfg.post_process.fisheye_enabled if (self.cfg and self.cfg.post_process) else False
            fe_strength = self.cfg.post_process.fisheye_strength if (self.cfg and self.cfg.post_process) else 0.5
            
            fov = base_fov
            if fe_active:
                # Boost FOV significantly for the fisheye capture (e.g. up to 140 degrees)
                fov = base_fov + (140.0 - base_fov) * fe_strength
                fov = min(fov, 170.0) # Cap at 170 to avoid extreme inversion artifacts
                
            # --- View matrices ---
            proj   = _mat4_perspective(fov, aspect, 0.1, 5000.0)
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
                    
                    # --- Weather Uniforms for Sky ---
                    if self.cfg and self.cfg.weather:
                        w = self.cfg.weather
                        w_types = {"clear":0, "cloudy":1, "rain":2, "stormy":3, "snow":4, "foggy":5}
                        self._set_i(sky_prog, "uWeatherType", w_types.get(w.type, 0))
                        self._set_f(sky_prog, "uWeatherIntensity", w.intensity)
                        self._set_f(sky_prog, "uThunderFlash", thunder_flash)
                    try:
                        glActiveTexture(GL_TEXTURE0)
                        glBindTexture(GL_TEXTURE_2D, int(env_tex))
                        self._set_i(sky_prog, "uEnvMap", 0)
                        glBindVertexArray(v_id)
                        glDrawElements(GL_TRIANGLES, self._sky_idx_count, GL_UNSIGNED_INT, None)
                    except Exception as e:
                        print(f"[GL] Skybox render error: {e}")
                    sky_prog.release()

            # ── Grid floor ─────────────────────────────────────────────────────
            if getattr(self, '_gizmo_prog', None) is not None:
                # Hide the floor grid when ocean is on to avoid Z-fighting/grid artifacts
                if not (self.cfg and self.cfg.ocean.enabled):
                    self._draw_grid_floor(proj, view)

            # ── PBR scene objects ──────────────────────────────────────────────────
            if pbr_prog is not None:
                pbr_prog.bind()
                glDepthMask(GL_TRUE) # Ensure objects write to depth buffer
                glEnable(GL_DEPTH_TEST)
                glDepthFunc(GL_LESS)
                self._set_v3(pbr_prog, "uCamPos", eye[0], eye[1], eye[2])
                self._set_f(pbr_prog, "uEnvStrength", self._env_strength)
                self._set_f(pbr_prog, "uLightIntensity", base_light)
                self._set_v3(pbr_prog, "uLightDir", 0.3, 0.8, 0.6) # Coming from front-right-top

                # --- Weather Uniforms ---
                if self.cfg and self.cfg.weather:
                    w = self.cfg.weather
                    w_types = {"clear":0, "cloudy":1, "rain":2, "stormy":3, "snow":4, "foggy":5}
                    self._set_i(pbr_prog, "uWeatherType", w_types.get(w.type, 0))
                    self._set_f(pbr_prog, "uWeatherIntensity", w.intensity)
                    self._set_f(pbr_prog, "uFogDensity", w.fog_density)
                else:
                    self._set_i(pbr_prog, "uWeatherType", 0)
                    self._set_f(pbr_prog, "uWeatherIntensity", 0.0)
                    self._set_f(pbr_prog, "uFogDensity", 0.0)

                # Draw each object from the scene configuration
                for obj in getattr(self, 'cfg', None).scene_objects if getattr(self, 'cfg', None) else []:
                    if not obj.visible: continue

                    mesh_path = getattr(obj.config, 'mesh_path', '')
                    if mesh_path and mesh_path not in self._mesh_cache:
                        # Lazy load new meshes discovered in the scene config
                        self.load_obj_mesh(mesh_path)
                        self._mesh_needs_upload = True

                    mesh = self._mesh_cache.get(mesh_path)
                    v_id = mesh["vao"] if mesh else (int(self._sphere_vao) if self._sphere_vao else 0)
                    idx_count = mesh["count"] if mesh else self._sphere_idx_count
                    albedo_tex = mesh["albedo"] if mesh else self._albedo_tex

                    if v_id is not None:
                        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE if self._wireframe else GL_FILL)
                        
                        # Uniforms
                        # Use a neutral light gray if no albedo provided
                        self._set_v3(pbr_prog, "uAlbedo", 0.9, 0.9, 0.9)
                        self._set_f(pbr_prog, "uMetallic", getattr(obj, 'metallic', 0.0))
                        self._set_f(pbr_prog, "uRoughness", getattr(obj, 'roughness', 0.8))

                        # Textures
                        # ALWAYS bind env map to Unit 0 for every object
                        if env_tex is not None:
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D, int(env_tex))
                            self._set_i(pbr_prog, "uEnvMap", 0)

                        glActiveTexture(GL_TEXTURE0 + 1) # Unit 1
                        # Use obj.config.tex_albedo if set, else fallback to mesh default
                        a_tex = None
                        if obj.config.tex_albedo:
                            a_tex = self._get_or_create_texture(obj.config.tex_albedo)
                        if not a_tex: a_tex = albedo_tex
                        
                        if a_tex:
                            glBindTexture(GL_TEXTURE_2D, a_tex)
                            self._set_i(pbr_prog, "uAlbedoMap", 1)
                            self._set_i(pbr_prog, "uHasAlbedoMap", 1)
                        else:
                            self._set_i(pbr_prog, "uHasAlbedoMap", 0)

                        # Normal map binding
                        glActiveTexture(GL_TEXTURE0 + 2) # Unit 2
                        n_tex = self._get_or_create_texture(obj.config.tex_normal)
                        if n_tex:
                            glBindTexture(GL_TEXTURE_2D, n_tex)
                            self._set_i(pbr_prog, "uNormalMap", 2)
                            self._set_i(pbr_prog, "uHasNormalMap", 1)
                        else:
                            self._set_i(pbr_prog, "uHasNormalMap", 0)


                        # Model matrix
                        model = _mat4_translate(obj.pos_x, obj.pos_y, obj.pos_z)
                        model = _mat4_mul(model, _mat4_rot_z(obj.rot_z))
                        model = _mat4_mul(model, _mat4_rot_y(obj.rot_y))
                        model = _mat4_mul(model, _mat4_rot_x(obj.rot_x))
                        model = _mat4_mul(model, _mat4_scale(obj.scale))
                        self._set_mat4(pbr_prog, "uModel", model)
                        self._set_mat4(pbr_prog, "uView",  view)
                        self._set_mat4(pbr_prog, "uProj",  proj)

                        normal_m = _mat3_inverse_transpose(model)
                        self._set_mat3(pbr_prog, "uNormalMat", normal_m)

                        glBindVertexArray(v_id)
                        glDrawElements(GL_TRIANGLES, idx_count, GL_UNSIGNED_INT, None)

                glBindVertexArray(0)
                pbr_prog.release()

            # Moved after PBR for correct alpha blending
            glDepthMask(GL_FALSE) # Don't write depth for the ocean
            self._draw_ocean(proj, view, eye, base_light)
            glDepthMask(GL_TRUE)
            
            # ── Transform Gizmo (selected object)
            if getattr(self, '_gizmo_prog', None) is not None and self._selected_obj:
                # The gizmo needs its own draw logic which I'll update to use _selected_obj
                self._draw_gizmo(proj, view)

            # ── Weather Overlay (Precipitation) ───────────────────
            self._draw_weather(W, H, thunder_flash, view)

            # ── Orientation Gizmo (LAST — uses QPainter which breaks GL state)
            if getattr(self, '_gizmo_prog', None) is not None:
                self._draw_orientation_gizmo(proj, view)

            # ── Final Blit with Post-Process ──────────────────────
            # Bind back to the default FBO of the widget (critical for Qt)
            target_fbo = self.defaultFramebufferObject()
            glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
            
            # ONLY clear if we are about to blit from an offscreen FBO
            if use_fbo:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if self._pp_prog and self._weather_vao and self._pp_tex:
                glDisable(GL_DEPTH_TEST)
                self._pp_prog.bind()
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._pp_tex)
                self._set_i(self._pp_prog, "uScreenTex", 0)
                
                fe = self.cfg.post_process.fisheye_enabled if (self.cfg and self.cfg.post_process) else False
                fs = self.cfg.post_process.fisheye_strength if (self.cfg and self.cfg.post_process) else 0.5
                
                self._set_i(self._pp_prog, "uFisheyeEnabled", 1 if fe else 0)
                self._set_f(self._pp_prog, "uFisheyeStrength", fs)
                self._set_f(self._pp_prog, "uAspect", W / max(H, 1.0))
                
                glBindVertexArray(self._weather_vao) # reuse full-screen quad
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
                glBindVertexArray(0)
                self._pp_prog.release()
                glEnable(GL_DEPTH_TEST)
            else:
                # If post-process failed, at least we should have something on screen?
                # Actually, if we bound FBO earlier, the scene is in self._pp_tex.
                # If blit fails, we might see nothing.
                if self._pp_tex:
                    # Try a simple unshaded blit if possible, or just log
                    pass

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

    def _build_pp_program(self):
        try:
            prog = QOpenGLShaderProgram(self)
            ok1 = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, _PP_VERT)
            ok2 = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _PP_FRAG)
            if not ok1 or not ok2:
                print(f"[GL] Post-process shader compile failed: {prog.log()}")
                return
            
            if not prog.link():
                print(f"[GL] Post-process shader link failed: {prog.log()}")
                return
                
            self._pp_prog = prog
            print("[GL] Post-process program built OK")
        except Exception as e:
            print(f"[GL] _build_pp_program error: {e}")

    def _setup_pp_fbo(self, w, h):
        from OpenGL.GL import (
            glGenFramebuffers, glBindFramebuffer, glGenTextures, glBindTexture, 
            GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_LINEAR, glTexParameteri,
            GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, glFramebufferTexture2D,
            GL_COLOR_ATTACHMENT0, glGenRenderbuffers, glBindRenderbuffer,
            glRenderbufferStorage, GL_DEPTH24_STENCIL8, glFramebufferRenderbuffer,
            GL_DEPTH_STENCIL_ATTACHMENT, GL_FRAMEBUFFER, GL_RENDERBUFFER,
            glDeleteFramebuffers, glDeleteTextures, glDeleteRenderbuffers, GL_MAX_TEXTURE_SIZE
        )
        
        # Cleanup safely
        if self._pp_fbo:
            try:
                if glIsFramebuffer(self._pp_fbo):
                    glDeleteFramebuffers(1, [int(self._pp_fbo)])
            except: pass
        if self._pp_tex:
            try:
                glDeleteTextures(1, [int(self._pp_tex)])
            except: pass
        if self._pp_depth:
            try:
                if glIsRenderbuffer(self._pp_depth):
                    glDeleteRenderbuffers(1, [int(self._pp_depth)])
            except: pass

        self._pp_fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, self._pp_fbo)

        self._pp_tex = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self._pp_tex)
        # Use GL_RGBA for better compatibility with different drivers
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._pp_tex, 0)

        self._pp_depth = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self._pp_depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self._pp_depth)

        # Check status
        from OpenGL.GL import glCheckFramebufferStatus, GL_FRAMEBUFFER_COMPLETE
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"[GL] FBO incomplete: {status}")
            self._pp_fbo = None

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

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
        if not self._selected_obj or self._gizmo_prog is None:
            return
        
        ox, oy, oz = self._selected_obj.pos_x, self._selected_obj.pos_y, self._selected_obj.pos_z

        # Extract camera right & up from the view matrix (column-major)
        import math
        dist = math.sqrt(sum((self._camera_pos()[i]-ox)**2 for i in range(3)))
        size = 0.5 * (dist / 10.0) # approx Unity behavior
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
    def _billboard_tri(cx, cy, cz, axis_dir, cam_r, cam_up, size, col):
        """Create a camera-facing triangle pointing along axis_dir at (cx,cy,cz)."""
        # Compute perpendicular to axis in camera plane
        # Cross axis_dir with cam_forward to get a screen-space perp
        import math
        ax, ay, az = axis_dir
        # Use cam_right and cam_up to build two perp offsets
        # Project cam_r and cam_up onto the plane perpendicular to axis
        # Simple approach: just use cam_up and cam_right scaled
        p1 = [cx - cam_r[i]*size + ax*size*1.5 for i in range(3)]
        p2 = [cx + cam_r[i]*size + ax*size*1.5 for i in range(3)]
        p3 = [cx + ax*size*3 for i in range(3)]
        # Better: compute the triangle tip and two base corners
        tip = [cx + ax*size*2.5, cy + ay*size*2.5, cz + az*size*2.5]
        b1 = [cx - cam_r[0]*size - cam_up[0]*size*0.3,
              cy - cam_r[1]*size - cam_up[1]*size*0.3,
              cz - cam_r[2]*size - cam_up[2]*size*0.3]
        b2 = [cx + cam_r[0]*size + cam_up[0]*size*0.3,
              cy + cam_r[1]*size + cam_up[1]*size*0.3,
              cz + cam_r[2]*size + cam_up[2]*size*0.3]
        return [*b1, *col, *tip, *col, *b2, *col]

    # ── Move Gizmo: arrows with camera-facing cone tips ────────────────────
    def _gizmo_translate_verts(self, ox, oy, oz, cam_r, cam_up):
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
            xT, oy-cam_up[1]*tip-cam_r[1]*tip, oz-cam_up[2]*tip-cam_r[2]*tip, *rx,
            xT+tip*2, oy, oz, *rx,
            xT, oy+cam_up[1]*tip+cam_r[1]*tip, oz+cam_up[2]*tip+cam_r[2]*tip, *rx,
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
            ox-cam_r[0]*tip, oy-cam_up[1]*tip, zT, *bx,
            ox, oy, zT+tip*2, *bx,
            ox+cam_r[0]*tip, oy+cam_up[1]*tip, zT, *bx,
        ]

        # Center box (camera-facing quad for free movement)
        bs = 0.07  # box half-size
        wc = [0.9, 0.9, 0.3]  # yellow/white color
        # Two triangles forming a small camera-facing square
        cr, cu = cam_r, cam_up
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

    def _build_ocean_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog:
            ok_v = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _OCEAN_VERT)
            ok_f = prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _OCEAN_FRAG)
            if not ok_v:
                print(f"[GL] Ocean VERT compile error:\n{prog.log()}")
            if not ok_f:
                print(f"[GL] Ocean FRAG compile error:\n{prog.log()}")
            if not prog.link():
                print(f"[GL] Ocean Shader Link Error: {prog.log()}")
            else:
                print("[GL] Ocean shader compiled & linked OK")
            self._ocean_prog = prog

    def _upload_ocean(self):
        """Build a high-resolution displacement grid for the ocean using NumPy for speed."""
        if not HAS_OPENGL: return
        res = 256 # Balanced resolution for performance vs detail
        size = 500.0 
        
        # Create vertex grid (x, y, z, u, v)
        x = np.linspace(-0.5, 0.5, res + 1, dtype=np.float32) * size
        z = np.linspace(-0.5, 0.5, res + 1, dtype=np.float32) * size
        xv, zv = np.meshgrid(x, z)
        
        u = np.linspace(0, 1, res + 1, dtype=np.float32)
        v = np.linspace(0, 1, res + 1, dtype=np.float32)
        uv, vv = np.meshgrid(u, v)
        
        # Pack into [xx, 0, zz, u, v]
        vertices = np.zeros((res + 1, res + 1, 5), dtype=np.float32)
        vertices[..., 0] = xv
        vertices[..., 1] = 0.0
        vertices[..., 2] = zv
        vertices[..., 3] = uv
        vertices[..., 4] = vv
        
        # Create indices
        grid_indices = np.arange((res + 1) * (res + 1), dtype=np.uint32).reshape(res + 1, res + 1)
        quad_a = grid_indices[:-1, :-1].ravel()
        quad_b = grid_indices[:-1, 1:].ravel()
        quad_c = grid_indices[1:, :-1].ravel()
        quad_d = grid_indices[1:, 1:].ravel()
        
        indices = np.stack([quad_a, quad_b, quad_d, quad_a, quad_d, quad_c], axis=-1).ravel()
        
        self._ocean_idx_count = len(indices)
        data = vertices.ravel()
        idx_data = indices

        self._ocean_vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._ocean_vao)
        self._ocean_vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._ocean_vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        self._ocean_ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ocean_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

    def _draw_ocean(self, proj, view, eye, base_light):
        """Render the ocean surface if enabled in SceneConfig."""
        if not self.cfg or not self.cfg.ocean.enabled:
            return
        if not self._ocean_prog or not self._ocean_vao:
            return

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Ensure we don't inherit wireframe mode from the scene objects
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        self._ocean_prog.bind()
        self._set_mat4(self._ocean_prog, "uProj", proj)
        self._set_mat4(self._ocean_prog, "uView", view)
        
        # New Unity-style eye/camera handling
        if isinstance(eye, (list, tuple, np.ndarray)):
            self._set_v3(self._ocean_prog, "uCamPos", eye[0], eye[1], eye[2])
        else: # QPointF or alike
            self._set_v3(self._ocean_prog, "uCamPos", eye.x(), eye.y(), eye.z())
        
        o = self.cfg.ocean
        # Use relative time (modulo) to maintain floating point precision in shaders.
        # time.time() is too large (~1.7B) and causes jitter/flatness in GLSL.
        rel_time = time.time() % 3600.0
        self._set_f(self._ocean_prog, "uTime",          rel_time * o.time_multiplier)
        self._set_f(self._ocean_prog, "uRepetitionSize", o.repetition_size)
        self._set_f(self._ocean_prog, "uLevel",         o.level)
        self._set_f(self._ocean_prog, "uWindSpeed",     o.wind_speed)
        self._set_f(self._ocean_prog, "uWindDirection", o.wind_direction)
        self._set_f(self._ocean_prog, "uChoppiness",    o.choppiness)
        self._set_f(self._ocean_prog, "uChaos",         o.chaos)
        self._set_f(self._ocean_prog, "uWaveAmplitude",  o.wave_amplitude)
        
        self._set_f(self._ocean_prog, "uBand0Multiplier", o.band0_multiplier)
        self._set_f(self._ocean_prog, "uBand1Multiplier", o.band1_multiplier)
        
        # Currents
        self._set_f(self._ocean_prog, "uCurrentSpeed",       o.current_speed)
        self._set_f(self._ocean_prog, "uCurrentOrientation", o.current_orientation)
        
        # Ripples
        self._set_i(self._ocean_prog, "uRipplesEnabled",   1 if o.ripples_enabled else 0)
        self._set_f(self._ocean_prog, "uRipplesWindSpeed", o.ripples_wind_speed)
        self._set_f(self._ocean_prog, "uRipplesWindDir",   o.ripples_wind_dir)
        self._set_f(self._ocean_prog, "uRipplesChaos",     o.ripples_chaos)

        # Colors & Material
        self._set_v3(self._ocean_prog, "uRefractionColor", o.refraction_color[0], o.refraction_color[1], o.refraction_color[2])
        self._set_v3(self._ocean_prog, "uScatteringColor", o.scattering_color[0], o.scattering_color[1], o.scattering_color[2])
        self._set_f(self._ocean_prog,  "uAbsorptionDistance", o.absorption_distance)
        self._set_f(self._ocean_prog,  "uAmbientScattering",  o.ambient_scattering)
        self._set_f(self._ocean_prog,  "uHeightScattering",   o.height_scattering)
        self._set_f(self._ocean_prog,  "uDisplacementScattering", o.displacement_scattering)
        self._set_f(self._ocean_prog,  "uDirectLightTipScattering", o.direct_light_tip_scattering)
        self._set_f(self._ocean_prog,  "uDirectLightBodyScattering", o.direct_light_body_scattering)
        
        self._set_f(self._ocean_prog,  "uSmoothness",         o.smoothness)
        self._set_f(self._ocean_prog,  "uTransparency",       o.transparency)
        self._set_f(self._ocean_prog,  "uEnvStrength",        o.reflection)
        
        # Caustics
        self._set_i(self._ocean_prog, "uCausticsEnabled",   1 if o.caustics_enabled else 0)
        self._set_f(self._ocean_prog, "uCausticsIntensity", o.caustics_intensity)
        
        # Foam
        self._set_i(self._ocean_prog, "uFoamEnabled", 1 if o.foam_enabled else 0)
        self._set_f(self._ocean_prog, "uFoamAmount",  o.foam_amount)

        # --- Weather Uniforms ---
        if self.cfg and self.cfg.weather:
            w = self.cfg.weather
            w_types = {"clear":0, "cloudy":1, "rain":2, "stormy":3, "snow":4, "foggy":5}
            self._set_i(self._ocean_prog, "uWeatherType", w_types.get(w.type, 0))
            self._set_f(self._ocean_prog, "uWeatherIntensity", w.intensity)
            self._set_f(self._ocean_prog, "uFogDensity", w.fog_density)
            
            # Boost uStorm if weather is stormy
            if w.type == "stormy":
                self._set_f(self._ocean_prog, "uStorm", max(o.storm_intensity, w.intensity))

        # Sun / directional light (matches the PBR scene light)
        self._set_v3(self._ocean_prog, "uLightDir", 0.3, 0.8, 0.6)
        self._set_f(self._ocean_prog, "uLightIntensity", base_light)
        
        # Always bind a valid texture to unit 0 — Core Profile makes sampling
        # from an unbound sampler undefined.  We keep a 1x1 black stub texture
        # for exactly this case and swap in the real HDRI if one is loaded.
        glActiveTexture(GL_TEXTURE0)
        env_tex = getattr(self, '_env_tex', None)
        # Use real HDRI if path is set, otherwise use procedural sky fallback
        if env_tex and self._hdri_path:
            glBindTexture(GL_TEXTURE_2D, int(env_tex))
            self._set_i(self._ocean_prog, "uHasEnvMap", 1)
        else:
            # Procedural fallback is better than sampling a 1x1 black stub
            if env_tex: glBindTexture(GL_TEXTURE_2D, int(env_tex))
            self._set_i(self._ocean_prog, "uHasEnvMap", 0)
            # Create a reusable stub if we don't have one yet
            stub = getattr(self, '_stub_env_tex', None)
            if stub is None:
                stub = int(glGenTextures(1))
                glBindTexture(GL_TEXTURE_2D, stub)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0,
                             GL_RGB, GL_UNSIGNED_BYTE, b'\x00\x00\x00')
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                self._stub_env_tex = stub
            glBindTexture(GL_TEXTURE_2D, stub)
            self._set_i(self._ocean_prog, "uHasEnvMap", 0)
        self._set_i(self._ocean_prog, "uEnvMap", 0)

        glBindVertexArray(self._ocean_vao)
        glDrawElements(GL_TRIANGLES, self._ocean_idx_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        self._ocean_prog.release()
        glDisable(GL_BLEND)

    def _build_sky_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog is not None:
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _SKY_VERT)
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _SKY_FRAG)
            if not prog.link():
                print(f"[GL] Sky Shader Link Error: {prog.log()}")
            self._sky_prog = prog

    def _build_weather_program(self):
        prog = QOpenGLShaderProgram(self)
        if prog:
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex,   _WEATHER_VERT)
            prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, _WEATHER_FRAG)
            if not prog.link():
                print(f"[GL] Weather Shader Link Error: {prog.log()}")
            self._weather_prog = prog

    def _upload_weather(self):
        if not HAS_OPENGL: return
        self._weather_vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._weather_vao)
        
        # Simple fullscreen quad
        verts = np.array([
            -1, -1, 0,
             1, -1, 0,
             1,  1, 0,
            -1,  1, 0
        ], dtype=np.float32)
        
        vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        ebo = int(glGenBuffers(1))
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        self._weather_vbo = vbo
        self._weather_ebo = ebo

    def _draw_weather(self, W, H, thunder_flash, view_mat):
        if not self.cfg or not self.cfg.weather: return
        w = self.cfg.weather
        wt = {"clear":0, "cloudy":1, "rain":2, "stormy":3, "snow":4, "foggy":5}.get(w.type, 0)
        
        if wt not in [2, 3, 4] or not getattr(self, '_weather_prog', None): return
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        
        self._weather_prog.bind()
        self._set_f(self._weather_prog, "uTime", time.time() % 1000.0)
        self._set_i(self._weather_prog, "uWeatherType", wt)
        self._set_f(self._weather_prog, "uWeatherIntensity", w.intensity)
        self._set_f(self._weather_prog, "uAspect", W / max(H, 1.0))
        self._set_f(self._weather_prog, "uThunderFlash", thunder_flash)

        # Extract camera axes directly from View Matrix for stability
        # View Matrix (Column Major): [ R.x, U.x, -F.x, 0,  R.y, U.y, -F.y, 0, ... ]
        # In Row-Major/List format: 
        # m[0,4,8] = Right, m[1,5,9] = Up, m[2,6,10] = -Forward
        m = view_mat
        right = [m[0], m[4], m[8]]
        up    = [m[1], m[5], m[9]]
        fwd   = [-m[2], -m[6], -m[10]]
        
        eye = self._camera_pos()
        self._set_v3(self._weather_prog, "uCamPos",   eye[0],   eye[1],   eye[2])
        self._set_v3(self._weather_prog, "uCamRight", right[0], right[1], right[2])
        self._set_v3(self._weather_prog, "uCamUp",    up[0],    up[1],    up[2])
        self._set_v3(self._weather_prog, "uCamFwd",   fwd[0],   fwd[1],   fwd[2])
        
        glBindVertexArray(self._weather_vao)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        self._weather_prog.release()
        
        glEnable(GL_DEPTH_TEST)

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

    def _upload_custom_mesh(self, verts, faces, normals=None, uvs=None, albedo_path=""):
        """Upload custom mesh data into interleaved VBO and bind albedo texture if provided."""
        if not HAS_OPENGL or not self._gl_ready:
            return
        import numpy as np
        from OpenGL.GL import (
            glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
            glGenerateMipmap, GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE,
            GL_LINEAR, GL_CLAMP_TO_EDGE, glActiveTexture, GL_TEXTURE0,
            glIsVertexArray, glVertexAttribPointer, glEnableVertexAttribArray,
            GL_FLOAT,
        )
        from PySide6.QtGui import QImage

        # Interleaved layout: [pos(3), norm(3), uv(2)] per vertex
        num_v = len(verts)
        data = np.zeros((num_v, 8), dtype=np.float32)
        data[:, 0:3] = verts
        if normals is not None and len(normals) == num_v:
            data[:, 3:6] = normals
        else:
            data[:, 4] = 1.0  # fallback up-normal
        if uvs is not None and len(uvs) == num_v:
            data[:, 6:8] = uvs

        flat_data = np.ascontiguousarray(data)
        self._sphere_idx_count = faces.size
        idx_data = np.ascontiguousarray(faces.astype(np.uint32).flatten())

        v_id = int(self._sphere_vao) if self._sphere_vao is not None else -1
        print(f"[GL] _upload_custom_mesh: vao={v_id} verts={num_v} idxs={self._sphere_idx_count}")
        if v_id >= 0 and glIsVertexArray(v_id):
            glBindVertexArray(v_id)

            glBindBuffer(GL_ARRAY_BUFFER, int(self._sphere_vbo))
            glBufferData(GL_ARRAY_BUFFER, flat_data.nbytes, flat_data, GL_STATIC_DRAW)

            # Re-declare attribute pointers after each upload
            stride = 8 * 4  # 8 floats × 4 bytes
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
            glEnableVertexAttribArray(2)

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(self._sphere_ebo))
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_data.nbytes, idx_data, GL_STATIC_DRAW)

            # Load albedo texture if path provided
            if albedo_path and os.path.exists(albedo_path):
                img = QImage(albedo_path).convertToFormat(QImage.Format.Format_RGB888)
                w, h = img.width(), img.height()
                img_data = img.bits().asstring(w * h * 3)
                tex_id = int(glGenTextures(1))
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glGenerateMipmap(GL_TEXTURE_2D)
                self._albedo_tex = tex_id
                print(f"[GL] Albedo texture loaded ({w}x{h}): {os.path.basename(albedo_path)}")
            else:
                self._albedo_tex = None

            glBindVertexArray(0)
            print(f"[GL] _upload_custom_mesh complete")
        else:
            print(f"[GL] _upload_custom_mesh SKIPPED — VAO {v_id} not valid")

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
    camera_active    = Signal()   # user is dragging/scrolling
    camera_idle      = Signal(float, float, float)   # user stopped moving for 400ms, carries (dist_world, elev, azim)
    generate_clicked = Signal()
    pause_clicked    = Signal()
    stop_clicked     = Signal()
    object_selected  = Signal(object) # NEW: 3D pick result
    object_moved     = Signal(object) # NEW: gizmo result
    texture_discovered = Signal(str, str) # NEW: mesh_path, texture_path

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

        # ── GL + Inspector in a horizontal split ──────────────────────
        self._gl_insp_row = QWidget()
        gi_layout = QHBoxLayout(self._gl_insp_row)
        gi_layout.setContentsMargins(0, 0, 0, 0)
        gi_layout.setSpacing(0)

        if HAS_OPENGL:
            self._gl = GL3DPreview(self._gl_insp_row)
            # Stop idle timer before forwarding signal
            def _on_gl_active():
                if hasattr(self, '_idle_timer'):
                    self._idle_timer.stop()
                self.camera_active.emit()
            self._gl.camera_active.connect(self.camera_active.emit)

            self._gl.object_selected.connect(self.object_selected.emit)
            self._gl.object_moved.connect(self.object_moved.emit)
            self._gl.texture_discovered.connect(self._on_texture_discovered)
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

    def _on_texture_discovered(self, mesh_path: str, tex_path: str):
        """Internal: bubble up auto-discovered textures to MainWindow."""
        self.texture_discovered.emit(mesh_path, tex_path)

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

    def set_scene_config(self, cfg):
        """Pass full scene state to the GL widget."""
        if self._gl:
            self._gl.set_scene_config(cfg)

    def set_selected_object(self, obj):
        """Sync selected object with the GL widget."""
        if self._gl:
            self._gl.set_selected_object(obj)

    def load_mesh(self, path: str):
        """Forward mesh path to GL widget for caching/loading."""
        if self._gl:
            self._gl.load_obj_mesh(path)

