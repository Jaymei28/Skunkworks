"""
engine/mesh_loader.py
=====================
Unified 3D asset loader supporting FBX, GLTF/GLB, OBJ, PLY.

Priority loader order:
  1. trimesh  — handles GLTF/GLB/OBJ/PLY natively (pure Python)
  2. pyassimp — handles FBX + everything else via Assimp native DLL

Returns a LoadedMesh dataclass with:
  - vertices, normals, uvs, indices  (numpy arrays)
  - textures dict: { 'albedo': path, 'normal': path, 'roughness': path, 'metallic': path }
  - name, center, scale_hint
"""

import os
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ── Assimp DLL discovery (Python 3.8+ requires os.add_dll_directory) ─────────
# pyassimp uses ctypes.util.find_library which on Python 3.8+ on Windows
# only searches os.add_dll_directory paths, not PATH.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "env", "Scripts"))

# Always add project root to both PATH and DLL search dirs
os.environ["PATH"] = _PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_PROJECT_ROOT)
    except Exception:
        pass
    try:
        os.add_dll_directory(_SCRIPTS_DIR)
    except Exception:
        pass


# ── LoadedMesh ────────────────────────────────────────────────────────────────

@dataclass
class LoadedMesh:
    """Standardized mesh data ready for uploading to GPU."""
    name:     str = "Unnamed"
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    normals:  np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    uvs:      np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    indices:  np.ndarray = field(default_factory=lambda: np.zeros((0,),   dtype=np.uint32))
    # PBR texture absolute paths (empty string = not present)
    tex_albedo:    str = ""
    tex_normal:    str = ""
    tex_roughness: str = ""
    tex_metallic:  str = ""
    # Derived stats
    center:     np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale_hint: float = 1.0   # suggested scale so object is ~1 unit tall


# ── Public API ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".fbx", ".obj", ".gltf", ".glb", ".ply", ".stl", ".dae"}


def load_mesh(path: str) -> LoadedMesh:
    """
    Load a 3D file and return a LoadedMesh.
    Raises RuntimeError if no loader can handle the file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    name = os.path.splitext(os.path.basename(path))[0]

    # Try trimesh first for ALL formats (including FBX — trimesh 4.x has
    # built-in FBX support via its own resolver, no native DLL needed).
    try:
        return _load_trimesh(path, name)
    except Exception:
        pass

    # Fall back to pyassimp (requires native Assimp DLL on Windows)
    try:
        return _load_assimp(path, name)
    except ImportError:
        raise RuntimeError(
            f"Could not load '{os.path.basename(path)}' with trimesh.\n"
            "pyassimp fallback also unavailable.\n"
            "Install: pip install pyassimp\n"
            "Also download assimp-5.x native DLL for Windows:\n"
            "  https://github.com/assimp/assimp/releases"
        )
    except Exception as e2:
        raise RuntimeError(f"Failed to load '{os.path.basename(path)}': {e2}")


# ── trimesh loader ────────────────────────────────────────────────────────────

def _load_trimesh(path: str, name: str) -> LoadedMesh:
    import trimesh

    scene_or_mesh = trimesh.load(path, force="mesh", process=True)

    # If it's a Scene with multiple meshes, concatenate them
    if hasattr(scene_or_mesh, "geometry"):
        meshes = list(scene_or_mesh.geometry.values())
        if not meshes:
            raise RuntimeError("No geometry found in file.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene_or_mesh

    # Ensure triangulated
    # mesh = mesh.triangulate()  # trimesh loads meshes already triangulated

    verts  = np.array(mesh.vertices,          dtype=np.float32)
    norms  = np.array(mesh.vertex_normals,     dtype=np.float32)
    idxs   = np.array(mesh.faces.flatten(),    dtype=np.uint32)

    # UVs — trimesh stores them in mesh.visual.uv if available
    uvs = np.zeros((len(verts), 2), dtype=np.float32)
    try:
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            raw_uv = np.array(mesh.visual.uv, dtype=np.float32)
            if raw_uv.shape[0] == len(verts):
                uvs = raw_uv
    except Exception:
        pass

    # Texture paths from material
    albedo_path = ""
    try:
        mat = mesh.visual.material
        if hasattr(mat, "image") and mat.image is not None:
            # PIL image — save temp PNG
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            mat.image.save(tmp.name)
            albedo_path = tmp.name
    except Exception:
        pass

    center, scale = _compute_normalization(verts)
    return LoadedMesh(
        name     = name,
        vertices = verts,
        normals  = norms,
        uvs      = uvs,
        indices  = idxs,
        tex_albedo  = albedo_path,
        center   = center,
        scale_hint = scale,
    )


# ── pyassimp loader ───────────────────────────────────────────────────────────

def _load_assimp(path: str, name: str) -> LoadedMesh:
    import pyassimp
    import pyassimp.postprocess as pp

    flags = (
        pp.aiProcess_Triangulate      |
        pp.aiProcess_GenSmoothNormals |
        pp.aiProcess_CalcTangentSpace |
        pp.aiProcess_JoinIdenticalVertices |
        pp.aiProcess_FlipUVs          |
        pp.aiProcess_PreTransformVertices
    )

    with pyassimp.load(path, processing=flags) as scene:
        if not scene.meshes:
            raise RuntimeError("No meshes found in file.")

        all_verts, all_norms, all_uvs, all_idxs = [], [], [], []
        tex_albedo = ""
        idx_offset = 0

        for mesh in scene.meshes:
            v = np.array(mesh.vertices, dtype=np.float32)
            n = np.array(mesh.normals,  dtype=np.float32) if len(mesh.normals) else np.zeros_like(v)

            # UVs: first channel
            if mesh.texturecoords is not None and len(mesh.texturecoords) > 0:
                uv = np.array(mesh.texturecoords[0][:, :2], dtype=np.float32)
            else:
                uv = np.zeros((len(v), 2), dtype=np.float32)

            # Faces: pyassimp 5.x + Assimp v6 returns numpy arrays directly (Nx3)
            # Older versions returned Face objects with .indices attribute
            raw_faces = mesh.faces
            if isinstance(raw_faces, np.ndarray) and raw_faces.ndim == 2:
                faces = raw_faces.astype(np.uint32)
            else:
                faces = np.array([[f.indices[0], f.indices[1], f.indices[2]] for f in raw_faces], dtype=np.uint32)

            all_verts.append(v)
            all_norms.append(n)
            all_uvs.append(uv)
            all_idxs.append(faces.flatten() + idx_offset)
            idx_offset += len(v)

            # Extract albedo texture path from material (wrapped in try/except
            # because pyassimp 5.x MaterialProperty can raise during dict())
            if not tex_albedo and mesh.materialindex < len(scene.materials):
                try:
                    mat = scene.materials[mesh.materialindex]
                    props = {k: v for k, v in mat.properties.items()}
                    tex_key = ("file", 1)  # aiTextureType_DIFFUSE
                    if tex_key in props:
                        raw_tex = props[tex_key]
                        if isinstance(raw_tex, bytes):
                            raw_tex = raw_tex.decode("utf-8", errors="ignore").strip("\x00")
                        # Resolve relative to mesh file
                        tex_abs = os.path.join(os.path.dirname(path), raw_tex)
                        if os.path.exists(tex_abs):
                            tex_albedo = tex_abs
                except Exception:
                    pass  # Material extraction failed, continue without texture

    verts = np.concatenate(all_verts, axis=0)
    norms = np.concatenate(all_norms, axis=0)
    uvs   = np.concatenate(all_uvs,   axis=0)
    idxs  = np.concatenate(all_idxs,  axis=0)

    # Fallback: scan directory for common texture filenames
    if not tex_albedo:
        tex_albedo = _find_texture_fallback(path)

    center, scale = _compute_normalization(verts)
    return LoadedMesh(
        name      = name,
        vertices  = verts,
        normals   = norms,
        uvs       = uvs,
        indices   = idxs,
        tex_albedo = tex_albedo,
        center    = center,
        scale_hint = scale,
    )


# ── Texture fallback discovery ────────────────────────────────────────────────

_TEX_PATTERNS = [
    "basecolor", "base_color", "diffuse", "albedo", "color", "texture",
]
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp"}

def _find_texture_fallback(mesh_path: str) -> str:
    """Scan the mesh's directory for a likely albedo texture file."""
    mesh_dir = os.path.dirname(mesh_path)
    if not os.path.isdir(mesh_dir):
        return ""
    for fname in os.listdir(mesh_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in _IMG_EXTS:
            continue
        lower = fname.lower()
        for pattern in _TEX_PATTERNS:
            if pattern in lower:
                found = os.path.join(mesh_dir, fname)
                print(f"[MeshLoader] Auto-discovered texture: {fname}")
                return found
    return ""


# ── Normalization helper ──────────────────────────────────────────────────────

def _compute_normalization(verts: np.ndarray):
    """Compute center and a scale hint (~unit height)."""
    if len(verts) == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    center = ((mn + mx) / 2).astype(np.float32)
    height = float(mx[1] - mn[1]) or 1.0
    return center, 1.0 / height

