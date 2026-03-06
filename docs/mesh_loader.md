# Mesh Loader — `app/engine/mesh_loader.py`

## Overview

The unified mesh loader provides a single-function API (`load_mesh`) for importing 3D assets into Skunkworks. It returns a standardized `LoadedMesh` dataclass regardless of input format.

## Supported Formats

| Extension | Loader   | Notes                                      |
|-----------|----------|--------------------------------------------|
| `.obj`    | trimesh  | Wavefront OBJ — most widely supported      |
| `.gltf`   | trimesh  | glTF 2.0 JSON                              |
| `.glb`    | trimesh  | glTF 2.0 binary (self-contained)           |
| `.ply`    | trimesh  | Stanford polygon format                    |
| `.stl`    | trimesh  | Stereolithography (no UVs/textures)        |
| `.dae`    | trimesh  | Collada                                    |
| `.fbx`    | pyassimp | Autodesk FBX — requires native Assimp DLL  |

## Dependencies

```bash
pip install trimesh pyassimp
```

### Windows — Assimp native DLL (FBX only)

`pyassimp` is a thin Python binding around the native Assimp C++ library. On Windows you must place the Assimp DLL where Python can find it:

1. Download the latest **Assimp release** from https://github.com/assimp/assimp/releases (e.g. `assimp-5.4.x-win-amd64.zip`).
2. Extract `assimp-vc143-mt.dll` (or similar) and place it in the **project root** (`D:\Jaymei\Skunkworks\`) or somewhere on your `PATH`.
3. If you only work with OBJ/GLTF/PLY/STL/DAE, the Assimp DLL is **not required** — trimesh handles those natively.

## Usage

```python
from app.engine.mesh_loader import load_mesh, LoadedMesh

mesh: LoadedMesh = load_mesh("path/to/model.obj")

print(mesh.vertices.shape)   # (N, 3)  float32
print(mesh.normals.shape)    # (N, 3)  float32
print(mesh.uvs.shape)        # (N, 2)  float32
print(mesh.indices.shape)    # (M,)    uint32
print(mesh.tex_albedo)       # absolute path or ""
```

## `LoadedMesh` Dataclass Fields

| Field          | Type            | Description                                      |
|----------------|-----------------|--------------------------------------------------|
| `name`         | `str`           | Derived from the filename                        |
| `vertices`     | `ndarray(N,3)`  | Vertex positions (float32)                       |
| `normals`      | `ndarray(N,3)`  | Per-vertex normals (float32)                     |
| `uvs`          | `ndarray(N,2)`  | Texture coordinates (float32, zeros if missing)  |
| `indices`      | `ndarray(M,)`   | Triangle face indices (uint32, flattened)         |
| `tex_albedo`   | `str`           | Absolute path to albedo/diffuse texture          |
| `tex_normal`   | `str`           | Absolute path to normal map                      |
| `tex_roughness`| `str`           | Absolute path to roughness map                   |
| `tex_metallic` | `str`           | Absolute path to metallic map                    |
| `center`       | `ndarray(3,)`   | Bounding-box center of the raw mesh              |
| `scale_hint`   | `float`         | Scale factor to normalize object to ~1 unit tall |

## Adding New Texture Types

1. Add a new `str` field to `LoadedMesh` (e.g. `tex_emissive: str = ""`).
2. In `_load_trimesh`, look for the material property in `mesh.visual.material` and extract the path.
3. In `_load_assimp`, add the corresponding Assimp texture key (e.g. `("file", 5)` for emissive) and resolve the path relative to the mesh file.
4. Update `_upload_custom_mesh` in `viewport.py` to bind the new texture to a GL texture unit.

## Loader Priority

```
load_mesh(path)
  ├─ ext != .fbx → trimesh (fast, pure Python)
  └─ ext == .fbx → pyassimp (needs native DLL)
```

If trimesh fails for a non-FBX file, the error is raised immediately. For FBX, pyassimp is the only path, and a clear error message is shown if the DLL is missing.
