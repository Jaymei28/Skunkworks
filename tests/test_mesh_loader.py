import os
import sys
# Ensure project root is in path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.abspath(os.path.join(_this_dir, '..'))
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.engine.mesh_loader import load_mesh

# Create a simple cube using trimesh and save to temporary OBJ
import trimesh
mesh = trimesh.creation.box(extents=(1,1,1))
temp_path = os.path.join(_this_dir, 'temp_cube.obj')
mesh.export(temp_path)

loaded = load_mesh(temp_path)
print('Loaded mesh name:', loaded.name)
print('Vertices shape:', loaded.vertices.shape)
print('Normals shape:', loaded.normals.shape)
print('UVs shape:', loaded.uvs.shape)
print('Indices shape:', loaded.indices.shape)
print('Albedo texture path:', loaded.tex_albedo)
# Cleanup
os.remove(temp_path)
