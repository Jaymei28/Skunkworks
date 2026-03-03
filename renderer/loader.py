import torch
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import os

class MeshLoader:
    """Helper class to load 3D models into PyTorch3D Meshes."""
    _cache = {}  # (file_path, device) -> Mesh
    
    @staticmethod
    def load(file_path, device="cpu"):
        """Loads a mesh from a file path."""
        cache_key = (file_path, str(device))
        if cache_key in MeshLoader._cache:
            return MeshLoader._cache[cache_key]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Mesh file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".obj":
            mesh = load_objs_as_meshes([file_path], device=device)
        elif ext == ".ply":
            verts, faces = load_ply(file_path)
            # Create a Meshes object for PLY
            mesh = Meshes(verts=[verts], faces=[faces]).to(device)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Fallback for meshes without textures
        if mesh.textures is None:
            # Add default uniform white texture
            verts_rgb = torch.ones_like(mesh.verts_padded()) # (N, V, 3)
            mesh.textures = TexturesVertex(verts_features=verts_rgb)
            
        MeshLoader._cache[cache_key] = mesh
        return mesh
