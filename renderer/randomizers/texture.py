import torch
import random
from .base import BaseRandomizer
from pytorch3d.renderer import TexturesVertex

class TextureRandomizer(BaseRandomizer):
    """Randomizes the colors/textures of the mesh."""
    
    def apply(self, renderer, mesh=None, **kwargs):
        if mesh is None:
            return
            
        num_verts = mesh.verts_packed().shape[0]
        # Generate random vertex colors
        random_colors = torch.rand((1, num_verts, 3), device=renderer.device)
        
        # Replace textures
        mesh.textures = TexturesVertex(verts_features=random_colors)
