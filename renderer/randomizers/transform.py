import torch
import random
import math
from .base import BaseRandomizer
from pytorch3d.transforms import Rotate, Translate, Scale

class TransformationRandomizer(BaseRandomizer):
    """Randomizes the rotation and scale of the mesh."""
    
    def __init__(self, scale_range=(0.8, 1.2), rotation_range=(0, 360), translation_range=(-10.0, 10.0)):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def apply(self, mesh, device="cpu"):
        # 1. Random Scale
        s = random.uniform(*self.scale_range)
        
        # 2. Random Rotation
        rx = random.uniform(*self.rotation_range)
        ry = random.uniform(*self.rotation_range)
        rz = random.uniform(*self.rotation_range)
        
        # 3. Random Translation
        tx = random.uniform(*self.translation_range)
        ty = random.uniform(*self.translation_range)
        tz = random.uniform(*self.translation_range)
        
        angles = torch.tensor([rx, ry, rz]) * (math.pi / 180.0)
        
        from pytorch3d.transforms import euler_angles_to_matrix, Rotate, Scale, Translate
        
        R = euler_angles_to_matrix(angles[None, ...], convention="XYZ")
        
        rot = Rotate(R=R, device=device)
        scale = Scale(s, device=device)
        transl = Translate(tx, ty, tz, device=device)
        
        # Combine transformations (Scale -> Rotate -> Translate)
        transform = scale.compose(rot).compose(transl)
        
        new_verts = transform.transform_points(mesh.verts_padded())
        
        return mesh.update_padded(new_verts)
