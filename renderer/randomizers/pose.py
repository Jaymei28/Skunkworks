import torch
import random
from .base import BaseRandomizer
from pytorch3d.renderer import look_at_view_transform

class PoseRandomizer(BaseRandomizer):
    """Randomizes the pose (rotation and translation) of the object or camera."""
    
    def __init__(self, dist_range=(2.0, 5.0), elev_range=(0, 90), azim_range=(0, 360)):
        self.dist_range = dist_range
        self.elev_range = elev_range
        self.azim_range = azim_range

    def apply(self, renderer, **kwargs):
        dist = random.uniform(*self.dist_range)
        elev = random.uniform(*self.elev_range)
        azim = random.uniform(*self.azim_range)
        
        R, T = look_at_view_transform(dist, elev, azim)
        renderer.cameras.R = R.to(renderer.device)
        renderer.cameras.T = T.to(renderer.device)
