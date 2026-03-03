import torch
import random
from .base import BaseRandomizer
from pytorch3d.renderer import look_at_view_transform

class PoseRandomizer(BaseRandomizer):
    """Randomizes the pose (rotation and translation) of the object or camera."""
    
    def __init__(self, dist_range=(400.0, 800.0), elev_range=(0, 90), azim_range=(0, 360),
                 dist_min=None, dist_max=None, elev_min=None, elev_max=None,
                 azim_min=None, azim_max=None):
        # Accept both tuple-style and individual min/max kwargs (from UI widget params)
        self.dist_range = (dist_min, dist_max) if dist_min is not None and dist_max is not None else dist_range
        self.elev_range = (elev_min, elev_max) if elev_min is not None and elev_max is not None else elev_range
        self.azim_range = (azim_min, azim_max) if azim_min is not None and azim_max is not None else azim_range

    def apply(self, renderer, **kwargs):
        """Samples and applies a random pose to the renderer's camera."""
        R, T = self.sample_pose()
        renderer.cameras.R = R.to(renderer.device)
        renderer.cameras.T = T.to(renderer.device)

    def sample_pose(self):
        dist = random.uniform(*self.dist_range)
        elev = random.uniform(*self.elev_range)
        azim = random.uniform(*self.azim_range)
        return look_at_view_transform(dist, elev, azim)

    def get_specific_pose(self, dist, elev, azim):
        return look_at_view_transform(dist, elev, azim)
