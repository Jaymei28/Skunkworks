import torch
import random
from .base import BaseRandomizer

class LightingRandomizer(BaseRandomizer):
    """Randomizes the lighting conditions."""
    
    def __init__(self, brightness_range=(0.3, 1.5)):
        self.brightness_range = brightness_range

    def apply(self, renderer, **kwargs):
        # Example for a single point light
        brightness = random.uniform(*self.brightness_range)
        location = [[random.uniform(-200, 200), random.uniform(200, 400), random.uniform(-200, 200)]]
        
        renderer.lights.location = torch.tensor(location, device=renderer.device)
        renderer.lights.ambient_color = torch.full((1, 3), brightness * 0.2, device=renderer.device)
        renderer.lights.diffuse_color = torch.full((1, 3), brightness * 0.8, device=renderer.device)
