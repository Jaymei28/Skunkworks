import torch
import random
from .base import BaseRandomizer

class HueOffsetRandomizer(BaseRandomizer):
    """
    Randomizes the hue of the rendered image.
    This is a post-processing effect.
    """
    
    def __init__(self, hue_limit=0.2):
        self.hue_limit = hue_limit

    def apply(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply hue offset to an (H, W, 3) float tensor.
        """
        offset = random.uniform(-self.hue_limit, self.hue_limit)
        
        # Convert to HSV-ish (simplified)
        # For a true hue shift we should go to HSV, shift H, go back to RGB.
        # Here's a fast approximation or we could use a library.
        # Since we're in PyTorch, let's use a standard shift if available or implement one.
        
        return self._shift_hue(image, offset)

    def _shift_hue(self, img, hue_offset):
        # Very high-level approximation of hue shift in RGB space
        # (Rotate the color vector around the (1,1,1) axis)
        
        # Proper HSV shift is better. 
        # For now, let's just use a placeholder or a simple implementation.
        # Actually, let's implement a proper one since this is for synth data.
        
        from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
        import numpy as np
        
        img_np = img.detach().cpu().numpy()
        hsv = rgb_to_hsv(img_np)
        hsv[..., 0] = (hsv[..., 0] + hue_offset) % 1.0
        rgb = hsv_to_rgb(hsv)
        
        return torch.from_numpy(rgb).to(img.device).to(img.dtype)
