import torch
import numpy as np
import cv2
import math

class HDRIBackground:
    """Handles HDRI background loading and camera-based sampling."""
    
    def __init__(self, hdri_path, device="cpu"):
        self.device = torch.device(device)
        print(f"Loading HDRI from {hdri_path}...")
        # Load HDR image using OpenCV (returns BGR)
        hdri_bgr = cv2.imread(hdri_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdri_bgr is None:
            raise FileNotFoundError(f"Could not load HDRI from {hdri_path}")
        # Convert BGR to RGB
        hdri_rgb = cv2.cvtColor(hdri_bgr, cv2.COLOR_BGR2RGB)
        # Convert to tensor and move to device
        self.hdri_tensor = torch.from_numpy(hdri_rgb).to(self.device)
        self.h, self.w, _ = self.hdri_tensor.shape

    def get_background(self, renderer, cameras, rotation_deg=0):
        """
        Generates a background image by sampling the HDRI based on camera view directions.
        """
        image_size = renderer.image_size
        
        # 1. Get pixel directions in CAMERA space
        if not hasattr(self, 'pixel_dirs_cam'):
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, image_size, device=self.device),
                torch.linspace(-1, 1, image_size, device=self.device),
                indexing='ij'
            )
            self.pixel_dirs_cam = torch.stack([x, -y, torch.ones_like(x)], dim=-1)
            self.pixel_dirs_cam = torch.nn.functional.normalize(self.pixel_dirs_cam, dim=-1)

        # 2. Rotate directions to WORLD space
        R_inv = cameras.R[0].transpose(0, 1)
        world_dirs = torch.matmul(self.pixel_dirs_cam.view(-1, 3), R_inv) 
        
        # 3. Spherical mapping with custom rotation
        theta = torch.atan2(world_dirs[:, 0], world_dirs[:, 2]) + (rotation_deg * math.pi / 180.0)
        phi = torch.asin(world_dirs[:, 1].clamp(-1, 1))
        
        u = (theta / (2 * math.pi) + 0.5) % 1.0
        v = (phi / math.pi + 0.5).clamp(0, 1)
        
        u_px = (u * (self.w - 1)).long()
        v_px = ((1 - v) * (self.h - 1)).long()
        
        # Final safety clamp
        u_px = u_px.clamp(0, self.w - 1)
        v_px = v_px.clamp(0, self.h - 1)
        
        bg_flat = self.hdri_tensor[v_px, u_px]
        bg_image = bg_flat.view(image_size, image_size, 3)
        
        return bg_image
