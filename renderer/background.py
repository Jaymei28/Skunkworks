import torch
import numpy as np
import cv2
import math
import random
import os

# Enable OpenEXR support for HDRIs
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class HDRIBackground:
    """Handles HDRI background loading, camera-based sampling, and strength randomization."""
    _cache = {} # (path, device) -> tensor

    def __init__(self, hdri_path, device="cpu", strength_range=(0.5, 2.0)):
        """
        Args:
            hdri_path: Path to the .hdr / .exr file.
            device: Torch device string.
            strength_range: (min, max) multiplier for HDRI brightness.
                            1.0 = neutral, >1.0 brighter, <1.0 darker.
        """
        self.device = torch.device(device)
        self.strength_range = strength_range
        self.strength = 1.0  # current strength, randomized per-frame

        cache_key = (hdri_path, str(device))
        if cache_key in HDRIBackground._cache:
            self.hdri_tensor = HDRIBackground._cache[cache_key]
        elif not hdri_path:
            # Procedural fallback: Neutral flat gray
            print("No HDRI path provided. Using neutral procedural background.")
            self.hdri_tensor = torch.full((64, 128, 3), 0.25, device=self.device, dtype=torch.float32)
            HDRIBackground._cache[cache_key] = self.hdri_tensor
        else:
            print(f"Loading HDRI from {hdri_path}...")
            # Load HDR image using OpenCV (returns BGR float32)
            hdri_bgr = cv2.imread(hdri_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if hdri_bgr is None:
                # Instead of crashing, let's use the procedural fallback but warn
                print(f"[Warning] Could not load HDRI from {hdri_path}. Using fallback.")
                self.hdri_tensor = torch.full((64, 128, 3), 0.25, device=self.device, dtype=torch.float32)
            else:
                # Convert BGR → RGB and store as float32 tensor
                hdri_rgb = cv2.cvtColor(hdri_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                self.hdri_tensor = torch.from_numpy(hdri_rgb).to(self.device)
            HDRIBackground._cache[cache_key] = self.hdri_tensor

        self.h, self.w, _ = self.hdri_tensor.shape
        # Cache for pixel directions (rebuilt if image_size changes)
        self._cached_image_size = None
        self.pixel_dirs_cam = None

    # ------------------------------------------------------------------
    # Strength helpers
    # ------------------------------------------------------------------

    def randomize_strength(self):
        """Sample a new random HDRI strength within strength_range."""
        self.strength = random.uniform(*self.strength_range)
        return self.strength

    def set_strength(self, value: float):
        """Manually set HDRI strength."""
        self.strength = value

    # ------------------------------------------------------------------
    # Background sampling
    # ------------------------------------------------------------------

    def get_background(self, renderer, cameras, rotation_deg=0):
        """
        Samples the HDRI to produce a background image.

        The current `self.strength` is applied as a linear multiplier
        before the tensor is returned (already in [0, ∞) HDR space).
        Call `randomize_strength()` before this each frame to vary brightness.

        Returns:
            Tensor (H, W, 3) float32 – linear HDR values scaled by strength.
        """
        image_size = renderer.image_size

        # Rebuild pixel direction cache if image_size changed
        if self._cached_image_size != image_size:
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, image_size, device=self.device),
                torch.linspace(-1, 1, image_size, device=self.device),
                indexing='ij'
            )
            self.pixel_dirs_cam = torch.stack([x, -y, torch.ones_like(x)], dim=-1)
            self.pixel_dirs_cam = torch.nn.functional.normalize(
                self.pixel_dirs_cam, dim=-1
            )
            self._cached_image_size = image_size

        # Rotate pixel directions into world space
        R_inv = cameras.R[0].transpose(0, 1)
        world_dirs = torch.matmul(self.pixel_dirs_cam.view(-1, 3), R_inv)

        # Equirectangular (spherical) mapping with optional yaw rotation
        theta = (
            torch.atan2(world_dirs[:, 0], world_dirs[:, 2])
            + (rotation_deg * math.pi / 180.0)
        )
        phi = torch.asin(world_dirs[:, 1].clamp(-1, 1))

        u = (theta / (2 * math.pi) + 0.5) % 1.0
        v = (phi / math.pi + 0.5).clamp(0, 1)

        u_px = (u * (self.w - 1)).long().clamp(0, self.w - 1)
        v_px = ((1 - v) * (self.h - 1)).long().clamp(0, self.h - 1)

        bg_flat = self.hdri_tensor[v_px, u_px]
        bg_image = bg_flat.view(image_size, image_size, 3)

        # Apply HDRI strength (linear scale)
        bg_image = bg_image * self.strength

        return bg_image
