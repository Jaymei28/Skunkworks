"""
depth_scale.py
==============
Depth-based object size normalisation for synthetic data generation.

Problem
-------
When a 3D object is placed at different depths (camera distances), its
projected size in the rendered image changes drastically.  For training
object detectors we typically want the object to span a **consistent
fraction of the image** regardless of where it sits in the scene.

This module provides:

    * ``DepthScaler`` – computes the correct mesh scale multiplier so that
      an object at a given depth projects to a target apparent size.
    * ``DepthAwareTransformRandomizer`` – drop-in replacement for
      ``TransformationRandomizer`` that ensures depth-correct sizing.

Maths
-----
For a perspective camera with vertical FoV ``fov_y``:

    apparent_height_px = (physical_height / depth) * (H / (2 * tan(fov_y/2)))

We want ``apparent_height_px ≈ target_fraction * H``, so:

    physical_height = target_fraction * depth * 2 * tan(fov_y/2)

Because the mesh has a known bounding-sphere radius ``r_mesh`` loaded at
unit scale, the required scale is:

    scale = (target_fraction * depth * 2 * tan(fov_y/2)) / (2 * r_mesh)
          = (target_fraction * depth * tan(fov_y/2)) / r_mesh

Usage::

    from renderer.randomizers.depth_scale import DepthScaler, DepthAwareTransformRandomizer

    # --- Option A: Manual ---
    scaler = DepthScaler(fov_y_deg=60, image_size=512)
    mesh   = MeshLoader.load(path, device)
    scale  = scaler.compute_scale(mesh, depth=500.0, target_fraction=0.25)
    mesh   = scaler.scale_mesh(mesh, scale, device)

    # --- Option B: Use DepthAwareTransformRandomizer ---
    transform_rand = DepthAwareTransformRandomizer(
        dist_range=(300, 700),
        target_fraction_range=(0.10, 0.35),
        fov_y_deg=60,
        image_size=512,
    )
    mesh, depth = transform_rand.apply(mesh, device=device)
"""

import random
import math

import torch
from pytorch3d.transforms import euler_angles_to_matrix, Rotate, Scale, Translate


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

class DepthScaler:
    """
    Computes the world-space scale factor needed so that an object at a
    known depth fills a target fraction of the rendered image height.

    Args:
        fov_y_deg:  Vertical field-of-view of the camera, in degrees.
        image_size: Height (and width, assuming square) of the rendered image.
    """

    def __init__(self, fov_y_deg: float = 60.0, image_size: int = 512):
        self.fov_y_deg  = fov_y_deg
        self.image_size = image_size
        self._tan_half  = math.tan(math.radians(fov_y_deg / 2.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bounding_radius(self, mesh) -> float:
        """
        Return the bounding-sphere radius of a mesh (assumed to be at unit scale).
        The radius is the max distance from the mesh centroid to any vertex.
        """
        verts = mesh.verts_packed()  # (V, 3)
        centroid = verts.mean(dim=0)
        r = (verts - centroid).norm(dim=-1).max().item()
        return max(r, 1e-6)  # guard against degenerate meshes

    def compute_scale(
        self,
        mesh,
        depth: float,
        target_fraction: float = 0.25,
    ) -> float:
        """
        Return the scale multiplier that makes the object span
        ``target_fraction`` of the image height at a given camera distance.

        Args:
            mesh:            Loaded PyTorch3D Meshes object (unit scale).
            depth:           Camera-to-object distance in world units.
            target_fraction: Desired apparent height as fraction of image H.

        Returns:
            float scale factor.
        """
        r_mesh = self.bounding_radius(mesh)
        # Desired world-space half-height = target_fraction * depth * tan(fov/2)
        target_world_r = target_fraction * depth * self._tan_half
        return target_world_r / r_mesh

    def scale_mesh(self, mesh, scale: float, device="cpu"):
        """Apply a uniform world-space scale to a mesh and return the result."""
        s_transform = Scale(scale, device=device)
        new_verts   = s_transform.transform_points(mesh.verts_padded())
        return mesh.update_padded(new_verts)

    def randomize_and_scale(
        self,
        mesh,
        depth: float,
        target_fraction_range: tuple = (0.10, 0.40),
        scale_jitter: float = 0.15,
        device="cpu",
    ):
        """
        Pick a random target fraction, compute the correct scale, add a small
        jitter for variety, and return the scaled mesh + chosen parameters.

        Args:
            mesh:                   Source mesh (unscaled).
            depth:                  Camera distance.
            target_fraction_range:  (min, max) fraction of image height.
            scale_jitter:           ±relative jitter applied on top of computed scale.
            device:                 Torch device string.

        Returns:
            (scaled_mesh, scale_used, target_fraction)
        """
        target_fraction = random.uniform(*target_fraction_range)
        scale = self.compute_scale(mesh, depth, target_fraction)
        # Apply a small multiplicative jitter so not every image has the same fraction
        jitter = random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        scale *= jitter
        scaled_mesh = self.scale_mesh(mesh, scale, device)
        return scaled_mesh, scale, target_fraction


# ---------------------------------------------------------------------------
# Drop-in randomizer (replaces TransformationRandomizer)
# ---------------------------------------------------------------------------

class DepthAwareTransformRandomizer:
    """
    Combines:
        * Random depth (distance from camera) sampling.
        * Depth-correct object scaling (via DepthScaler).
        * Random rotation and optional XY translation.

    This is a drop-in replacement for ``TransformationRandomizer`` that
    ensures objects maintain a consistent apparent size in the image.

    Args:
        dist_range:             (min, max) camera-to-object distance.
        target_fraction_range:  (min, max) fraction of image height the object spans.
        scale_jitter:           Relative scale jitter on top of computed scale.
        rotation_range:         (min, max) rotation in degrees for each axis.
        translation_range:      (min, max) lateral XY translation in world units.
                                Set to (0, 0) for centred objects.
        fov_y_deg:              Camera FoV (must match renderer camera).
        image_size:             Rendered image size (square assumed).
    """

    def __init__(
        self,
        dist_range:            tuple = (200.0, 800.0),
        target_fraction_range: tuple = (0.10, 0.40),
        scale_jitter:          float = 0.15,
        rotation_range:        tuple = (0, 360),
        translation_range:     tuple = (-20.0, 20.0),
        fov_y_deg:             float = 60.0,
        image_size:            int   = 512,
    ):
        self.dist_range            = dist_range
        self.target_fraction_range = target_fraction_range
        self.scale_jitter          = scale_jitter
        self.rotation_range        = rotation_range
        self.translation_range     = translation_range
        self.scaler = DepthScaler(fov_y_deg=fov_y_deg, image_size=image_size)

        # Last sampled depth (accessible after apply())
        self.last_depth            = None
        self.last_scale            = None
        self.last_target_fraction  = None

    def apply(self, mesh, device="cpu"):
        """
        Apply depth-aware scale + random rotation + random XY translation.

        Args:
            mesh:   Source PyTorch3D Meshes object (clone before calling).
            device: Torch device string.

        Returns:
            (transformed_mesh, depth_used)
        """
        # 1. Sample random depth
        depth = random.uniform(*self.dist_range)
        self.last_depth = depth

        # 2. Depth-correct scale (with jitter)
        scaled_mesh, scale, frac = self.scaler.randomize_and_scale(
            mesh,
            depth=depth,
            target_fraction_range=self.target_fraction_range,
            scale_jitter=self.scale_jitter,
            device=device,
        )
        self.last_scale           = scale
        self.last_target_fraction = frac

        # 3. Random rotation
        rx = random.uniform(*self.rotation_range)
        ry = random.uniform(*self.rotation_range)
        rz = random.uniform(*self.rotation_range)
        angles = torch.tensor([rx, ry, rz]) * (math.pi / 180.0)
        R  = euler_angles_to_matrix(angles[None, ...], convention="XYZ")
        rot = Rotate(R=R, device=device)
        new_verts = rot.transform_points(scaled_mesh.verts_padded())
        scaled_mesh = scaled_mesh.update_padded(new_verts)

        # 4. Random XY lateral translation
        tx = random.uniform(*self.translation_range)
        ty = random.uniform(*self.translation_range)
        transl = Translate(tx, ty, 0.0, device=device)
        new_verts = transl.transform_points(scaled_mesh.verts_padded())
        scaled_mesh = scaled_mesh.update_padded(new_verts)

        return scaled_mesh, depth
