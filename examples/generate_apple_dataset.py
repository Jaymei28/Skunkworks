import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from pytorch3d.structures import Meshes, join_meshes_as_batch

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.core import Renderer3D
from renderer.loader import MeshLoader
from renderer.background import HDRIBackground
from renderer.randomizers.pose import PoseRandomizer
from renderer.randomizers.lighting import LightingRandomizer
from renderer.randomizers.depth_scale import DepthAwareTransformRandomizer
from renderer.randomizers.weather import WeatherRandomizer
from renderer.randomizers.post_process import PostProcessRandomizer
from renderer.annotators.common import DepthAnnotator, MaskAnnotator
from renderer.annotators.bbox import BBoxAnnotator
from renderer.annotators.metadata import MetadataAnnotator
from pytorch3d.renderer import BlendParams


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Shared camera / depth range – keeps PoseRandomizer and DepthScaler in sync.
DIST_RANGE = (400.0, 800.0)
IMAGE_SIZE  = 512
FOV_Y_DEG   = 60.0          # must match Renderer3D camera FoV


def generate_dataset(num_images=2000):
    print(f"Starting synthetic data generation for Apple (Target: {num_images} images)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths ---
    root_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    apple_path  = os.path.join(root_dir, "tests",  "apple.obj")
    hdri_path   = os.path.join(root_dir, "HDRI",   "citrus_orchard_road_puresky_4k.hdr")
    output_base = os.path.join(root_dir, "dataset_apple")

    os.makedirs(os.path.join(output_base, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "masks"),  exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels"), exist_ok=True)

    # --- 1. Renderer ---
    renderer = Renderer3D(image_size=IMAGE_SIZE, device=device)
    renderer.shader.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

    # --- 2. HDRI background (now with strength randomization) ---
    hdri_bg = HDRIBackground(
        hdri_path,
        device=device,
        strength_range=(0.4, 2.5),   # 0.4 = dim dusk, 2.5 = bright noon
    )

    # --- 3. Pose randomizer (dist_range matches depth_scale) ---
    pose_rand = PoseRandomizer(dist_range=DIST_RANGE)

    # --- 4. Light randomizer ---
    light_rand = LightingRandomizer(brightness_range=(0.5, 1.6))

    # --- 5. Depth-aware transform randomizer (replaces TransformationRandomizer) ---
    #   Objects will always appear between 10 – 35 % of the image height
    #   regardless of how far away they are.
    transform_rand = DepthAwareTransformRandomizer(
        dist_range=DIST_RANGE,
        target_fraction_range=(0.10, 0.35),
        scale_jitter=0.15,
        rotation_range=(0, 360),
        translation_range=(-50.0, 50.0),
        fov_y_deg=FOV_Y_DEG,
        image_size=IMAGE_SIZE,
    )

    # --- 6. Weather randomizer ---
    weather_rand = WeatherRandomizer(
        weights={
            "clear":    0.35,
            "rain":     0.20,
            "fog":      0.20,
            "dust":     0.10,
            "overcast": 0.15,
        },
        intensity_range=(0.15, 0.75),
    )

    # --- 7. Post-processing randomizer ---
    pp_rand = PostProcessRandomizer(
        exposure_range=(-0.5, 0.5),
        bloom_intensity_range=(0.0, 0.30),
        bloom_threshold=0.70,
        noise_mode="random",                 # randomly picks "small" or "large" grain
        noise_intensity_range=(0.0, 0.06),
        ao_intensity_range=(0.0, 0.35),
        white_balance_temp_range=(3500, 8500),
        blur_sigma_range=(0.0, 1.8),
        enabled={                             # toggle any effect off here
            "exposure":          True,
            "bloom":             True,
            "noise":             True,
            "ambient_occlusion": True,
            "white_balance":     True,
            "blur":              True,
        },
    )

    # --- 8. Annotators ---
    mask_ann = MaskAnnotator()
    bbox_ann = BBoxAnnotator()
    meta_ann = MetadataAnnotator()

    # --- 9. Load base mesh ---
    apple_base = MeshLoader.load(apple_path, device=device)

    # ---------------------------------------------------------------------------
    # Generation loop
    # ---------------------------------------------------------------------------
    for i in tqdm(range(num_images)):

        # ---- Randomize scene-level settings ----

        # HDRI strength (Feature 1)
        hdri_strength = hdri_bg.randomize_strength()

        # Weather type for this frame (Feature 3) – draw before rendering
        weather_type, weather_intensity = weather_rand.randomize()

        # Post-processing parameters (Feature 4)
        pp_rand.randomize()

        # Number of apples: 10 % background-only
        num_apples = random.choices([0, 1, 2, 3], weights=[0.1, 0.4, 0.3, 0.2])[0]

        # ---- Build apple meshes with depth-aware scaling (Feature 2) ----
        apple_list = []
        for _ in range(num_apples):
            scaled_apple, depth_used = transform_rand.apply(
                apple_base.clone(), device=device
            )
            apple_list.append(scaled_apple)

        # ---- Randomize camera pose and lighting ----
        pose_rand.apply(renderer)
        light_rand.apply(renderer)

        # ---- Render ----
        if num_apples > 0:
            all_verts    = []
            all_faces    = []
            all_uv_verts = []
            all_uv_faces = []
            v_offset  = 0
            uv_offset = 0

            for m in apple_list:
                all_verts.append(m.verts_list()[0])
                all_faces.append(m.faces_list()[0] + v_offset)
                v_offset += m.verts_list()[0].shape[0]

                tex = m.textures
                all_uv_verts.append(tex.verts_uvs_list()[0])
                all_uv_faces.append(tex.faces_uvs_list()[0] + uv_offset)
                uv_offset += tex.verts_uvs_list()[0].shape[0]

            merged_verts    = torch.cat(all_verts,    dim=0)
            merged_faces    = torch.cat(all_faces,    dim=0)
            merged_uv_verts = torch.cat(all_uv_verts, dim=0)
            merged_uv_faces = torch.cat(all_uv_faces, dim=0)

            from pytorch3d.renderer import TexturesUV
            joined_tex = TexturesUV(
                maps=apple_list[0].textures.maps_list(),
                faces_uvs=[merged_uv_faces],
                verts_uvs=[merged_uv_verts],
            ).to(device)

            joined_scene = Meshes(
                verts=[merged_verts], faces=[merged_faces], textures=joined_tex
            ).to(device)

            raw_image = renderer.render(joined_scene)
            mask      = mask_ann.annotate(renderer, joined_scene)
            is_foreground = (mask[0] >= 0).float()

            batched_scene = join_meshes_as_batch(apple_list)
            bbox       = bbox_ann.annotate(renderer, batched_scene)
            class_list = ["Apple"] * num_apples
        else:
            raw_image     = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 4), device=device)
            is_foreground = torch.zeros((IMAGE_SIZE, IMAGE_SIZE), device=device)
            bbox          = []
            class_list    = []

        # ---- Sample HDRI background (strength already set above) ----
        bg_rot   = random.uniform(0, 360)
        bg_image = hdri_bg.get_background(renderer, renderer.cameras, rotation_deg=bg_rot)

        # ---- Composite foreground over background ----
        apple_rgb = raw_image[0, ..., :3]
        # Normalise HDR background to [0, 1] range for compositing
        bg_norm   = (bg_image / (bg_image.max() + 1e-6)).clamp(0, 1)
        composite = (apple_rgb * is_foreground[..., None]) + (bg_norm * (1 - is_foreground[..., None]))

        # ---- Apply weather (Feature 3) ----
        composite = weather_rand.apply(composite, weather_type, weather_intensity)

        # ---- Apply post-processing overrides (Feature 4) ----
        composite = pp_rand.apply(composite)

        # ---- Save ----
        img_name  = f"apple_{i:04d}.png"
        mask_name = f"mask_{i:04d}.png"
        json_name = f"meta_{i:04d}.json"

        comp_np = composite.detach().cpu().numpy().clip(0, 1)
        plt.imsave(os.path.join(output_base, "images", img_name), comp_np)

        mask_np = is_foreground.detach().cpu().numpy()
        plt.imsave(os.path.join(output_base, "masks", mask_name), mask_np, cmap="gray")

        meta_path = os.path.join(output_base, "labels", json_name)
        meta_ann.annotate(bbox, class_list, meta_path)

    print(f"\nDataset generation complete! Files saved to: {output_base}")


if __name__ == "__main__":
    generate_dataset(num_images=2000)

