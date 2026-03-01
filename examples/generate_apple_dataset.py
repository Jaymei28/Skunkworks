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
from renderer.randomizers.transform import TransformationRandomizer
from renderer.annotators.common import DepthAnnotator, MaskAnnotator
from renderer.annotators.bbox import BBoxAnnotator
from renderer.annotators.metadata import MetadataAnnotator
from pytorch3d.renderer import BlendParams

def generate_dataset(num_images=2000):
    print(f"Starting synthetic data generation for Apple (Target: {num_images} images)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 512
    
    # Paths
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    apple_path = os.path.join(root_dir, "tests", "apple.obj")
    hdri_path = os.path.join(root_dir, "HDRI", "citrus_orchard_road_puresky_4k.hdr")
    output_base = os.path.join(root_dir, "dataset_apple")
    
    os.makedirs(os.path.join(output_base, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "labels"), exist_ok=True)
    
    # 1. Initialize Components
    renderer = Renderer3D(image_size=image_size, device=device)
    
    # Use solid black background for the raw render to make compositing easier
    renderer.shader.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))
    
    hdri_bg = HDRIBackground(hdri_path, device=device)
    
    # Randomizers (Zoomed out significantly)
    pose_rand = PoseRandomizer(dist_range=(400.0, 800.0)) 
    light_rand = LightingRandomizer(brightness_range=(0.6, 1.4))
    transform_rand = TransformationRandomizer(
        scale_range=(0.7, 1.3),
        translation_range=(-80.0, 80.0) # Wider translation for multi-object
    )
    
    # Annotators
    mask_ann = MaskAnnotator()
    bbox_ann = BBoxAnnotator()
    meta_ann = MetadataAnnotator()
    
    # 2. Load Base Mesh
    apple_base = MeshLoader.load(apple_path, device=device)
    
    # 3. Generation Loop
    
    for i in tqdm(range(num_images)):
        # Decide number of apples (0 to 3)
        # 10% chance of background-only (good for YOLO)
        num_apples = random.choices([0, 1, 2, 3], weights=[0.1, 0.4, 0.3, 0.2])[0]
        
        apple_list = []
        for _ in range(num_apples):
            transformed_apple = transform_rand.apply(apple_base.clone(), device=device)
            apple_list.append(transformed_apple)
        
        # Randomize Pose and Lighting
        pose_rand.apply(renderer)
        light_rand.apply(renderer)
        
        # Render and Annotate
        if num_apples > 0:
            # Merge all individual apple meshes into a single Mesh object for rendering in one image
            all_verts = []
            all_faces = []
            all_uv_verts = []
            all_uv_faces = []
            v_offset = 0
            uv_offset = 0
            
            for m in apple_list:
                all_verts.append(m.verts_list()[0])
                all_faces.append(m.faces_list()[0] + v_offset)
                v_offset += m.verts_list()[0].shape[0]
                
                # Assume TexturesUV for apple.obj
                tex = m.textures
                all_uv_verts.append(tex.verts_uvs_list()[0])
                all_uv_faces.append(tex.faces_uvs_list()[0] + uv_offset)
                uv_offset += tex.verts_uvs_list()[0].shape[0]
            
            merged_verts = torch.cat(all_verts, dim=0)
            merged_faces = torch.cat(all_faces, dim=0)
            merged_uv_verts = torch.cat(all_uv_verts, dim=0)
            merged_uv_faces = torch.cat(all_uv_faces, dim=0)
            
            from pytorch3d.renderer import TexturesUV
            joined_tex = TexturesUV(
                maps=apple_list[0].textures.maps_list(), # Re-use map from first apple
                faces_uvs=[merged_uv_faces],
                verts_uvs=[merged_uv_verts]
            ).to(device)
            
            joined_scene = Meshes(verts=[merged_verts], faces=[merged_faces], textures=joined_tex).to(device)
            
            # Rendering
            raw_image = renderer.render(joined_scene)
            
            # Mask generation (Instance segmentation)
            mask = mask_ann.annotate(renderer, joined_scene)
            is_foreground = (mask[0] >= 0).float()
            
            # BBox needs batch for separate boxes per apple
            batched_scene = join_meshes_as_batch(apple_list)
            bbox = bbox_ann.annotate(renderer, batched_scene)
            class_list = ["Apple"] * num_apples
        else: # Empty Scene
            raw_image = torch.zeros((1, image_size, image_size, 4), device=device)
            is_foreground = torch.zeros((image_size, image_size), device=device)
            bbox = []
            class_list = []

        # Sample HDRI Background with random rotation
        bg_rot = random.uniform(0, 360)
        bg_image = hdri_bg.get_background(renderer, renderer.cameras, rotation_deg=bg_rot)
        
        # Composite
        apple_rgb = raw_image[0, ..., :3]
        composite = (apple_rgb * is_foreground[..., None]) + (bg_image * (1 - is_foreground[..., None]))
        
        # Save Files
        img_name = f"apple_{i:04d}.png"
        mask_name = f"mask_{i:04d}.png"
        json_name = f"meta_{i:04d}.json"
        
        comp_np = composite.detach().cpu().numpy().clip(0, 1)
        plt.imsave(os.path.join(output_base, "images", img_name), comp_np)
        
        mask_np = is_foreground.detach().cpu().numpy()
        plt.imsave(os.path.join(output_base, "masks", mask_name), mask_np, cmap='gray')
        
        meta_path = os.path.join(output_base, "labels", json_name)
        meta_ann.annotate(bbox, class_list, meta_path)

    print(f"\nDataset generation complete! Files saved to: {output_base}")

if __name__ == "__main__":
    generate_dataset(num_images=2000)
