import torch
import os
import sys
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer import TexturesUV

# Add root to path
sys.path.append(os.getcwd())

from renderer.core import Renderer3D
from renderer.loader import MeshLoader
from renderer.randomizers.transform import TransformationRandomizer

def debug_merge():
    device = torch.device('cpu')
    renderer = Renderer3D(image_size=512, device=device)
    apple_base = MeshLoader.load('tests/apple.obj', device=device)
    
    transform_rand = TransformationRandomizer(scale_range=(0.8, 1.2), translation_range=(-50, 50))
    
    # Create 2 apples
    apple1 = transform_rand.apply(apple_base.clone(), device=device)
    apple2 = transform_rand.apply(apple_base.clone(), device=device)
    apple_list = [apple1, apple2]
    
    print("Merging meshes...")
    all_verts = [m.verts_list()[0] for m in apple_list]
    all_faces = []
    curr_v_offset = 0
    for m in apple_list:
        all_faces.append(m.faces_list()[0] + curr_v_offset)
        curr_v_offset += m.verts_list()[0].shape[0]
    
    merged_verts = torch.cat(all_verts, dim=0)
    merged_faces = torch.cat(all_faces, dim=0)
    
    print(f"Merged Verts: {merged_verts.shape}")
    print(f"Merged Faces: {merged_faces.shape}")
    
    # TexturesUV merge
    all_verts_uvs = [m.textures.verts_uvs_list()[0] for m in apple_list]
    all_faces_uvs = []
    curr_uv_offset = 0
    for m in apple_list:
        all_faces_uvs.append(m.textures.faces_uvs_list()[0] + curr_uv_offset)
        curr_uv_offset += m.textures.verts_uvs_list()[0].shape[0]
        
    merged_verts_uvs = torch.cat(all_verts_uvs, dim=0)
    merged_faces_uvs = torch.cat(all_faces_uvs, dim=0)
    
    print(f"Merged UV Verts: {merged_verts_uvs.shape}")
    print(f"Merged UV Faces: {merged_faces_uvs.shape}")
    
    joined_tex = TexturesUV(
        maps=apple_list[0].textures.maps_list(),
        faces_uvs=[merged_faces_uvs],
        verts_uvs=[merged_verts_uvs]
    )
    
    joined_scene = Meshes(verts=[merged_verts], faces=[merged_faces], textures=joined_tex).to(device)
    
    print("Rendering...")
    try:
        image = renderer.render(joined_scene)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    debug_merge()
