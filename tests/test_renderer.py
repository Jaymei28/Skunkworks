import torch
import matplotlib.pyplot as plt
import os
import sys

# Add the root directory to sys.path so we can import the renderer module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from renderer.core import Renderer3D
    from renderer.loader import MeshLoader
    from renderer.randomizers.pose import PoseRandomizer
    from renderer.randomizers.lighting import LightingRandomizer
    from renderer.annotators.common import DepthAnnotator, MaskAnnotator
    from renderer.annotators.bbox import BBoxAnnotator
    from renderer.annotators.metadata import MetadataAnnotator
except ImportError as e:
    print(f"Error importing library components: {e}")
    print("Ensure you are running this from the tests directory or the root directory is in your PYTHONPATH.")
    sys.exit(1)

def run_test():
    print("Testing PyTorch3D Rendering Library...")
    
    # Check for PyTorch3D
    try:
        import pytorch3d
        print(f"PyTorch3D version: {pytorch3d.__version__}")
    except ImportError:
        print("PyTorch3D not found. Please install it first.")
        return

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    mesh_path = os.path.join(os.path.dirname(__file__), "cube.obj")
    
    # 1. Initialize Renderer
    renderer = Renderer3D(image_size=512, device=device)
    
    # 2. Load Mesh
    print(f"Loading mesh from {mesh_path}...")
    obj_template = MeshLoader.load(mesh_path, device=device)
    
    # Create two instances
    # Instance 1: Offset to the left
    obj1 = obj_template.clone()
    verts1 = obj1.verts_list()[0]
    obj1 = obj1.update_padded(verts1[None, ...] + torch.tensor([[-2.0, 0.0, 0.0]], device=device))
    
    # Instance 2: Offset to the right
    obj2 = obj_template.clone()
    verts2 = obj2.verts_list()[0]
    obj2 = obj2.update_padded(verts2[None, ...] + torch.tensor([[2.0, 0.0, 0.0]], device=device))
    
    # Combine into a single scene (Merge into one Meshes object)
    from pytorch3d.structures import Meshes
    verts_merged = torch.cat([obj1.verts_packed(), obj2.verts_packed()], dim=0)
    faces_merged = torch.cat([obj1.faces_packed(), obj2.faces_packed() + obj1.num_verts_per_mesh()[0]], dim=0)
    
    # Textures: Use distinct colors
    from pytorch3d.renderer import TexturesVertex
    tex1 = torch.ones_like(obj1.verts_packed()) * torch.tensor([1.0, 0.2, 0.2], device=device)
    tex2 = torch.ones_like(obj2.verts_packed()) * torch.tensor([0.2, 0.2, 1.0], device=device)
    tex_merged = torch.cat([tex1, tex2], dim=0)
    
    joined_scene = Meshes(
        verts=[verts_merged], 
        faces=[faces_merged], 
        textures=TexturesVertex(verts_features=[tex_merged])
    ).to(device)
    
    scene_list = [obj1, obj2]
    class_names = ["left_cube", "right_cube"]
    
    # 3. Apply Randomization
    print("Applying random pose and lighting...")
    PoseRandomizer(dist_range=(8.0, 15.0)).apply(renderer)
    LightingRandomizer().apply(renderer)
    
    # 4. Render
    print("Rendering scene with 2 instances...")
    image = renderer.render(joined_scene) 
    
    # 5. Annotations
    print("Generating annotations...")
    depth = DepthAnnotator().annotate(renderer, joined_scene)
    mask = MaskAnnotator().annotate(renderer, joined_scene)
    
    # BBox needs the separate meshes to identify each one
    from pytorch3d.structures import join_meshes_as_batch
    batched_scene = join_meshes_as_batch(scene_list)
    bbox = BBoxAnnotator().annotate(renderer, batched_scene)
    
    # 6. Save results
    print(f"Saving results to {output_dir}...")
    
    # Save Images (Clamp RGB to 0-1 to avoid matplotlib errors)
    img_np = image[0, ..., :3].detach().cpu().numpy()
    img_np = img_np.clip(0, 1)
    plt.imsave(os.path.join(output_dir, "render.png"), img_np)
    
    # Save Depth (Normalize 0-1 for visualization)
    depth_np = depth[0].detach().cpu().numpy()
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max > depth_min:
        depth_vis = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_vis = depth_np * 0
    plt.imsave(os.path.join(output_dir, "depth.png"), depth_vis, cmap='magma')
    
    # Save Mask (Instance ID visualization)
    mask_np = mask[0].detach().cpu().numpy()
    # Map -1 (bg) to 0, and instance IDs (0, 1...) to 1, 2...
    mask_vis = (mask_np + 1.0) / (len(scene_list) + 1.0)
    plt.imsave(os.path.join(output_dir, "mask.png"), mask_vis, cmap='viridis')
    
    # Save Metadata JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    MetadataAnnotator().annotate(bbox, class_names, metadata_path)
    
    print("\nTest passed successfully!")
    print(f"Outputs generated in: {output_dir}")
    print("- render.png (RGB)")
    print("- depth.png (Depth Map)")
    print("- mask.png (Instance Segmentation)")
    print("- metadata.json (BBoxes & Classes)")
    print("\nInstance Summary:")
    for idx, (box, cls) in enumerate(zip(bbox, class_names)):
        print(f"  {idx}: {cls} -> [x1: {box[0]:.1f}, y1: {box[1]:.1f}, x2: {box[2]:.1f}, y2: {box[3]:.1f}]")

if __name__ == "__main__":
    run_test()
