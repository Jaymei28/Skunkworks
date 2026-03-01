import torch

class DepthAnnotator:
    """Generates depth maps from the renderer."""
    
    def annotate(self, renderer, meshes):
        fragments = renderer.rasterizer(meshes)
        # fragments.zbuf is of shape (B, H, W, K) where K is faces_per_pixel
        depth = fragments.zbuf[..., 0] 
        return depth

class MaskAnnotator:
    """Generates instance segmentation masks where each pixel is the instance ID."""
    
    def annotate(self, renderer, meshes):
        fragments = renderer.rasterizer(meshes)
        # pix_to_face is (B, H, W, K)
        pix_to_face = fragments.pix_to_face[..., 0] 
        
        # Create a mapping from face index to mesh index
        face_to_mesh = meshes.faces_packed_to_mesh_idx()
        
        # Map pixels to mesh indices
        # We need to handle background (-1) carefully
        mask = torch.full_like(pix_to_face, -1)
        mask_valid = pix_to_face >= 0
        if mask_valid.any():
            mask[mask_valid] = face_to_mesh[pix_to_face[mask_valid]]
            
        return mask
