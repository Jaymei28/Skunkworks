import torch

class BBoxAnnotator:
    """Calculates 2D bounding boxes for each mesh in the batch."""
    
    def annotate(self, renderer, meshes):
        bboxes = []
        # Get point projections for the entire batch
        # transform_points_screen expects (N, P, 3) or (P, 3)
        verts_list = meshes.verts_list()
        
        for i, verts in enumerate(verts_list):
            # Project verts to screen space for this instance
            verts_screen = renderer.cameras.transform_points_screen(
                verts[None, ...], 
                image_size=((renderer.image_size, renderer.image_size),)
            )[0]
            
            # Get min/max coordinates
            x_min = verts_screen[:, 0].min().item()
            y_min = verts_screen[:, 1].min().item()
            x_max = verts_screen[:, 0].max().item()
            y_max = verts_screen[:, 1].max().item()
            
            # Clip to image boundaries
            x_min = max(0, min(renderer.image_size, x_min))
            y_min = max(0, min(renderer.image_size, y_min))
            x_max = max(0, min(renderer.image_size, x_max))
            y_max = max(0, min(renderer.image_size, y_max))
            
            bboxes.append([x_min, y_min, x_max, y_max])
            
        return bboxes # List of [x1, y1, x2, y2]
