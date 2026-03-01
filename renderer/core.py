import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftPhongShader,
    BlendParams
)

from pytorch3d.structures import join_meshes_as_batch

class Renderer3D:
    def __init__(self, image_size=512, device="cpu"):
        self.device = torch.device(device)
        self.image_size = image_size
        
        # Initialize camera
        self.cameras = FoVPerspectiveCameras(device=self.device, znear=0.1, zfar=1000.0)
        
        # Initialize lights
        self.lights = PointLights(device=self.device)
        
        # Initialize rasterizer
        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )
        
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, 
            raster_settings=raster_settings
        )
        
        # Initialize shader
        blend_params = BlendParams(background_color=(0.1, 0.1, 0.1))
        self.shader = SoftPhongShader(
            device=self.device, 
            cameras=self.cameras,
            lights=self.lights,
            blend_params=blend_params
        )
        
        # Main renderer
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def render(self, meshes):
        """Renders the given meshes (supports singleton or list)."""
        if isinstance(meshes, list):
            meshes = join_meshes_as_batch(meshes)
        return self.renderer(meshes)

    def set_camera(self, R, T):
        self.cameras.R = R.to(self.device)
        self.cameras.T = T.to(self.device)
        
    def set_image_size(self, size):
        self.rasterizer.raster_settings.image_size = size
