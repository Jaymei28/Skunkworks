"""
engine/scene_state.py
=====================
Central configuration dataclass for the entire generation pipeline.
SceneConfig is the single source of truth – the UI reads from and writes to it,
and the Worker reads from it when generating.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid


@dataclass
class RandomizerInstance:
    """A single active randomizer on an object."""
    type:   str = "ObjectPlacementRandomizer"
    params: dict = field(default_factory=dict)

@dataclass
class ObjectConfig:
    """Settings for a single 3D model type (template / prefab)."""
    name:       str = "New Object"
    class_name: str = "Object"
    mesh_path:  str = ""
    count_min:  int = 1
    count_max:  int = 1

    # PBR texture overrides (empty = auto-detected from mesh file)
    tex_albedo:    str = ""
    tex_normal:    str = ""
    tex_roughness: str = ""
    tex_metallic:  str = ""

    # Perception Randomizers (multi-select / list)
    randomizers: List[RandomizerInstance] = field(default_factory=list)


@dataclass
class SceneObject:
    """A live, placed instance of an ObjectConfig in the 3D scene."""
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    config:      ObjectConfig = field(default_factory=ObjectConfig)
    label:       str = ""           # class name assigned by user
    # World-space transform
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0
    scale: float = 1.0
    visible: bool = True
    # PBR material overrides
    metallic:  float = 0.0
    roughness: float = 0.8
    floating:  bool  = False        # If true, reacts to ocean waves (buoyancy)
    buoyancy_offset: float = 0.0    # Vertical offset for floating objects (waterline adjustment)
    float_bob:  float = 1.0         # Bobbing intensity
    float_tilt: float = 1.0         # Tilting intensity
    is_focus_target: bool = False   # If true, camera orbits THIS object during generation
    
    # Per-Object Randomization
    rand_pos:   bool = True
    rand_rot:   bool = True
    rand_scale: bool = True
    rand_translation_max: float = 1.0
    rand_scale_jitter:    float = 0.15
    rand_rot_min:         float = 0.0
    rand_rot_max:         float = 360.0

    @property
    def display_name(self) -> str:
        base = self.config.name or "Object"
        return f"{base} [{self.label}]" if self.label else base

@dataclass
class WeatherConfig:
    enabled:      bool  = True
    type:         str   = "clear"       # clear, cloudy, rain, stormy, snow, foggy
    intensity:    float = 0.5           # 0..1 intensity of rain/snow/clouds
    fog_density:  float = 0.05
    
    weights: Dict[str, float] = field(default_factory=lambda: {
        "clear":    0.35,
        "rain":     0.20,
        "fog":      0.20,
        "dust":     0.10,
        "overcast": 0.15,
    })
    intensity_min: float = 0.15
    intensity_max: float = 0.75


@dataclass
class PostProcessConfig:
    exposure_enabled:  bool  = True
    exposure_min:      float = -0.5
    exposure_max:      float =  0.5

    bloom_enabled:     bool  = True
    bloom_min:         float =  0.0
    bloom_max:         float =  0.30
    bloom_threshold:   float =  0.70

    noise_enabled:     bool  = True
    noise_mode:        str   = "random"   # "small" | "large" | "random"
    noise_min:         float =  0.0
    noise_max:         float =  0.06

    ao_enabled:        bool  = True
    
    # Fish-eye Distortion
    fisheye_enabled:   bool  = False
    fisheye_intensity: float = 0.5

    wb_enabled:        bool  = True

    wb_enabled:        bool  = True
    wb_temp_min:       float =  3500.0
    wb_temp_max:       float =  8500.0

    blur_enabled:      bool  = True
    blur_min:          float =  0.0
    blur_max:          float =  1.8


@dataclass
class OceanConfig:
    enabled:      bool  = True
    surface_type: str   = "Ocean"      # Ocean, River, Pool
    level:        float = 0.0

    # 🌊 Simulation (Unity HDRP-Style)
    time_multiplier: float = 1.0
    repetition_size: float = 500.0
    
    # Large Waves (Band 0 & 1)
    wind_speed:      float = 30.0      # Maps to largeWindSpeed
    wind_direction:  float = 0.0       # Maps to largeWindOrientation
    choppiness:      float = 1.3
    chaos:           float = 0.8       # Maps to largeChaos
    wave_amplitude:  float = 1.0
    
    band0_multiplier: float = 1.0
    band1_multiplier: float = 1.0
    
    # Currents
    current_speed:       float = 0.0
    current_orientation: float = 0.0
    
    # Ripples (Fine detail)
    ripples_enabled:     bool  = True
    ripples_wind_speed:  float = 8.0
    ripples_wind_dir:    float = 0.0
    ripples_chaos:       float = 0.8
    
    # ✨ Material & Scattering (HDRP-precise)
    refraction_color: List[float] = field(default_factory=lambda: [0.10, 0.70, 0.85]) # Vibrant Cyan
    scattering_color: List[float] = field(default_factory=lambda: [0.00, 0.40, 0.55]) # Tropical Deep
    absorption_distance: float = 12.0
    ambient_scattering:  float = 0.2
    height_scattering:   float = 1.2
    displacement_scattering: float = 0.5
    direct_light_tip_scattering:  float = 1.0
    direct_light_body_scattering: float = 0.8
    
    smoothness:          float = 0.98 # Very glossy
    transparency:        float = 0.85 # More solid surface
    reflection:          float = 1.0
    
    # 🧼 Foam
    foam_enabled:    bool  = True
    foam_amount:     float = 0.3
    foam_smoothness: float = 1.0
    
    # 💎 Caustics
    caustics_enabled:   bool  = True
    caustics_intensity: float = 0.5
    
    # 🔥 Physics
    storm_intensity: float = 0.0
    buoyancy:        float = 0.8


@dataclass
class SceneConfig:
    # ── Asset paths ──────────────────────────────────────────────
    models:     List[ObjectConfig] = field(default_factory=list)
    hdri_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    
    # ── Randomization Toggles ────────────────────────────────────
    rand_pose:      bool = True
    rand_hdri:      bool = True
    rand_lighting:  bool = True
    rand_transform: bool = True
    rand_weather:   bool = True
    
    # ── Global / HDRI Randomizers ────────────────────────────────
    hdri_randomizers: List[RandomizerInstance] = field(default_factory=list)

    # ── Scene objects (placed instances in viewport) ──────────────
    scene_objects: List["SceneObject"] = field(default_factory=list)

    # ── Annotation ───────────────────────────────────────────────
    annotation_type: str = "bbox2d"  # bbox2d | bbox3d | segmentation | keypoints | depth
    export_format:   str = "yolo"    # yolo | coco | kitti | pascal_voc

    # ── Generation ───────────────────────────────────────────────
    num_images:     int   = 500
    image_width:    int   = 640
    image_height:   int   = 480
    num_obj_max:    int   = 3
    bg_only_prob:   float = 0.10
    class_name:     str   = "Object" # Default / global
    output_format:  str   = "yolo"    # "yolo" | "coco"

    # ── HDRI ─────────────────────────────────────────────────────
    hdri_strength_min: float = 0.4
    hdri_strength_max: float = 2.5

    # ── Camera / Pose ─────────────────────────────────────────────
    # Standard normalized range (objects are ~2 units large)
    dist_min:  float =   2.0
    dist_max:  float =  12.0
    elev_min:  float =   5.0
    elev_max:  float =  60.0
    azim_min:  float =   0.0
    azim_max:  float = 360.0
    
    # ── Camera Lens (Focal Length mm -> FOV) ──
    focal_length: float = 50.0  # Standard 50mm lens
    fov_y:        float = 26.99 # Calculated from focal_length (Standard 35mm Full Frame 24mm height)

    # ── Lighting ─────────────────────────────────────────────────
    brightness_min: float = 0.5
    brightness_max: float = 1.6

    # ── Depth-aware transform ──────────────────────────────────────
    target_frac_min: float = 0.10
    target_frac_max: float = 0.35
    scale_jitter:    float = 0.15
    translation_max: float = 2.0

    # ── Sub-configs ───────────────────────────────────────────────
    weather:      WeatherConfig     = field(default_factory=WeatherConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    ocean:        OceanConfig       = field(default_factory=OceanConfig)

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

    @staticmethod
    def from_unity_json(data: dict) -> "SceneConfig":
        """Maps Unity PascalCase JSON (from JsonUtility) to Python snake_case SceneConfig."""
        cfg = SceneConfig()
        
        # 1. Global Settings
        cfg.num_images = data.get("numImages", cfg.num_images)
        cfg.image_width = data.get("imageSize", cfg.image_width)
        cfg.image_height = data.get("imageSize", cfg.image_height)
        
        # 2. Camera Ranges
        cfg.dist_min = data.get("distMin", cfg.dist_min)
        cfg.dist_max = data.get("distMax", cfg.dist_max)
        cfg.elev_min = data.get("elevMin", cfg.elev_min)
        cfg.elev_max = data.get("elevMax", cfg.elev_max)
        
        # 3. Import Scene Objects
        unity_objs = data.get("sceneObjects", [])
        cfg.scene_objects = []
        
        for uo in unity_objs:
            obj = SceneObject()
            obj.instance_id = str(uo.get("instanceId", "obj_" + str(uuid.uuid4())[:4]))
            obj.label = uo.get("label", "Object")
            
            # Position
            pos = uo.get("position", {"x": 0, "y": 0, "z": 0})
            obj.pos_x = pos.get("x", 0.0)
            obj.pos_y = pos.get("y", 0.0)
            obj.pos_z = pos.get("z", 0.0)
            
            # Rotation
            rot = uo.get("rotation", {"x": 0, "y": 0, "z": 0})
            obj.rot_x = rot.get("x", 0.0)
            obj.rot_y = rot.get("y", 0.0)
            obj.rot_z = rot.get("z", 0.0)
            
            # Scale
            obj.scale = uo.get("scale", 1.0)
            
            # PBR
            obj.metallic = uo.get("metallic", 0.0)
            obj.roughness = uo.get("roughness", 0.8)
            obj.floating = uo.get("floating", False)
            obj.is_focus_target = uo.get("isFocusTarget", False)
            
            cfg.scene_objects.append(obj)
            
        return cfg

    def validate(self) -> list[str]:
        """Return a list of validation error strings (empty = OK)."""
        errors = []
        if not self.models:
            errors.append("No 3D models imported.")
        else:
            for i, m in enumerate(self.models):
                if not m.mesh_path:
                    errors.append(f"Model {i+1} has no file path.")

        if not self.hdri_paths:
            errors.append("No HDRI backgrounds loaded.")
        if not self.output_dir:
            errors.append("No output directory set.")
        if self.num_images < 1:
            errors.append("Number of images must be ≥ 1.")
        return errors
