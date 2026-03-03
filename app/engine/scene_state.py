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

    @property
    def display_name(self) -> str:
        base = self.config.name or "Object"
        return f"{base} [{self.label}]" if self.label else base

@dataclass
class WeatherConfig:
    enabled: bool = True
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
    ao_min:            float =  0.0
    ao_max:            float =  0.35

    wb_enabled:        bool  = True
    wb_temp_min:       float =  3500.0
    wb_temp_max:       float =  8500.0

    blur_enabled:      bool  = True
    blur_min:          float =  0.0
    blur_max:          float =  1.8


@dataclass
class SceneConfig:
    # ── Asset paths ──────────────────────────────────────────────
    models:     List[ObjectConfig] = field(default_factory=list)
    hdri_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    
    # ── Global / HDRI Randomizers ────────────────────────────────
    hdri_randomizers: List[RandomizerInstance] = field(default_factory=list)

    # ── Scene objects (placed instances in viewport) ──────────────
    scene_objects: List["SceneObject"] = field(default_factory=list)

    # ── Annotation ───────────────────────────────────────────────
    annotation_type: str = "bbox2d"  # bbox2d | bbox3d | segmentation | keypoints | depth
    export_format:   str = "yolo"    # yolo | coco | kitti | pascal_voc

    # ── Generation ───────────────────────────────────────────────
    num_images:     int   = 500
    image_size:     int   = 512
    num_obj_max:    int   = 3
    bg_only_prob:   float = 0.10
    class_name:     str   = "Object" # Default / global
    output_format:  str   = "yolo"    # "yolo" | "coco"

    # ── HDRI ─────────────────────────────────────────────────────
    hdri_strength_min: float = 0.4
    hdri_strength_max: float = 2.5

    # ── Camera / Pose ─────────────────────────────────────────────
    dist_min:  float = 400.0
    dist_max:  float = 800.0
    elev_min:  float =   0.0
    elev_max:  float =  90.0
    azim_min:  float =   0.0
    azim_max:  float = 360.0
    fov_y:     float =  60.0

    # ── Lighting ─────────────────────────────────────────────────
    brightness_min: float = 0.5
    brightness_max: float = 1.6

    # ── Depth-aware transform ──────────────────────────────────────
    target_frac_min: float = 0.10
    target_frac_max: float = 0.35
    scale_jitter:    float = 0.15
    translation_max: float = 50.0

    # ── Sub-configs ───────────────────────────────────────────────
    weather:      WeatherConfig     = field(default_factory=WeatherConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)

    def to_dict(self):
        from dataclasses import asdict
        return asdict(self)

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
