"""
engine/worker.py
================
QThread-based generation worker.

Signals emitted:
    progress(int, int)         – (current, total)
    preview_ready(QImage)      – latest rendered frame (for viewport)
    log_message(str)           – text line for console
    finished(bool, str)        – (success, message)
    stats_updated(dict)        – live stats dict for the status bar
"""

import os
import sys
import random as _rng
random = _rng
import traceback

from PySide6.QtCore  import QThread, Signal
from PySide6.QtGui   import QImage

# Add project root so renderer imports work
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.renderer   import BlendParams, TexturesUV

from renderer.core                          import Renderer3D
from renderer.loader                        import MeshLoader
from renderer.background                    import HDRIBackground
from renderer.randomizers.pose              import PoseRandomizer
from renderer.randomizers.lighting          import LightingRandomizer
from renderer.randomizers.depth_scale       import DepthAwareTransformRandomizer
from renderer.randomizers.weather           import WeatherRandomizer
from renderer.randomizers.post_process      import PostProcessRandomizer
from renderer.annotators.common             import MaskAnnotator
from renderer.annotators.bbox               import BBoxAnnotator
from renderer.annotators.metadata           import MetadataAnnotator

# ── Logging Helper ──────────────────────────────────────────────────────────


class GeneratorWorker(QThread):
    """Runs the full PyTorch3D generation pipeline on a background thread."""

    progress      = Signal(int, int)       # (current, total)
    preview_ready = Signal(QImage)         # latest rendered frame
    log_message   = Signal(str)            # console text
    finished      = Signal(bool, str)      # (success, message)
    stats_updated = Signal(dict)           # live metrics

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config   = config
        self._stop    = False
        self._paused  = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def stop(self):
        self._stop = True

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _get_randomizer_inst(self, script_name: str, params: dict):
        """Dynamically load and instantiate a randomizer class from a script."""
        import importlib
        import inspect
        
        try:
            module_path = f"renderer.randomizers.{script_name}"
            # Reload if already imported to pick up script changes
            if module_path in sys.modules:
                module = importlib.reload(sys.modules[module_path])
            else:
                module = importlib.import_module(module_path)
            
            # 1. Try to find a class named ScriptnameRandomizer (e.g. PoseRandomizer)
            class_name = script_name.capitalize() + "Randomizer"
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                return cls(**params)
            
            # 2. Fallback: find any class that has an 'apply' method and isn't a base class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "apply") and name not in ["RandomizerBase", "BaseRandomizer"]:
                    return obj(**params)
                    
        except Exception as e:
            self._log(f"[Error] Failed to load randomizer '{script_name}': {e}")
        
        return None

    def _tensor_to_qimage(self, tensor) -> QImage:
        """Convert (H, W, 3) float32 [0,1] tensor → QImage RGB888."""
        import numpy as np
        arr = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        return QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self):
        try:
            self._run_generation()
        except Exception:
            tb = traceback.format_exc()
            self._log(f"[ERROR] {tb}")
            self.finished.emit(False, "Generation failed – see log for details.")

    def _run_generation(self):
        cfg    = self.config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log(f"[Init] Using device: {device}")

        # ---- Output dirs ----
        out_images = os.path.join(cfg.output_dir, "images")
        out_masks  = os.path.join(cfg.output_dir, "masks")
        out_labels = os.path.join(cfg.output_dir, "labels")
        for d in [out_images, out_masks, out_labels]:
            os.makedirs(d, exist_ok=True)

        # ---- Renderer ----
        self._log("[Init] Building renderer...")
        renderer = Renderer3D(image_size=cfg.image_size, device=device)
        renderer.shader.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

        mask_ann = MaskAnnotator()
        bbox_ann = BBoxAnnotator()
        meta_ann = MetadataAnnotator()

        # ---- Load meshes ----
        loaded_meshes = []
        for m_cfg in cfg.models:
            self._log(f"[Init] Loading mesh: {os.path.basename(m_cfg.mesh_path)} ({m_cfg.class_name})")
            m = MeshLoader.load(m_cfg.mesh_path, device=device)

            # ── Baseline normalization ─────────────────────────────────────────
            # OBJ files can be in any unit (mm, cm, m). Without normalization a
            # millimeter-scale mesh at dist=400-800 is sub-pixel in the render.
            # We normalize to a 240-unit bounding sphere (same as viewport/preview)
            # so objects are always visible. DepthScaleRandomizer rescales further.
            v = m.verts_packed()
            v_min, _ = v.min(dim=0)
            v_max, _ = v.max(dim=0)
            center = (v_min + v_max) / 2.0
            m.offset_verts_(-center.expand_as(v))
            span = (v_max - v_min).max().clamp(min=1e-6).item()
            m.scale_verts_(240.0 / span)
            self._log(f"[Init]   ↳ normalized: span={span:.4f} → scale factor={240.0/span:.2f}")
            # ─────────────────────────────────────────────────────────────────

            loaded_meshes.append((m_cfg, m))
        
        if not loaded_meshes:
            raise ValueError("No models loaded to generate.")

        self._log(f"[Init] {len(loaded_meshes)} model types loaded.")

        num_images = cfg.num_images
        self._log(f"[Start] Generating {num_images} images → {cfg.output_dir}")

        preview_every = max(1, num_images // 40)   # ~40 preview updates

        for i in range(num_images):

            # ---- Pause / stop ----
            while self._paused and not self._stop:
                self.msleep(100)
            if self._stop:
                self._log("[Stopped] Generation cancelled by user.")
                self.finished.emit(False, "Generation stopped.")
                return

            hdri_path = random.choice(cfg.hdri_paths) if cfg.hdri_paths else ""
            
            hdri_bg = HDRIBackground(
                hdri_path,
                device=device,
                strength_range=(cfg.hdri_strength_min, cfg.hdri_strength_max),
            )
            # HDRI strength stays at 1.0 (neutral) unless HdriStrengthRandomizer
            # component is added by the user below.
            hdri_bg.set_strength(1.0)

            # ---- Apply HDRI-level randomizers from UI ----
            weather_type      = "clear"
            weather_intensity = 0.0
            pp_rand_active    = None

            for r_inst in cfg.hdri_randomizers:
                try:
                    r_obj = self._get_randomizer_inst(r_inst.type, r_inst.params)
                    if r_obj is None:
                        continue

                    # Dispatch based on known script types
                    t = r_inst.type.lower()

                    if "hdristrength" in t or "hdri_strength" in t:
                        # HdriStrengthRandomizer — randomize strength
                        strength = random.uniform(
                            float(r_inst.params.get("strength_min", cfg.hdri_strength_min)),
                            float(r_inst.params.get("strength_max", cfg.hdri_strength_max)),
                        )
                        hdri_bg.set_strength(strength)

                    elif "weather" in t:
                        # WeatherRandomizer — pick type + intensity
                        weights = {
                            "clear":    float(r_inst.params.get("weight_clear",    0.35)),
                            "rain":     float(r_inst.params.get("weight_rain",     0.20)),
                            "fog":      float(r_inst.params.get("weight_fog",      0.20)),
                            "dust":     float(r_inst.params.get("weight_dust",     0.10)),
                            "overcast": float(r_inst.params.get("weight_overcast", 0.15)),
                        }
                        int_min = float(r_inst.params.get("intensity_min", 0.15))
                        int_max = float(r_inst.params.get("intensity_max", 0.75))
                        wr = WeatherRandomizer(
                            weights=weights,
                            intensity_range=(int_min, int_max),
                        )
                        wr.randomize()
                        weather_type, weather_intensity = wr.last_type, wr.last_intensity

                    elif "postprocess" in t or "post_process" in t:
                        # PostProcessRandomizer — configure and randomize
                        pc = cfg.post_process
                        pp_rand_active = PostProcessRandomizer(
                            exposure_range=(
                                float(r_inst.params.get("exposure_min", pc.exposure_min)),
                                float(r_inst.params.get("exposure_max", pc.exposure_max)),
                            ),
                            bloom_intensity_range=(
                                float(r_inst.params.get("bloom_min", pc.bloom_min)),
                                float(r_inst.params.get("bloom_max", pc.bloom_max)),
                            ),
                            bloom_threshold=float(r_inst.params.get("bloom_threshold", pc.bloom_threshold)),
                            noise_mode=r_inst.params.get("noise_mode", pc.noise_mode),
                            noise_intensity_range=(
                                float(r_inst.params.get("noise_min", pc.noise_min)),
                                float(r_inst.params.get("noise_max", pc.noise_max)),
                            ),
                            ao_intensity_range=(
                                float(r_inst.params.get("ao_min", pc.ao_min)),
                                float(r_inst.params.get("ao_max", pc.ao_max)),
                            ),
                            white_balance_temp_range=(
                                float(r_inst.params.get("wb_temp_min", pc.wb_temp_min)),
                                float(r_inst.params.get("wb_temp_max", pc.wb_temp_max)),
                            ),
                            blur_sigma_range=(
                                float(r_inst.params.get("blur_min", pc.blur_min)),
                                float(r_inst.params.get("blur_max", pc.blur_max)),
                            ),
                            enabled={
                                "exposure":          bool(r_inst.params.get("exposure_enabled", pc.exposure_enabled)),
                                "bloom":             bool(r_inst.params.get("bloom_enabled",    pc.bloom_enabled)),
                                "noise":             bool(r_inst.params.get("noise_enabled",    pc.noise_enabled)),
                                "ambient_occlusion": bool(r_inst.params.get("ao_enabled",       pc.ao_enabled)),
                                "white_balance":     bool(r_inst.params.get("wb_enabled",       pc.wb_enabled)),
                                "blur":              bool(r_inst.params.get("blur_enabled",      pc.blur_enabled)),
                            },
                        )
                        pp_rand_active.randomize()

                    else:
                        # Generic fallback: call apply on the renderer
                        r_obj.apply(renderer)

                except Exception as e:
                    self._log(f"[Warn] HDRI Rand '{r_inst.type}' failed: {e}")

            # ---- Build object list ----
            obj_list = []
            cls_list = []

            for m_cfg, mesh in loaded_meshes:
                num = random.randint(m_cfg.count_min, m_cfg.count_max)
                for _ in range(num):
                    m_instance = mesh.clone()

                    for r_inst in m_cfg.randomizers:
                        try:
                            r_obj = self._get_randomizer_inst(r_inst.type, r_inst.params)
                            if r_obj is None:
                                continue

                            t = r_inst.type.lower()

                            if "depthscale" in t or "depth_scale" in t:
                                # DepthScaleRandomizer — apply depth-aware transform
                                has_depth_scale = True
                                dist_min = float(r_inst.params.get("dist_min", cfg.dist_min))
                                dist_max = float(r_inst.params.get("dist_max", cfg.dist_max))
                                tfr = DepthAwareTransformRandomizer(
                                    dist_range=(dist_min, dist_max),
                                    target_fraction_range=(
                                        float(r_inst.params.get("target_fraction_min", cfg.target_frac_min)),
                                        float(r_inst.params.get("target_fraction_max", cfg.target_frac_max)),
                                    ),
                                    scale_jitter=float(r_inst.params.get("scale_jitter", cfg.scale_jitter)),
                                    translation_range=(
                                        -float(r_inst.params.get("translation_max", cfg.translation_max)),
                                         float(r_inst.params.get("translation_max", cfg.translation_max)),
                                    ),
                                    fov_y_deg=float(r_inst.params.get("fov_y", cfg.fov_y)),
                                    image_size=cfg.image_size,
                                )
                                m_instance, _ = tfr.apply(m_instance, device=device)

                            elif "pose" in t:
                                has_pose = True
                                pose_r = PoseRandomizer(
                                    dist_range=(
                                        float(r_inst.params.get("dist_min", cfg.dist_min)),
                                        float(r_inst.params.get("dist_max", cfg.dist_max)),
                                    ),
                                    elev_range=(
                                        float(r_inst.params.get("elev_min", cfg.elev_min)),
                                        float(r_inst.params.get("elev_max", cfg.elev_max)),
                                    ),
                                    azim_range=(
                                        float(r_inst.params.get("azim_min", cfg.azim_min)),
                                        float(r_inst.params.get("azim_max", cfg.azim_max)),
                                    ),
                                )
                                pose_r.apply(renderer)

                            elif "light" in t:
                                has_lighting = True
                                light_r = LightingRandomizer(
                                    brightness_range=(cfg.brightness_min, cfg.brightness_max)
                                )
                                light_r.apply(renderer)

                            else:
                                # Generic per-object randomizer
                                result = r_obj.apply(renderer, mesh=m_instance, device=device)
                                if result is not None:
                                    m_instance = result

                        except Exception as e:
                            self._log(f"[Warn] Object Rand '{r_inst.type}' failed: {e}")

                    obj_list.append(m_instance)
                    cls_list.append(m_cfg.class_name)

            num_objs = len(obj_list)

            # ---- Render each object individually and composite ----
            if num_objs > 0:
                comp_rgb   = None
                fg_accum   = None
                all_bboxes = []

                for m in obj_list:
                    tex = m.textures
                    try:
                        joined_tex = TexturesUV(
                            maps=tex.maps_list(),
                            faces_uvs=[tex.faces_uvs_list()[0]],
                            verts_uvs=[tex.verts_uvs_list()[0]],
                        ).to(device)
                        scene_i = Meshes(
                            verts=[m.verts_list()[0]],
                            faces=[m.faces_list()[0]],
                            textures=joined_tex,
                        ).to(device)
                    except Exception:
                        scene_i = m   # vertex-colour fallback

                    raw_i  = renderer.render(scene_i)
                    mask_i = mask_ann.annotate(renderer, scene_i)
                    fg_i   = (mask_i[0] >= 0).float()

                    if comp_rgb is None:
                        comp_rgb = raw_i[0, ..., :3].clone()
                        fg_accum = fg_i.clone()
                    else:
                        comp_rgb = raw_i[0, ..., :3] * fg_i[..., None] + comp_rgb * (1 - fg_i[..., None])
                        fg_accum = (fg_accum + fg_i).clamp(0, 1)

                    try:
                        bb = bbox_ann.annotate(renderer, scene_i)
                        all_bboxes.extend(bb)
                    except Exception:
                        pass

                raw_rgb = comp_rgb
                fg      = fg_accum
                bbox    = all_bboxes

            else:
                S       = cfg.image_size
                raw_rgb = torch.zeros((S, S, 3), device=device)
                fg      = torch.zeros((S, S), device=device)
                bbox    = []
                cls_list = []

            # ---- Composite over HDRI ----
            bg_rot = random.uniform(0, 360)
            bg     = hdri_bg.get_background(renderer, renderer.cameras, rotation_deg=bg_rot)
            # Tone-map using the configured strength (set by hdri_bg.set_strength or
            # randomize_strength). Dividing by bg.max() would ignore strength entirely.
            strength = getattr(hdri_bg, "strength", 1.0)
            p99  = torch.quantile(bg.reshape(-1), 0.99).clamp(min=1e-6)
            bg_n = (bg / p99).clamp(0, 1)
            comp = raw_rgb * fg[..., None] + bg_n * (1 - fg[..., None])

            # ---- Post-process (only if PostProcessRandomizer was added) ----
            if weather_intensity > 0 and weather_type != "clear":
                wr_apply = WeatherRandomizer(weights={"clear": 1.0}, intensity_range=(0, 0))
                wr_apply.last_type      = weather_type
                wr_apply.last_intensity = weather_intensity
                comp = wr_apply.apply(comp, weather_type, weather_intensity)

            if pp_rand_active is not None:
                comp = pp_rand_active.apply(comp)

            # ---- Save ----
            name    = f"{cfg.class_name.lower()}_{i:05d}"
            comp_np = comp.detach().cpu().numpy().clip(0, 1)
            plt.imsave(os.path.join(out_images, name + ".png"), comp_np)

            fg_np = fg.detach().cpu().numpy()
            plt.imsave(os.path.join(out_masks, "mask_" + name + ".png"), fg_np, cmap="gray")

            meta_ann.annotate(bbox, cls_list, os.path.join(out_labels, name + ".json"))

            # ---- Signals ----
            self.progress.emit(i + 1, num_images)

            if i % preview_every == 0:
                qi = self._tensor_to_qimage(comp)
                self.preview_ready.emit(qi)

            if i % max(1, num_images // 20) == 0 or i == num_images - 1:
                pct = (i + 1) / num_images * 100
                self._log(
                    f"[{pct:5.1f}%] {i+1}/{num_images}  "
                    f"weather={weather_type}  objs={num_objs}"
                )
                self.stats_updated.emit({
                    "generated": i + 1,
                    "total":     num_images,
                    "weather":   weather_type,
                    "device":    device,
                })

        self._log(f"[Done] {num_images} images saved to {cfg.output_dir}")
        self.finished.emit(True, f"Done! {num_images} images saved.")






# ─────────────────────────────────────────────────────────────────────────────
# Preview Worker  –  renders a single frame, no saving
# ─────────────────────────────────────────────────────────────────────────────

class PreviewWorker(QThread):
    """
    Renders one composited frame from the current SceneConfig and emits it
    as a QImage.  Used for the live viewport preview before generation starts.

    Signals:
        preview_ready(QImage)  – the composited render
        log_message(str)       – status text for the console
        finished()             – emitted when done (success or failure)
    """

    preview_ready = Signal(QImage)
    log_message   = Signal(str)
    finished      = Signal()

    def __init__(self, config, manual_pose=None, parent=None):
        super().__init__(parent)
        self.config = config
        self.manual_pose = manual_pose

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def _tensor_to_qimage(self, tensor) -> QImage:
        import numpy as np
        arr = (tensor.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        h, w, _ = arr.shape
        return QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()

    def run(self):
        try:
            self._render_preview()
        except Exception:
            tb = traceback.format_exc()
            self._log(f"[Preview Error] {tb}")
        finally:
            self.finished.emit()

    def _render_preview(self):
        cfg    = self.config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Render preview at a fixed 512px for maximum responsiveness
        preview_size = 512

        self._log(f"[Preview] Rendering {preview_size}px frame (HD mode) on {device}…")

        renderer = Renderer3D(image_size=preview_size, device=device)
        renderer.shader.blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

        # ── Load HDRI ──────────────────────────────────────────────────
        if not cfg.hdri_paths:
            # Fallback: create a blank neutral HDRI path logic or just skip HDRI loading
            hdri_path = ""
        else:
            hdri_path = random.choice(cfg.hdri_paths)

        hdri_bg = HDRIBackground(
            hdri_path,
            device=device,
            strength_range=(cfg.hdri_strength_min, cfg.hdri_strength_max),
        )
        hdri_bg.set_strength(
            (cfg.hdri_strength_min + cfg.hdri_strength_max) / 2.0
        )

        # ── Load meshes ────────────────────────────────────────────────
        if not cfg.models:
            self._log("[Preview Error] No models added.")
            return

        # Preview usually just one of the models
        m_cfg = cfg.models[0]
        base_mesh = MeshLoader.load(m_cfg.mesh_path, device=device)

        # ── Setup camera pose ──────────────────────────────────────────
        has_pose = any("pose" in r.type.lower() for m in cfg.models for r in m.randomizers)
        
        if self.manual_pose:
            d, e, a = self.manual_pose
            pose_gen = PoseRandomizer(dist_range=(d,d), elev_range=(e,e), azim_range=(a,a))
            R, T = pose_gen.get_specific_pose(d, e, a)
        elif has_pose:
            pose_gen = PoseRandomizer(
                dist_range=(cfg.dist_min, cfg.dist_max),
                elev_range=(cfg.elev_min, cfg.elev_max),
                azim_range=(cfg.azim_min, cfg.azim_max),
            )
            R, T = pose_gen.sample_pose()
        else:
            # Fixed default pose
            pose_gen = PoseRandomizer(dist_range=(cfg.dist_min, cfg.dist_min), elev_range=(cfg.elev_min, cfg.elev_min), azim_range=(cfg.azim_min, cfg.azim_min))
            R, T = pose_gen.get_specific_pose((cfg.dist_min+cfg.dist_max)/2, (cfg.elev_min+cfg.elev_max)/2, (cfg.azim_min+cfg.azim_max)/2)
            
        renderer.set_camera(R, T)

        # ── Lighting ──────────────────────────────────────────────────
        has_light = any("light" in r.type.lower() for m in cfg.models for r in m.randomizers)
        if has_light:
            light_rand = LightingRandomizer(brightness_range=(cfg.brightness_min, cfg.brightness_max))
            light_rand.apply(renderer)
        else:
            # Neutral fixed light
            renderer.lights.location = torch.tensor([[0.0, 10.0, 0.0]], device=device)
            renderer.lights.ambient_color = torch.tensor([[0.5, 0.5, 0.5]], device=device)
            renderer.lights.diffuse_color = torch.tensor([[0.5, 0.5, 0.5]], device=device)

        # ── Scale / Transform ─────────────────────────────────────────
        has_depth = any(("depth" in r.type.lower() or "scale" in r.type.lower()) for m in cfg.models for r in m.randomizers)
        if has_depth:
            transform_rand = DepthAwareTransformRandomizer(
                dist_range=(cfg.dist_min, cfg.dist_max),
                target_fraction_range=(0.20, 0.30),
                scale_jitter=0.05,
                rotation_range=(0, 360),
                translation_range=(-10.0, 10.0),
                fov_y_deg=cfg.fov_y,
                image_size=preview_size,
            )
            scaled_mesh, _ = transform_rand.apply(base_mesh.clone(), device=device)
        else:
            # Default scale: fit roughly in view
            scaled_mesh = base_mesh.clone()
            
            # Basic normalization for visibility
            verts_packed = scaled_mesh.verts_packed()
            v_min = verts_packed.min(dim=0)[0]
            v_max = verts_packed.max(dim=0)[0]
            center = (v_min + v_max) / 2.0
            
            # PyTorch3D offset_verts_ expects (sum(V), 3)
            offsets = -center.expand(verts_packed.shape[0], 3)
            scaled_mesh.offset_verts_(offsets)
            
            # Scale such that it fits nicely at the world-scale distances (dist=400-800)
            # Viewport Scale (~240) vs World Distance (~600)
            scale = 240.0 / (v_max - v_min).max().clamp(min=1e-6)
            scaled_mesh.scale_verts_(float(scale))

        # ── Detect texture type and render ────────────────────────────
        tex = scaled_mesh.textures
        try:
            # TexturesUV path (standard .obj)
            joined_tex = TexturesUV(
                maps=tex.maps_list(),
                faces_uvs=[tex.faces_uvs_list()[0]],
                verts_uvs=[tex.verts_uvs_list()[0]],
            ).to(device)
            scene = Meshes(
                verts=[scaled_mesh.verts_list()[0]],
                faces=[scaled_mesh.faces_list()[0]],
                textures=joined_tex,
            ).to(device)
        except Exception:
            # TexturesVertex fallback (.ply / vertex colours)
            scene = scaled_mesh

        raw = renderer.render(scene)

        # ── Mask / foreground ─────────────────────────────────────────
        mask_ann  = MaskAnnotator()
        mask      = mask_ann.annotate(renderer, scene)
        fg        = (mask[0] >= 0).float()

        # ── Composite over HDRI ───────────────────────────────────────
        bg_rot = _rng.uniform(0, 360)
        bg     = hdri_bg.get_background(renderer, renderer.cameras, rotation_deg=bg_rot)
        
        # Proper Tone mapping (ACES or val/(val+1)) for HDR backgrounds
        # Skunkworks neutral gray (0.25) should NOT become 1.0 (white)
        # We use a simple exposure + tone-mapping logic
        exposure = 1.0
        bg_n   = (bg * exposure) / (bg * exposure + 1.0)
        bg_n   = (bg_n * 1.5).clamp(0, 1) # Boost for preview visibility
        
        obj_rgb = raw[0, ..., :3]
        # Tone map the object render too
        obj_n   = (obj_rgb * exposure) / (obj_rgb * exposure + 1.0)
        obj_n   = (obj_n * 1.5).clamp(0, 1)

        comp    = obj_n * fg[..., None] + bg_n * (1 - fg[..., None])
        comp    = comp.clamp(0, 1)
        
        self._log(f"[Preview] Render complete: {comp.shape[1]}x{comp.shape[0]} @ {device}")

        qi = self._tensor_to_qimage(comp)
        self._log("[Preview] Done.")
        self.preview_ready.emit(qi)


# ─────────────────────────────────────────────────────────────────────────────
# GL-based workers  —  use the same OpenGL shaders as the live viewport.
# Drop-in replacements: identical Signal API to GeneratorWorker / PreviewWorker.
# No PyTorch3D dependency.
# ─────────────────────────────────────────────────────────────────────────────

class GLPreviewWorker(QThread):
    """
    Renders a single composited frame using the GL offscreen renderer.
    Replaces PreviewWorker — output is visually consistent with the live GL viewport.
    """
    preview_ready = Signal(QImage)
    log_message   = Signal(str)
    finished      = Signal()

    def __init__(self, config, azim: float = 45.0, elev: float = 25.0,
                 dist: float = 3.0, parent=None):
        super().__init__(parent)
        self.config = config
        self.azim   = azim
        self.elev   = elev
        self.dist   = dist
        # Surface must be created on the main thread
        from app.engine.gl_offscreen_renderer import GLOffscreenRenderer
        self._renderer = GLOffscreenRenderer(512, 512)

    def run(self):
        try:
            self._renderer.init_gl()

            cfg = self.config
            if cfg.hdri_paths:
                self._renderer.load_hdri(cfg.hdri_paths[0])

            for obj in cfg.scene_objects:
                if obj.config.mesh_path:
                    self._renderer.load_mesh_from_path(
                        obj.config.mesh_path, obj.config.tex_albedo)

            frame_np, _ = self._renderer.render_frame(
                cfg, self.azim, self.elev, self.dist, t=0.0)

            h, w, _ = frame_np.shape
            qi = QImage(frame_np.tobytes(), w, h, w * 3,
                        QImage.Format.Format_RGB888).copy()
            self.preview_ready.emit(qi)
        except Exception:
            tb = traceback.format_exc()
            self.log_message.emit(f"[GLPreview Error] {tb}")
        finally:
            try:
                self._renderer.cleanup()
            except Exception:
                pass
            self.finished.emit()


class GLGeneratorWorker(QThread):
    """
    Full batch generation pipeline using OpenGL — same renderer as the live preview.
    Drop-in replacement for GeneratorWorker (identical Signal API).
    """
    progress      = Signal(int, int)
    preview_ready = Signal(QImage)
    log_message   = Signal(str)
    finished      = Signal(bool, str)
    stats_updated = Signal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config  = config
        self._stop   = False
        self._paused = False
        # Surface created on main thread
        from app.engine.gl_offscreen_renderer import GLOffscreenRenderer
        sz = config.image_size
        self._renderer = GLOffscreenRenderer(sz, sz)

    def stop(self):    self._stop   = True
    def pause(self):   self._paused = True
    def resume(self):  self._paused = False

    def _log(self, msg: str):
        self.log_message.emit(msg)

    def run(self):
        try:
            self._run_generation()
        except Exception:
            tb = traceback.format_exc()
            self._log(f"[ERROR] {tb}")
            self.finished.emit(False, "GL generation failed — see log.")

    def _run_generation(self):
        import random as _rng
        import json
        cfg = self.config

        self._log("[GL Init] Initialising OpenGL offscreen renderer…")
        self._renderer.init_gl()
        self._log("[GL Init] Context ready.")

        # ── Output directories ────────────────────────────────────────────────
        out_images = os.path.join(cfg.output_dir, "images")
        out_labels = os.path.join(cfg.output_dir, "labels")
        out_masks  = os.path.join(cfg.output_dir, "masks")
        for d in [out_images, out_labels, out_masks]:
            os.makedirs(d, exist_ok=True)

        # ── Load HDRI ─────────────────────────────────────────────────────────
        if cfg.hdri_paths:
            hdri = _rng.choice(cfg.hdri_paths)
            self._log(f"[GL Init] Loading HDRI: {os.path.basename(hdri)}")
            self._renderer.load_hdri(hdri)

        # ── Load meshes ───────────────────────────────────────────────────────
        if not cfg.scene_objects:
            self._log("[GL Init] No scene objects to render — aborting.")
            self.finished.emit(False, "No scene objects in scene.")
            return

        loaded_paths = set()
        for obj in cfg.scene_objects:
            p = obj.config.mesh_path
            if p and p not in loaded_paths:
                self._log(f"[GL Init] Loading mesh: {os.path.basename(p)}")
                self._renderer.load_mesh_from_path(p, obj.config.tex_albedo)
                loaded_paths.add(p)

        self._log(f"[GL Start] Generating {cfg.num_images} images → {cfg.output_dir}")
        preview_every = max(1, cfg.num_images // 40)

        for i in range(cfg.num_images):
            # ── Pause / stop ──────────────────────────────────────────────────
            while self._paused and not self._stop:
                self.msleep(100)
            if self._stop:
                self._log("[Stopped] Generation cancelled.")
                self.finished.emit(False, "Generation stopped.")
                return

            # ── Randomize camera pose ─────────────────────────────────────────
            azim = _rng.uniform(cfg.azim_min, cfg.azim_max)
            elev = _rng.uniform(cfg.elev_min, cfg.elev_max)
            dist = _rng.uniform(cfg.dist_min, cfg.dist_max)
            t    = i * 0.12   # simulated time for wave animation

            # ── Render ────────────────────────────────────────────────────────
            frame_np, bboxes = self._renderer.render_frame(cfg, azim, elev, dist, t)

            # ── Save image ────────────────────────────────────────────────────
            name = f"{cfg.class_name.lower()}_{i:05d}"
            try:
                from PIL import Image
                Image.fromarray(frame_np).save(os.path.join(out_images, name + ".png"))
            except ImportError:
                import cv2
                cv2.imwrite(os.path.join(out_images, name + ".png"),
                            cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))

            # ── Save YOLO labels ──────────────────────────────────────────────
            label_lines = []
            # Build class name → index map
            cls_map: dict[str, int] = {}
            idx_counter = 0
            for cls_name, xc, yc, w, h in bboxes:
                if cls_name not in cls_map:
                    cls_map[cls_name] = idx_counter
                    idx_counter += 1
                label_lines.append(
                    f"{cls_map[cls_name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            label_path = os.path.join(out_labels, name + ".txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            # ── Signals ───────────────────────────────────────────────────────
            self.progress.emit(i + 1, cfg.num_images)

            if i % preview_every == 0:
                h, w, _ = frame_np.shape
                qi = QImage(frame_np.tobytes(), w, h, w * 3,
                            QImage.Format.Format_RGB888).copy()
                self.preview_ready.emit(qi)

            if i % max(1, cfg.num_images // 20) == 0 or i == cfg.num_images - 1:
                pct = (i + 1) / cfg.num_images * 100
                self._log(f"[{pct:5.1f}%] {i+1}/{cfg.num_images}  "
                          f"cam=az{azim:.0f}/el{elev:.0f}/d{dist:.0f}  "
                          f"objs={len(bboxes)}")
                self.stats_updated.emit({
                    "generated": i + 1,
                    "total":     cfg.num_images,
                    "weather":   "—",
                    "device":    "GL",
                })

        self._log(f"[GL Done] {cfg.num_images} images saved to {cfg.output_dir}")
        try:
            self._renderer.cleanup()
        except Exception:
            pass
        self.finished.emit(True, f"Done! {cfg.num_images} images saved.")
