import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QDoubleSpinBox, QComboBox, QStackedWidget, QCheckBox
)
from PySide6.QtCore import Qt, Signal
from app.engine.scene_state import RandomizerInstance

# Defined categories based on user arrangement
GLOBAL_RANDOMIZERS = [
    "LightingRandomizer",
    "LightRandomizer", 
    "HdriStrengthRandomizer",
    "WeatherRandomizer",
    "PostProcessRandomizer",
    "BloomRandomizer",
    "ExposureRandomizer",
    "NoiseRandomizer",
    "WhiteBalanceRandomizer",
    "CameraPoseRandomizer",
    "HueOffsetRandomizer",
    "AtmosphereRandomizer"
]

OBJECT_RANDOMIZERS = [
    "TransformationRandomizer",
    "DepthAwareTransformRandomizer",
    "DepthScaleRandomizer",
    "TextureRandomizer",
    "PoseRandomizer"
]

class RandomizerComponentWidget(QFrame):
    """A single randomizer 'component' for objects or HDRI."""
    removed = Signal()
    updated = Signal()

    def __init__(self, instance: RandomizerInstance, mode: str = "object", 
                 show_type_selector: bool = True, show_enabled_toggle: bool = True, 
                 show_delete_button: bool = True, show_header: bool = True, parent=None):
        super().__init__(parent)
        self.instance = instance
        self.mode = mode
        self.setObjectName("randomizer_component")
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        main_l = QVBoxLayout(self)
        main_l.setContentsMargins(10, 8, 10, 8)
        main_l.setSpacing(8)

        if show_header:
            header = QHBoxLayout()
            
            self.expand_btn = QPushButton("▼")
            self.expand_btn.setFixedSize(20, 20)
            self.expand_btn.setStyleSheet("QPushButton { background: transparent; color: #4fc3f7; border: none; font-weight: bold; }")
            self.expand_btn.setCheckable(True)
            self.expand_btn.setChecked(True)
            self.expand_btn.clicked.connect(self._on_toggle_expand)
            header.addWidget(self.expand_btn)

            if show_type_selector:
                self.type_dropdown = QComboBox()
                # Populate based on mode
                if mode == "global":
                    scripts = GLOBAL_RANDOMIZERS
                else:
                    scripts = OBJECT_RANDOMIZERS
                self.type_dropdown.addItems(scripts)
                if instance.type in scripts:
                    self.type_dropdown.setCurrentText(instance.type)
                elif scripts:
                    instance.type = scripts[0]
                    self.type_dropdown.setCurrentText(scripts[0])
                self.type_dropdown.setFixedHeight(22)
                header.addWidget(self.type_dropdown, stretch=1)
            else:
                self.type_lbl = QLabel(instance.type)
                self.type_lbl.setStyleSheet("color: #cdd6f4; font-weight: bold; font-size: 11px;")
                header.addWidget(self.type_lbl)
                header.addStretch()
            
            if show_enabled_toggle:
                self.enabled_chk = QCheckBox()
                self.enabled_chk.setChecked(instance.enabled)
                self.enabled_chk.setToolTip("Enable/Disable")
                self.enabled_chk.clicked.connect(self._on_toggle_enabled)
                header.addWidget(self.enabled_chk)

            if show_delete_button:
                del_btn = QPushButton("✕")
                del_btn.setObjectName("danger_btn_tiny")
                del_btn.setFixedSize(20, 20)
                del_btn.clicked.connect(self.removed.emit)
                header.addWidget(del_btn)
            
            main_l.addLayout(header)
        else:
            # We still need an expand btn somewhere? 
            # If no header, we just show the params stack.
            pass

        # Parameter Area
        self.params_stack = QStackedWidget()
        
        # 0. Placement (Reuse logic)
        self.placement_panel = QWidget()
        place_l = QHBoxLayout(self.placement_panel)
        place_l.addWidget(QLabel("Radius:"))
        self.place_radius = QSpinBox()
        self.place_radius.setRange(0, 1000)
        self.place_radius.setValue(instance.params.get("radius", 250))
        place_l.addWidget(self.place_radius)
        self.params_stack.addWidget(self.placement_panel)

        # 1. Rotation
        self.rotation_panel = QWidget()
        rot_l = QHBoxLayout(self.rotation_panel)
        rot_l.addWidget(QLabel("Range:"))
        self.rot_min = QSpinBox()
        self.rot_min.setRange(-360, 360)
        self.rot_min.setValue(instance.params.get("rot_min", 0))
        self.rot_max = QSpinBox()
        self.rot_max.setRange(-360, 360)
        self.rot_max.setValue(instance.params.get("rot_max", 360))
        rot_l.addWidget(self.rot_min)
        rot_l.addWidget(QLabel("to"))
        rot_l.addWidget(self.rot_max)
        self.params_stack.addWidget(self.rotation_panel)

        # 2. Hue
        self.hue_panel = QWidget()
        hue_l = QHBoxLayout(self.hue_panel)
        hue_l.addWidget(QLabel("Limit:"))
        self.hue_val = QDoubleSpinBox()
        self.hue_val.setRange(0, 1)
        self.hue_val.setValue(instance.params.get("hue_limit", 0.2))
        hue_l.addWidget(self.hue_val)
        self.params_stack.addWidget(self.hue_panel)

        # 3. Pose params (dist/elev/azim ranges)
        self.pose_panel = QWidget()
        pose_l = QHBoxLayout(self.pose_panel)
        pose_l.setSpacing(6)
        pose_l.addWidget(QLabel("Dist:"))
        self.pose_dist_min = QDoubleSpinBox()
        self.pose_dist_min.setRange(0, 10000)
        self.pose_dist_min.setValue(instance.params.get("dist_min", 400.0))
        self.pose_dist_max = QDoubleSpinBox()
        self.pose_dist_max.setRange(0, 10000)
        self.pose_dist_max.setValue(instance.params.get("dist_max", 800.0))
        pose_l.addWidget(self.pose_dist_min)
        pose_l.addWidget(QLabel("–"))
        pose_l.addWidget(self.pose_dist_max)
        pose_l.addWidget(QLabel("  Elev:"))
        self.pose_elev_min = QDoubleSpinBox()
        self.pose_elev_min.setRange(-90, 90)
        self.pose_elev_min.setValue(instance.params.get("elev_min", 0.0))
        self.pose_elev_max = QDoubleSpinBox()
        self.pose_elev_max.setRange(-90, 90)
        self.pose_elev_max.setValue(instance.params.get("elev_max", 90.0))
        pose_l.addWidget(self.pose_elev_min)
        pose_l.addWidget(QLabel("–"))
        pose_l.addWidget(self.pose_elev_max)
        self.params_stack.addWidget(self.pose_panel)

        # 4. Global: Placeholder/Generic
        self.params_stack.addWidget(QLabel("(No parameters for this randomizer)"))

        # 5. Global: Hdri Intensity
        self.hdri_panel = QWidget()
        hdri_l = QHBoxLayout(self.hdri_panel)
        hdri_l.addWidget(QLabel("Range:"))
        self.hdri_min = QDoubleSpinBox()
        self.hdri_min.setRange(0, 10)
        self.hdri_min.setValue(instance.params.get("strength_min", 0.5))
        self.hdri_max = QDoubleSpinBox()
        self.hdri_max.setRange(0, 10)
        self.hdri_max.setValue(instance.params.get("strength_max", 1.5))
        hdri_l.addWidget(self.hdri_min)
        hdri_l.addWidget(QLabel("–"))
        hdri_l.addWidget(self.hdri_max)
        self.params_stack.addWidget(self.hdri_panel)

        # 6. Global: Lighting/Sun
        self.light_panel = QWidget()
        light_l = QHBoxLayout(self.light_panel)
        light_l.addWidget(QLabel("Luminance:"))
        self.light_min = QDoubleSpinBox()
        self.light_min.setRange(0, 20)
        self.light_min.setValue(instance.params.get("brightness_min", 1.0))
        self.light_max = QDoubleSpinBox()
        self.light_max.setRange(0, 20)
        self.light_max.setValue(instance.params.get("brightness_max", 5.0))
        light_l.addWidget(self.light_min)
        light_l.addWidget(QLabel("–"))
        light_l.addWidget(self.light_max)
        self.params_stack.addWidget(self.light_panel)

        # 7. Global: Weather
        self.weather_panel = QWidget()
        weather_l = QHBoxLayout(self.weather_panel)
        weather_l.addWidget(QLabel("Intensity:"))
        self.weather_val_min = QDoubleSpinBox()
        self.weather_val_min.setRange(0, 1)
        self.weather_val_min.setValue(instance.params.get("intensity_min", 0.2))
        self.weather_val_max = QDoubleSpinBox()
        self.weather_val_max.setRange(0, 1)
        self.weather_val_max.setValue(instance.params.get("intensity_max", 0.8))
        weather_l.addWidget(self.weather_val_min)
        weather_l.addWidget(QLabel("–"))
        weather_l.addWidget(self.weather_val_max)
        self.params_stack.addWidget(self.weather_panel)

        # 8. Global: PostProcess (Fisheye)
        self.post_panel = QWidget()
        post_l = QHBoxLayout(self.post_panel)
        post_l.addWidget(QLabel("Distortion:"))
        self.post_val = QDoubleSpinBox()
        self.post_val.setRange(-1, 1)
        self.post_val.setValue(instance.params.get("fisheye_strength", 0.0))
        post_l.addWidget(self.post_val)
        self.params_stack.addWidget(self.post_panel)

        # 9. Global: Atmosphere (Fog)
        self.atmos_panel = QWidget()
        atmos_l = QHBoxLayout(self.atmos_panel)
        atmos_l.addWidget(QLabel("Fog Density:"))
        self.atmos_fog_min = QDoubleSpinBox()
        self.atmos_fog_min.setRange(0, 0.5)
        self.atmos_fog_min.setSingleStep(0.01)
        self.atmos_fog_min.setValue(instance.params.get("fog_min", 0.01))
        self.atmos_fog_max = QDoubleSpinBox()
        self.atmos_fog_max.setRange(0, 0.5)
        self.atmos_fog_max.setSingleStep(0.01)
        self.atmos_fog_max.setValue(instance.params.get("fog_max", 0.10))
        atmos_l.addWidget(self.atmos_fog_min)
        atmos_l.addWidget(QLabel("–"))
        atmos_l.addWidget(self.atmos_fog_max)
        self.params_stack.addWidget(self.atmos_panel)

        # 10. Global: Bloom
        self.bloom_panel = QWidget()
        bloom_l = QHBoxLayout(self.bloom_panel)
        bloom_l.addWidget(QLabel("Intensity:"))
        self.bloom_min = QDoubleSpinBox()
        self.bloom_min.setRange(0, 5)
        self.bloom_min.setValue(instance.params.get("bloom_min", 0.0))
        self.bloom_max = QDoubleSpinBox()
        self.bloom_max.setRange(0, 5)
        self.bloom_max.setValue(instance.params.get("bloom_max", 0.5))
        bloom_l.addWidget(self.bloom_min)
        bloom_l.addWidget(QLabel("–"))
        bloom_l.addWidget(self.bloom_max)
        self.params_stack.addWidget(self.bloom_panel)

        # 11. Global: Exposure
        self.exposure_panel = QWidget()
        exp_l = QHBoxLayout(self.exposure_panel)
        exp_l.addWidget(QLabel("Range:"))
        self.exp_min = QDoubleSpinBox()
        self.exp_min.setRange(-5, 5)
        self.exp_min.setValue(instance.params.get("exposure_min", -0.5))
        self.exp_max = QDoubleSpinBox()
        self.exp_max.setRange(-5, 5)
        self.exp_max.setValue(instance.params.get("exposure_max", 0.5))
        exp_l.addWidget(self.exp_min)
        exp_l.addWidget(QLabel("–"))
        exp_l.addWidget(self.exp_max)
        self.params_stack.addWidget(self.exposure_panel)

        # 12. Global: Noise
        self.noise_panel = QWidget()
        noise_l = QHBoxLayout(self.noise_panel)
        noise_l.addWidget(QLabel("Intensity:"))
        self.noise_max = QDoubleSpinBox()
        self.noise_max.setRange(0, 0.5)
        self.noise_max.setValue(instance.params.get("noise_max", 0.05))
        noise_l.addWidget(self.noise_max)
        self.params_stack.addWidget(self.noise_panel)

        # 13. Global: White Balance
        self.wb_panel = QWidget()
        wb_l = QHBoxLayout(self.wb_panel)
        wb_l.addWidget(QLabel("Temp Range:"))
        self.wb_min = QDoubleSpinBox()
        self.wb_min.setRange(2000, 12000)
        self.wb_min.setSingleStep(500)
        self.wb_min.setValue(instance.params.get("wb_temp_min", 4500))
        self.wb_max = QDoubleSpinBox()
        self.wb_max.setRange(2000, 12000)
        self.wb_max.setSingleStep(500)
        self.wb_max.setValue(instance.params.get("wb_temp_max", 8500))
        wb_l.addWidget(self.wb_min)
        wb_l.addWidget(QLabel("–"))
        wb_l.addWidget(self.wb_max)
        self.params_stack.addWidget(self.wb_panel)
        
        main_l.addWidget(self.params_stack)

        # Signals
        if hasattr(self, "type_dropdown"):
            self.type_dropdown.currentTextChanged.connect(self._on_type_changed)
        self.place_radius.valueChanged.connect(self._sync)
        self.rot_min.valueChanged.connect(self._sync)
        self.rot_max.valueChanged.connect(self._sync)
        self.hue_val.valueChanged.connect(self._sync)
        self.hdri_min.valueChanged.connect(self._sync)
        self.hdri_max.valueChanged.connect(self._sync)
        self.light_min.valueChanged.connect(self._sync)
        self.light_max.valueChanged.connect(self._sync)
        self.weather_val_min.valueChanged.connect(self._sync)
        self.weather_val_max.valueChanged.connect(self._sync)
        self.post_val.valueChanged.connect(self._sync)
        self.atmos_fog_min.valueChanged.connect(self._sync)
        self.atmos_fog_max.valueChanged.connect(self._sync)
        self.bloom_min.valueChanged.connect(self._sync)
        self.bloom_max.valueChanged.connect(self._sync)
        self.exp_min.valueChanged.connect(self._sync)
        self.exp_max.valueChanged.connect(self._sync)
        self.noise_max.valueChanged.connect(self._sync)
        self.wb_min.valueChanged.connect(self._sync)
        self.wb_max.valueChanged.connect(self._sync)
        
        if hasattr(self, "type_dropdown"):
            self._on_type_changed(self.type_dropdown.currentText())
        else:
            self._on_type_changed(self.instance.type)

    def _on_toggle_enabled(self, checked):
        self.instance.enabled = checked
        self.updated.emit()

    def _on_toggle_expand(self, checked):
        if checked:
            self.expand_btn.setText("▼")
            self.params_stack.show()
        else:
            self.expand_btn.setText("▶")
            self.params_stack.hide()

    def _on_type_changed(self, text):
        t = text.lower()
        if "camera" in t or "pose" in t:
            idx = 3  # pose panel
        elif "hdri" in t:
            idx = 5  # hdri panel
        elif "lighting" in t:
            idx = 6  # light panel
        elif "weather" in t:
            idx = 7  # weather panel
        elif "post" in t:
            idx = 8  # post process
        elif "atmos" in t:
            idx = 9  # atmosphere
        elif "bloom" in t:
            idx = 10 # bloom
        elif "exposure" in t:
            idx = 11 # exposure
        elif "noise" in t:
            idx = 12 # noise
        elif "white" in t or "balance" in t:
            idx = 13 # wb
        elif "hue" in t or "texture" in t:
            idx = 2  # hue
        elif "transform" in t or "rotation" in t:
            idx = 1  # rotation
        elif "placement" in t or "radius" in t:
            idx = 0  # placement
        else:
            idx = 4  # generic
        
        self.params_stack.setCurrentIndex(idx)
        self._sync()

    def _sync(self):
        if hasattr(self, "type_dropdown"):
            self.instance.type = self.type_dropdown.currentText()
        t = self.instance.type.lower()
        
        if "pose" in t:
            self.instance.params = {
                "dist_min": self.pose_dist_min.value(),
                "dist_max": self.pose_dist_max.value(),
                "elev_min": self.pose_elev_min.value(),
                "elev_max": self.pose_elev_max.value(),
            }
        elif "hdri" in t:
            self.instance.params = {"strength_min": self.hdri_min.value(), "strength_max": self.hdri_max.value()}
        elif "light" in t:
            self.instance.params = {"brightness_min": self.light_min.value(), "brightness_max": self.light_max.value()}
        elif "weather" in t:
            self.instance.params = {"intensity_min": self.weather_val_min.value(), "intensity_max": self.weather_val_max.value()}
        elif "post" in t:
            self.instance.params = {"fisheye_strength": self.post_val.value()}
        elif "atmos" in t:
            self.instance.params = {"fog_min": self.atmos_fog_min.value(), "fog_max": self.atmos_fog_max.value()}
        elif "bloom" in t:
            self.instance.params = {"bloom_min": self.bloom_min.value(), "bloom_max": self.bloom_max.value()}
        elif "exposure" in t:
            self.instance.params = {"exposure_min": self.exp_min.value(), "exposure_max": self.exp_max.value()}
        elif "noise" in t:
            self.instance.params = {"noise_max": self.noise_max.value()}
        elif "white" in t or "balance" in t:
            self.instance.params = {"wb_temp_min": self.wb_min.value(), "wb_temp_max": self.wb_max.value()}
        elif "hue" in t or "texture" in t:
            self.instance.params = {"hue_limit": self.hue_val.value()}
        elif "transform" in t or "rotation" in t:
            self.instance.params = {"rot_min": self.rot_min.value(), "rot_max": self.rot_max.value()}
        else:
            self.instance.params = {"radius": self.place_radius.value()}
            
        self.updated.emit()
