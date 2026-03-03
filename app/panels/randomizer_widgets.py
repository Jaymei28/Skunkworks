import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QDoubleSpinBox, QComboBox, QStackedWidget
)
from PySide6.QtCore import Qt, Signal
from app.engine.scene_state import RandomizerInstance

class RandomizerComponentWidget(QFrame):
    """A single randomizer 'component' for objects or HDRI."""
    removed = Signal()
    updated = Signal()

    def __init__(self, instance: RandomizerInstance, parent=None):
        super().__init__(parent)
        self.instance = instance
        self.setObjectName("randomizer_component")
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        main_l = QVBoxLayout(self)
        main_l.setContentsMargins(10, 8, 10, 8)
        main_l.setSpacing(8)

        # Path to randomizers — relative to CWD so it works on any machine
        self.base_path = os.path.join(os.getcwd(), "renderer", "randomizers")

        # Header: Type and Remove
        header = QHBoxLayout()
        self.type_dropdown = QComboBox()
        
        # Populate from scripts
        scripts = []
        if os.path.exists(self.base_path):
            scripts = [f.replace(".cs", "") for f in os.listdir(self.base_path) 
                       if f.endswith(".cs")]
        
        if not scripts:
            scripts = ["MyFirstRandomizer"] # Fallback

        self.type_dropdown.addItems(scripts)
        
        if instance.type in scripts:
            self.type_dropdown.setCurrentText(instance.type)

        header.addWidget(self.type_dropdown)
        header.addStretch()
        
        del_btn = QPushButton("✕")
        del_btn.setObjectName("danger_btn_tiny")
        del_btn.setFixedSize(24, 24)
        del_btn.clicked.connect(self.removed.emit)
        header.addWidget(del_btn)
        main_l.addLayout(header)

        # Parameter Area
        self.params_stack = QStackedWidget()
        
        # 1. Placement (Reuse logic)
        self.placement_panel = QWidget()
        place_l = QHBoxLayout(self.placement_panel)
        place_l.addWidget(QLabel("Radius:"))
        self.place_radius = QSpinBox()
        self.place_radius.setRange(0, 1000)
        self.place_radius.setValue(instance.params.get("radius", 250))
        place_l.addWidget(self.place_radius)
        self.params_stack.addWidget(self.placement_panel)

        # 2. Rotation
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

        # 3. Hue
        self.hue_panel = QWidget()
        hue_l = QHBoxLayout(self.hue_panel)
        hue_l.addWidget(QLabel("Limit:"))
        self.hue_val = QDoubleSpinBox()
        self.hue_val.setRange(0, 1)
        self.hue_val.setValue(instance.params.get("hue_limit", 0.2))
        hue_l.addWidget(self.hue_val)
        self.params_stack.addWidget(self.hue_panel)

        # 4. Pose params (dist/elev/azim ranges)
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

        # Placeholder for other scripts
        self.params_stack.addWidget(QLabel("(Parameters handled globally via Settings)"))
        
        main_l.addWidget(self.params_stack)

        # Signals
        self.type_dropdown.currentTextChanged.connect(self._on_type_changed)
        self.place_radius.valueChanged.connect(self._sync)
        self.rot_min.valueChanged.connect(self._sync)
        self.rot_max.valueChanged.connect(self._sync)
        self.hue_val.valueChanged.connect(self._sync)
        self.pose_dist_min.valueChanged.connect(self._sync)
        self.pose_dist_max.valueChanged.connect(self._sync)
        self.pose_elev_min.valueChanged.connect(self._sync)
        self.pose_elev_max.valueChanged.connect(self._sync)
        
        self._on_type_changed(self.type_dropdown.currentText())

    def _on_type_changed(self, text):
        t = text.lower()
        # Map script name → param panel index
        if "pose" in t:
            idx = 3  # pose panel
        elif "transform" in t or "rotation" in t:
            idx = 1  # rotation panel
        elif "texture" in t or "hue" in t:
            idx = 2  # hue panel
        elif "placement" in t or "radius" in t:
            idx = 0  # placement panel
        else:
            idx = 4  # generic placeholder
        self.params_stack.setCurrentIndex(idx)
        self._sync()

    def _sync(self):
        self.instance.type = self.type_dropdown.currentText()
        t = self.instance.type.lower()
        if "pose" in t:
            self.instance.params = {
                "dist_min": self.pose_dist_min.value(),
                "dist_max": self.pose_dist_max.value(),
                "elev_min": self.pose_elev_min.value(),
                "elev_max": self.pose_elev_max.value(),
                "azim_min": 0.0,
                "azim_max": 360.0,
            }
        elif "transform" in t or "rotation" in t:
            self.instance.params = {
                "rot_min": self.rot_min.value(),
                "rot_max": self.rot_max.value(),
            }
        elif "texture" in t or "hue" in t:
            self.instance.params = {"hue_limit": self.hue_val.value()}
        else:
            self.instance.params = {"radius": self.place_radius.value()}
        self.updated.emit()
