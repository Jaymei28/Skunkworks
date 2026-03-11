"""
panels/settings_view.py
======================
A view for global generation settings (image size, dataset format, etc).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
    QComboBox, QFileDialog, QPushButton, QFrame, QSpacerItem, 
    QSizePolicy, QCheckBox, QDoubleSpinBox, QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, Signal
import os

from app.engine.scene_state import SceneConfig


class SettingsPanel(QWidget):
    """View for configuring global dataset parameters."""
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setObjectName("settings_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        # Title
        title_v = QVBoxLayout()
        title = QLabel("Global Settings")
        title.setObjectName("view_title")
        title_v.addWidget(title)
        subtitle = QLabel("Configure image resolution, dataset size, and export format.")
        subtitle.setObjectName("view_subtitle")
        title_v.addWidget(subtitle)
        layout.addLayout(title_v)

        # ── Scrollable Area ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self.main_v = QVBoxLayout(content)
        self.main_v.setContentsMargins(0, 0, 40, 0)
        self.main_v.setSpacing(30)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # ── Group: Project & Output ────────────────────────────────
        project_card = QFrame()
        project_card.setObjectName("settings_card")
        pl = QGridLayout(project_card)
        pl.setContentsMargins(25, 25, 25, 25)
        pl.setSpacing(15)
        
        l_proj = QLabel("Asset & Output Settings")
        l_proj.setObjectName("settings_section_title")
        pl.addWidget(l_proj, 0, 0, 1, 2)

        def add_prop(grid, text, widget, row, col_idx=0):
            lbl = QLabel(text)
            lbl.setObjectName("property_label")
            lbl.setFixedWidth(140)
            grid.addWidget(lbl, row, col_idx)
            if isinstance(widget, QWidget):
                widget.setMaximumWidth(220)
                grid.addWidget(widget, row, col_idx + 1, Qt.AlignLeft)
            else:
                grid.addLayout(widget, row, col_idx + 1, Qt.AlignLeft)

        # 1. Resolution
        res_h = QHBoxLayout()
        self.w_spin = QSpinBox()
        self.w_spin.setRange(128, 4096)
        self.w_spin.setValue(self.cfg.image_width)
        self.w_spin.setFixedWidth(80)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(128, 4096)
        self.h_spin.setValue(self.cfg.image_height)
        self.h_spin.setFixedWidth(80)
        res_h.addWidget(self.w_spin)
        res_h.addWidget(QLabel("×"))
        res_h.addWidget(self.h_spin)
        res_h.addWidget(QLabel("px"))
        add_prop(pl, "Target Resolution", res_h, 1)

        # 2. Batch Size
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 10000)
        self.num_spin.setValue(self.cfg.num_images)
        self.num_spin.setFixedWidth(100)
        add_prop(pl, "Dataset Capacity", self.num_spin, 2)

        # 3. Path
        path_h = QHBoxLayout()
        self.out_label = QLabel(self.cfg.output_dir or "Select a folder...")
        self.out_label.setObjectName("path_label_small")
        path_h.addWidget(self.out_label)
        btn_browse = QPushButton("Browse")
        btn_browse.setFixedWidth(70)
        btn_browse.setObjectName("secondary_btn_small")
        btn_browse.clicked.connect(self._on_browse)
        path_h.addWidget(btn_browse)
        add_prop(pl, "Export Directory", path_h, 3)

        self.main_v.addWidget(project_card)

        # ── Group: Perception Pipeline ──────────────────────────────
        anno_card = QFrame()
        anno_card.setObjectName("settings_card")
        al = QGridLayout(anno_card)
        al.setContentsMargins(25, 25, 25, 25)
        al.setSpacing(15)

        l_anno = QLabel("Perception & Format")
        l_anno.setObjectName("settings_section_title")
        al.addWidget(l_anno, 0, 0, 1, 2)

        self.anno_combo = QComboBox()
        self.anno_combo.addItems(["BBox 2D", "BBox 3D", "Segmentation", "Keypoints", "Depth"])
        anno_map = {"bbox2d": "BBox 2D", "bbox3d": "BBox 3D", "segmentation": "Segmentation", "keypoints": "Keypoints", "depth": "Depth"}
        self.anno_combo.setCurrentText(anno_map.get(self.cfg.annotation_type, "BBox 2D"))
        add_prop(al, "Annotation Type", self.anno_combo, 1)

        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["YOLO", "COCO", "Pascal VOC", "KITTI"])
        self.fmt_combo.setCurrentText(self.cfg.export_format.upper())
        add_prop(al, "Export Target", self.fmt_combo, 2)

        self.main_v.addWidget(anno_card)

        # ── Group: Camera & Lens ──────────────────────────────
        cam_card = QFrame()
        cam_card.setObjectName("settings_card")
        cl = QGridLayout(cam_card)
        cl.setContentsMargins(25, 25, 25, 25)
        cl.setSpacing(15)

        l_cam = QLabel("Camera & Lens")
        l_cam.setObjectName("settings_section_title")
        cl.addWidget(l_cam, 0, 0, 1, 2)

        # Presets
        self.lens_combo = QComboBox()
        self.lens_combo.addItems([
            "Wide (14mm)", "Wide (24mm)", "Standard (35mm)", 
            "Standard (50mm)", "Portrait (85mm)", "Telephoto (200mm)", "Custom"
        ])
        fl = getattr(self.cfg, 'focal_length', 50.0)
        preset_map = {14: "Wide (14mm)", 24: "Wide (24mm)", 35: "Standard (35mm)", 50: "Standard (50mm)", 85: "Portrait (85mm)", 200: "Telephoto (200mm)"}
        self.lens_combo.setCurrentText(preset_map.get(int(fl), "Custom"))
        add_prop(cl, "Lens Preset", self.lens_combo, 1)

        self.fl_spin = QDoubleSpinBox()
        self.fl_spin.setRange(1.0, 1000.0)
        self.fl_spin.setValue(fl)
        self.fl_spin.setSuffix(" mm")
        add_prop(cl, "Focal Length", self.fl_spin, 2)

        self.fov_lbl = QLabel(f"Field of View: {self.cfg.fov_y:.1f}°")
        self.fov_lbl.setObjectName("property_label")
        self.fov_lbl.setStyleSheet("color: #4fc3f7;")
        cl.addWidget(self.fov_lbl, 3, 1)

        self.main_v.addWidget(cam_card)

        self.main_v.addStretch()

        # Wire signals
        self.w_spin.valueChanged.connect(self._sync)
        self.h_spin.valueChanged.connect(self._sync)
        self.num_spin.valueChanged.connect(self._sync)
        self.fmt_combo.currentTextChanged.connect(self._sync)
        self.anno_combo.currentIndexChanged.connect(self._sync)
        self.fmt_combo.currentIndexChanged.connect(self._sync)
        self.lens_combo.currentTextChanged.connect(self._on_preset_changed)
        self.fl_spin.valueChanged.connect(self._sync)

    def _on_preset_changed(self, text):
        mapping = {
            "Wide (14mm)": 14.0, "Wide (24mm)": 24.0, "Standard (35mm)": 35.0,
            "Standard (50mm)": 50.0, "Portrait (85mm)": 85.0, "Telephoto (200mm)": 200.0
        }
        if text in mapping:
            self.fl_spin.setValue(mapping[text])

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.cfg.output_dir = path
            self.out_label.setText(path)
            self._sync()

    def _sync(self):
        """Push UI values into the SceneConfig object."""
        self.cfg.image_width = self.w_spin.value()
        self.cfg.image_height = self.h_spin.value()
        self.cfg.num_images = self.num_spin.value()
        
        # Annotation Map
        name_map = {"BBox 2D": "bbox2d", "BBox 3D": "bbox3d", "Segmentation": "segmentation", "Keypoints": "keypoints", "Depth": "depth"}
        self.cfg.annotation_type = name_map.get(self.anno_combo.currentText(), "bbox2d")
        self.cfg.export_format = self.fmt_combo.currentText().lower().replace(" ", "_")
        
        # (Global randomization sync now handled via Hierarchy)
        
        # Sync Camera
        import math
        fl = self.fl_spin.value()
        self.cfg.focal_length = fl
        self.cfg.fov_y = math.degrees(2.0 * math.atan(12.0 / max(0.1, fl)))
        self.fov_lbl.setText(f"Field of View: {self.cfg.fov_y:.1f}°")

        self.config_updated.emit()
