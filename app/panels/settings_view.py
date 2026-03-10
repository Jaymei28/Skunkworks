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

        # ── Group: Randomization Engine ─────────────────────────────
        rand_card = QFrame()
        rand_card.setObjectName("settings_card")
        rl = QGridLayout(rand_card)
        rl.setContentsMargins(25, 25, 25, 25)
        rl.setSpacing(15)

        l_rand = QLabel("Randomization Engine")
        l_rand.setObjectName("settings_section_title")
        rl.addWidget(l_rand, 0, 0, 1, 4)

        # Toggles in 2 columns
        self.chk_pose = QCheckBox("Camera Motion")
        self.chk_pose.setChecked(self.cfg.rand_pose)
        rl.addWidget(self.chk_pose, 1, 0, 1, 2)

        self.chk_hdri = QCheckBox("Environment")
        self.chk_hdri.setChecked(self.cfg.rand_hdri)
        rl.addWidget(self.chk_hdri, 1, 2, 1, 2)

        self.chk_light = QCheckBox("Sun/Lighting")
        self.chk_light.setChecked(self.cfg.rand_lighting)
        rl.addWidget(self.chk_light, 2, 0, 1, 2)

        self.chk_trans = QCheckBox("Object Jitter")
        self.chk_trans.setChecked(self.cfg.rand_transform)
        rl.addWidget(self.chk_trans, 2, 2, 1, 2)

        self.chk_weather = QCheckBox("Weather System")
        self.chk_weather.setChecked(self.cfg.rand_weather)
        self.chk_weather.setToolTip("Randomize fog and rain for every image.")
        rl.addWidget(self.chk_weather, 3, 0, 1, 2)

        # Fisheye nicely integrated
        fe_box = QHBoxLayout()
        self.chk_fisheye = QCheckBox("Lens Distortion")
        self.chk_fisheye.setChecked(self.cfg.post_process.fisheye_enabled)
        fe_box.addWidget(self.chk_fisheye)
        fe_box.addSpacing(5)
        self.fe_spin = QDoubleSpinBox()
        self.fe_spin.setRange(-1.0, 1.0)
        self.fe_spin.setValue(self.cfg.post_process.fisheye_intensity)
        self.fe_spin.setFixedWidth(60)
        fe_box.addWidget(self.fe_spin)
        rl.addLayout(fe_box, 3, 2, 1, 2)

        # Separator for ranges
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Plain)
        sep.setStyleSheet("background-color: #30363d; margin: 10px 0;")
        rl.addWidget(sep, 4, 0, 1, 4)

        # Global Ranges
        def add_range_row(row_idx, label, attr_min, attr_max, r_min=0, r_max=100):
            lbl = QLabel(label)
            lbl.setObjectName("property_label")
            rl.addWidget(lbl, row_idx, 0)
            
            spin_min = QDoubleSpinBox()
            spin_min.setRange(r_min, r_max)
            spin_min.setValue(attr_min)
            spin_min.setFixedWidth(70)
            
            spin_max = QDoubleSpinBox()
            spin_max.setRange(r_min, r_max)
            spin_max.setValue(attr_max)
            spin_max.setFixedWidth(70)

            rl.addWidget(QLabel("Min:"), row_idx, 1, Qt.AlignRight)
            rl.addWidget(spin_min, row_idx, 2, Qt.AlignLeft)
            rl.addWidget(QLabel("Max:"), row_idx, 3, Qt.AlignRight)
            rl.addWidget(spin_max, row_idx, 3, Qt.AlignLeft) # Wait, adjust column indices
            
            # Re-do layout for range row to be more precise
            range_h = QHBoxLayout()
            range_h.addWidget(QLabel("Min:"))
            range_h.addWidget(spin_min)
            range_h.addSpacing(15)
            range_h.addWidget(QLabel("Max:"))
            range_h.addWidget(spin_max)
            range_h.addStretch()
            rl.addLayout(range_h, row_idx, 1, 1, 3)
            return spin_min, spin_max

        self.cam_dist_min, self.cam_dist_max = add_range_row(5, "Distance Window", self.cfg.dist_min, self.cfg.dist_max, 0.5, 50)
        self.cam_elev_min, self.cam_elev_max = add_range_row(6, "Elevation Range", self.cfg.elev_min, self.cfg.elev_max, -90, 90)
        self.hdri_str_min, self.hdri_str_max = add_range_row(7, "Env Intensity", self.cfg.hdri_strength_min, self.cfg.hdri_strength_max, 0, 10)
        self.light_int_min, self.light_int_max = add_range_row(8, "Sun Luminance", self.cfg.brightness_min, self.cfg.brightness_max, 0, 10)

        self.main_v.addWidget(rand_card)
        self.main_v.addStretch()

        # Wire signals
        self.w_spin.valueChanged.connect(self._sync)
        self.h_spin.valueChanged.connect(self._sync)
        self.num_spin.valueChanged.connect(self._sync)
        self.fmt_combo.currentTextChanged.connect(self._sync)
        self.anno_combo.currentIndexChanged.connect(self._sync)
        self.fmt_combo.currentIndexChanged.connect(self._sync)
        self.chk_pose.stateChanged.connect(self._sync)
        self.chk_hdri.stateChanged.connect(self._sync)
        self.chk_light.stateChanged.connect(self._sync)
        self.chk_trans.stateChanged.connect(self._sync)
        self.chk_weather.stateChanged.connect(self._sync)
        self.chk_fisheye.stateChanged.connect(self._sync)
        self.fe_spin.valueChanged.connect(self._sync)
        
        self.cam_dist_min.valueChanged.connect(self._sync)
        self.cam_dist_max.valueChanged.connect(self._sync)
        self.cam_elev_min.valueChanged.connect(self._sync)
        self.cam_elev_max.valueChanged.connect(self._sync)
        self.hdri_str_min.valueChanged.connect(self._sync)
        self.hdri_str_max.valueChanged.connect(self._sync)
        self.light_int_min.valueChanged.connect(self._sync)
        self.light_int_max.valueChanged.connect(self._sync)

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
        
        # Sync Toggles
        self.cfg.rand_pose = self.chk_pose.isChecked()
        self.cfg.rand_hdri = self.chk_hdri.isChecked()
        self.cfg.rand_lighting = self.chk_light.isChecked()
        self.cfg.rand_transform = self.chk_trans.isChecked()
        self.cfg.rand_weather = self.chk_weather.isChecked()
        
        self.cfg.post_process.fisheye_enabled = self.chk_fisheye.isChecked()
        self.cfg.post_process.fisheye_intensity = self.fe_spin.value()
        
        self.cfg.dist_min = self.cam_dist_min.value()
        self.cfg.dist_max = self.cam_dist_max.value()
        self.cfg.elev_min = self.cam_elev_min.value()
        self.cfg.elev_max = self.cam_elev_max.value()
        self.cfg.hdri_strength_min = self.hdri_str_min.value()
        self.cfg.hdri_strength_max = self.hdri_str_max.value()
        self.cfg.brightness_min = self.light_int_min.value()
        self.cfg.brightness_max = self.light_int_max.value()
        
        self.config_updated.emit()
