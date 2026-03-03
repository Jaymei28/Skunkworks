"""
panels/settings_view.py
======================
A view for global generation settings (image size, dataset format, etc).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
    QComboBox, QFileDialog, QPushButton, QFrame, QSpacerItem, 
    QSizePolicy
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

        # ── Settings Grid ──────────────────────────────────────────
        grid = QFrame()
        grid.setObjectName("settings_card")
        gl = QVBoxLayout(grid)
        gl.setContentsMargins(20, 20, 20, 20)
        gl.setSpacing(20)

        # 1. Output Size
        res_h = QHBoxLayout()
        res_h.addWidget(QLabel("Image Resolution (px):"))
        self.res_spin = QSpinBox()
        self.res_spin.setRange(128, 2048)
        self.res_spin.setSingleStep(128)
        self.res_spin.setValue(self.cfg.image_size)
        res_h.addWidget(self.res_spin)
        res_h.addStretch()
        gl.addLayout(res_h)

        # 2. Number of Images
        num_h = QHBoxLayout()
        num_h.addWidget(QLabel("Number of Images:"))
        self.num_spin = QSpinBox()
        self.num_spin.setRange(1, 10000)
        self.num_spin.setValue(self.cfg.num_images)
        num_h.addWidget(self.num_spin)
        num_h.addStretch()
        gl.addLayout(num_h)

        # 3. Export Format
        fmt_h = QHBoxLayout()
        fmt_h.addWidget(QLabel("Export Format:"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.addItems(["YOLO", "COCO"])
        self.fmt_combo.setCurrentText(self.cfg.output_format.upper())
        fmt_h.addWidget(self.fmt_combo)
        fmt_h.addStretch()
        gl.addLayout(fmt_h)

        # 4. Output Directory
        out_h = QHBoxLayout()
        out_h.addWidget(QLabel("Output Folder:"))
        self.out_label = QLabel(self.cfg.output_dir or "Not set...")
        self.out_label.setObjectName("path_label_small")
        out_h.addWidget(self.out_label)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName("secondary_btn_small")
        browse_btn.clicked.connect(self._on_browse)
        out_h.addWidget(browse_btn)
        gl.addLayout(out_h)

        layout.addWidget(grid)
        layout.addStretch()

        # Wire signals
        self.res_spin.valueChanged.connect(self._sync)
        self.num_spin.valueChanged.connect(self._sync)
        self.fmt_combo.currentTextChanged.connect(self._sync)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.cfg.output_dir = path
            self.out_label.setText(path)
            self._sync()

    def _sync(self):
        self.cfg.image_size = self.res_spin.value()
        self.cfg.num_images = self.num_spin.value()
        self.cfg.output_format = self.fmt_combo.currentText().lower()
        self.config_updated.emit()
