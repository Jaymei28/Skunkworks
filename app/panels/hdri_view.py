"""
panels/hdri_view.py
===================
A view for managing multiple HDRI backgrounds.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QFrame, QFileDialog, QSpacerItem, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
import os

from app.engine.scene_state import SceneConfig, RandomizerInstance
from .randomizer_widgets import RandomizerComponentWidget


class HDRIEntryWidget(QFrame):
    """A single card for one imported HDRI."""
    removed = Signal()

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("hdri_entry")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedHeight(50)
        
        main_l = QHBoxLayout(self)
        main_l.setContentsMargins(12, 0, 12, 0)
        main_l.setSpacing(10)
        
        self.path_lbl = QLabel(os.path.basename(path))
        self.path_lbl.setToolTip(path)
        main_l.addWidget(self.path_lbl)
        
        main_l.addStretch()
        
        del_btn = QPushButton("✕")
        del_btn.setObjectName("danger_btn_tiny")
        del_btn.setFixedSize(24, 24)
        del_btn.clicked.connect(self.removed.emit)
        main_l.addWidget(del_btn)


class HDRIPanel(QWidget):
    """View for importing and managing multiple HDRI backgrounds."""
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setObjectName("hdri_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title
        title_v = QVBoxLayout()
        title = QLabel("HDRI Backgrounds")
        title.setObjectName("view_title")
        title_v.addWidget(title)
        subtitle = QLabel("Import environment maps and attach randomizers.")
        subtitle.setObjectName("view_subtitle")
        title_v.addWidget(subtitle)
        layout.addLayout(title_v)

        # Container for scroll area + buttons
        self.center_container = QWidget()
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        self.add_btn = QPushButton("Add New HDRI")
        self.add_btn.setObjectName("primary_btn")
        self.add_btn.setMinimumHeight(45)
        self.add_btn.clicked.connect(self._on_add_hdri)
        center_layout.addWidget(self.add_btn)

        # Scrollable list of files
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("hdri_scroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setMaximumHeight(200) # Keep files list reasonable
        
        self.file_list_container = QWidget()
        self.file_list_layout = QVBoxLayout(self.file_list_container)
        self.file_list_layout.setContentsMargins(0, 0, 10, 0)
        self.file_list_layout.setSpacing(8)
        self.file_list_layout.addStretch()
        
        self.scroll.setWidget(self.file_list_container)
        center_layout.addWidget(self.scroll)
        
        # ── Global Randomizers Section ──────────────────────────────
        rand_header = QHBoxLayout()
        rl = QLabel("Global / HDRI Randomizers")
        rl.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        rand_header.addWidget(rl)
        rand_header.addStretch()
        
        self.add_rand_btn = QPushButton("+ Add Randomizer")
        self.add_rand_btn.setObjectName("secondary_btn_small")
        self.add_rand_btn.clicked.connect(self._on_add_randomizer)
        rand_header.addWidget(self.add_rand_btn)
        center_layout.addLayout(rand_header)
        
        self.rand_list_layout = QVBoxLayout()
        self.rand_list_layout.setSpacing(6)
        center_layout.addLayout(self.rand_list_layout)

        layout.addWidget(self.center_container)
        
        self._refresh_list()
        self._refresh_randomizers()

    def _refresh_list(self):
        # Clear existing
        while self.file_list_layout.count() > 1:
            item = self.file_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add current config
        for idx, path in enumerate(self.cfg.hdri_paths):
            entry = HDRIEntryWidget(path)
            entry.removed.connect(lambda i=idx: self._on_remove_hdri(i))
            self.file_list_layout.insertWidget(idx, entry)

    def _refresh_randomizers(self):
        while self.rand_list_layout.count():
            item = self.rand_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for idx, inst in enumerate(self.cfg.hdri_randomizers):
            comp = RandomizerComponentWidget(inst)
            comp.removed.connect(lambda i=idx: self._on_remove_randomizer(i))
            comp.updated.connect(self.config_updated.emit)
            self.rand_list_layout.addWidget(comp)

    def _on_add_randomizer(self):
        self.cfg.hdri_randomizers.append(RandomizerInstance())
        self._refresh_randomizers()
        self.config_updated.emit()

    def _on_remove_randomizer(self, index):
        if 0 <= index < len(self.cfg.hdri_randomizers):
            self.cfg.hdri_randomizers.pop(index)
            self._refresh_randomizers()
            self.config_updated.emit()

    def _on_add_hdri(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import HDRIs", "",
            "HDR Images (*.hdr *.exr);;All Files (*)"
        )
        if paths:
            self.cfg.hdri_paths.extend(paths)
            self._refresh_list()
            self.config_updated.emit()

    def _on_remove_hdri(self, index: int):
        if 0 <= index < len(self.cfg.hdri_paths):
            self.cfg.hdri_paths.pop(index)
            self._refresh_list()
            self.config_updated.emit()
