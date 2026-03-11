"""
panels/import_view.py
=====================
A central view for managing imported models. 
Allows adding multiple models, setting class names, counts, and per-object overrides.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QLineEdit, QSpinBox, QDoubleSpinBox,
    QFileDialog, QSpacerItem, QSizePolicy, QComboBox, QStackedWidget,
    QGridLayout,
)
from PySide6.QtCore import Qt, Signal
import os

from app.engine.scene_state import SceneConfig, ObjectConfig, RandomizerInstance
from .randomizer_widgets import RandomizerComponentWidget


class ObjectEntryWidget(QFrame):
    """A single row/card for one imported model."""
    removed = Signal()
    updated = Signal()

    def __init__(self, obj_cfg: ObjectConfig, parent=None):
        super().__init__(parent)
        self.cfg = obj_cfg
        self.setObjectName("object_entry")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header: Tag and Remove
        header = QHBoxLayout()
        self.header_label = QLabel(f"Tag: {obj_cfg.class_name}")
        self.header_label.setObjectName("entry_title_blue")
        header.addWidget(self.header_label)
        header.addStretch()
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("danger_btn_small")
        remove_btn.setFixedWidth(80)
        remove_btn.clicked.connect(self.removed.emit)
        header.addWidget(remove_btn)
        main_layout.addLayout(header)

        # File path
        path_lbl = QLabel(f"Path: ...{os.path.basename(obj_cfg.mesh_path)}")
        path_lbl.setObjectName("path_label_small")
        main_layout.addWidget(path_lbl)

        # Class Name Edit
        class_h = QHBoxLayout()
        class_h.addWidget(QLabel("Class Name:"))
        self.class_edit = QLineEdit(obj_cfg.class_name)
        self.class_edit.textChanged.connect(self._on_class_changed)
        class_h.addWidget(self.class_edit)
        main_layout.addLayout(class_h)

        # Count/Scale global-ish toggles (optional but keeping simple for now)
        
        # Add Randomizer Component Button
        comp_header = QHBoxLayout()
        l = QLabel("Randomizer Components")
        l.setStyleSheet("font-weight: bold; color: #8b949e;")
        comp_header.addWidget(l)
        comp_header.addStretch()
        add_rand_btn = QPushButton("+ Add Randomizer")
        add_rand_btn.setObjectName("secondary_btn_small")
        add_rand_btn.clicked.connect(self._on_add_randomizer)
        comp_header.addWidget(add_rand_btn)
        main_layout.addLayout(comp_header)

        # List of randomizers
        self.rand_list_layout = QVBoxLayout()
        self.rand_list_layout.setSpacing(4)
        main_layout.addLayout(self.rand_list_layout)

        self._refresh_randomizers()

    def _on_class_changed(self, text):
        self.cfg.class_name = text
        self.cfg.name = text
        self.header_label.setText(f"Tag: {text}")
        self.updated.emit()

    def _refresh_randomizers(self):
        # Clear
        while self.rand_list_layout.count():
            item = self.rand_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Build
        for idx, inst in enumerate(self.cfg.randomizers):
            comp = RandomizerComponentWidget(inst, mode="object")
            comp.removed.connect(lambda i=idx: self._on_remove_randomizer(i))
            comp.updated.connect(self.updated.emit)
            self.rand_list_layout.addWidget(comp)

    def _on_add_randomizer(self):
        self.cfg.randomizers.append(RandomizerInstance())
        self._refresh_randomizers()
        self.updated.emit()

    def _on_remove_randomizer(self, index):
        if 0 <= index < len(self.cfg.randomizers):
            self.cfg.randomizers.pop(index)
            self._refresh_randomizers()
            self.updated.emit()


class ImportPanel(QWidget):
    """Central panel for managing multiple model imports."""
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setObjectName("import_panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Title section
        title_row = QHBoxLayout()
        title_v = QVBoxLayout()
        title = QLabel("Import Models")
        title.setObjectName("view_title")
        title_v.addWidget(title)
        
        subtitle = QLabel("Add one or more 3D models to your synthetic scene.")
        subtitle.setObjectName("view_subtitle")
        title_v.addWidget(subtitle)
        title_row.addLayout(title_v)
        
        title_row.addStretch()
        layout.addLayout(title_row)

        # Container for scroll area + floating button
        self.center_container = QWidget()
        center_layout = QVBoxLayout(self.center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add Button (Max Width at top of list area)
        self.add_btn = QPushButton("Add New Model")
        self.add_btn.setObjectName("primary_btn")
        self.add_btn.setMinimumHeight(50)
        self.add_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.add_btn.clicked.connect(self._on_add_model)
        center_layout.addWidget(self.add_btn)

        # Scrollable list of models
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("model_scroll")
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 10, 0)
        self.list_layout.setSpacing(16)
        self.list_layout.addStretch() # spacer at bottom
        
        self.scroll.setWidget(self.list_container)
        center_layout.addWidget(self.scroll)
        
        layout.addWidget(self.center_container)

    def _refresh_list(self):
        # Clear existing
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add current config
        for idx, obj in enumerate(self.cfg.models):
            entry = ObjectEntryWidget(obj)
            entry.removed.connect(lambda i=idx: self._on_remove_model(i))
            entry.updated.connect(self.config_updated.emit)
            self.list_layout.insertWidget(idx, entry)

    def _on_add_model(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import 3D Models", "",
            "3D Models (*.fbx *.obj *.gltf *.glb *.ply *.stl *.dae);;"
            "FBX (*.fbx);;OBJ (*.obj);;GLTF/GLB (*.gltf *.glb);;All Files (*)"
        )
        if paths:
            for path in paths:
                new_obj = ObjectConfig(
                    name=os.path.basename(path).split('.')[0].capitalize(),
                    mesh_path=path,
                    count_min=1, count_max=2
                )
                self.cfg.models.append(new_obj)
            self._refresh_list()
            self.config_updated.emit()

    def _on_remove_model(self, index: int):
        if 0 <= index < len(self.cfg.models):
            self.cfg.models.pop(index)
            self._refresh_list()
            self.config_updated.emit()
