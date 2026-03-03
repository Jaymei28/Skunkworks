"""
panels/scene_hierarchy.py
==========================
Unity-style Scene Hierarchy panel.
Shows all SceneObjects in the scene as a tree list with:
  - Colored class label badge (Blender-style colors)
  - Randomizer dice icon (future)
  - Visibility toggle
  - Click to select (emits object_selected signal)
  - + Add Object button
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QFileDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QColor, QPainter, QPixmap, QIcon, QFont

from app.engine.scene_state import SceneObject, ObjectConfig, SceneConfig
from app.engine.label_registry import REGISTRY


# ── Object Row Widget ─────────────────────────────────────────────────────────

class ObjectRowWidget(QFrame):
    """A single row representing one SceneObject in the hierarchy."""
    selected   = Signal(object)  # emits SceneObject
    removed    = Signal(object)  # emits SceneObject
    visibility_toggled = Signal(object, bool)

    def __init__(self, scene_obj: SceneObject, parent=None):
        super().__init__(parent)
        self.scene_obj = scene_obj
        self.setFixedHeight(34)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("hierarchy_row")
        self._is_selected = False
        self._apply_style(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        # ── Colored label badge ──────────────────────────────────────
        lbl_name = scene_obj.label or scene_obj.config.class_name or "unlabeled"
        color = REGISTRY.get_qcolor(lbl_name)
        self._badge = QLabel(lbl_name or "?")
        self._badge.setFixedHeight(18)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setStyleSheet(
            f"background: {color.name()}; color: #111; border-radius: 3px; "
            f"padding: 0 6px; font-size: 10px; font-weight: bold;"
        )
        self._badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # ── Object name ──────────────────────────────────────────────
        self._name_lbl = QLabel(scene_obj.config.name or "Object")
        self._name_lbl.setStyleSheet("color: #cdd6f4; font-size: 12px;")
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # ── Visibility toggle ────────────────────────────────────────
        self._vis_btn = QPushButton("👁")
        self._vis_btn.setFixedSize(22, 22)
        self._vis_btn.setCheckable(True)
        self._vis_btn.setChecked(scene_obj.visible)
        self._vis_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 11px; color: #8b949e; }"
            "QPushButton:checked { color: #cdd6f4; }"
        )
        self._vis_btn.clicked.connect(self._on_visibility)

        layout.addWidget(self._name_lbl)
        layout.addWidget(self._badge)
        layout.addWidget(self._vis_btn)

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "QFrame#hierarchy_row { background: #1a2744; border-left: 2px solid #4fc3f7; }"
                "QFrame#hierarchy_row:hover { background: #1f3060; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#hierarchy_row { background: transparent; border-left: 2px solid transparent; }"
                "QFrame#hierarchy_row:hover { background: rgba(255,255,255,0.05); }"
            )

    def set_selected(self, v: bool):
        self._is_selected = v
        self._apply_style(v)

    def _on_visibility(self, checked):
        self.scene_obj.visible = checked
        self.visibility_toggled.emit(self.scene_obj, checked)

    def update_badge(self, label: str):
        """Refresh the badge after label changes."""
        lbl_name = label or "unlabeled"
        color = REGISTRY.get_qcolor(lbl_name)
        self._badge.setText(lbl_name)
        self._badge.setStyleSheet(
            f"background: {color.name()}; color: #111; border-radius: 3px; "
            f"padding: 0 6px; font-size: 10px; font-weight: bold;"
        )

    def mousePressEvent(self, event):
        self.selected.emit(self.scene_obj)
        super().mousePressEvent(event)


# ── Scene Hierarchy Panel ─────────────────────────────────────────────────────

class SceneHierarchyPanel(QWidget):
    """The full scene hierarchy panel."""
    object_selected  = Signal(object)  # emits SceneObject or None
    object_added     = Signal(object)  # emits SceneObject
    object_removed   = Signal(object)  # emits SceneObject
    scene_changed    = Signal()        # general refresh needed

    # Supported formats
    _MESH_FILTER = (
        "3D Models (*.fbx *.obj *.gltf *.glb *.ply *.stl *.dae);;"
        "FBX (*.fbx);;"
        "OBJ (*.obj);;"
        "GLTF/GLB (*.gltf *.glb);;"
        "All Files (*)"
    )

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._rows: dict[str, ObjectRowWidget] = {}  # instance_id -> row
        self._selected: SceneObject | None = None
        self.setObjectName("scene_hierarchy")
        self.setStyleSheet("QWidget#scene_hierarchy { background: #161b27; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ───────────────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(36)
        header.setStyleSheet("background: #1c2333; border-bottom: 1px solid #252d3d;")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(10, 0, 6, 0)
        title = QLabel("Scene Hierarchy")
        title.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        h_layout.addWidget(title)
        h_layout.addStretch()
        root.addWidget(header)

        # ── Scrollable rows ──────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 4, 0, 4)
        self._list_layout.setSpacing(1)
        self._list_layout.addStretch()
        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, stretch=1)

        # ── Bottom bar ───────────────────────────────────────────────
        bottom = QWidget()
        bottom.setFixedHeight(36)
        bottom.setStyleSheet("background: #1c2333; border-top: 1px solid #252d3d;")
        b_layout = QHBoxLayout(bottom)
        b_layout.setContentsMargins(6, 0, 6, 0)
        b_layout.setSpacing(4)

        add_btn = QPushButton("+ Add Object")
        add_btn.setStyleSheet(
            "QPushButton { background: #1f3060; color: #4fc3f7; border: 1px solid #4fc3f7; "
            "border-radius: 3px; font-size: 11px; padding: 3px 8px; }"
            "QPushButton:hover { background: #243875; }"
        )
        add_btn.clicked.connect(self._on_add_object)
        b_layout.addWidget(add_btn)
        b_layout.addStretch()
        root.addWidget(bottom)

    # ── Public API ────────────────────────────────────────────────────

    def refresh(self):
        """Rebuild the list from cfg.scene_objects."""
        # Clear existing rows
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for obj in self.cfg.scene_objects:
            self._add_row(obj)

    def _add_row(self, obj: SceneObject):
        row = ObjectRowWidget(obj)
        row.selected.connect(self._on_row_selected)
        row.visibility_toggled.connect(lambda o, v: self.scene_changed.emit())
        self._list_layout.insertWidget(self._list_layout.count() - 1, row)
        self._rows[obj.instance_id] = row
        return row

    def update_row_badge(self, obj: SceneObject):
        """Refresh a single row's badge after label edit."""
        if obj.instance_id in self._rows:
            self._rows[obj.instance_id].update_badge(obj.label)

    def _on_row_selected(self, obj: SceneObject):
        # Deselect all
        for row in self._rows.values():
            row.set_selected(False)
        # Select clicked
        if obj.instance_id in self._rows:
            self._rows[obj.instance_id].set_selected(True)
        self._selected = obj
        self.object_selected.emit(obj)

    def _on_add_object(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import 3D Models", "", self._MESH_FILTER
        )
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0].capitalize()
            obj_cfg = ObjectConfig(name=name, mesh_path=path)
            scene_obj = SceneObject(config=obj_cfg, label=obj_cfg.class_name)
            self.cfg.scene_objects.append(scene_obj)
            # Also add to legacy models list for backward compat
            self.cfg.models.append(obj_cfg)
            row = self._add_row(scene_obj)
            self._on_row_selected(scene_obj)
            self.object_added.emit(scene_obj)
            self.scene_changed.emit()
