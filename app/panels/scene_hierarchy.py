"""
panels/scene_hierarchy.py
==========================
Unity-style Scene Hierarchy panel.
Each row shows:
  - Base Map + Normal Map thumbnail slots (click to pick texture)
  - Object name
  - Colored class label badge
  - Eye (visibility) toggle
  - ✕ remove button
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QFileDialog, QLayout,
    QDoubleSpinBox, QCheckBox, QColorDialog, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QColor, QPixmap, QFont

from app.engine.scene_state import SceneObject, ObjectConfig, SceneConfig
from app.engine.label_registry import REGISTRY


# ── Texture slot thumbnail ────────────────────────────────────────────────────

_IMG_FILTER = "Images (*.png *.jpg *.jpeg *.tga *.bmp *.hdr);;All Files (*)"

class _TexThumb(QLabel):
    """Clickable 48×36 thumbnail for a texture slot."""
    clicked = Signal()

    def __init__(self, slot_label: str, parent=None):
        super().__init__(parent)
        self._slot = slot_label
        self.setFixedSize(48, 36)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_empty()

    def _set_empty(self):
        self.clear()
        self.setText(self._slot)
        self.setStyleSheet(
            "border: 1px solid #3a4255; background: #1c2333; color: #555; "
            "font-size: 8px; border-radius: 2px;"
        )

    def set_path(self, path: str):
        if path and os.path.isfile(path):
            pix = QPixmap(path).scaled(
                self.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(pix)
            self.setStyleSheet("border: 1px solid #4fc3f7; border-radius: 2px;")
        else:
            self._set_empty()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# ── Object Row Widget ─────────────────────────────────────────────────────────

class ObjectRowWidget(QFrame):
    """A single row in the hierarchy representing one SceneObject."""
    selected           = Signal(object)        # emits SceneObject
    removed            = Signal(object)        # emits SceneObject
    texture_changed    = Signal(object)        # emits SceneObject
    visibility_toggled = Signal(object, bool)

    def __init__(self, scene_obj: SceneObject, parent=None):
        super().__init__(parent)
        self.scene_obj = scene_obj
        self.setFixedHeight(68)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("hierarchy_row")
        self._is_selected = False
        self._apply_style(False)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 4, 8, 4)
        root.setSpacing(2)

        # ── Top row: name + badge + eye + ✕ ────────────────────────
        top = QHBoxLayout()
        top.setSpacing(6)
        top.setContentsMargins(0, 0, 0, 0)

        self._name_lbl = QLabel(scene_obj.config.name or "Object")
        self._name_lbl.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: 500;")
        self._name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._name_lbl.setMinimumWidth(80)

        lbl_name = scene_obj.label or scene_obj.config.class_name or "unlabeled"
        color = REGISTRY.get_qcolor(lbl_name)
        self._badge = QLabel(lbl_name or "?")
        self._badge.setFixedHeight(16)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setStyleSheet(
            f"background: {color.name()}; color: #111; border-radius: 3px; "
            f"padding: 0 5px; font-size: 9px; font-weight: bold;"
        )
        self._badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._vis_btn = QPushButton("👁")
        self._vis_btn.setFixedSize(20, 20)
        self._vis_btn.setCheckable(True)
        self._vis_btn.setChecked(scene_obj.visible)
        self._vis_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 10px; color: #555; }"
            "QPushButton:checked { color: #cdd6f4; }"
        )
        self._vis_btn.clicked.connect(self._on_visibility)

        self._rm_btn = QPushButton("✕")
        self._rm_btn.setFixedSize(18, 18)
        self._rm_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 10px; color: #555; }"
            "QPushButton:hover { color: #f04040; }"
        )
        self._rm_btn.clicked.connect(self._on_remove)

        top.addWidget(self._name_lbl)
        top.addWidget(self._badge)
        top.addWidget(self._vis_btn)
        top.addWidget(self._rm_btn)

        # ── Bottom row: texture thumbnails ───────────────────────────
        bot = QHBoxLayout()
        bot.setSpacing(4)
        bot.setContentsMargins(0, 0, 0, 0)

        self._base_thumb  = _TexThumb("Base\nMap")
        self._norm_thumb  = _TexThumb("Normal\nMap")

        # Pre-fill if mesh already has textures
        self._base_thumb.set_path(scene_obj.config.tex_albedo)
        self._norm_thumb.set_path(scene_obj.config.tex_normal)

        self._base_thumb.clicked.connect(lambda: self._pick_texture("albedo"))
        self._norm_thumb.clicked.connect(lambda: self._pick_texture("normal"))

        bot.addWidget(self._base_thumb)
        bot.addWidget(self._norm_thumb)
        bot.addStretch()

        root.addLayout(top)
        root.addLayout(bot)

    # ── Styling ───────────────────────────────────────────────────────────────

    def _apply_style(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "QFrame#hierarchy_row { background: #1a2744; border-left: 2px solid #4fc3f7; "
                "border-bottom: 1px solid #252d3d; }"
            )
        else:
            self.setStyleSheet(
                "QFrame#hierarchy_row { background: transparent; border-left: 2px solid transparent; "
                "border-bottom: 1px solid #1e2538; }"
                "QFrame#hierarchy_row:hover { background: rgba(255,255,255,0.04); }"
            )

    def set_selected(self, v: bool):
        self._is_selected = v
        self._apply_style(v)

    def update_badge(self, label: str):
        lbl_name = label or "unlabeled"
        color = REGISTRY.get_qcolor(lbl_name)
        self._badge.setText(lbl_name)
        self._badge.setStyleSheet(
            f"background: {color.name()}; color: #111; border-radius: 3px; "
            f"padding: 0 5px; font-size: 9px; font-weight: bold;"
        )

    def refresh_thumbnails(self):
        """Reload thumbnails from the current config (e.g. after auto-discovery)."""
        self._base_thumb.set_path(self.scene_obj.config.tex_albedo)
        self._norm_thumb.set_path(self.scene_obj.config.tex_normal)

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_visibility(self, checked):
        self.scene_obj.visible = checked
        self.visibility_toggled.emit(self.scene_obj, checked)

    def _on_remove(self):
        self.removed.emit(self.scene_obj)

    def _pick_texture(self, slot: str):
        """Open file dialog and assign a texture to albedo or normal slot."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Texture", "", _IMG_FILTER)
        if not path:
            return
        if slot == "albedo":
            self.scene_obj.config.tex_albedo = path
            self._base_thumb.set_path(path)
        else:
            self.scene_obj.config.tex_normal = path
            self._norm_thumb.set_path(path)
        self.texture_changed.emit(self.scene_obj)

    def mousePressEvent(self, event):
        # Don't propagate if clicking buttons
        self.selected.emit(self.scene_obj)
        super().mousePressEvent(event)


# ── Compact Control Widgets ──────────────────────────────────────────────────

class CompactSlider(QWidget):
    """A tiny labeled slider/spinbox hybrid for the hierarchy."""
    valueChanged = Signal(float)
    def __init__(self, label, val, min_v, max_v, step=0.1, parent=None):
        super().__init__(parent)
        l = QHBoxLayout(self)
        l.setContentsMargins(12, 2, 12, 2)
        l.setSpacing(4)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #8b949e; font-size: 9px;")
        l.addWidget(lbl)
        
        self.sb = QDoubleSpinBox()
        self.sb.setRange(min_v, max_v)
        self.sb.setSingleStep(step)
        self.sb.setValue(val)
        self.sb.setFixedHeight(18)
        self.sb.setStyleSheet("""
            QDoubleSpinBox { background: #0d1117; color: #cdd6f4; border: 1px solid #252d3d; 
                             border-radius: 2px; font-size: 9px; padding: 0 2px; }
        """)
        self.sb.valueChanged.connect(self.valueChanged.emit)
        l.addWidget(self.sb)

class CompactColor(QWidget):
    """Tiny color picker for hierarchy."""
    colorChanged = Signal()
    def __init__(self, label, color_list, parent=None):
        super().__init__(parent)
        self.color_list = color_list
        l = QHBoxLayout(self)
        l.setContentsMargins(12, 2, 12, 2)
        
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #8b949e; font-size: 9px;")
        l.addWidget(lbl)
        
        self.btn = QPushButton()
        self.btn.setFixedSize(16, 12)
        self._update_btn()
        self.btn.clicked.connect(self._pick)
        l.addWidget(self.btn)
        l.addStretch()

    def _update_btn(self):
        c = QColor.fromRgbF(self.color_list[0], self.color_list[1], self.color_list[2])
        self.btn.setStyleSheet(f"background: {c.name()}; border: 1px solid #30363d; border-radius: 2px;")

    def _pick(self):
        c = QColor.fromRgbF(self.color_list[0], self.color_list[1], self.color_list[2])
        nc = QColorDialog.getColor(c, self)
        if nc.isValid():
            self.color_list[0] = nc.redF()
            self.color_list[1] = nc.greenF()
            self.color_list[2] = nc.blueF()
            self._update_btn()
            self.colorChanged.emit()

class CollapsibleSection(QWidget):
    """A section with a toggleable header and content area."""
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self.header = QPushButton(f"▼ {title.upper()}")
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setStyleSheet("""
            QPushButton {
                background: #1c2333;
                color: #8b949e;
                border: none;
                border-bottom: 1px solid #252d3d;
                text-align: left;
                padding: 6px 12px;
                font-size: 10px;
                font-weight: bold;
                letter-spacing: 0.8px;
            }
            QPushButton:hover { background: #242d3d; color: #cdd6f4; }
            QPushButton:checked { color: #4fc3f7; }
        """)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 4)
        self.content_layout.setSpacing(0)
        
        self.layout.addWidget(self.header)
        self.layout.addWidget(self.content_widget)
        self.header.toggled.connect(self._on_toggle)
        
    def _on_toggle(self, checked):
        title = self.header.text()[2:]
        if checked:
            self.header.setText(f"▼ {title}")
            self.content_widget.show()
        else:
            self.header.setText(f"▶ {title}")
            self.content_widget.hide()

    def add_widget(self, widget):
        if self.content_layout.count() > 0:
            last_item = self.content_layout.itemAt(self.content_layout.count() - 1)
            if last_item.spacerItem():
                self.content_layout.takeAt(self.content_layout.count() - 1)
        
        self.content_layout.addWidget(widget)
        self.content_layout.addStretch()

    def clear(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class HDRIRowWidget(QFrame):
    removed = Signal(str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFixedHeight(32)
        self.setStyleSheet("""
            QFrame { background: transparent; border-bottom: 1px solid #1e2538; }
            QFrame:hover { background: rgba(255,255,255,0.03); }
        """)
        
        l = QHBoxLayout(self)
        l.setContentsMargins(12, 0, 8, 0)
        
        icon_lbl = QLabel("🖼")
        icon_lbl.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        
        name_lbl = QLabel(os.path.basename(path))
        name_lbl.setStyleSheet("color: #cdd6f4; font-size: 10px; font-weight: 400;")
        name_lbl.setToolTip(path)
        
        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(16, 16)
        rm_btn.setStyleSheet("""
            QPushButton { color: #555; background: transparent; border: none; font-size: 9px; }
            QPushButton:hover { color: #f04040; }
        """)
        rm_btn.clicked.connect(lambda: self.removed.emit(self.path))
        
        l.addWidget(icon_lbl)
        l.addWidget(name_lbl)
        l.addStretch()
        l.addWidget(rm_btn)


class GlobalRowWidget(QFrame):
    clicked = Signal(str) # category: 'ocean', 'weather'

    def __init__(self, title, icon="⚡", category="ocean", parent=None):
        super().__init__(parent)
        self.category = category
        self.setFixedHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            QFrame { background: transparent; border-bottom: 1px solid #1e2538; }
            QFrame:hover { background: rgba(255,255,255,0.03); }
        """)
        
        l = QHBoxLayout(self)
        l.setContentsMargins(12, 0, 12, 0)
        
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("color: #fbc02d; font-size: 11px;")
        
        name_lbl = QLabel(title)
        name_lbl.setStyleSheet("color: #cdd6f4; font-size: 10px; font-weight: 500;")
        
        l.addWidget(icon_lbl)
        l.addWidget(name_lbl)
        l.addStretch()
        
        arrow = QLabel("›")
        arrow.setStyleSheet("color: #444; font-size: 14px;")
        l.addWidget(arrow)

    def mousePressEvent(self, event):
        self.clicked.emit(self.category)
        super().mousePressEvent(event)


# ── Scene Hierarchy Panel ─────────────────────────────────────────────────────

class SceneHierarchyPanel(QWidget):
    """The full scene hierarchy panel."""
    object_selected  = Signal(object)   # emits SceneObject or None
    object_added     = Signal(object)   # emits SceneObject
    object_removed   = Signal(object)   # emits SceneObject
    texture_changed  = Signal(object)   # emits SceneObject
    scene_changed    = Signal()         # general refresh needed
    hdri_changed     = Signal()         # specific HDRI load needed
    navigation_requested = Signal(str)  # emits 'ocean', 'hdri', 'settings', etc.

    _MESH_FILTER = (
        "3D Models (*.fbx *.obj *.gltf *.glb *.ply *.stl *.dae);;"
        "FBX (*.fbx);;OBJ (*.obj);;GLTF/GLB (*.gltf *.glb);;All Files (*)"
    )

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._rows: dict[str, ObjectRowWidget] = {}  # instance_id → row
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
        self._main_layout = QVBoxLayout(self._list_widget)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # ── Categorized Sections ──────────────────────────────
        self.section_objects = CollapsibleSection("Objects")
        self.section_hdri    = CollapsibleSection("HDRI's")
        self.section_physics = CollapsibleSection("Physics & Global")

        self._main_layout.addWidget(self.section_objects)
        self._main_layout.addWidget(self.section_hdri)
        self._main_layout.addWidget(self.section_physics)
        self._main_layout.addStretch()

        self._scroll.setWidget(self._list_widget)
        root.addWidget(self._scroll, stretch=1)

    # ── Public API ────────────────────────────────────────────────────

    def refresh(self):
        """Rebuild all sections from current config."""
        self.section_objects.clear()
        self.section_hdri.clear()
        self.section_physics.clear()
        self._rows.clear()

        # 1. Objects
        for obj in self.cfg.scene_objects:
            self._add_object_row(obj)
        
        add_obj = QPushButton("+ Add Object")
        add_obj.setStyleSheet("QPushButton { background: #1f3060; color: #4fc3f7; border: 1px solid #4fc3f7; "
                             "font-size: 10px; margin: 8px; padding: 4px; border-radius: 2px; }")
        add_obj.clicked.connect(self._on_add_object)
        self.section_objects.add_widget(add_obj)

        # 2. HDRI
        for path in self.cfg.hdri_paths:
            h_row = HDRIRowWidget(path)
            h_row.removed.connect(self._on_remove_hdri)
            self.section_hdri.add_widget(h_row)
        
        add_hdri = QPushButton("+ Add HDRI")
        add_hdri.setStyleSheet("QPushButton { background: #1f3060; color: #4fc3f7; border: 1px solid #4fc3f7; "
                              "font-size: 10px; margin: 8px; padding: 4px; border-radius: 2px; }")
        add_hdri.clicked.connect(self._on_add_hdri_btn)
        self.section_hdri.add_widget(add_hdri)

        # 3. Physics / Global
        self._build_ocean_dropdown()
        self._build_weather_dropdown()
        self._build_camera_dropdown()

    def _build_ocean_dropdown(self):
        container = CollapsibleSection("Ocean Surface Settings")
        container.header.setStyleSheet(container.header.styleSheet().replace("#1c2333", "#121620"))
        container.header.setText("\u25b6 OCEAN SURFACE")
        container.header.setChecked(False)
        
        o = self.cfg.ocean
        
        cb = QCheckBox("Enable Ocean")
        cb.setChecked(o.enabled)
        cb.setStyleSheet("color: #cdd6f4; font-size: 10px; margin-left: 12px; margin-top: 4px;")
        cb.toggled.connect(self._sync_ocean_enabled)
        container.add_widget(cb)

        def add_sub(text):
            from PySide6.QtWidgets import QLabel
            lbl = QLabel(text.upper())
            lbl.setStyleSheet("color: #4fc3f7; font-size: 8px; font-weight: bold; margin-left: 12px; margin-top: 8px;")
            container.add_widget(lbl)

        add_sub("Simulation")
        container.add_widget(CompactSlider("Level (m)", o.level, -20, 20))
        container.add_widget(CompactSlider("Repetition", o.repetition_size, 10, 2000, 50))
        
        add_sub("Waves & Bands")
        container.add_widget(CompactSlider("Amplitude", o.wave_amplitude, 0, 10))
        container.add_widget(CompactSlider("Wind Speed", o.wind_speed, 0, 100))
        container.add_widget(CompactSlider("Choppiness", o.choppiness, 0, 5))
        container.add_widget(CompactSlider("Band 0 Mul", o.band0_multiplier, 0, 2, 0.1))
        container.add_widget(CompactSlider("Band 1 Mul", o.band1_multiplier, 0, 2, 0.1))

        add_sub("Currents & Ripples")
        container.add_widget(CompactSlider("Curr Speed", o.current_speed, 0, 20, 0.5))
        container.add_widget(CompactSlider("Rip Speed", o.ripples_wind_speed, 0, 50, 0.5))

        add_sub("Scattering & Material")
        w_col = CompactColor("Refraction", o.refraction_color)
        w_col.colorChanged.connect(lambda c: self._sync_ocean_attr("refraction_color", [c.redF(), c.greenF(), c.blueF()]))
        container.add_widget(w_col)
        
        w_scat = CompactColor("Scattering", o.scattering_color)
        w_scat.colorChanged.connect(lambda c: self._sync_ocean_attr("scattering_color", [c.redF(), c.greenF(), c.blueF()]))
        container.add_widget(w_scat)
        
        container.add_widget(CompactSlider("Tip Scat", o.direct_light_tip_scattering, 0, 2, 0.1))
        container.add_widget(CompactSlider("Body Scat", o.direct_light_body_scattering, 0, 2, 0.1))
        container.add_widget(CompactSlider("Smoothness", o.smoothness, 0, 1, 0.05))

        # Re-wire logic
        from PySide6.QtWidgets import QLabel
        for i in range(container.content_layout.count()):
            it = container.content_layout.itemAt(i)
            if not it: continue
            widget = it.widget()
            if isinstance(widget, CompactSlider):
                mapping = {
                    "Level (m)": "level", "Repetition": "repetition_size",
                    "Amplitude": "wave_amplitude", "Wind Speed": "wind_speed",
                    "Choppiness": "choppiness", "Band 0 Mul": "band0_multiplier",
                    "Band 1 Mul": "band1_multiplier", "Curr Speed": "current_speed",
                    "Rip Speed": "ripples_wind_speed", "Tip Scat": "direct_light_tip_scattering",
                    "Body Scat": "direct_light_body_scattering", "Smoothness": "smoothness"
                }
                for child in widget.findChildren(QLabel):
                    if child.text() in mapping:
                        attr = mapping[child.text()]
                        widget.valueChanged.connect(lambda v, a=attr: self._sync_ocean_attr(a, v))

        # Navigation row
        nav_row = QWidget()
        nav_l = QHBoxLayout(nav_row)
        nav_l.setContentsMargins(12, 8, 12, 8)
        nav_btn = QPushButton("↗ Open Advanced Ocean Editor")
        nav_btn.setStyleSheet("""
            QPushButton { background: #1f3060; color: #4fc3f7; border: 1px solid #4fc3f7; 
                          font-size: 9px; padding: 4px; border-radius: 2px; }
            QPushButton:hover { background: #2a4080; }
        """)
        nav_btn.clicked.connect(lambda: self.navigation_requested.emit("ocean"))
        nav_l.addWidget(nav_btn)
        container.add_widget(nav_row)
        self.section_physics.add_widget(container)

    def _build_weather_dropdown(self):
        container = CollapsibleSection("Weather & Post-Process")
        container.header.setStyleSheet(container.header.styleSheet().replace("#1c2333", "#121620"))
        container.header.setText("▶ ATMOSPHERE & RENDER")
        container.header.setChecked(False)

        w = self.cfg.weather
        pp = self.cfg.post_process

        # 1. Weather Type & Intensity
        type_row = QWidget()
        type_row.setStyleSheet("background: transparent; margin: 4px 8px;")
        th = QHBoxLayout(type_row)
        th.setContentsMargins(0, 0, 0, 0)
        t_lbl = QLabel("Weather Type")
        t_lbl.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["clear", "cloudy", "rain", "stormy", "snow", "foggy"])
        self.weather_combo.setCurrentText(w.type)
        self.weather_combo.setFixedHeight(22)
        self.weather_combo.setStyleSheet("""
            QComboBox { background: #0d1117; color: #cdd6f4; border: 1px solid #30363d; 
                        border-radius: 3px; font-size: 10px; padding: 2px 5px; }
            QComboBox::drop-down { border: none; }
        """)
        self.weather_combo.currentTextChanged.connect(self._sync_weather_type)
        
        th.addWidget(t_lbl)
        th.addStretch()
        th.addWidget(self.weather_combo)
        container.add_widget(type_row)

        intensity = CompactSlider("Intensity", w.intensity, 0, 1, 0.05)
        intensity.valueChanged.connect(self._sync_weather_intensity)
        container.add_widget(intensity)

        fog = CompactSlider("Fog Density", w.fog_density, 0, 1, 0.01)
        fog.valueChanged.connect(self._sync_weather_fog)
        container.add_widget(fog)

        exp = CompactSlider("Global Exposure", pp.exposure_max, -2, 2, 0.1)
        exp.valueChanged.connect(lambda v: self._sync_pp_attr("exposure_max", v))
        container.add_widget(exp)

        bloom = CompactSlider("Bloom Intensity", pp.bloom_max, 0, 1, 0.05)
        bloom.valueChanged.connect(lambda v: self._sync_pp_attr("bloom_max", v))
        container.add_widget(bloom)

        noise = CompactSlider("Grain / Noise", pp.noise_max, 0, 0.2, 0.01)
        noise.valueChanged.connect(lambda v: self._sync_pp_attr("noise_max", v))
        container.add_widget(noise)

        # Dataset globals
        size = CompactSlider("Dataset Resolution", self.cfg.image_size, 128, 1024, 128)
        size.valueChanged.connect(self._sync_global_res)
        container.add_widget(size)
        
        self.section_physics.add_widget(container)

    def _build_camera_dropdown(self):
        container = CollapsibleSection("Camera Lens")
        container.header.setStyleSheet(container.header.styleSheet().replace("#1c2333", "#121620"))
        container.header.setText("▶ CAMERA LENS")
        container.header.setChecked(False)

        # Lens Type Presets 
        type_row = QWidget()
        type_row.setStyleSheet("background: transparent; margin: 4px 8px;")
        th = QHBoxLayout(type_row)
        th.setContentsMargins(0, 0, 0, 0)
        t_lbl = QLabel("Lens Type")
        t_lbl.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        
        self.lens_combo = QComboBox()
        self.lens_combo.addItems([
            "Wide (14mm)", "Wide (24mm)", "Standard (35mm)", 
            "Standard (50mm)", "Portrait (85mm)", "Telephoto (200mm)", "Custom"
        ])
        
        fl = getattr(self.cfg, 'focal_length', 50.0)
        if fl == 14: self.lens_combo.setCurrentText("Wide (14mm)")
        elif fl == 24: self.lens_combo.setCurrentText("Wide (24mm)")
        elif fl == 35: self.lens_combo.setCurrentText("Standard (35mm)")
        elif fl == 50: self.lens_combo.setCurrentText("Standard (50mm)")
        elif fl == 85: self.lens_combo.setCurrentText("Portrait (85mm)")
        elif fl == 200: self.lens_combo.setCurrentText("Telephoto (200mm)")
        else: self.lens_combo.setCurrentText("Custom")
        
        self.lens_combo.setFixedHeight(20)
        self.lens_combo.setStyleSheet("""
            QComboBox { background: #0d1117; color: #cdd6f4; border: 1px solid #30363d; 
                        border-radius: 3px; font-size: 9px; padding: 1px 4px; }
            QComboBox::drop-down { border: none; }
        """)
        self.lens_combo.currentTextChanged.connect(self._on_lens_preset_changed)
        
        th.addWidget(t_lbl)
        th.addStretch()
        th.addWidget(self.lens_combo)
        container.add_widget(type_row)

        self.fl_spinner = CompactSlider("Focal Length (mm)", fl, 1.0, 1000.0, 1.0)
        self.fl_spinner.valueChanged.connect(self._sync_focal_length)
        container.add_widget(self.fl_spinner)

        # Info label for FOV
        self.fov_info = QLabel(f"Field of View: {self.cfg.fov_y:.1f}°")
        self.fov_info.setStyleSheet("color: #555; font-size: 9px; margin-left: 14px; margin-bottom: 4px;")
        container.add_widget(self.fov_info)

        # --- Fisheye Settings ---
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("background-color: #30363d; margin: 4px 8px;")
        container.add_widget(line)

        fe_row = QWidget()
        fe_row.setStyleSheet("background: transparent; margin: 2px 8px;")
        fe_l = QHBoxLayout(fe_row)
        fe_l.setContentsMargins(0, 0, 0, 0)
        
        fe_lbl = QLabel("Fisheye Effect")
        fe_lbl.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        from PySide6.QtWidgets import QCheckBox
        self.fe_active = QCheckBox()
        self.fe_active.setChecked(self.cfg.post_process.fisheye_enabled)
        self.fe_active.toggled.connect(lambda v: self._sync_pp_attr('fisheye_enabled', v))
        
        fe_l.addWidget(fe_lbl)
        fe_l.addStretch()
        fe_l.addWidget(self.fe_active)
        container.add_widget(fe_row)

        self.fe_strength = CompactSlider("Distortion", self.cfg.post_process.fisheye_strength, 0.0, 2.0, 0.05)
        self.fe_strength.valueChanged.connect(lambda v: self._sync_pp_attr('fisheye_strength', v))
        container.add_widget(self.fe_strength)

        self.section_physics.add_widget(container)

    def _sync_focal_length(self, val):
        import math
        self.cfg.focal_length = val
        # FOV_y = 2 * atan(sensor_height / (2 * focal_length))
        # Sensor height for 35mm full frame is 24mm.
        self.cfg.fov_y = math.degrees(2.0 * math.atan(12.0 / max(0.1, val)))
        if hasattr(self, 'fov_info'):
            self.fov_info.setText(f"Field of View: {self.cfg.fov_y:.1f}°")
        self.scene_changed.emit()

    def _on_lens_preset_changed(self, text):
        mapping = {
            "Wide (14mm)": 14.0, "Wide (24mm)": 24.0, "Standard (35mm)": 35.0,
            "Standard (50mm)": 50.0, "Portrait (85mm)": 85.0, "Telephoto (200mm)": 200.0
        }
        if text in mapping:
            self.fl_spinner.sb.setValue(mapping[text])

    def _sync_pp_attr(self, attr, val):
        setattr(self.cfg.post_process, attr, val)
        self.scene_changed.emit()

    def _sync_ocean_enabled(self, v):
        self.cfg.ocean.enabled = v
        self.scene_changed.emit()

    def _sync_ocean_attr(self, attr, val):
        setattr(self.cfg.ocean, attr, val)
        self.scene_changed.emit()

    def _sync_global_res(self, v):
        self.cfg.image_size = int(v)
        self.scene_changed.emit()

    def _sync_weather_fog(self, v):
        self.cfg.weather.fog_density = v
        self.scene_changed.emit()

    def _sync_weather_type(self, t):
        print(f"[UI] Weather changed to: {t}")
        if self.cfg and self.cfg.weather:
            self.cfg.weather.type = t
            self.scene_changed.emit()

    def _sync_weather_intensity(self, v):
        self.cfg.weather.intensity = v
        self.scene_changed.emit()

    def _add_object_row(self, obj: SceneObject) -> ObjectRowWidget:
        row = ObjectRowWidget(obj)
        row.selected.connect(self._on_row_selected)
        row.removed.connect(self._on_remove_object)
        row.texture_changed.connect(self.texture_changed)
        row.visibility_toggled.connect(lambda o, v: self.scene_changed.emit())
        self.section_objects.add_widget(row)
        self._rows[obj.instance_id] = row
        return row

    def update_row_badge(self, obj: SceneObject):
        if obj.instance_id in self._rows:
            self._rows[obj.instance_id].update_badge(obj.label)

    def update_row_thumbnails(self, obj: SceneObject):
        if obj.instance_id in self._rows:
            self._rows[obj.instance_id].refresh_thumbnails()

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_row_selected(self, obj: SceneObject):
        for row in self._rows.values():
            row.set_selected(False)
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
            self.cfg.models.append(obj_cfg)
            row = self._add_object_row(scene_obj)
            self._on_row_selected(scene_obj)
            self.object_added.emit(scene_obj)
            self.scene_changed.emit()

    def _on_remove_hdri(self, path):
        if path in self.cfg.hdri_paths:
            self.cfg.hdri_paths.remove(path)
            self.refresh()
            self.hdri_changed.emit()
            self.scene_changed.emit()

    def _on_add_hdri_btn(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import HDRIs", "",
            "HDR Images (*.hdr *.exr);;All Files (*)"
        )
        if paths:
            self.cfg.hdri_paths.extend(paths)
            self.refresh()
            self.hdri_changed.emit()
            self.scene_changed.emit()

    def _on_remove_object(self, obj: SceneObject):
        # Remove from scene state
        self.cfg.scene_objects = [o for o in self.cfg.scene_objects if o.instance_id != obj.instance_id]
        self.cfg.models = [m for m in self.cfg.models if m.mesh_path != obj.config.mesh_path]

        # Remove row widget
        if obj.instance_id in self._rows:
            row = self._rows.pop(obj.instance_id)
            row.setParent(None)
            row.deleteLater()

        # Deselect if it was selected
        if self._selected and self._selected.instance_id == obj.instance_id:
            self._selected = None

        self.object_removed.emit(obj)
        self.scene_changed.emit()
