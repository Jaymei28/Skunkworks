"""
panels/object_props.py
======================
Object Properties panel — shown when a SceneObject is selected.
Tabs: Transform | Label | Randomizers (Anim in Phase 2)
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QDoubleSpinBox, QCheckBox,
    QTabWidget, QFrame, QSizePolicy, QScrollArea, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QColor

from app.engine.scene_state import SceneObject
from app.engine.label_registry import REGISTRY


# ── Color swatch helper ───────────────────────────────────────────────────────

class ColorSwatch(QLabel):
    """A small colored rectangle showing the class color."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(18, 18)
        self.setStyleSheet("border-radius: 3px; border: 1px solid #444;")

    def set_color(self, qcolor: QColor):
        self.setStyleSheet(
            f"background: {qcolor.name()}; border-radius: 3px; border: 1px solid #444;"
        )


# ── Object Properties Panel ───────────────────────────────────────────────────

class ObjectPropertiesPanel(QWidget):
    """Tabbed panel for the selected SceneObject's properties."""
    label_changed    = Signal(object)   # emits SceneObject
    transform_changed = Signal(object)  # emits SceneObject

    def __init__(self, parent=None):
        super().__init__(parent)
        self._obj: SceneObject | None = None
        self.setObjectName("object_props")
        self.setStyleSheet("QWidget#object_props { background: #161b27; }")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ───────────────────────────────────────────────────
        self._header = QWidget()
        self._header.setFixedHeight(36)
        self._header.setStyleSheet("background: #1c2333; border-bottom: 1px solid #252d3d; border-top: 1px solid #252d3d;")
        h_layout = QHBoxLayout(self._header)
        h_layout.setContentsMargins(10, 0, 10, 0)
        self._title = QLabel("No Object Selected")
        self._title.setStyleSheet("color: #8b949e; font-size: 11px; font-weight: bold;")
        h_layout.addWidget(self._title)
        root.addWidget(self._header)

        # ── Tabs ─────────────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #161b27; }
            QTabBar::tab { background: #1c2333; color: #8b949e; padding: 5px 12px;
                           border: none; font-size: 11px; }
            QTabBar::tab:selected { background: #161b27; color: #cdd6f4;
                                    border-bottom: 2px solid #4fc3f7; }
            QTabBar::tab:hover { color: #cdd6f4; }
        """)
        root.addWidget(self._tabs, stretch=1)

        self._tabs.addTab(self._build_label_tab(),     "Label")
        self._tabs.addTab(self._build_transform_tab(), "Transform")
        self._tabs.addTab(self._build_rand_tab(),      "Randomizers")

        self._set_enabled(False)

    # ── Tab builders ──────────────────────────────────────────────────────────

    def _build_label_tab(self) -> QWidget:
        tab = QWidget()
        tab.setStyleSheet("background: #161b27;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Class name row
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.addItems(REGISTRY.all_names())
        self._class_combo.setStyleSheet(
            "QComboBox { background: #1c2333; color: #cdd6f4; border: 1px solid #333; "
            "border-radius: 3px; padding: 3px 6px; }"
        )
        self._class_combo.currentTextChanged.connect(self._on_class_changed)
        self._color_swatch = ColorSwatch()
        cls_row.addWidget(self._class_combo, stretch=1)
        cls_row.addWidget(self._color_swatch)
        layout.addLayout(cls_row)

        # Instance ID row
        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Instance ID:"))
        self._id_label = QLabel("—")
        self._id_label.setStyleSheet("color: #8b949e; font-family: monospace;")
        id_row.addWidget(self._id_label, stretch=1)
        layout.addLayout(id_row)

        # Visible to camera
        self._visible_check = QCheckBox("Visible to Camera")
        self._visible_check.setChecked(True)
        self._visible_check.stateChanged.connect(self._on_visibility_changed)
        layout.addWidget(self._visible_check)

        layout.addStretch()

        # Label all instances button
        style = (
            "QPushButton { background: #1f3060; color: #4fc3f7; border: 1px solid #4fc3f7; "
            "border-radius: 3px; padding: 5px; }"
            "QPushButton:hover { background: #243875; }"
        )
        apply_btn = QPushButton("Apply Label to All Objects of This Type")
        apply_btn.setStyleSheet(style)
        apply_btn.clicked.connect(self._on_apply_all)
        layout.addWidget(apply_btn)
        return tab

    def _build_transform_tab(self) -> QWidget:
        tab = QWidget()
        tab.setStyleSheet("background: #161b27;")
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def _spin(lo=-1000.0, hi=1000.0, step=0.1, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setStyleSheet(
                "QDoubleSpinBox { background: #1c2333; color: #cdd6f4; "
                "border: 1px solid #333; border-radius: 3px; padding: 2px 6px; }"
            )
            return s

        self._px = _spin(); self._py = _spin(); self._pz = _spin()
        self._rx = _spin(-360, 360, 1, 1); self._ry = _spin(-360, 360, 1, 1); self._rz = _spin(-360, 360, 1, 1)
        self._sc = _spin(0.001, 100, 0.1, 3)
        self._sc.setValue(1.0)

        layout.addRow("Pos X:", self._px)
        layout.addRow("Pos Y:", self._py)
        layout.addRow("Pos Z:", self._pz)
        layout.addRow("Rot X:", self._rx)
        layout.addRow("Rot Y:", self._ry)
        layout.addRow("Rot Z:", self._rz)
        layout.addRow("Scale:",  self._sc)

        for sp in [self._px, self._py, self._pz, self._rx, self._ry, self._rz, self._sc]:
            sp.valueChanged.connect(self._on_transform_changed)

        return tab

    def _build_rand_tab(self) -> QWidget:
        tab = QWidget()
        tab.setStyleSheet("background: #161b27;")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        info = QLabel("Per-object randomizers coming in Phase 3.")
        info.setStyleSheet("color: #8b949e; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()
        return tab

    # ── Public API ────────────────────────────────────────────────────────────

    def set_object(self, obj: SceneObject | None):
        self._obj = obj
        if obj is None:
            self._title.setText("No Object Selected")
            self._set_enabled(False)
            return

        self._title.setText(f"Properties — {obj.config.name}")
        self._set_enabled(True)

        # Populate Label tab
        label = obj.label or obj.config.class_name or ""
        idx = self._class_combo.findText(label)
        if idx >= 0:
            self._class_combo.setCurrentIndex(idx)
        else:
            self._class_combo.setCurrentText(label)
        self._id_label.setText(obj.instance_id)
        self._visible_check.setChecked(obj.visible)
        self._update_swatch(label)

        # Populate Transform tab (block signals to avoid feedback loop)
        for sp, val in [(self._px, obj.pos_x), (self._py, obj.pos_y), (self._pz, obj.pos_z),
                        (self._rx, obj.rot_x), (self._ry, obj.rot_y), (self._rz, obj.rot_z),
                        (self._sc, obj.scale)]:
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def _set_enabled(self, v: bool):
        self._tabs.setEnabled(v)
        self._tabs.setStyleSheet(self._tabs.styleSheet())  # force repaint

    def _update_swatch(self, label: str):
        if label:
            self._color_swatch.set_color(REGISTRY.get_qcolor(label))
        else:
            self._color_swatch.setStyleSheet("background: #333; border-radius: 3px; border: 1px solid #444;")

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_class_changed(self, text: str):
        if self._obj is None:
            return
        self._obj.label = text.strip().lower()
        REGISTRY.add_class(text)
        self._update_swatch(text)
        self.label_changed.emit(self._obj)

    def _on_visibility_changed(self, state: int):
        if self._obj:
            self._obj.visible = bool(state)

    def _on_transform_changed(self):
        if self._obj is None:
            return
        self._obj.pos_x = self._px.value()
        self._obj.pos_y = self._py.value()
        self._obj.pos_z = self._pz.value()
        self._obj.rot_x = self._rx.value()
        self._obj.rot_y = self._ry.value()
        self._obj.rot_z = self._rz.value()
        self._obj.scale = self._sc.value()
        self.transform_changed.emit(self._obj)

    def _on_apply_all(self):
        """Apply this label to all objects with the same mesh filename."""
        if self._obj is None:
            return
        # (future: iterate scene_objects and apply to matching mesh)
        pass
