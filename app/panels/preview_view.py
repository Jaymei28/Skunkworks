"""
panels/preview_view.py
======================
High-level preview controls and viewport switch logic.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame,
    QHBoxLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
from app.engine.scene_state import SceneConfig


class PreviewPanel(QWidget):
    """Placeholder view for Preview Scene tab. Switches focus to the Viewport."""
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        title = QLabel("Scene Preview")
        title.setObjectName("view_title")
        layout.addWidget(title)

        info = QLabel("Use the main viewport to inspect the 3D scene. "
                      "This panel provides high-level scene verification tools.")
        info.setObjectName("view_subtitle")
        info.setWordWrap(True)
        layout.addWidget(info)

        card = QFrame()
        card.setObjectName("settings_card")
        cl = QVBoxLayout(card)
        
        btn = QPushButton("Refresh Viewport")
        btn.setObjectName("action_btn")
        btn.clicked.connect(self.config_updated.emit)
        cl.addWidget(btn)
        
        layout.addWidget(card)
        layout.addStretch()
