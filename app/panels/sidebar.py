"""
panels/sidebar.py
=================
Left navigation sidebar with icon buttons for panel switching.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QSpacerItem,
    QSizePolicy, QFrame, QToolButton, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui  import QFont, QAction


class SidebarButton(QPushButton):
    """Single navigation button with icon + label."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar_btn")
        self.setCheckable(True)
        self.setText(label)
        self.setMinimumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(label)


class Sidebar(QWidget):
    """Left navigation sidebar."""

    page_changed = Signal(str)   # emits page key when user clicks

    PAGES = [
        ("Models",        "models"),
        ("HDRI",          "hdri"),
        ("Preview Scene", "preview"),
        ("Settings",      "settings"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 16)
        layout.setSpacing(2)

        # ── Logo ────────────────────────────────────────
        logo = QLabel("SKUNKWORKS")
        logo.setObjectName("logo_label")
        logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(logo)

        sub = QLabel("SYNTHETIC DATA ENGINE")
        sub.setObjectName("logo_sub")
        sub.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(sub)

        # ── Options Menu ────────────────────────────────
        from PySide6.QtWidgets import QToolButton, QMenu
        from PySide6.QtGui import QAction
        
        self.options_btn = QToolButton()
        self.options_btn.setObjectName("sidebar_options_btn")
        self.options_btn.setText("Options  ▾")
        self.options_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.options_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.options_btn.setFixedHeight(28)
        self.options_btn.setStyleSheet("""
            QToolButton {
                background: #21262d; border: 1px solid #30363d;
                border-radius: 4px; color: #8b949e;
                font-size: 11px; font-weight: 500;
                margin: 8px 4px 4px 4px;
            }
            QToolButton:hover { background: #30363d; color: #cdd6f4; }
            QToolButton::menu-indicator { image: none; }
        """)
        
        self.options_menu = QMenu(self)
        self.options_menu.setStyleSheet("""
            QMenu { background-color: #161b22; border: 1px solid #30363d; padding: 4px 0px; }
            QMenu::item { padding: 6px 28px 6px 14px; color: #8b949e; font-size: 11px; }
            QMenu::item:selected { background-color: #1a2744; color: #4fc3f7; }
        """)
        
        self.export_action = QAction("Export for Unity", self)
        self.options_menu.addAction(self.export_action)
        self.options_btn.setMenu(self.options_menu)
        layout.addWidget(self.options_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addSpacing(6)

        # ── Nav buttons ──────────────────────────────────
        self._buttons: dict[str, SidebarButton] = {}
        for label, key in self.PAGES:
            btn = SidebarButton(label)
            btn.clicked.connect(lambda checked, k=key: self._on_click(k))
            layout.addWidget(btn)
            self._buttons[key] = btn

        layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        )

        # ── Version footer ───────────────────────────────
        ver = QLabel("v0.1.0-alpha")
        ver.setObjectName("logo_sub")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ver)

        # Select first page
        self._activate("models")

    def _activate(self, key: str):
        for k, btn in self._buttons.items():
            btn.setChecked(k == key)
            btn.setProperty("active", "true" if k == key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_click(self, key: str):
        self._activate(key)
        self.page_changed.emit(key)
