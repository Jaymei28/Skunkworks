import os
import subprocess
import random

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QLineEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

from app.engine.scene_state import SceneConfig


class RandomizerEntryWidget(QFrame):
    """A list item representing a randomizer script."""
    clicked = Signal(str)  # Emits the absolute path

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("script_entry")
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6) # More compact margins
        layout.setSpacing(10)

        # Icon 
        icon_lbl = QLabel("C#")
        icon_lbl.setStyleSheet("font-size: 10px; font-weight: bold; color: #4fc3f7; background: #1a1a1a; padding: 2px 4px; border-radius: 4px;")
        layout.addWidget(icon_lbl)

        # Name
        name_lbl = QLabel(self.file_name)
        name_lbl.setStyleSheet("font-weight: 500; font-size: 12px; color: #cdd6f4;") # Smaller text
        layout.addWidget(name_lbl, stretch=1)

        # Open in VS Code button
        self.open_btn = QPushButton("Edit")
        self.open_btn.setFixedSize(54, 24) # Slightly wider to avoid clipping
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 4px;
                color: #8b949e;
                font-size: 10px;
                font-weight: bold;
                padding: 0;
            }
            QPushButton:hover {
                background-color: #30363d;
                color: #c9d1d9;
                border-color: #484f58;
            }
        """)
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.open_btn)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(lambda: self.clicked.emit(self.file_path))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.file_path)


class ControlsPanel(QWidget):
    """
    Randomizer Manager Panel.
    Displays a list of all randomizer scripts in the renderer/randomizers directory.
    Allows opening them in VS Code and adding new ones.
    """
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.setObjectName("right_panel")
        
        # Make adjustable
        self.setMinimumWidth(250)
        self.setMaximumWidth(600)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        
        self.cfg = cfg
        
        # Path to randomizers
        self.base_path = os.path.join("d:\\Jaymei\\Skunkworks", "renderer", "randomizers")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 16, 12, 16)
        root.setSpacing(12)

        # Header 
        header = QHBoxLayout()
        title = QLabel("Randomizers")
        title.setObjectName("view_title")
        title.setStyleSheet("font-size: 16px; font-weight: 700;") # Slightly smaller header
        header.addWidget(title)
        
        header.addStretch()
        
        self.add_btn = QPushButton("+ New")
        self.add_btn.setFixedSize(72, 24) # Wider for better text fit
        self.add_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a6fa8, stop:1 #4fc3f7);
                color: #0d1117;
                border: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
                padding: 0;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e8bd4, stop:1 #67d3ff);
            }
        """)
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        header.addWidget(self.add_btn)
        root.addLayout(header)

        # Subtitle
        subtitle = QLabel("Click to edit randomizer logic.")
        subtitle.setObjectName("view_subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        # Scroll Area for script list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6) # Tighter spacing
        self.list_layout.addStretch()
        
        self.scroll.setWidget(self.list_container)
        root.addWidget(self.scroll, stretch=1)

        # Signals
        self.add_btn.clicked.connect(self._on_add_randomizer)

        # Initial scan
        self.refresh()

    def refresh(self):
        """Scans the directory and rebuilds the list."""
        # Clear layout
        while self.list_layout.count() > 1:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

        scripts = [f for f in os.listdir(self.base_path) if f.endswith(".cs")]
        scripts.sort()

        for s in scripts:
            full_path = os.path.join(self.base_path, s)
            entry = RandomizerEntryWidget(full_path)
            entry.clicked.connect(self._open_in_vscode)
            self.list_layout.insertWidget(self.list_layout.count() - 1, entry)

    def _open_in_vscode(self, path: str):
        """Launches VS Code for the given path."""
        try:
            # Using 'code' command (assuming it's in PATH)
            subprocess.Popen(["code", path], shell=True)
        except Exception as e:
            print(f"Error launching VS Code: {e}")

    def _on_add_randomizer(self):
        """Creates a new C# randomizer script from a Unity template."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New C# Randomizer", "Script Name (e.g. MyPoseRandomizer):")
        if ok and name:
            if not name.endswith(".cs"):
                name += ".cs"
            
            target = os.path.join(self.base_path, name)
            if os.path.exists(target):
                return # Already exists
            
            class_name = name.replace(".cs", "")
            
            template = (
                "using System;\n"
                "using UnityEngine;\n"
                "using UnityEngine.Perception.GroundTruth;\n"
                "using UnityEngine.Perception.GroundTruth.Randomizers;\n\n"
                "[Serializable]\n"
                "public class " + class_name + " : Randomizer\n"
                "{\n"
                "    // Unity-style Randomizer logic\n"
                "    protected override void OnIterationStart()\n"
                "    {\n"
                "        // TODO: Implement C# randomization logic here\n"
                "    }\n"
                "}\n"
            )
            
            with open(target, "w", encoding="utf-8") as f:
                f.write(template)
            
            self.refresh()
            self._open_in_vscode(target)
