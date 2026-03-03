"""
main_window.py
==============
Top-level QMainWindow that assembles all panels and wires the engine.
"""

import os
import json

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QFileDialog, QMessageBox, QLabel,
    QStackedWidget, QPushButton, QToolButton, QMenu
)
from PySide6.QtGui  import QImage, QAction
from PySide6.QtCore import Qt, Signal

from app.engine.scene_state import SceneConfig
from app.engine.worker      import GeneratorWorker, PreviewWorker
from app.panels.sidebar     import Sidebar
from app.panels.viewport    import ViewportWidget
from app.panels.controls    import ControlsPanel
from app.panels.console     import ConsolePanel
from app.panels.import_view import ImportPanel
from app.panels.hdri_view   import HDRIPanel
from app.panels.settings_view import SettingsPanel


class MainWindow(QMainWindow):
    """
    Layout:
    ┌────────┬─────────────────────┬────────┐
    │Sidebar │   Viewport          │Controls│
    │        │                     │        │
    ├────────┴─────────────────────┴────────┤
    │            Console / Bottom           │
    └───────────────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self.setMinimumSize(1280, 800)

        # ── Shared state ─────────────────────────────────────────
        self.cfg    = SceneConfig()
        self.worker:         GeneratorWorker | None = None
        self.preview_worker: PreviewWorker   | None = None
        self._paused = False

        # ── Build panels ─────────────────────────────────────────
        self.sidebar      = Sidebar()
        self.viewport     = ViewportWidget()
        self.import_view  = ImportPanel(self.cfg)
        self.hdri_view    = HDRIPanel(self.cfg)
        self.settings_view = SettingsPanel(self.cfg)
        self.controls     = ControlsPanel(self.cfg)
        self.console      = ConsolePanel()

        # Stacked center area
        self.stack = QStackedWidget() # Index mapping:
        self.stack.addWidget(self.import_view)   # 0: Models
        self.stack.addWidget(self.hdri_view)     # 1: HDRI
        self.stack.addWidget(self.viewport)      # 2: Preview Scene
        self.stack.addWidget(self.settings_view) # 3: Settings

        # ── Central layout ───────────────────────────────────────
        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        # Splitter: sidebar + center stack + controls
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.sidebar)
        top_splitter.addWidget(self.stack)
        top_splitter.addWidget(self.controls)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setStretchFactor(2, 0)
        top_splitter.setHandleWidth(1)

        root_v.addWidget(top_splitter, stretch=1)
        root_v.addWidget(self.console, stretch=0)

        # ── Wire signals ─────────────────────────────────────────
        self.sidebar.page_changed.connect(self._on_page)
        
        # System Actions
        self.sidebar.export_action.triggered.connect(self._on_export_unity)

        # Viewport Play Controls (Unity style)
        self.viewport.generate_clicked.connect(self._on_generate)
        self.viewport.stop_clicked.connect(self._on_stop)
        self.viewport.pause_clicked.connect(self._on_pause)
        
        self.controls.config_updated.connect(self._on_config_updated)
        self.import_view.config_updated.connect(self._on_config_updated)
        self.hdri_view.config_updated.connect(self._on_config_updated)
        self.settings_view.config_updated.connect(self._on_config_updated)
        # Camera lifecycle: viewport emits these; MainWindow just reacts
        self.viewport.camera_active.connect(self.viewport.clear_preview)
        self.viewport.camera_idle.connect(self._on_camera_idle)

        # ── Initial log ───────────────────────────────────────────
        self.console.append_log("Skunkworks initialised.  Import a model and HDRI to begin.")

        # Pre-populate the GL viewport with whatever cfg already has
        self._on_config_updated()

    def _on_export_unity(self):
        """Export the current SceneConfig to a JSON file for Unity ingestion."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Unity Config", "unity_config.json",
            "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                data = self.cfg.to_dict()
                data["version"] = "1.0.0"
                data["exported_at"] = os.path.basename(path)
                
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                
                self.console.append_log(f"[Export] Saved Unity configuration to: {os.path.basename(path)}")
                QMessageBox.information(self, "Export Successful", f"Unity configuration saved to:\n{path}")
            except Exception as e:
                self.console.append_log(f"[Export] Error: {e}")
                QMessageBox.critical(self, "Export Failed", f"Failed to export configuration: {e}")

    def _on_page(self, key: str):
        """Handle left sidebar navigation."""
        mapping = {
            "models":   0,
            "hdri":     1,
            "preview":  2,
            "settings": 3
        }
        idx = mapping.get(key, 0)
        self.stack.setCurrentIndex(idx)
        if key == "preview":
            # Always re-sync the GL viewport when switching to the preview tab
            self._on_config_updated()
            self._maybe_preview()

    def _on_config_updated(self):
        info = self._base_info()
        self.viewport.set_scene_info(info)
        # ── Link Config to Real-time Viewport ─────────────────────
        if self.cfg.hdri_paths:
            self.viewport.set_hdri(self.cfg.hdri_paths[0])
        else:
            self.viewport.set_hdri("")

        if self.cfg.models:
            self.viewport.load_mesh(self.cfg.models[0].mesh_path)

    def _base_info(self) -> dict:
        return {
            "dist_min":  int(self.cfg.dist_min),
            "dist_max":  int(self.cfg.dist_max),
            "weather":   "—",
            "generated": 0,
            "total":     self.cfg.num_images,
            "device":    "cuda" if self._cuda_available() else "cpu",
        }

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _on_camera_idle(self, dist, elev, azim):
        """Called by viewport 400ms after camera stops. Fire PyTorch3D ONLY if
        a model and HDRI are configured; otherwise just let the live GL show."""
        if self.cfg.models and self.cfg.hdri_paths:
            self._run_preview(manual_pose=(dist, elev, azim))

    def _maybe_preview(self):
        if self.cfg.models and self.cfg.hdri_paths: # Check paths exists
            self._run_preview()

    def _run_preview(self, manual_pose=None):
        if self.worker:
            return  # Don't preview while generating
            
        # If a preview is already running, terminate it to start the fresh one
        if self.preview_worker:
            self.preview_worker.terminate()
            self.preview_worker.wait(500)
            self.preview_worker = None
        if not self.cfg.models:
            return
        if manual_pose is None:
            self.viewport.set_loading(True)

        self.preview_worker = PreviewWorker(self.cfg, manual_pose=manual_pose)
        self.preview_worker.preview_ready.connect(self._on_preview)
        self.preview_worker.log_message.connect(self.console.append_log)
        self.preview_worker.finished.connect(self._on_preview_finished)
        self.preview_worker.start()

    def _on_preview_finished(self):
        self.viewport.set_loading(False)
        self.preview_worker = None

    def _on_generate(self):
        errors = self.cfg.validate()
        if errors:
            QMessageBox.warning(self, "Config Error", "\n• ".join(errors))
            return

        self.console.reset()
        self.console.set_generating(True)
        self.viewport.play_btn.setEnabled(False)
        self.viewport.stop_btn.setEnabled(True)
        self.viewport.pause_btn.setEnabled(True)

        self.console.start_timer()
        self.console.append_log(f"Starting generation: {self.cfg.num_images} images...")
        self.viewport.clear_preview()
        self.stack.setCurrentIndex(2) # Switch to Preview/Viewport index

        self.worker = GeneratorWorker(self.cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.preview_ready.connect(self._on_preview)
        self.worker.log_message.connect(self.console.append_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.stats_updated.connect(self._on_stats)
        self.worker.start()

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
            self.console.append_log("[User] Stop requested…")

    def _on_pause(self):
        if self.worker:
            self._paused = not self._paused
            if self._paused:
                self.worker.pause()
            else:
                self.worker.resume()
            self.console.append_log("Paused" if self._paused else "Resumed")

    def _on_progress(self, current: int, total: int):
        self.console.update_progress(current, total)

    def _on_preview(self, qimage: QImage):
        self.viewport.set_preview(qimage)

    def _on_stats(self, stats: dict):
        self.console.update_stats(stats)
        self.viewport.set_scene_info(stats)

    def _on_finished(self, success: bool, message: str):
        self.console.set_generating(False)
        self.viewport.play_btn.setEnabled(True)
        self.viewport.stop_btn.setEnabled(False)
        self.viewport.pause_btn.setEnabled(False)
        
        self.console.append_log(f"[Done] {message}")
        if success:
            QMessageBox.information(self, "Complete", message)
        self.worker = None
        self._paused = False

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(2000)
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.terminate()
            self.preview_worker.wait(1000)
        event.accept()
