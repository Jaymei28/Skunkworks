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
    QStackedWidget, QPushButton, QToolButton, QMenu,
    QTabBar, QStyle
)
from PySide6.QtGui  import QImage, QAction
from PySide6.QtCore import Qt, Signal

from app.engine.scene_state import SceneConfig
from app.engine.worker      import GLGeneratorWorker as GeneratorWorker, GLPreviewWorker as PreviewWorker
from app.panels.sidebar     import Sidebar
from app.panels.viewport    import ViewportWidget
from app.panels.controls    import ControlsPanel
from app.panels.console     import ConsolePanel
from app.panels.import_view import ImportPanel
from app.panels.hdri_view   import HDRIPanel
from app.panels.ocean_view  import OceanPanel
from app.panels.settings_view import SettingsPanel
from app.panels.scene_hierarchy import SceneHierarchyPanel
from app.panels.object_props   import ObjectPropertiesPanel


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
        self.viewport      = ViewportWidget()
        self.ocean_view    = OceanPanel(self.cfg)
        self.settings_view = SettingsPanel(self.cfg)
        self.console       = ConsolePanel()
        self.hierarchy     = SceneHierarchyPanel(self.cfg)
        self.obj_props     = ObjectPropertiesPanel()
        self.controls      = ControlsPanel(self.cfg)

        # Stacked center area
        self.stack = QStackedWidget() 
        self.stack.addWidget(self.viewport)      # Index 0
        self.stack.addWidget(self.settings_view) # Index 1
        self.stack.addWidget(self.ocean_view)    # Index 2

        # ── Central layout ───────────────────────────────────────
        central = QWidget()
        central.setObjectName("central_widget")
        self.setCentralWidget(central)

        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        # ── Global Navigation (Top Tabs) ──────────────────────────
        self._top_tabs = QTabBar()
        self._top_tabs.setExpanding(False)
        self._top_tabs.addTab("Preview Scene")
        self._top_tabs.addTab("Settings")
        self._top_tabs.addTab("Ocean")
        self._top_tabs.setDrawBase(False)
        self._top_tabs.setStyleSheet("""
            QTabBar::tab { background: #1c2333; color: #8b949e; padding: 8px 20px;
                           border-top-left-radius: 4px; border-top-right-radius: 4px;
                           margin-right: 2px; font-weight: bold; font-size: 11px; }
            QTabBar::tab:selected { background: #161b27; color: #4fc3f7; border-bottom: 2px solid #4fc3f7; }
            QTabBar::tab:hover { color: #cdd6f4; }
        """)
        self._top_tabs.currentChanged.connect(self._on_tab_changed)

        # Top Bar Container (Logo + Tabs)
        top_bar = QWidget()
        top_bar.setStyleSheet("background: #0d1117; border-bottom: 1px solid #252d3d;")
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(15, 0, 15, 0)
        
        logo_lbl = QLabel("SKUNKWORKS")
        logo_lbl.setStyleSheet("color: #4fc3f7; font-weight: 900; font-size: 14px; letter-spacing: 1px;")
        tb_layout.addWidget(logo_lbl)
        tb_layout.addSpacing(40)
        tb_layout.addWidget(self._top_tabs)
        tb_layout.addStretch()

        # Options Button (moved to top bar)
        self.options_btn = QToolButton()
        self.options_btn.setText("File")
        self.options_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.options_btn.setStyleSheet("""
            QToolButton { background: transparent; color: #8b949e; border: 1px solid #333; border-radius: 3px; padding: 3px 10px; }
            QToolButton:hover { background: #1c2333; color: #cdd6f4; }
        """)
        self.options_menu = QMenu(self)
        self.export_action = QAction("Export for Unity", self)
        self.options_menu.addAction(self.export_action)
        self.options_btn.setMenu(self.options_menu)
        tb_layout.addWidget(self.options_btn)

        root_v.addWidget(top_bar)

        # ── Main Content Area ─────────────────────────────────────
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 1. Left: Scene Hierarchy
        main_splitter.addWidget(self.hierarchy)
        self.hierarchy.setMinimumWidth(260)
        
        # 2. Center: Stack (Viewport / Settings)
        main_splitter.addWidget(self.stack)
        
        # 3. Right: Object Properties
        main_splitter.addWidget(self.obj_props)
        self.obj_props.setMinimumWidth(300)

        main_splitter.setStretchFactor(0, 0) # hierarchy
        main_splitter.setStretchFactor(1, 1) # view
        main_splitter.setStretchFactor(2, 0) # props
        main_splitter.setHandleWidth(1)
        
        root_v.addWidget(main_splitter, stretch=1)
        root_v.addWidget(self.console, stretch=0)

        # ── Wire signals ─────────────────────────────────────────
        
        # System Actions
        self.export_action.triggered.connect(self._on_export_unity)

        # Viewport Play Controls (Unity style)
        self.viewport.generate_clicked.connect(self._on_generate)
        self.viewport.stop_clicked.connect(self._on_stop)
        self.viewport.pause_clicked.connect(self._on_pause)
        
        self.settings_view.config_updated.connect(self._on_config_updated)
        # Camera lifecycle: viewport emits these; MainWindow just reacts
        self.viewport.camera_active.connect(self.viewport.clear_preview)
        self.viewport.camera_idle.connect(self._on_camera_idle)

        # ── Scene Hierarchy ↔ Object Properties wiring ───────────
        self.hierarchy.object_selected.connect(self._on_object_selected)
        self.hierarchy.object_added.connect(self._on_object_added)
        self.hierarchy.object_removed.connect(self._on_object_removed)
        self.hierarchy.texture_changed.connect(self._on_texture_changed)
        self.hierarchy.hdri_changed.connect(self._on_hdri_changed)
        self.hierarchy.scene_changed.connect(lambda: self.viewport.update())
        self.hierarchy.navigation_requested.connect(self._on_page)
        self.obj_props.label_changed.connect(self._on_label_changed)

        # Viewport Selection ↔ UI sync
        self.viewport.object_selected.connect(self._on_viewport_object_selected)
        self.viewport.object_moved.connect(self._on_viewport_object_moved)
        self.viewport.texture_discovered.connect(self._on_texture_discovered)

        # ── Initial log ───────────────────────────────────────────
        self.console.append_log("Skunkworks initialised.  Import a model and HDRI to begin.")

        # Pre-populate the GL viewport with whatever cfg already has
        self._on_config_updated()

    def _on_hdri_changed(self):
        """Hierarchy or HDRI panel changed → push to viewport immediately."""
        if self.cfg.hdri_paths:
            self.viewport.set_hdri(self.cfg.hdri_paths[0])
        else:
            self.viewport.set_hdri("")
        self.viewport.update()

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

    def _on_tab_changed(self, index: int):
        """Top tab changed → switch stack."""
        self.stack.setCurrentIndex(index)

    def _on_page(self, key: str):
        """Page navigation from hierarchy or other signals."""
        if key == "settings":
            self._top_tabs.setCurrentIndex(1)
        elif key == "ocean":
            self._top_tabs.setCurrentIndex(2)
        else:
            self._top_tabs.setCurrentIndex(0)
                
        self.viewport.update()

    def _on_config_updated(self):
        info = self._base_info()
        self.viewport.set_scene_info(info)
        # Sync full config (all objects) to viewport
        self.viewport.set_scene_config(self.cfg)
        # Ensure hierarchy is in sync
        self.hierarchy.refresh()
        
        # ── HDRI ─────────────────────
        if self.cfg.hdri_paths:
            self.viewport.set_hdri(self.cfg.hdri_paths[0])
        else:
            self.viewport.set_hdri("")

        # Pre-load all meshes in config
        for model in self.cfg.models:
            self.viewport.load_mesh(model.mesh_path)


    # ── Scene Hierarchy / Object Properties handlers ──────────────────────

    def _on_object_selected(self, obj):
        """Hierarchy row clicked → populate properties panel."""
        self.obj_props.set_object(obj)
        self.viewport.set_selected_object(obj)

    def _on_viewport_object_selected(self, obj):
        """Object clicked in 3D viewport → select in Hierarchy."""
        self.hierarchy.select_object(obj)
        self.obj_props.set_object(obj)

    def _on_viewport_object_moved(self, obj):
        """Object transform changed in 3D viewport → refresh Properties."""
        self.obj_props.refresh_transform()


    def _on_object_added(self, obj):
        """New object added via hierarchy + Add Object button."""
        if obj and obj.config.mesh_path:
            self.viewport.load_mesh(obj.config.mesh_path)
        self.console.append_log(
            f"[Scene] Added object: {obj.config.name} ({obj.label})"
        )

    def _on_label_changed(self, obj):
        """Label edited in Object Properties → refresh hierarchy badge."""
        self.hierarchy.update_row_badge(obj)

    def _on_object_removed(self, obj):
        """Object removed from hierarchy → clear properties and reset viewport."""
        self.obj_props.set_object(None)
        self.console.append_log(f"[Scene] Removed object: {obj.config.name}")
        # If there are other objects, show the first one; otherwise reset sphere
        remaining = self.cfg.scene_objects
        if remaining:
            next_obj = remaining[0]
            self.viewport.load_mesh(next_obj.config.mesh_path)
        else:
            self.viewport.load_mesh("")

    def _on_texture_changed(self, obj):
        """Texture picked in hierarchy row → reload mesh with new texture."""
        if obj and obj.config.mesh_path:
            self.viewport.load_mesh(obj.config.mesh_path)
        # Also refresh thumbnails in obj_props if this object is selected
        self.obj_props.set_object(obj)
        # Update hierarchy row thumbnails with auto-discovered paths
        self.hierarchy.update_row_thumbnails(obj)

    def _on_texture_discovered(self, mesh_path: str, tex_path: str):
        """Viewport auto-discovered a texture → sync to config and UI."""
        updated = False
        for obj in self.cfg.scene_objects:
            if obj.config.mesh_path == mesh_path:
                if not obj.config.tex_albedo:
                    obj.config.tex_albedo = tex_path
                    updated = True
                    # Update hierarchy thumbnails
                    self.hierarchy.refresh_item_thumbnails(obj)
        
        if updated:
            # Refresh properties panel if current object was updated
            curr = self.obj_props._obj
            if curr and curr.config.mesh_path == mesh_path:
                self.obj_props.set_object(curr)

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
        if not self.cfg.models and not self.cfg.ocean.enabled:
            return
        if manual_pose is None:
            self.viewport.set_loading(True)

        # GLPreviewWorker takes explicit azim/elev/dist args (not a tuple)
        if manual_pose is not None:
            dist, elev, azim = manual_pose
            self.preview_worker = PreviewWorker(
                self.cfg, azim=azim, elev=elev, dist=dist)
        else:
            # Default pose: mid-range values from config
            cfg = self.cfg
            azim = (cfg.azim_min + cfg.azim_max) / 2.0
            elev = (cfg.elev_min + cfg.elev_max) / 2.0
            dist = (cfg.dist_min + cfg.dist_max) / 2.0
            self.preview_worker = PreviewWorker(
                self.cfg, azim=azim, elev=elev, dist=dist)

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
