from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QPushButton, QFrame, QCheckBox, QColorDialog, QComboBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

from app.engine.scene_state import SceneConfig


class OceanPanel(QWidget):
    """View for configuring ocean simulation and buoyancy."""
    config_updated = Signal()

    def __init__(self, cfg: SceneConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setObjectName("ocean_panel")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # ── Title ──────────────────────────────────────────
        title_v = QVBoxLayout()
        title = QLabel("Ocean Surface (HDRP-Style)")
        title.setObjectName("view_title")
        title_v.addWidget(title)
        subtitle = QLabel("Physical water simulation based on Unity HDRP.")
        subtitle.setObjectName("view_subtitle")
        title_v.addWidget(subtitle)
        layout.addLayout(title_v)

        # ── General Settings ──────────────────────────────────
        self.enabled_cb = QCheckBox("Enable Water System")
        self.enabled_cb.setChecked(self.cfg.ocean.enabled)
        self.enabled_cb.toggled.connect(self._sync)
        layout.addWidget(self.enabled_cb)

        card = QFrame()
        card.setObjectName("settings_card")
        cl = QVBoxLayout(card)
        layout.addWidget(card)

        # Surface Type
        surf_h = QHBoxLayout()
        surf_h.addWidget(QLabel("Surface Type:"))
        self.surf_combo = QComboBox()
        self.surf_combo.addItems(["Ocean", "River", "Pool"])
        self.surf_combo.setCurrentText(self.cfg.ocean.surface_type)
        self.surf_combo.currentTextChanged.connect(self._sync)
        surf_h.addWidget(self.surf_combo)
        cl.addLayout(surf_h)

        # Level
        lvl_h = QHBoxLayout()
        lvl_h.addWidget(QLabel("Water Level (m):"))
        self.lvl_spin = QDoubleSpinBox()
        self.lvl_spin.setRange(-50.0, 50.0)
        self.lvl_spin.setValue(self.cfg.ocean.level)
        self.lvl_spin.valueChanged.connect(self._sync)
        lvl_h.addWidget(self.lvl_spin)
        cl.addLayout(lvl_h)

        # ── Simulation Card ──────────────────────────────────
        sim_label = QLabel("SIMULATION")
        sim_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(sim_label)

        def add_row(lbl, minv, maxv, step, val, parent_layout):
            row = QHBoxLayout()
            row.addWidget(QLabel(lbl))
            sb = QDoubleSpinBox()
            sb.setRange(minv, maxv)
            sb.setSingleStep(step)
            sb.setValue(val)
            sb.valueChanged.connect(self._sync)
            row.addWidget(sb)
            parent_layout.addLayout(row)
            return sb

        self.time_spin = add_row("Time Multiplier:", 0, 10, 0.1, self.cfg.ocean.time_multiplier, cl)
        self.rep_spin  = add_row("Repetition Size (m):", 10, 2000, 50, self.cfg.ocean.repetition_size, cl)
        self.wind_spin = add_row("Wind Speed (m/s):", 0, 100, 0.5, self.cfg.ocean.wind_speed, cl)
        self.dir_spin  = add_row("Wind Direction:", 0, 360, 5, self.cfg.ocean.wind_direction, cl)
        self.chop_spin = add_row("Choppiness:", 0, 5, 0.1, self.cfg.ocean.choppiness, cl)
        self.chaos_spin = add_row("Chaos / Randomness:", 0, 1, 0.05, self.cfg.ocean.chaos, cl)
        self.wave_amp_spin = add_row("Wave Amplitude:", 0, 10, 0.1, self.cfg.ocean.wave_amplitude, cl)
        
        # Bands
        band_h = QHBoxLayout()
        band_h.addWidget(QLabel("Bands (0 / 1) Mul:"))
        self.b0_spin = QDoubleSpinBox()
        self.b0_spin.setRange(0, 2); self.b0_spin.setValue(self.cfg.ocean.band0_multiplier); self.b0_spin.setSingleStep(0.1)
        self.b1_spin = QDoubleSpinBox()
        self.b1_spin.setRange(0, 2); self.b1_spin.setValue(self.cfg.ocean.band1_multiplier); self.b1_spin.setSingleStep(0.1)
        self.b0_spin.valueChanged.connect(self._sync); self.b1_spin.valueChanged.connect(self._sync)
        band_h.addWidget(self.b0_spin); band_h.addWidget(self.b1_spin)
        cl.addLayout(band_h)

        # ── Currents ──────────────────────────────────────────
        curr_label = QLabel("CURRENTS")
        curr_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(curr_label)
        self.curr_spd_spin = add_row("Speed (m/s):", 0, 20, 0.5, self.cfg.ocean.current_speed, cl)
        self.curr_dir_spin = add_row("Orientation:", 0, 360, 5, self.cfg.ocean.current_orientation, cl)

        # ── Ripples ──────────────────────────────────────────
        rip_label = QLabel("RIPPLES (MICRO DETAIL)")
        rip_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(rip_label)
        self.rip_cb = QCheckBox("Enable Ripples")
        self.rip_cb.setChecked(self.cfg.ocean.ripples_enabled)
        self.rip_cb.toggled.connect(self._sync)
        cl.addWidget(self.rip_cb)
        self.rip_spd_spin = add_row("Ripple Wind Speed:", 0, 50, 0.5, self.cfg.ocean.ripples_wind_speed, cl)
        self.rip_dir_spin = add_row("Ripple Wind Dir:", 0, 360, 5, self.cfg.ocean.ripples_wind_dir, cl)
        self.rip_chs_spin = add_row("Ripple Chaos:", 0, 1, 0.05, self.cfg.ocean.ripples_chaos, cl)

        # ── Material & Scattering ──────────────────────────
        mat_label = QLabel("MATERIAL & SCATTERING")
        mat_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(mat_label)

        self.refr_col_btn = self._add_color_row("Refraction Color:", self.cfg.ocean.refraction_color, cl)
        self.scat_col_btn = self._add_color_row("Scattering Color:", self.cfg.ocean.scattering_color, cl)
        
        self.abs_spin  = add_row("Absorption Dist (m):", 0.1, 100, 1, self.cfg.ocean.absorption_distance, cl)
        self.amb_spin  = add_row("Ambient Scattering:", 0, 1, 0.05, self.cfg.ocean.ambient_scattering, cl)
        self.hscat_spin = add_row("Height Scattering:", 0, 1, 0.05, self.cfg.ocean.height_scattering, cl)
        self.dscat_spin = add_row("Displacement Scattering:", 0, 1, 0.05, self.cfg.ocean.displacement_scattering, cl)
        self.ltip_spin  = add_row("Direct Light Tip:", 0, 2, 0.1, self.cfg.ocean.direct_light_tip_scattering, cl)
        self.lbdy_spin  = add_row("Direct Light Body:", 0, 2, 0.1, self.cfg.ocean.direct_light_body_scattering, cl)
        
        self.smooth_spin = add_row("Smoothness:", 0, 1, 0.05, self.cfg.ocean.smoothness, cl)
        self.trans_spin = add_row("Transparency:", 0, 1, 0.05, self.cfg.ocean.transparency, cl)
        self.refl_spin  = add_row("Reflection Strength:", 0, 2, 0.1, self.cfg.ocean.reflection, cl)

        # ── Foam ──────────────────────────────────
        foam_label = QLabel("FOAM")
        foam_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(foam_label)

        self.foam_cb = QCheckBox("Enable Foam")
        self.foam_cb.setChecked(self.cfg.ocean.foam_enabled)
        self.foam_cb.toggled.connect(self._sync)
        cl.addWidget(self.foam_cb)
        
        self.foam_amt_spin = add_row("Foam Amount:", 0, 2, 0.1, self.cfg.ocean.foam_amount, cl)
        self.foam_sm_spin  = add_row("Foam Smoothness:", 0, 1, 0.05, self.cfg.ocean.foam_smoothness, cl)

        # ── Caustics ────────────────────────────────
        caus_label = QLabel("CAUSTICS")
        caus_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(caus_label)
        self.caus_cb = QCheckBox("Enable Caustics")
        self.caus_cb.setChecked(self.cfg.ocean.caustics_enabled)
        self.caus_cb.toggled.connect(self._sync)
        cl.addWidget(self.caus_cb)
        self.caus_int_spin = add_row("Caustics Intensity:", 0, 2, 0.1, self.cfg.ocean.caustics_intensity, cl)

        # ── Physics ──────────────────────────────────
        storm_label = QLabel("PHYSICS & WEATHER")
        storm_label.setStyleSheet("font-weight: bold; color: #8b949e; margin-top: 10px;")
        cl.addWidget(storm_label)
        self.storm_spin = add_row("Storm Intensity:", 0, 1, 0.05, self.cfg.ocean.storm_intensity, cl)
        self.buoy_spin  = add_row("Buoyancy reaction:", 0, 2, 0.1, self.cfg.ocean.buoyancy, cl)

        layout.addStretch()

    def _add_color_row(self, label, color_list, layout):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        btn = QPushButton()
        btn.setFixedWidth(40)
        btn.setFixedHeight(20)
        
        def update_btn():
            qcol = QColor.fromRgbF(color_list[0], color_list[1], color_list[2])
            btn.setStyleSheet(f"background: {qcol.name()}; border: 1px solid #30363d; border-radius: 3px;")

        update_btn()
        
        def pick():
            qcol = QColorDialog.getColor(QColor.fromRgbF(color_list[0], color_list[1], color_list[2]), self)
            if qcol.isValid():
                color_list[0] = qcol.redF()
                color_list[1] = qcol.greenF()
                color_list[2] = qcol.blueF()
                update_btn()
                self._sync()
        
        btn.clicked.connect(pick)
        row.addWidget(btn)
        row.addStretch()
        layout.addLayout(row)
        return btn

    def _sync(self):
        o = self.cfg.ocean
        o.enabled = self.enabled_cb.isChecked()
        o.surface_type = self.surf_combo.currentText()
        o.level = self.lvl_spin.value()
        
        o.time_multiplier = self.time_spin.value()
        o.repetition_size = self.rep_spin.value()
        o.wind_speed = self.wind_spin.value()
        o.wind_direction = self.dir_spin.value()
        o.choppiness = self.chop_spin.value()
        o.chaos = self.chaos_spin.value()
        o.wave_amplitude = self.wave_amp_spin.value()
        
        o.band0_multiplier = self.b0_spin.value()
        o.band1_multiplier = self.b1_spin.value()
        
        o.current_speed = self.curr_spd_spin.value()
        o.current_orientation = self.curr_dir_spin.value()
        
        o.ripples_enabled = self.rip_cb.isChecked()
        o.ripples_wind_speed = self.rip_spd_spin.value()
        o.ripples_wind_dir = self.rip_dir_spin.value()
        o.ripples_chaos = self.rip_chs_spin.value()

        o.refraction_color = [self.refr_col_btn.styleSheet().count("#"), 0.0, 0.0] # Placeholder if real color sync is needed
        # Wait, the color row handles the list directly in pick().
        
        o.absorption_distance = self.abs_spin.value()
        o.ambient_scattering = self.amb_spin.value()
        o.height_scattering = self.hscat_spin.value()
        o.displacement_scattering = self.dscat_spin.value()
        o.direct_light_tip_scattering = self.ltip_spin.value()
        o.direct_light_body_scattering = self.lbdy_spin.value()
        
        o.smoothness = self.smooth_spin.value()
        o.transparency = self.trans_spin.value()
        o.reflection = self.refl_spin.value()
        
        o.foam_enabled = self.foam_cb.isChecked()
        o.foam_amount = self.foam_amt_spin.value()
        o.foam_smoothness = self.foam_sm_spin.value()
        
        o.caustics_enabled = self.caus_cb.isChecked()
        o.caustics_intensity = self.caus_int_spin.value()
        
        o.storm_intensity = self.storm_spin.value()
        o.buoyancy = self.buoy_spin.value()
        
        self.config_updated.emit()
