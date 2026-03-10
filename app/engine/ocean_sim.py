"""
engine/ocean_sim.py
===================
CPU-side simulation of Gerstner waves for buoyancy physics.
Synchronized with viewport.py High-Fidelity Ocean Shaders.
"""

import numpy as np
import time

class GerstnerWaveSim:
    def __init__(self):
        # Matches waves[8] in viewport.py
        # [dirX, dirZ, steepness, wavelength_factor]
        self.waves = [
            [1.0, 0.5,  0.50, 1.2], [0.8, 1.0,  0.45, 0.9], 
            [0.2, 1.2,  0.40, 0.7], [-0.5, 0.8, 0.35, 0.5], 
            [0.1, -1.0, 0.30, 0.3], [-0.7, -0.2, 0.25, 0.15], 
            [0.5, 0.5,  0.20, 0.08],[-0.2, 0.8, 0.15, 0.04]
        ]

    def get_wave_height(self, x: float, z: float, t: float, 
                        wind_speed: float = 30.0, 
                        wind_direction: float = 0.0, 
                        choppiness: float = 1.3, 
                        wave_amplitude: float = 1.0,
                        repetition_size: float = 500.0,
                        band0_multiplier: float = 1.0,
                        band1_multiplier: float = 1.0,
                        chaos: float = 0.8,
                        storm_intensity: float = 0.0,
                        **kwargs) -> float:
        """
        Calculates the vertical displacement (Y) at world-space (x, z).
        Now uses a 3-iteration iterative solver to account for horizontal 
        Gerstner displacement, ensuring the boat 'stays' on top of physical peaks.
        """
        # 1. Start with initial guess (Base position = World position)
        guess_x, guess_z = x, z
        
        # 2. Iterate to find the Base position that DISPLACES to World(x, z)
        # 3 iterations is industry standard for stable buoyancy
        for _ in range(3):
            dx, _, dz, _ = self._calculate_gerstner_full(
                guess_x, guess_z, t, wind_speed, wind_direction, choppiness,
                wave_amplitude, repetition_size, band0_multiplier, band1_multiplier, chaos, storm_intensity
            )
            # Find the error (where we are vs where we want to be)
            err_x = (guess_x + dx) - x
            err_z = (guess_z + dz) - z
            # Newton-style step
            guess_x -= err_x * 0.7
            guess_z -= err_z * 0.7

        # 3. Final height evaluation at the derived base point
        _, height, _, _ = self._calculate_gerstner_full(
            guess_x, guess_z, t, wind_speed, wind_direction, choppiness,
            wave_amplitude, repetition_size, band0_multiplier, band1_multiplier, chaos, storm_intensity
        )
        return height

    def _calculate_gerstner_full(self, x, z, t, wind_speed, wind_dir, choppiness, 
                                amp, rep_size, b0, b1, chaos, storm):
        """Internal helper to match EXACT compute pattern of viewport.py shader."""
        wind_angle = np.radians(wind_dir)
        cos_w = np.cos(wind_angle)
        sin_w = np.sin(wind_angle)
        
        # Boost based on Storm intensity (matches visual intent)
        storm_boost = 1.0 + storm
        global_amp = (wind_speed / 10.0) * amp * storm_boost
        global_chop = choppiness * 1.8 * (1.0 + storm * 0.5)
        
        disp_x, disp_y, disp_z = 0.0, 0.0, 0.0
        max_crest = 0.0
        
        for i, w in enumerate(self.waves):
            # w = [dirX, dirZ, steepness, wl_factor]
            # Match GLSL direction math: normalize(windRot * w.xy)
            # where windRot = [ cos  sin ] / [ -sin  cos ] (Column Major transpose)
            # Resulting in: X' = cos*x + sin*z, Z' = -sin*x + cos*z
            orig_dx, orig_dz = w[0], w[1]
            dx_rot =  orig_dx * cos_w + orig_dz * sin_w
            dz_rot = -orig_dx * sin_w + orig_dz * cos_w
            
            dlen = np.sqrt(dx_rot**2 + dz_rot**2)
            nx, nz = dx_rot/dlen, dz_rot/dlen
            
            band_mul = b0 if i < 4 else b1
            wl = w[3] * rep_size * 0.2
            k = 2.0 * np.pi / max(wl, 0.01)
            a = (wl / 40.0) * global_amp * band_mul
            q = (w[2] * global_chop * chaos) / (k * a * 8.0 + 0.001)

            speed = np.sqrt(9.81 / k)
            # Phase: k * (dot(dir, pos) - speed * t)
            phase = k * (nx * x + nz * z - speed * t)
            
            sin_p = np.sin(phase)
            cos_p = np.cos(phase)
            
            disp_x += q * a * nx * cos_p
            disp_y += a * sin_p
            disp_z += q * a * nz * cos_p
            max_crest += a

        return disp_x, disp_y, disp_z, max_crest

    def get_surface_normal(self, x: float, z: float, t: float, **kwargs) -> np.ndarray:
        """Estimate the normal for tilting. Still uses heightmap-style finite diff."""
        eps = 1.2 # Slightly wider for stability on choppy water
        h0 = self.get_wave_height(x, z, t, **kwargs)
        hx = self.get_wave_height(x + eps, z, t, **kwargs)
        hz = self.get_wave_height(x, z + eps, t, **kwargs)
        
        tx = np.array([eps, hx - h0, 0.0])
        tz = np.array([0.0, hz - h0, eps])
        
        norm = np.cross(tz, tx)
        mag = np.linalg.norm(norm)
        if mag < 1e-6: return np.array([0, 1, 0], dtype=np.float32)
        return (norm / mag).astype(np.float32)

ocean_physics = GerstnerWaveSim()
