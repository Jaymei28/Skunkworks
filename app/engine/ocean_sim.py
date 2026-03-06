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
                        band0_mul: float = 1.0,
                        band1_mul: float = 1.0,
                        **kwargs) -> float:
        """
        Calculate the Y displacement at a given world (x, z) coordinate.
        Matches calculateWave() in viewport.py.
        """
        wind_angle = np.radians(wind_direction)
        cos_w = np.cos(wind_angle)
        sin_w = np.sin(wind_angle)
        
        global_amp = (wind_speed / 10.0) * wave_amplitude
        global_chop = choppiness * 1.8
        
        y = 0.0
        # For Gerstner waves, x and z also displace. 
        # For simplicity of lookup, we assume the input (x,z) is the 'base' position.
        
        for i, w in enumerate(self.waves):
            dx_orig, dz_orig, q_factor, wl_factor = w
            
            # Rotate direction
            dx_rot = dx_orig * cos_w - dz_orig * sin_w
            dz_rot = dx_orig * sin_w + dz_orig * cos_w
            
            # Normalize direction
            dlen = np.sqrt(dx_rot*dx_rot + dz_rot*dz_rot)
            ndx, ndz = dx_rot/dlen, dz_rot/dlen
            
            band_mul = band0_mul if i < 4 else band1_mul
            
            wavelength = wl_factor * repetition_size * 0.2
            k = 2.0 * np.pi / max(wavelength, 0.01)
            a = (wavelength / 40.0) * global_amp * band_mul
            
            speed = np.sqrt(9.81 / k)
            phase = k * (ndx * x + ndz * z - speed * t)
            
            y += a * np.sin(phase)
            
        return y

    def get_surface_normal(self, x: float, z: float, t: float, **kwargs) -> np.ndarray:
        """Estimate the normal for tilting floating objects."""
        eps = 1.0 # Larger eps for smoother tilting on large waves
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
