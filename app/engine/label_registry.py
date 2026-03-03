"""
engine/label_registry.py
========================
Blender-style class label registry — each class name gets a unique,
deterministic color (same algorithm Blender uses for material preview colors).

Usage:
    from app.engine.label_registry import REGISTRY
    color = REGISTRY.get_color("car")      # -> (R, G, B) 0-255 tuple
    idx   = REGISTRY.get_index("car")     # -> int class index
    names = REGISTRY.all_names()
"""

import hashlib
import colorsys
from typing import Dict, List, Optional, Tuple


class LabelRegistry:
    """Singleton registry mapping class name → color + index."""

    # Default classes to pre-register (like Blender's default materials)
    DEFAULT_CLASSES = [
        "car", "truck", "bus", "motorcycle", "bicycle",
        "person", "animal", "box", "cone", "sphere",
        "tree", "building", "road", "sky", "ground",
    ]

    def __init__(self):
        self._classes: List[str] = []
        self._color_cache: Dict[str, Tuple[int, int, int]] = {}
        # Pre-register defaults
        for c in self.DEFAULT_CLASSES:
            self._register(c)

    def _register(self, name: str) -> int:
        """Add class if new, return its index."""
        name = name.strip().lower()
        if name not in self._classes:
            self._classes.append(name)
        return self._classes.index(name)

    def get_index(self, name: str) -> int:
        """Return the integer class index (0-based). Registers if new."""
        return self._register(name.strip().lower())

    def get_color(self, name: str) -> Tuple[int, int, int]:
        """
        Return (R, G, B) 0-255 for this class name.
        Uses a hash-based HSV approach matching Blender's object color.
        """
        key = name.strip().lower()
        if key in self._color_cache:
            return self._color_cache[key]

        # Deterministic hue from name hash (same approach as Blender)
        h_bytes = hashlib.md5(key.encode()).digest()
        hue = int.from_bytes(h_bytes[:2], "big") / 65535.0
        # Blender uses high saturation + value for vivid colors
        sat = 0.75
        val = 0.90
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        color = (int(r * 255), int(g * 255), int(b * 255))
        self._color_cache[key] = color
        return color

    def get_qcolor(self, name: str):
        """Return a PySide6 QColor for this class."""
        from PySide6.QtGui import QColor
        r, g, b = self.get_color(name)
        return QColor(r, g, b)

    def all_names(self) -> List[str]:
        """Return all registered class names in index order."""
        return list(self._classes)

    def add_class(self, name: str) -> int:
        """Manually add a class (e.g. from user input)."""
        return self._register(name)

    def rename(self, old: str, new: str):
        """Rename a class, preserving its index."""
        old_key = old.strip().lower()
        new_key = new.strip().lower()
        if old_key in self._classes:
            idx = self._classes.index(old_key)
            self._classes[idx] = new_key
            if old_key in self._color_cache:
                self._color_cache[new_key] = self._color_cache.pop(old_key)


# ── Module-level singleton ────────────────────────────────────────────────────
REGISTRY = LabelRegistry()
