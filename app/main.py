"""
app/main.py
===========
Application entry point.
"""

import sys
import os

# Make sure the project root is on sys.path regardless of where we launch from
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore    import Qt
from PySide6.QtGui     import QFontDatabase, QFont

from app.main_window import MainWindow


def _load_stylesheet() -> str:
    qss_path = os.path.join(os.path.dirname(__file__), "styles", "dark.qss")
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def main():
    # High-DPI scaling (PySide6 ≥ 6.3 handles this automatically)
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("Skunkworks")
    app.setOrganizationName("Jaymei")
    app.setApplicationDisplayName("Skunkworks – Synthetic Data Engine")

    # Apply dark stylesheet
    app.setStyleSheet(_load_stylesheet())

    # Use a clean sans-serif font
    app.setFont(QFont("San Francisco", 10))
    app.font().setFamilies([".AppleSystemUIFont", "San Francisco", "SF Pro Text", "SF Pro Display", "Helvetica Neue", "Segoe UI", "Inter", "Arial"])

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
