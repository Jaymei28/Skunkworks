"""
panels/console.py
=================
Bottom panel – progress bar, status line, and log console.
Generation controls have been moved to the top header for a Unity-style workflow.
"""

import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QPlainTextEdit, QSizePolicy, QFrame,
)
from PySide6.QtCore    import Qt, Signal
from PySide6.QtGui     import QTextCursor, QFont, QColor


class ConsolePanel(QWidget):
    """Bottom panel with progress and log view."""

    preview_clicked  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottom_panel")
        self.setFixedHeight(180) # Slightly shorter now that buttons are gone

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        #Top bar: Status and Stats
        top = QHBoxLayout()

        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("stat_label")
        top.addWidget(self.status_lbl)

        top.addStretch()

        # Stats pills
        self._stat_pills: dict[str, QLabel] = {}
        for key, default in [("generated", "0 / 0"), ("weather", "—"), ("device", "CPU")]:
            pill = QLabel(default)
            pill.setObjectName("stat_label")
            top.addWidget(pill)
            self._stat_pills[key] = pill
            top.addSpacing(4)

        root.addLayout(top)

        #Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

        #Progress label row
        prog_row = QHBoxLayout()
        self.prog_lbl = QLabel("0 / 0")
        self.prog_lbl.setObjectName("value_label")
        prog_row.addWidget(self.prog_lbl)
        prog_row.addStretch()

        self.eta_lbl = QLabel("ETA: —")
        self.eta_lbl.setObjectName("stat_label")
        prog_row.addWidget(self.eta_lbl)
        root.addLayout(prog_row)

        #Log console
        self.log = QPlainTextEdit()
        self.log.setObjectName("log_console")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1000) # Keep more logs
        root.addWidget(self.log)

        self._paused      = False
        self._start_time  = None
        self._total       = 0

    #Public API

    def set_generating(self, running: bool):
        """Update status label based on generation state."""
        if running:
            self.status_lbl.setText("Generating…")
            self.status_lbl.setStyleSheet("color: #4fc3f7;")
        else:
            self.status_lbl.setText("Ready")
            self.status_lbl.setStyleSheet("")

    def update_progress(self, current: int, total: int):
        self._total = total
        pct = int(current / total * 100) if total > 0 else 0
        self.progress.setValue(pct)
        self.prog_lbl.setText(f"{current} / {total}  ({pct}%)")

        if self._start_time and current > 0:
            import time
            elapsed = time.monotonic() - self._start_time
            rate    = current / elapsed
            remaining = (total - current) / rate if rate > 0 else 0
            eta_str = str(datetime.timedelta(seconds=int(remaining)))
            self.eta_lbl.setText(f"ETA: {eta_str}")

    def start_timer(self):
        import time
        self._start_time = time.monotonic()

    def append_log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}]  {msg}")
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def update_stats(self, stats: dict):
        generated = stats.get("generated", 0)
        total     = stats.get("total", 0)
        self._stat_pills["generated"].setText(f"{generated} / {total}")
        self._stat_pills["weather"].setText(stats.get("weather", "—"))
        self._stat_pills["device"].setText(stats.get("device", "cpu").upper())

    def reset(self):
        """Reset progress visuals."""
        self.progress.setValue(0)
        self.prog_lbl.setText("0 / 0")
        self.eta_lbl.setText("ETA: —")
        self._start_time = None
        self._paused     = False
