"""Scrolling, append-only log pane with prefix-based levels.

Deliberately separate from Python's `logging` module: the audience here is
the end user, not a developer, so we don't want framework noise from torch /
ultralytics / onnxruntime leaking into this pane. The levels are
presentational only — they drive the line prefix and the filter checkbox in
the main window.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit


class LogWidget(QPlainTextEdit):
    def __init__(self, parent=None, max_lines: int = 2000, show_debug: bool = True):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.setFont(font)
        self._show_debug = show_debug

    def set_show_debug(self, value: bool) -> None:
        self._show_debug = value

    def log(self, message: str, level: str = "LOG") -> None:
        if level == "DEBUG" and not self._show_debug:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.appendPlainText(f"{ts} [{level}] {message}")

    def debug(self, message: str) -> None:
        self.log(message, level="DEBUG")

    def info(self, message: str) -> None:
        self.log(message, level="LOG")

    def error(self, message: str) -> None:
        self.log(message, level="ERROR")
