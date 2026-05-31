"""App-wide visual theme (Qt style sheet).

A single dark theme applied to the QApplication so every window and dialog
shares one cohesive look. Kept as a flat QSS string targeting standard widget
types plus a couple of object names (`primary` for the main action button,
`vsep` for the group separators) — no per-widget styling scattered across the
UI code.

`apply_theme(app)` sets the Fusion base style (consistent across Windows
versions) then layers the QSS on top.
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QFrame

# Palette: a calm dark-slate scheme with a single blue accent for primary
# actions and selected/active states.
_BG = "#1e1f22"        # window background
_PANEL = "#2b2d31"     # raised surfaces (log, dropdown popups)
_PANEL_2 = "#34373c"   # inputs (combo, line edit)
_BORDER = "#44474d"
_BORDER_HI = "#5a5d63"
_TEXT = "#dcddde"
_TEXT_DIM = "#9aa0a6"
_TEXT_OFF = "#6c7176"
_ACCENT = "#4a7fd6"
_ACCENT_HI = "#5b8de0"
_ACCENT_LO = "#3f6fbf"

_QSS = f"""
QWidget {{
    background-color: {_BG};
    color: {_TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{ background-color: {_BG}; }}
QLabel {{ background: transparent; }}

QPushButton {{
    background-color: #3a3d42;
    border: 1px solid {_BORDER};
    border-radius: 5px;
    padding: 5px 12px;
}}
QPushButton:hover {{ background-color: {_BORDER}; }}
QPushButton:pressed {{ background-color: {_PANEL_2}; }}
QPushButton:checked {{ background-color: {_ACCENT}; border-color: {_ACCENT}; color: white; }}
QPushButton:disabled {{ color: {_TEXT_OFF}; background-color: {_PANEL}; }}

QPushButton#primary {{
    background-color: {_ACCENT};
    border-color: {_ACCENT};
    color: white;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: {_ACCENT_HI}; }}
QPushButton#primary:pressed {{ background-color: {_ACCENT_LO}; }}

QComboBox, QLineEdit {{
    background-color: {_PANEL_2};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 20px;
}}
QComboBox:hover, QLineEdit:focus {{ border-color: {_BORDER_HI}; }}
QComboBox:disabled {{ color: {_TEXT_OFF}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {_PANEL};
    border: 1px solid {_BORDER};
    selection-background-color: {_ACCENT};
    outline: none;
}}

QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {_BORDER_HI};
    background-color: {_PANEL_2};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {_ACCENT}; border-color: {_ACCENT};
}}

QSlider::groove:horizontal {{ height: 4px; background: {_BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {_ACCENT}; width: 14px; height: 14px;
    margin: -6px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: {_ACCENT_HI}; }}

QPlainTextEdit, QTextEdit {{
    background-color: {_PANEL};
    border: 1px solid #36393f;
    border-radius: 6px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
}}

QFrame#vsep {{ color: #3a3d42; }}
QLabel#section {{ color: {_TEXT_DIM}; font-size: 11px; font-weight: 600; }}

QStatusBar {{ background-color: #17181a; color: {_TEXT_DIM}; }}
QStatusBar::item {{ border: none; }}

QScrollBar:vertical {{ background: {_BG}; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {_BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {_BORDER_HI}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QToolTip {{
    background-color: {_PANEL}; color: {_TEXT};
    border: 1px solid {_BORDER}; padding: 4px;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the dark theme to the whole application."""
    app.setStyle("Fusion")
    app.setStyleSheet(_QSS)


def make_separator() -> QFrame:
    """A thin vertical rule for separating control-bar groups."""
    line = QFrame()
    line.setObjectName("vsep")
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    return line
