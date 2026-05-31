"""Configuration overrides dialog.

Surfaces every bundled config file (dungeons, classes, tag_rules) with
its current state ("using default" or "user override") and per-file
actions:

  - Customize this file → copies the bundled default to the user dir if
    it isn't already there, then opens it in the OS default text editor.
    From that point on the user copy wins.
  - Revert to default → deletes the user copy (with confirm). Next load
    falls back to the bundled version; future app updates that change
    the bundled file naturally appear.

The dialog reads filesystem state on demand — no in-memory cache — so
the "currently using" labels stay accurate even if the user edits files
outside the app between operations.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from wow_alert.paths import USER_CONFIG_DIR, defaults_config_dir


class ConfigOverridesDialog(QDialog):
    """List bundled files + per-file customize/revert controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration overrides")
        self.resize(640, 540)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        intro = QLabel(
            "Customize a file to make a personal copy you can edit. "
            "Revert to default deletes your copy and falls back to what "
            "shipped with the app. Files only override when present in "
            "your user config directory; you always get bundled updates "
            "for anything you haven't customized."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        # Action buttons that apply to ALL files at once.
        bulk = QHBoxLayout()
        bulk.addStretch(1)
        self._open_user_dir_btn = QPushButton("Open user config folder")
        self._open_user_dir_btn.clicked.connect(self._open_user_dir)
        bulk.addWidget(self._open_user_dir_btn)
        root.addLayout(bulk)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._populate_rows()
        self._rows_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(self._rows_container)
        scroll.setWidgetResizable(True)
        root.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(
            self.accept,
        )
        root.addWidget(buttons)

    # ---- list build ----

    def _populate_rows(self) -> None:
        """Walk the bundled defaults tree and render one row per file."""
        defaults = defaults_config_dir()
        if not defaults.exists():
            self._rows_layout.addWidget(QLabel(
                "Bundled defaults directory not found — install is broken.",
            ))
            return
        files = sorted(defaults.rglob("*.yaml"))
        for path in files:
            rel = path.relative_to(defaults)
            self._rows_layout.addWidget(self._build_row(rel))

    def _build_row(self, rel_path: Path) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)

        user_path = USER_CONFIG_DIR / rel_path
        status_label = QLabel(
            "[user override]" if user_path.exists() else "[default]"
        )
        status_label.setFixedWidth(110)
        status_label.setStyleSheet(
            "color: #4a7fd6;" if user_path.exists() else "color: gray;"
        )
        layout.addWidget(status_label)

        name_label = QLabel(str(rel_path).replace("\\", "/"))
        name_label.setMinimumWidth(260)
        layout.addWidget(name_label, stretch=1)

        customize_btn = QPushButton("Customize")
        customize_btn.setToolTip(
            "Copy the bundled default to your user config dir and open it "
            "in your default text editor. Already-customized files just open."
        )
        customize_btn.clicked.connect(
            lambda _checked=False, rel=rel_path, sl=status_label:
            self._customize(rel, sl)
        )
        layout.addWidget(customize_btn)

        revert_btn = QPushButton("Revert")
        revert_btn.setToolTip(
            "Delete the user copy. Bundled default takes over again on "
            "next app launch."
        )
        revert_btn.setEnabled(user_path.exists())
        revert_btn.clicked.connect(
            lambda _checked=False, rel=rel_path, sl=status_label, rb=revert_btn:
            self._revert(rel, sl, rb)
        )
        layout.addWidget(revert_btn)

        return row

    # ---- actions ----

    def _customize(self, rel_path: Path, status_label: QLabel) -> None:
        """Ensure a user copy exists, then open it in the OS text editor."""
        user_path = USER_CONFIG_DIR / rel_path
        if not user_path.exists():
            user_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(defaults_config_dir() / rel_path, user_path)
            except OSError as exc:
                QMessageBox.warning(
                    self, "Could not customize",
                    f"Failed to copy default to user dir: {exc}",
                )
                return
            status_label.setText("[user override]")
            status_label.setStyleSheet("color: #4a7fd6;")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_path)))

    def _revert(
        self,
        rel_path: Path,
        status_label: QLabel,
        revert_btn: QPushButton,
    ) -> None:
        """Delete the user copy after confirming."""
        user_path = USER_CONFIG_DIR / rel_path
        if not user_path.exists():
            revert_btn.setEnabled(False)
            return
        reply = QMessageBox.question(
            self, "Revert to default?",
            f"Delete your customized {rel_path}?\n\n"
            "Bundled default takes over on next app launch. Your edits "
            "to this file will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            user_path.unlink()
        except OSError as exc:
            QMessageBox.warning(
                self, "Could not revert",
                f"Failed to delete user copy: {exc}",
            )
            return
        status_label.setText("[default]")
        status_label.setStyleSheet("color: gray;")
        revert_btn.setEnabled(False)

    def _open_user_dir(self) -> None:
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(USER_CONFIG_DIR)))
