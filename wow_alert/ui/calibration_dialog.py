"""Post-calibration edit dialog.

After the LLM finishes, this dialog lets the user verify what was detected
and fix it before it's persisted. Useful because:
  - WoW truncates long names in party frames; the LLM correctly reads what
    it sees ("Shafte…") but the user knows the full name ("Shafter Joel").
  - The LLM sometimes returns null for hard-to-read slots; the user can
    fill those in by hand.
  - The dungeon name occasionally OCRs incorrectly or isn't in the
    screenshot; the user can override.

The dialog shows a thumbnail of each detected party slot next to its name
field so the user has visual context for what they're editing.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from wow_alert.calibration import Calibration, PartyMember

logger = logging.getLogger(__name__)


_THUMBNAIL_HEIGHT = 48  # px; just enough to read the slot, keeps dialog compact

# Role dropdown values. Index 0 is "unknown" — maps to None in the saved
# Calibration. The display labels are user-friendly; the saved values are
# the lowercase tokens the rule engine expects.
_ROLE_OPTIONS = [
    ("(unknown)", None),
    ("Tank", "tank"),
    ("Healer", "healer"),
    ("DPS", "dps"),
]


class CalibrationDialog(QDialog):
    """Edit pass over a fresh `Calibration` before it's persisted.

    Constructed with the source frame so we can render per-slot thumbnails.
    Call `result_calibration()` after `exec()` returns Accepted to get the
    edited Calibration; ignore it on Rejected and discard the calibration.
    """

    def __init__(
        self,
        cal: Calibration,
        source_frame: np.ndarray,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit calibration")
        self.resize(560, 600)

        self._cal = cal
        self._source = source_frame
        # (member_index, QLineEdit, QComboBox) — we keep indices rather than
        # copying PartyMember objects so bbox + identity stay tied to the
        # original detection. The combo holds the role selector.
        self._member_rows: list[tuple[int, QLineEdit, QComboBox]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        form = QFormLayout()
        self._dungeon_edit = QLineEdit(cal.dungeon_name or "")
        self._dungeon_edit.setPlaceholderText("e.g. Mists of Tirna Scithe")
        form.addRow("Dungeon:", self._dungeon_edit)
        root.addLayout(form)

        root.addWidget(self._make_section_label(
            f"Party members ({len(cal.party_members)} detected)"
        ))
        root.addWidget(self._build_member_list())

        root.addWidget(self._make_section_label(
            f"Cooldown icons: {len(cal.cooldown_icons)} detected "
            "(edit on a future iteration)"
        ))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def result_calibration(self) -> Calibration:
        """Build a new Calibration from the edited field values. Call only
        after the dialog was accepted."""
        edited_members: list[PartyMember] = []
        for idx, name_edit, role_combo in self._member_rows:
            original = self._cal.party_members[idx]
            name = name_edit.text().strip() or None
            role = role_combo.currentData()  # None for "(unknown)", else the lowercase token
            edited_members.append(
                PartyMember(name=name, role=role, bbox=original.bbox)
            )

        dungeon = self._dungeon_edit.text().strip() or None

        # deepcopy to preserve cooldown_icons / notes / calibrated_at without
        # mutating the original (the caller may want to compare before/after).
        return Calibration(
            party_members=edited_members,
            cooldown_icons=deepcopy(self._cal.cooldown_icons),
            dungeon_name=dungeon,
            notes=self._cal.notes,
            calibrated_at=self._cal.calibrated_at,
        )

    # ---- internals ----

    @staticmethod
    def _make_section_label(text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _build_member_list(self) -> QWidget:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        if not self._cal.party_members:
            empty = QLabel("(no party members detected)")
            empty.setStyleSheet("color: gray;")
            col.addWidget(empty)
        else:
            for idx, member in enumerate(self._cal.party_members):
                col.addWidget(self._build_member_row(idx, member))
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        return scroll

    def _build_member_row(self, idx: int, member: PartyMember) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # Slot thumbnail — gives the user visual context for what they're
        # editing, which matters most when the LLM read got truncated.
        thumb = QLabel()
        thumb.setFixedHeight(_THUMBNAIL_HEIGHT)
        pixmap = self._crop_pixmap(member.bbox)
        if pixmap is not None:
            thumb.setPixmap(pixmap)
        else:
            thumb.setText("(no preview)")
            thumb.setStyleSheet("color: gray;")
        layout.addWidget(thumb)

        name_edit = QLineEdit(member.name or "")
        name_edit.setPlaceholderText("Name (LLM couldn't read)" if not member.name else "")
        layout.addWidget(name_edit, stretch=1)

        # Role selector. Pre-selected to whatever Pass 2 returned (or "unknown"
        # if it couldn't tell). One click for the user to correct.
        role_combo = QComboBox()
        for label, value in _ROLE_OPTIONS:
            role_combo.addItem(label, userData=value)
        # Pick the index matching the LLM-returned role; default to 0 ("unknown").
        for i, (_, value) in enumerate(_ROLE_OPTIONS):
            if value == member.role:
                role_combo.setCurrentIndex(i)
                break
        role_combo.setFixedWidth(100)
        layout.addWidget(role_combo)

        self._member_rows.append((idx, name_edit, role_combo))
        return row

    def _crop_pixmap(self, bbox: tuple[int, int, int, int]) -> QPixmap | None:
        h, w = self._source.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bgr = self._source[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        # `.copy()` is essential — QImage doesn't take ownership of the
        # numpy buffer, and the slice would otherwise be reclaimed by the GC
        # while Qt was still rendering from it.
        qimg = QImage(rgb.data, cw, ch, cw * 3, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        # Scale to the row height, preserving aspect.
        return pixmap.scaledToHeight(_THUMBNAIL_HEIGHT, Qt.TransformationMode.SmoothTransformation)
