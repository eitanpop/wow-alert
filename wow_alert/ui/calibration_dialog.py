"""Roster editor.

The fast-changing slice of calibration: who's in the party, their roles, and
which row is "Me". Class/spec + screen regions live in the Region dialog,
not here — those rarely change between runs. Roster updates every dungeon.

The "Load party members" button takes a fresh screenshot, crops the saved
party region from the calibration, and OCRs names into the rows so the user
doesn't have to type. Existing rows are replaced. Manual add/remove and edit
are available too.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from wow_alert.calibration import (
    Calibration,
    PartyMember,
    ocr_party_members,
)

logger = logging.getLogger(__name__)


_THUMBNAIL_HEIGHT = 48

# Role dropdown values. Index 0 is "unknown" — maps to None on save.
_ROLE_OPTIONS = [
    ("(unknown)", None),
    ("Tank", "tank"),
    ("Healer", "healer"),
    ("DPS", "dps"),
]


class RosterDialog(QDialog):
    """Edit the per-run roster (party member names + roles + "Me").

    Constructed with the current Calibration (regions + class/spec stay
    untouched) and a way to grab a fresh frame for the OCR refresh button.
    Call `result_calibration()` after `exec()` returns Accepted to get the
    updated Calibration; on Reject keep the prior one.
    """

    def __init__(
        self,
        cal: Calibration,
        frame_provider,
        ocr,
        parent=None,
    ):
        """`frame_provider` is a no-arg callable returning the latest BGR
        frame (or None when no frame is available — happens before the
        worker captures its first one). `ocr` is the live OcrEngine.
        """
        super().__init__(parent)
        self.setWindowTitle("Roster")
        self.resize(560, 600)

        self._cal = cal
        self._frame_provider = frame_provider
        self._ocr = ocr
        # Each row owns its widgets; we rebuild the list from scratch on every
        # "Load party members" so we don't have to reconcile bbox identities.
        self._rows: list[_RosterRow] = []
        self._me_group = QButtonGroup(self)
        self._me_group.setExclusive(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header: live action button + count label
        header = QHBoxLayout()
        self._count_label = QLabel()
        self._load_btn = QPushButton("Load party members")
        self._load_btn.setToolTip(
            "Snapshot the screen and OCR names from the saved party region."
        )
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._add_btn = QPushButton("Add row")
        self._add_btn.clicked.connect(self._on_add_clicked)
        header.addWidget(self._count_label)
        header.addStretch(1)
        header.addWidget(self._add_btn)
        header.addWidget(self._load_btn)
        root.addLayout(header)

        hint = QLabel("Mark yourself with “Me” so self-defensive callouts work.")
        hint.setStyleSheet("color: gray;")
        root.addWidget(hint)

        # Scrolling list of rows
        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(self._rows_container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        root.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        # Seed from the saved calibration's existing roster (could be empty).
        self._set_rows_from_members(
            cal.party_members, source_frame=None,
        )
        self._refresh_count()

        # First-time open: if no roster yet but a party region is saved,
        # auto-run the OCR pass so the dialog isn't a confusing empty list.
        # The user can still click "Load party members" later to refresh.
        if not cal.party_members and cal.party_region is not None:
            QTimer.singleShot(0, self._on_load_clicked)

    # ---- public API ----

    def result_calibration(self) -> Calibration:
        """Build a Calibration with the edited roster, leaving UI fields
        (regions, class/spec, icons) untouched."""
        members: list[PartyMember] = []
        player_name: str | None = None
        for row in self._rows:
            name = row.name_edit.text().strip() or None
            role = row.role_combo.currentData()
            members.append(PartyMember(name=name, role=role, bbox=row.bbox))
            if row.me_radio.isChecked():
                player_name = name
        return self._cal.model_copy(update={
            "party_members": members,
            "player_name": player_name,
        })

    # ---- callbacks ----

    def _on_load_clicked(self) -> None:
        """OCR the saved party region from a fresh screenshot."""
        if self._cal.party_region is None:
            QMessageBox.warning(
                self, "No party region",
                "No party region is saved. Click Calibrate first to set "
                "where the party frames are on screen.",
            )
            return
        frame = self._frame_provider()
        if frame is None:
            QMessageBox.warning(
                self, "No frame",
                "The worker hasn't captured a frame yet — wait a moment and "
                "try again.",
            )
            return
        try:
            members = ocr_party_members(frame, self._cal.party_region, self._ocr)
        except Exception as exc:
            logger.exception("OCR party read failed")
            QMessageBox.critical(self, "OCR failed", str(exc))
            return
        # Convert dicts to PartyMembers (no role from OCR; default unknown).
        as_members = [
            PartyMember(name=m["name"], role=None, bbox=m["bbox"])
            for m in members
        ]
        self._set_rows_from_members(as_members, source_frame=frame)
        self._refresh_count()
        if not as_members:
            QMessageBox.information(
                self, "No names",
                "OCR didn't read any names. Check that the party region "
                "covers the party frames and try again.",
            )

    def _on_add_clicked(self) -> None:
        """Append a blank row for manual entry (no bbox → no thumbnail)."""
        self._append_row(PartyMember(name=None, role=None, bbox=(0, 0, 0, 0)),
                         source_frame=None)
        self._refresh_count()

    # ---- internals ----

    def _set_rows_from_members(
        self,
        members: list[PartyMember],
        source_frame: np.ndarray | None,
    ) -> None:
        """Wipe the current rows and rebuild from `members`."""
        for row in self._rows:
            row.widget.setParent(None)
            row.widget.deleteLater()
        self._rows.clear()
        for m in members:
            self._append_row(m, source_frame=source_frame)
        # Preserve the saved "Me" selection if it matches one of the rows.
        if self._cal.player_name:
            target = self._cal.player_name.strip().lower()
            for row in self._rows:
                if (row.name_edit.text() or "").strip().lower() == target:
                    row.me_radio.setChecked(True)
                    break

    def _append_row(
        self,
        member: PartyMember,
        source_frame: np.ndarray | None,
    ) -> None:
        row = _RosterRow(member, source_frame, self._me_group, self)
        # Insert before the trailing stretch.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row.widget)
        self._rows.append(row)
        row.remove_clicked.connect(lambda r=row: self._remove_row(r))

    def _remove_row(self, row: "_RosterRow") -> None:
        if row not in self._rows:
            return
        self._rows.remove(row)
        row.widget.setParent(None)
        row.widget.deleteLater()
        self._refresh_count()

    def _refresh_count(self) -> None:
        self._count_label.setText(
            f"{len(self._rows)} member(s) — optional, leave empty to skip"
        )


class _RosterRow:
    """One row in the roster list. Plain Python class wrapping the widgets so
    the dialog can iterate them without a separate model layer."""

    def __init__(
        self,
        member: PartyMember,
        source_frame: np.ndarray | None,
        me_group: QButtonGroup,
        dialog: QDialog,
    ):
        self.bbox = member.bbox
        self.widget = QWidget(dialog)
        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        thumb = QLabel()
        thumb.setFixedHeight(_THUMBNAIL_HEIGHT)
        pixmap = self._crop_pixmap(source_frame, member.bbox)
        if pixmap is not None:
            thumb.setPixmap(pixmap)
        else:
            thumb.setText("")
            thumb.setFixedWidth(80)
        layout.addWidget(thumb)

        self.name_edit = QLineEdit(member.name or "")
        self.name_edit.setPlaceholderText("Name")
        layout.addWidget(self.name_edit, stretch=1)

        self.role_combo = QComboBox()
        for label, value in _ROLE_OPTIONS:
            self.role_combo.addItem(label, userData=value)
        for i, (_, value) in enumerate(_ROLE_OPTIONS):
            if value == member.role:
                self.role_combo.setCurrentIndex(i)
                break
        self.role_combo.setFixedWidth(100)
        layout.addWidget(self.role_combo)

        self.me_radio = QRadioButton("Me")
        me_group.addButton(self.me_radio)
        layout.addWidget(self.me_radio)

        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("Remove this row")
        layout.addWidget(remove_btn)
        # Bubble up so the dialog can re-layout + update the count.
        # Use a small QObject as the signal carrier since the row isn't a QObject.
        self._signal_carrier = _RemoveSignal()
        self.remove_clicked = self._signal_carrier.fired
        remove_btn.clicked.connect(self._signal_carrier.emit_fired)

    @staticmethod
    def _crop_pixmap(
        source_frame: np.ndarray | None,
        bbox: tuple[int, int, int, int],
    ) -> QPixmap | None:
        if source_frame is None:
            return None
        h, w = source_frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad = 12
        x1 = max(0, min(x1 - pad, w))
        y1 = max(0, min(y1 - pad, h))
        x2 = max(0, min(x2 + pad, w))
        y2 = max(0, min(y2 + pad, h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop_bgr = source_frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(
            rgb.data, cw, ch, cw * 3, QImage.Format.Format_RGB888,
        ).copy()
        return QPixmap.fromImage(qimg).scaledToHeight(
            _THUMBNAIL_HEIGHT, Qt.TransformationMode.SmoothTransformation,
        )


class _RemoveSignal(QObject):
    """Carrier so the plain-Python _RosterRow can re-emit clicks upward."""
    fired = Signal()

    def emit_fired(self) -> None:
        self.fired.emit()
