"""Post-calibration icon labeling dialog.

After the matcher has done its best guess against the stock icon DB,
this dialog lets the user confirm or correct each calibrated icon's
identity. Each (icon → spell_id) pairing the user accepts has its
live-rendered crop saved into the icons dir as `<spell_id>.png`,
overwriting the stock reference for that spell.

Why this exists: Wowhead's stock icon JPGs differ from your client's
rendering in subtle ways — JPG compression artifacts, the native
cooldown manager's frame/charge-counter overlays, color-curve
treatments. Even visually-identical icons score ~0.6–0.7 across that
gap. Once the reference is your own crop from this exact client, live
and reference match at near-pixel level → 0.95+ scores.

Each row shows: bbox thumbnail · index + bbox coords · spell dropdown
pre-selected to the matcher's current best guess (or "skip"). Pick
through, accept; PNGs get written.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from wow_alert.calibration import Calibration, CooldownIcon
from wow_alert.class_library import ClassAction

logger = logging.getLogger(__name__)


class _SwallowWheelFilter(QObject):
    """Event filter that drops wheel events on whatever widget it's
    installed on. Default Qt behavior is that scrolling the mouse
    wheel over a QComboBox cycles its selection — a real UX trap
    when the user is scrolling the dialog body and their cursor
    happens to be over a dropdown."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if event.type() == QEvent.Type.Wheel:
            return True
        return False


_THUMBNAIL_PX = 96  # display size for each icon thumbnail

# Below this score, the matcher's "closest" is basically noise.
# Pre-selecting it would put the wrong spell in the dropdown and make
# the user click to fix it rather than the simpler "click to set" from
# a (skip) default. The threshold sits below the matcher's pass cutoff
# (0.70) so confident-but-not-quite matches still pre-select.
_PRESELECT_MIN_SCORE = 0.5

# Below this score, the closest reference is so unrelated to the live
# icon that showing it as "the system's guess" is misleading. We hide
# the reference thumbnail instead — empty placeholder with a "no match"
# note. The dropdown still works; user picks manually or skips.
_SHOW_REFERENCE_MIN_SCORE = 0.3


class IconLabelDialog(QDialog):
    """Confirm-or-correct each calibrated icon, then write per-icon
    PNGs to the icon DB as the new reference.

    When `diagnostics` is supplied (parallel to cal.cooldown_icons), each
    row also shows the closest reference + its score and pre-selects the
    closest match — useful when the matcher's best guesses are all below
    threshold so without this hint every dropdown would default to
    (skip) and the user would have to identify 15 icons from scratch.
    """

    def __init__(
        self,
        cal: Calibration,
        source_frame: np.ndarray,
        class_actions: list[ClassAction],
        icon_dir: Path,
        diagnostics: list[dict] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Label cooldown icons")
        self.resize(780, 780)

        self._cal = cal
        self._source = source_frame
        self._class_actions = class_actions
        self._icon_dir = Path(icon_dir)
        self._diagnostics = diagnostics or []
        # (icon_index, combo). Read combo.currentData() at accept time.
        self._icon_rows: list[tuple[int, QComboBox]] = []
        # Per-row widget so we can hide/show via the unrecognized toggle.
        self._row_widgets: list[tuple[QWidget, bool]] = []  # (widget, is_low_score)
        # One filter instance shared across all dropdowns to drop wheel
        # events. Without it, scrolling the dialog with the cursor over
        # a QComboBox silently cycles its selection.
        self._wheel_filter = _SwallowWheelFilter(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        explainer = QLabel(
            "<b>Help the system learn what your icons look like.</b><br>"
            "Each row shows two thumbnails: <b>your icon</b> from this "
            "calibration on the left, and the system's <b>best guess</b> "
            "from its database on the right.<br><br>"
            "<b>For each row:</b><br>"
            "• If the two icons look like the <b>same spell</b>: pick "
            "that spell from the dropdown (the system may have already "
            "picked it for you).<br>"
            "• If they look <b>different</b>: either pick the correct "
            "spell from the dropdown yourself, or leave <b>(skip)</b> "
            "for icons you don't want tracked (rotation abilities like "
            "Vivify or Renewing Mist — the rule engine only reacts to "
            "defensives, dispels, CC, etc., so rotation icons are fine "
            "to skip).<br><br>"
            "Hitting OK saves <b>your</b> icons as the system's new "
            "references, so future calibrations recognize them at a "
            "glance."
        )
        explainer.setTextFormat(Qt.TextFormat.RichText)
        explainer.setWordWrap(True)
        root.addWidget(explainer)

        # Column header strip.
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 0)
        header.setSpacing(10)
        for text, width in [
            ("Your icon", _THUMBNAIL_PX),
            ("", 20),  # spacer for the → arrow column
            ("Closest match", _THUMBNAIL_PX),
            ("Score", 90),
            ("Label as…", 0),
        ]:
            label = QLabel(text)
            label.setStyleSheet("font-weight: bold; color: #444;")
            if width > 0:
                label.setFixedWidth(width)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header.addWidget(label, stretch=(1 if width == 0 else 0))
        root.addLayout(header)

        root.addWidget(self._build_icon_list(), stretch=1)

        # Toggle to hide visually-dimmed low-score rows entirely. Off
        # by default — rows are dimmed in-place rather than hidden, so
        # nothing is invisible. On = collapse the dimmed rows out of
        # sight for a tighter focused view.
        self._unrecognized_label = QLabel()
        self._unrecognized_toggle = QCheckBox("Hide unrecognized icons")
        self._unrecognized_toggle.toggled.connect(self._apply_unrecognized_filter)
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self._unrecognized_label)
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(self._unrecognized_toggle)
        root.addLayout(bottom_bar)

        self._unrecognized_toggle.setChecked(False)
        self._apply_unrecognized_filter()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ---- public API ----

    def apply_labels(self) -> tuple[Calibration, int]:
        """Call after exec() returns Accepted.

        Writes per-icon PNGs to the icon dir for every row the user
        labeled (non-skip), and returns `(updated_calibration,
        n_written)`. The updated calibration carries the user-confirmed
        spell_ids on cooldown_icons so the watcher tracks them
        correctly.
        """
        self._icon_dir.mkdir(parents=True, exist_ok=True)
        updated: list[CooldownIcon] = []
        written = 0
        # Map spell_id back to action.id for human-readable logging.
        id_to_label = {
            a.spell_id: f"{a.label} [{a.id}]" for a in self._class_actions
        }
        for idx, combo in self._icon_rows:
            icon = self._cal.cooldown_icons[idx]
            chosen = combo.currentData()  # int spell_id, or None
            chosen_label = (
                id_to_label.get(chosen, f"spell_id={chosen}")
                if chosen is not None else "(skip)"
            )
            logger.info(
                "Label dialog: icon #%d at %s -> %s",
                idx, icon.bbox, chosen_label,
            )
            if chosen is None:
                updated.append(CooldownIcon(bbox=icon.bbox, spell_id=None))
                continue
            crop = self._crop_bbox(icon.bbox)
            if crop is None:
                logger.warning(
                    "Skipping reference write for icon #%d (degenerate bbox %s)",
                    idx, icon.bbox,
                )
                updated.append(CooldownIcon(bbox=icon.bbox, spell_id=chosen))
                continue
            target = self._icon_dir / f"{chosen}.png"
            ok = cv2.imwrite(str(target), crop)
            if ok:
                written += 1
            else:
                logger.warning("cv2.imwrite failed for %s", target)
            updated.append(CooldownIcon(bbox=icon.bbox, spell_id=chosen))
        return self._cal.model_copy(update={"cooldown_icons": updated}), written

    # ---- internals ----

    def _build_icon_list(self) -> QWidget:
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)

        if not self._cal.cooldown_icons:
            empty = QLabel("(no cooldown icons detected)")
            empty.setStyleSheet("color: gray;")
            col.addWidget(empty)
        else:
            for idx, icon in enumerate(self._cal.cooldown_icons):
                row_widget = self._build_row(idx, icon)
                diag = (
                    self._diagnostics[idx]
                    if idx < len(self._diagnostics) else None
                )
                score = diag["score"] if diag else 0.0
                is_low = score < _SHOW_REFERENCE_MIN_SCORE
                self._row_widgets.append((row_widget, is_low))
                col.addWidget(row_widget)
        col.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.StyledPanel)
        return scroll

    def _build_row(self, idx: int, icon: CooldownIcon) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        # Dim low-score rows so they're visible-but-de-emphasized. The
        # row stays interactive; the user can still label it if they
        # recognize the icon.
        diag_for_dim = (
            self._diagnostics[idx] if idx < len(self._diagnostics) else None
        )
        dim_score = diag_for_dim["score"] if diag_for_dim else 0.0
        if dim_score < _SHOW_REFERENCE_MIN_SCORE:
            row.setStyleSheet(
                "QWidget { background-color: #f4f4f4; color: #777; }"
            )

        # Thumbnail of the live bbox crop — what the matcher actually sees.
        live_thumb = QLabel()
        live_thumb.setFixedSize(_THUMBNAIL_PX, _THUMBNAIL_PX)
        live_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_pixmap = self._live_pixmap(icon.bbox)
        if live_pixmap is not None:
            live_thumb.setPixmap(live_pixmap)
        else:
            live_thumb.setText("?")
            live_thumb.setStyleSheet("color: gray; border: 1px dashed gray;")
        layout.addWidget(live_thumb)

        # Side-by-side: matcher's closest reference + score, when known.
        diag = self._diagnostics[idx] if idx < len(self._diagnostics) else None
        closest_id: int | None = diag["closest"] if diag else None
        score: float = diag["score"] if diag else 0.0
        passed: bool = diag["passed"] if diag else False

        arrow = QLabel("→")
        arrow.setStyleSheet("color: gray; font-size: 18pt;")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arrow.setFixedWidth(20)
        layout.addWidget(arrow)

        ref_thumb = QLabel()
        ref_thumb.setFixedSize(_THUMBNAIL_PX, _THUMBNAIL_PX)
        ref_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Hide the reference when the matcher's score is too low — the
        # closest is essentially noise and showing it would mislead
        # ("the system thinks this is X" when in fact the score says it
        # really doesn't know).
        ref_pixmap = (
            self._reference_pixmap(closest_id)
            if closest_id and score >= _SHOW_REFERENCE_MIN_SCORE
            else None
        )
        if ref_pixmap is not None:
            ref_thumb.setPixmap(ref_pixmap)
        else:
            ref_thumb.setText("(no match)")
            ref_thumb.setStyleSheet("color: gray; border: 1px dashed gray;")
        layout.addWidget(ref_thumb)

        # Metadata column: index, bbox, score.
        score_color = "green" if passed else ("orange" if score > 0.45 else "gray")
        meta = QLabel(
            f"#{idx}\n"
            f"<span style='color:gray'>{icon.bbox[0]},{icon.bbox[1]}</span><br>"
            f"<span style='color:{score_color}'>score {score:.2f}</span>"
        )
        meta.setTextFormat(Qt.TextFormat.RichText)
        meta.setStyleSheet("font-size: 9pt;")
        meta.setMinimumWidth(90)
        layout.addWidget(meta)

        combo = QComboBox()
        combo.installEventFilter(self._wheel_filter)
        # Stop Qt's default wheel-cycles-selection behavior on the
        # QAbstractItemView inside the combo too (the popup list).
        combo.view().installEventFilter(self._wheel_filter)
        combo.addItem("(skip)", userData=None)
        for action in self._class_actions:
            combo.addItem(
                f"{action.label}   [{action.id}]", userData=action.spell_id,
            )
        # Pre-select the matcher's closest guess only when we're at
        # least somewhat confident. Below `_PRESELECT_MIN_SCORE` the
        # closest is essentially noise — putting a wrong spell in the
        # dropdown is worse UX than the (skip) default. The
        # already-passed case (icon.spell_id set by the matcher when
        # score ≥ matcher.threshold) always pre-selects.
        target_id: int | None = None
        if icon.spell_id is not None:
            target_id = icon.spell_id
        elif closest_id is not None and score >= _PRESELECT_MIN_SCORE:
            target_id = closest_id
        if target_id is not None:
            for i in range(combo.count()):
                if combo.itemData(i) == target_id:
                    combo.setCurrentIndex(i)
                    break
        layout.addWidget(combo, stretch=1)

        self._icon_rows.append((idx, combo))
        return row

    def _live_pixmap(self, bbox: tuple[int, int, int, int]) -> QPixmap | None:
        """Thumbnail of the live bbox crop from the source frame."""
        crop = self._crop_bbox(bbox)
        if crop is None:
            return None
        return self._array_to_pixmap(crop)

    def _reference_pixmap(self, spell_id: int) -> QPixmap | None:
        """Thumbnail of the reference icon PNG for `spell_id`, or None
        if the file isn't on disk."""
        path = self._icon_dir / f"{spell_id}.png"
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        return self._array_to_pixmap(img)

    @staticmethod
    def _array_to_pixmap(bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # .copy() so QImage owns its own buffer, not a view into numpy.
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg).scaled(
            _THUMBNAIL_PX, _THUMBNAIL_PX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _apply_unrecognized_filter(self) -> None:
        """When the toggle is ON, hide low-score rows entirely. When
        OFF (default), every row is visible — low-score ones are
        visually dimmed via the per-row styling in `_build_row` so
        they're recognizable but don't compete with confident rows
        for attention. The bottom-bar label reports the hidden count
        when the user opts in to hiding."""
        hide_low = self._unrecognized_toggle.isChecked()
        hidden = 0
        for widget, is_low in self._row_widgets:
            if is_low and hide_low:
                widget.hide()
                hidden += 1
            else:
                widget.show()
        low_count = sum(1 for _, low in self._row_widgets if low)
        if hide_low and hidden:
            self._unrecognized_label.setText(
                f"{hidden} icon(s) hidden."
            )
        elif low_count:
            self._unrecognized_label.setText(
                f"{low_count} icon(s) dimmed — nothing in your class "
                f"library looks like them. Label them anyway if needed."
            )
        else:
            self._unrecognized_label.setText("")
        self._unrecognized_label.setStyleSheet("color: gray;")

    def _crop_bbox(self, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        h, w = self._source.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return self._source[y1:y2, x1:x2].copy()
