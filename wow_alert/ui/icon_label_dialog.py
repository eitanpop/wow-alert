"""Post-calibration icon labeling dialog.

After the matcher has done its best guess against the stock icon DB,
this dialog lets the user confirm or correct each calibrated icon's
identity. Each (icon → spell_id) pairing the user accepts has its
live-rendered crop saved to `config/icons/<spell_id>.png`, overwriting
the stock reference for that spell.

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
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
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


_THUMBNAIL_PX = 96  # display size for each icon thumbnail


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

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        explainer = QLabel(
            "Label each cooldown icon so its current rendering becomes "
            "the matcher's reference. Pre-selected to the matcher's "
            "best guess; leave on '(skip)' for icons you don't want "
            "tracked. Accepted icons overwrite "
            "config/icons/<spell_id>.png with your client's exact pixels."
        )
        explainer.setWordWrap(True)
        root.addWidget(explainer)

        root.addWidget(self._build_icon_list(), stretch=1)

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
        for idx, combo in self._icon_rows:
            icon = self._cal.cooldown_icons[idx]
            chosen = combo.currentData()  # int spell_id, or None
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
                logger.info(
                    "Wrote labeled reference for icon #%d -> %s", idx, target,
                )
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
                col.addWidget(self._build_row(idx, icon))
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
        ref_pixmap = self._reference_pixmap(closest_id) if closest_id else None
        if ref_pixmap is not None:
            ref_thumb.setPixmap(ref_pixmap)
        else:
            ref_thumb.setText("(no ref)")
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
        combo.addItem("(skip)", userData=None)
        for action in self._class_actions:
            combo.addItem(
                f"{action.label}   [{action.id}]", userData=action.spell_id,
            )
        # Pre-select the matcher's closest guess if it produced any —
        # even when score < threshold. A wrong pre-selection is one click
        # to fix; an absent one means re-identifying from scratch.
        target_id = closest_id if closest_id is not None else icon.spell_id
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
