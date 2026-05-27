"""Tests for the icon-labeling dialog's write path.

UI rendering is not tested (Qt event loop required); the test below
exercises apply_labels() directly by constructing the dialog,
mutating the dropdown selections to simulate user choices, and
verifying the on-disk PNGs + the returned Calibration.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wow_alert.calibration import Calibration, CooldownIcon, PartyMember
from wow_alert.class_library import ClassAction


pytest.importorskip(
    "PySide6.QtWidgets",
    reason="PySide6 not available; skipping icon-label-dialog tests",
)


@pytest.fixture
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _make_frame(w: int = 200, h: int = 100) -> np.ndarray:
    """Predictable frame with colored rectangles so we can verify
    bbox-extraction wrote the right pixels."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Two distinct colored blocks the dialog will crop out.
    frame[10:50, 20:60] = (255, 0, 0)    # blue 40x40 at (20,10)
    frame[10:50, 100:140] = (0, 255, 0)  # green 40x40 at (100,10)
    return frame


def _make_calibration(*bboxes) -> Calibration:
    return Calibration(
        party_members=[],
        cooldown_icons=[CooldownIcon(bbox=b) for b in bboxes],
        player_class="paladin",
        player_spec="holy",
    )


def _action(id_: str, label: str, spell_id: int) -> ClassAction:
    return ClassAction(
        id=id_, label=label, category="defensive", scope="single_target",
        spell_id=spell_id,
    )


class TestApplyLabels:
    def test_writes_png_for_labeled_icons(self, qapp, tmp_path: Path):
        from wow_alert.ui.icon_label_dialog import IconLabelDialog

        frame = _make_frame()
        cal = _make_calibration((20, 10, 60, 50), (100, 10, 140, 50))
        actions = [
            _action("bop", "BOP", 1022),
            _action("sac", "Sac", 6940),
        ]

        dialog = IconLabelDialog(cal, frame, actions, tmp_path)
        # Simulate user labeling row 0 as BoP and row 1 as Sac.
        row0_combo = dialog._icon_rows[0][1]
        row1_combo = dialog._icon_rows[1][1]
        for combo, target_spell_id in [(row0_combo, 1022), (row1_combo, 6940)]:
            for i in range(combo.count()):
                if combo.itemData(i) == target_spell_id:
                    combo.setCurrentIndex(i)
                    break

        updated_cal, written = dialog.apply_labels()
        assert written == 2
        assert (tmp_path / "1022.png").exists()
        assert (tmp_path / "6940.png").exists()

        # Verify the written PNG actually has the bbox's pixel content.
        bop_png = cv2.imread(str(tmp_path / "1022.png"))
        assert bop_png.shape == (40, 40, 3)
        # The blue block in the test frame was (255, 0, 0) in BGR.
        assert bop_png[20, 20].tolist() == [255, 0, 0]

        # And the spell_ids on cooldown_icons follow the user's labels.
        assert updated_cal.cooldown_icons[0].spell_id == 1022
        assert updated_cal.cooldown_icons[1].spell_id == 6940

    def test_skip_writes_nothing_clears_spell_id(self, qapp, tmp_path: Path):
        from wow_alert.ui.icon_label_dialog import IconLabelDialog

        frame = _make_frame()
        # Calibration starts with a matcher-set spell_id (the
        # pre-selection) — user "skips" it; we should not write a PNG
        # AND should clear the spell_id on the cooldown_icon.
        cal = Calibration(
            party_members=[],
            cooldown_icons=[CooldownIcon(bbox=(20, 10, 60, 50), spell_id=1022)],
            player_class="paladin", player_spec="holy",
        )
        actions = [_action("bop", "BOP", 1022)]

        dialog = IconLabelDialog(cal, frame, actions, tmp_path)
        row0_combo = dialog._icon_rows[0][1]
        # Force selection to the (skip) option at index 0.
        row0_combo.setCurrentIndex(0)
        assert row0_combo.currentData() is None

        updated_cal, written = dialog.apply_labels()
        assert written == 0
        assert not (tmp_path / "1022.png").exists()
        assert updated_cal.cooldown_icons[0].spell_id is None

    def test_preselects_matcher_guess(self, qapp, tmp_path: Path):
        from wow_alert.ui.icon_label_dialog import IconLabelDialog

        frame = _make_frame()
        cal = Calibration(
            party_members=[],
            # Matcher set spell_id 6940 for this icon — dialog should
            # pre-select Sac in the dropdown.
            cooldown_icons=[CooldownIcon(bbox=(20, 10, 60, 50), spell_id=6940)],
            player_class="paladin", player_spec="holy",
        )
        actions = [
            _action("bop", "BOP", 1022),
            _action("sac", "Sac", 6940),
        ]

        dialog = IconLabelDialog(cal, frame, actions, tmp_path)
        row0_combo = dialog._icon_rows[0][1]
        assert row0_combo.currentData() == 6940

    def test_degenerate_bbox_does_not_write(self, qapp, tmp_path: Path):
        from wow_alert.ui.icon_label_dialog import IconLabelDialog

        frame = _make_frame()
        cal = _make_calibration((10, 10, 11, 11))  # 1x1 — below MIN_REGION
        actions = [_action("bop", "BOP", 1022)]

        dialog = IconLabelDialog(cal, frame, actions, tmp_path)
        row0_combo = dialog._icon_rows[0][1]
        for i in range(row0_combo.count()):
            if row0_combo.itemData(i) == 1022:
                row0_combo.setCurrentIndex(i)
                break

        updated_cal, written = dialog.apply_labels()
        assert written == 0
        assert not (tmp_path / "1022.png").exists()
        # spell_id is still set on the cooldown_icon — the user said it's
        # BoP, we just couldn't extract a usable crop.
        assert updated_cal.cooldown_icons[0].spell_id == 1022
