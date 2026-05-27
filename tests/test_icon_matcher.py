"""Unit tests for the icon matcher.

Synthesizes deterministic "icon" images (distinct color patterns) so we
can characterize matcher behavior without depending on the real icon
database.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wow_alert.icon_matcher import IconMatcher


def _make_icon(seed: int, size: int = 64) -> np.ndarray:
    """Produce a deterministic, visually-distinct 'icon' indexed by seed."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    # A few colored bars across the center so adjacent seeds aren't
    # accidentally too close in random pixels.
    for stripe in range(4):
        y = (size // 5) * (stripe + 1)
        color = ((seed * 53 + stripe * 71) % 255,
                 (seed * 89 + stripe * 23) % 255,
                 (seed * 197 + stripe * 13) % 255)
        cv2.line(img, (4, y), (size - 4, y), color, 3)
    return img


def _write_icons(dirpath: Path, spell_ids: list[int]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for sid in spell_ids:
        cv2.imwrite(str(dirpath / f"{sid}.png"), _make_icon(sid))


class TestIconMatcher:
    def test_matches_identical_crop(self, tmp_path: Path):
        _write_icons(tmp_path, [1022, 6940, 465])
        matcher = IconMatcher(tmp_path)
        crop = _make_icon(6940)  # identical to the stored reference
        closest, score, passed = matcher.match(crop)
        assert closest == 6940
        assert passed is True
        assert score > 0.95  # near-perfect

    def test_below_threshold_still_returns_closest(self, tmp_path: Path):
        _write_icons(tmp_path, [1022])
        matcher = IconMatcher(tmp_path, threshold=0.9)
        # Random crop unrelated to any reference.
        rng = np.random.default_rng(99999)
        crop = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        closest, score, passed = matcher.match(crop)
        # Even though it didn't pass, we still report the closest reference
        # — diagnostic data the calibration log surfaces.
        assert closest == 1022
        assert score < 0.9
        assert passed is False

    def test_robust_to_scale(self, tmp_path: Path):
        _write_icons(tmp_path, [465])
        matcher = IconMatcher(tmp_path)
        # Live crop is 96×96 instead of the 64×64 reference.
        live = cv2.resize(_make_icon(465), (96, 96), interpolation=cv2.INTER_CUBIC)
        closest, score, passed = matcher.match(live)
        assert closest == 465
        # Cross-scale match loses some correlation vs. the identical case
        # (~0.75 here); above the 0.7 default threshold is what matters.
        assert score > matcher.threshold
        assert passed is True

    def test_robust_to_smaller_scale(self, tmp_path: Path):
        _write_icons(tmp_path, [853])
        matcher = IconMatcher(tmp_path)
        # Live crop is 40×40 — smaller than the reference.
        live = cv2.resize(_make_icon(853), (40, 40), interpolation=cv2.INTER_AREA)
        closest, score, passed = matcher.match(live)
        assert closest == 853
        assert score > matcher.threshold
        assert passed is True

    def test_picks_best_among_many(self, tmp_path: Path):
        spell_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        _write_icons(tmp_path, spell_ids)
        matcher = IconMatcher(tmp_path)
        # Match against seed 5 specifically.
        crop = _make_icon(5)
        closest, _, _ = matcher.match(crop)
        assert closest == 5

    def test_loads_all_pngs(self, tmp_path: Path):
        _write_icons(tmp_path, [1022, 6940, 465, 853])
        matcher = IconMatcher(tmp_path)
        assert len(matcher) == 4
        assert set(matcher.spell_ids()) == {1022, 6940, 465, 853}

    def test_missing_dir_is_safe(self, tmp_path: Path):
        # Nonexistent dir: matcher logs warning, loads zero, returns None.
        matcher = IconMatcher(tmp_path / "does_not_exist")
        assert len(matcher) == 0
        closest, score, passed = matcher.match(_make_icon(1))
        assert closest is None
        assert score == 0.0
        assert passed is False

    def test_ignores_non_numeric_filenames(self, tmp_path: Path):
        _write_icons(tmp_path, [1022])
        # Drop a non-numeric file alongside.
        cv2.imwrite(str(tmp_path / "BoP.png"), _make_icon(999))
        matcher = IconMatcher(tmp_path)
        assert matcher.spell_ids() == [1022]

    def test_empty_crop_returns_none(self, tmp_path: Path):
        _write_icons(tmp_path, [1022])
        matcher = IconMatcher(tmp_path)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        closest, score, passed = matcher.match(empty)
        assert closest is None
        assert score == 0.0
        assert passed is False

    def test_distinguishes_close_seeds(self, tmp_path: Path):
        # Seeds 10 and 11 are different inputs; matcher must not collapse them.
        _write_icons(tmp_path, [10, 11])
        matcher = IconMatcher(tmp_path)
        assert matcher.match(_make_icon(10))[0] == 10
        assert matcher.match(_make_icon(11))[0] == 11

    def test_non_square_live_crop_is_centered_squared(self, tmp_path: Path):
        # WoW cell bboxes commonly come back as e.g. 49x55 — taller than
        # wide because the bbox includes UI chrome below the icon.
        # Build a synthetic version: take a real icon, pad it with junk
        # pixels (8 rows top, 8 rows bottom — symmetric, so center-square
        # lands cleanly on the icon).
        _write_icons(tmp_path, [1022])
        matcher = IconMatcher(tmp_path)
        icon = _make_icon(1022)  # 64x64
        rng = np.random.default_rng(31337)
        pad_top = rng.integers(0, 255, size=(8, 64, 3), dtype=np.uint8)
        pad_bottom = rng.integers(0, 255, size=(8, 64, 3), dtype=np.uint8)
        tall = np.vstack([pad_top, icon, pad_bottom])  # 80x64
        closest, score, passed = matcher.match(tall)
        assert closest == 1022
        assert passed is True
        assert score > 0.95  # squaring strips noise → near-identical match

    def test_off_center_chrome_still_finds_closest(self, tmp_path: Path):
        # Asymmetric chrome (more on the bottom than top — realistic for
        # keybind labels) shifts the centered-square slightly off the
        # icon. The match still identifies the right reference but the
        # score may dip — important behavior to pin so future "tighten
        # threshold" changes don't silently break this path.
        _write_icons(tmp_path, [1022])
        matcher = IconMatcher(tmp_path)
        icon = _make_icon(1022)
        rng = np.random.default_rng(424242)
        pad_top = rng.integers(0, 255, size=(2, 64, 3), dtype=np.uint8)
        pad_bottom = rng.integers(0, 255, size=(14, 64, 3), dtype=np.uint8)
        tall = np.vstack([pad_top, icon, pad_bottom])  # 80x64, off-centered
        closest, _, _ = matcher.match(tall)
        assert closest == 1022
