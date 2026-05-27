"""Unit tests for cooldown-icon contour detection.

Synthesizes images with colored squares on dark backgrounds — the
detector's job is to find the squares and return bboxes in grid order.
Replaces the LLM-based per-icon bbox detection in calibration; making
sure it doesn't regress is important.
"""
from __future__ import annotations

import cv2
import numpy as np

from wow_alert.cooldown_grid import find_icon_bboxes


def _make_icon_grid(
    rows: list[list[tuple[int, int, int]]],
    icon_size: int = 50,
    spacing: int = 10,
    margin: int = 20,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Build a synthetic cooldown manager: a black background with
    colored squares in a grid. Returns (image, ground_truth_bboxes)."""
    n_cols = max(len(row) for row in rows)
    n_rows = len(rows)
    w = margin * 2 + n_cols * icon_size + (n_cols - 1) * spacing
    h = margin * 2 + n_rows * icon_size + (n_rows - 1) * spacing
    img = np.zeros((h, w, 3), dtype=np.uint8)
    expected: list[tuple[int, int, int, int]] = []
    for r, row in enumerate(rows):
        for c, color in enumerate(row):
            x1 = margin + c * (icon_size + spacing)
            y1 = margin + r * (icon_size + spacing)
            x2 = x1 + icon_size
            y2 = y1 + icon_size
            img[y1:y2, x1:x2] = color
            expected.append((x1, y1, x2, y2))
    return img, expected


class TestFindIconBboxes:
    def test_finds_single_row(self):
        # 3 saturated icons in a row, distinct hues.
        img, expected = _make_icon_grid([[(0, 0, 255), (0, 255, 0), (255, 0, 0)]])
        found = find_icon_bboxes(img)
        assert len(found) == 3
        for got, exp in zip(found, expected):
            # Each found bbox should be within a few pixels of the
            # synthetic ground truth (mostly exact, but the contour can
            # nibble off a row of pixels at the edges).
            assert abs(got[0] - exp[0]) <= 2
            assert abs(got[1] - exp[1]) <= 2
            assert abs(got[2] - exp[2]) <= 2
            assert abs(got[3] - exp[3]) <= 2

    def test_finds_two_rows_in_grid_order(self):
        # Two rows × three cols → six bboxes returned in reading order.
        img, expected = _make_icon_grid([
            [(0, 0, 255), (0, 255, 0), (255, 0, 0)],
            [(255, 255, 0), (0, 255, 255), (255, 0, 255)],
        ])
        found = find_icon_bboxes(img)
        assert len(found) == 6
        # Row 0 must come before row 1.
        row0_y = found[0][1]
        row1_y = found[3][1]
        assert row0_y < row1_y
        # Within each row, x-coords increase.
        for row_start in (0, 3):
            xs = [found[row_start + i][0] for i in range(3)]
            assert xs == sorted(xs)

    def test_ignores_low_saturation_noise(self):
        # Background near-gray (low saturation) should not produce
        # contours; only the saturated square is found.
        h, w = 200, 300
        img = np.full((h, w, 3), 50, dtype=np.uint8)  # near-gray
        img[40:100, 40:100] = (0, 0, 255)  # saturated red
        found = find_icon_bboxes(img)
        assert len(found) == 1
        x1, y1, x2, y2 = found[0]
        assert 38 <= x1 <= 42
        assert 38 <= y1 <= 42

    def test_rejects_non_square_contours(self):
        # A very-wide rectangle should be rejected by the aspect filter.
        img = np.zeros((150, 400, 3), dtype=np.uint8)
        img[40:90, 40:380] = (0, 0, 255)  # 340x50, aspect ~6.8
        found = find_icon_bboxes(img)
        assert found == []

    def test_rejects_too_small(self):
        # 10x10 icon falls under MIN_ICON_PX.
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:50, 40:50] = (0, 0, 255)
        found = find_icon_bboxes(img)
        assert found == []

    def test_empty_input(self):
        assert find_icon_bboxes(np.zeros((0, 0, 3), dtype=np.uint8)) == []
        assert find_icon_bboxes(None) == []  # type: ignore[arg-type]

    def test_handles_mixed_row_alignment(self):
        # Two rows where one icon is shifted slightly up — should still
        # cluster into two rows in reading order.
        img = np.zeros((250, 400, 3), dtype=np.uint8)
        # Row 0 at y=30
        for i, x in enumerate([30, 100, 170]):
            color = (0, 0, 255) if i % 2 == 0 else (255, 0, 0)
            img[30:80, x:x + 50] = color
        # Row 1 at y=150 (some y-jitter)
        for i, (x, y) in enumerate([(30, 152), (100, 150), (170, 148)]):
            img[y:y + 50, x:x + 50] = (0, 255, 0)
        found = find_icon_bboxes(img)
        assert len(found) == 6
        # First three are row 0 (smaller y); last three are row 1.
        for b in found[:3]:
            assert b[1] < 100
        for b in found[3:]:
            assert b[1] > 100
