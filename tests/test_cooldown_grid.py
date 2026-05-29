"""Unit tests for cooldown-icon contour detection.

Synthesizes images with colored squares on dark backgrounds — the
detector's job is to find the squares and return bboxes in grid order.
Replaces the LLM-based per-icon bbox detection in calibration; making
sure it doesn't regress is important.

Plus one deterministic regression test against a real WoW cooldown-
manager crop (`tests/fixtures/cooldown_manager_real.png`) — locks in
that the detector works on actual native-cooldown-manager pixels and
catches future tuning that breaks real-world detection.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from wow_alert.cooldown_grid import (
    _cluster_rows,
    _reconstruct_grid,
    find_icon_bboxes,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures"


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

    def test_ignores_bright_background_with_dark_borders(self):
        # The dark-border-based detector finds icons by their thin dark
        # borders. Here we paint a bright icon (red) surrounded by a
        # thin dark border on a bright background. The icon should be
        # detected; the bright background outside the dark border is
        # one connected region but its size filters it out (we cap
        # icon size at _MAX_ICON_PX = 200).
        h, w = 200, 300
        img = np.full((h, w, 3), 200, dtype=np.uint8)  # bright background
        # Dark border ring at (38..102, 38..102)
        img[38:102, 38:102] = (0, 0, 0)
        # Bright red icon inside the border at (40..100, 40..100)
        img[40:100, 40:100] = (40, 40, 255)
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

    def _assert_clean_detection(
        self,
        bboxes: list[tuple[int, int, int, int]],
        *,
        expected_min: int,
        expected_max: int,
        aspect_min: float = 0.5,
        aspect_max: float = 1.7,
        max_overlap_frac: float = 0.25,
    ) -> None:
        """Shared structural checks for real-fixture detections.

        Asserts the detection count falls in the expected range, every
        bbox is roughly square, no pair of bboxes overlaps significantly
        (the whole point of the detector is to find SEPARATE icons),
        and the bboxes come out in top-to-bottom reading order.
        """
        assert expected_min <= len(bboxes) <= expected_max, (
            f"expected {expected_min}-{expected_max} icons, got {len(bboxes)}"
        )
        for x1, y1, x2, y2 in bboxes:
            w, h = x2 - x1, y2 - y1
            aspect = w / h
            assert aspect_min <= aspect <= aspect_max, (
                f"non-square bbox {(x1, y1, x2, y2)} aspect={aspect:.2f}"
            )
        for i, a in enumerate(bboxes):
            for j in range(i + 1, len(bboxes)):
                b = bboxes[j]
                ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                overlap_frac = inter / min(area_a, area_b)
                assert overlap_frac < max_overlap_frac, (
                    f"bboxes {a} and {b} overlap by {overlap_frac:.0%}"
                )
        if len(bboxes) >= 2:
            first_y = (bboxes[0][1] + bboxes[0][3]) / 2
            last_y = (bboxes[-1][1] + bboxes[-1][3]) / 2
            assert first_y <= last_y, "bboxes not in top-to-bottom order"

    def test_real_cooldown_manager_monk(self):
        """Regression test on a real native-cooldown-manager crop —
        monk character, mid-tone dungeon background, 14 icons across
        three rows (mix of ~60 px and ~40 px icons)."""
        crop = cv2.imread(str(FIXTURES / "cooldown_manager_real.png"))
        assert crop is not None
        bboxes = find_icon_bboxes(crop)
        self._assert_clean_detection(bboxes, expected_min=12, expected_max=16)

    def test_real_cooldown_manager_paladin(self):
        """Different character (paladin), different bar layout, same
        environment — 14 visible icons. The bar has different icon-art
        characteristics (Lightsmith abilities with mostly-dark
        interiors) which has historically been the failure case for
        contour-based detectors."""
        crop = cv2.imread(str(FIXTURES / "cooldown_manager_paladin.png"))
        assert crop is not None
        bboxes = find_icon_bboxes(crop)
        self._assert_clean_detection(bboxes, expected_min=12, expected_max=16)

    def test_real_screenshot_handles_2x_nearest_upscale(self):
        """A player on a higher native resolution sees larger icons AND
        larger gaps between them (because the whole UI is rendered at
        a higher resolution, not interpolated up). INTER_NEAREST
        simulates that — sharp scale-up with no blur between icons.
        Detector must still find the same icons cleanly."""
        crop = cv2.imread(str(FIXTURES / "cooldown_manager_real.png"))
        h, w = crop.shape[:2]
        upscaled = cv2.resize(
            crop, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST,
        )
        bboxes = find_icon_bboxes(upscaled)
        self._assert_clean_detection(bboxes, expected_min=12, expected_max=18)

    def test_real_screenshot_handles_huge_pad(self):
        """When the user-confirmed cooldown region is loose (lots of
        surrounding game-world padding), the detector must still find
        the icons — game-world pixels shouldn't be mistaken for icons.
        """
        crop = cv2.imread(str(FIXTURES / "cooldown_manager_real.png"))
        # Pad with a copy of an adjacent region of game world. The
        # existing crop already includes some game world at top/right;
        # tile it for additional padding on all sides.
        h, w = crop.shape[:2]
        padded = np.zeros((h + 100, w + 100, 3), dtype=np.uint8)
        # Fill with a tiled sample of the game world (top-left corner of
        # the crop, which the existing fixture confirms is background).
        bg_sample = crop[0:40, 0:40]
        bg_h, bg_w = bg_sample.shape[:2]
        for y in range(0, padded.shape[0], bg_h):
            for x in range(0, padded.shape[1], bg_w):
                yend = min(y + bg_h, padded.shape[0])
                xend = min(x + bg_w, padded.shape[1])
                padded[y:yend, x:xend] = bg_sample[: yend - y, : xend - x]
        # Drop the actual cooldown crop in the center.
        padded[50:50 + h, 50:50 + w] = crop
        bboxes = find_icon_bboxes(padded)
        # Same 14 icons, just offset by (50, 50) in the padded image.
        self._assert_clean_detection(bboxes, expected_min=12, expected_max=18)

    def test_tiny_scale_fails_gracefully(self):
        """At small UI scales the icons drop below _MIN_ICON_PX (20)
        and aren't detected. This must be a clean 'no detections'
        return, not a crash — the calibration flow surfaces the empty
        result and the user re-confirms a tighter region."""
        crop = cv2.imread(str(FIXTURES / "cooldown_manager_real.png"))
        h, w = crop.shape[:2]
        tiny = cv2.resize(
            crop, (w // 3, h // 3), interpolation=cv2.INTER_AREA,
        )
        bboxes = find_icon_bboxes(tiny)
        # Don't assert exact 0 — extremely small icons might still be
        # ambiguously detected. Just verify we don't crash and don't
        # over-detect from a low-resolution input.
        assert len(bboxes) <= 5

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


class TestGridReconstruction:
    """Rebuilding a row from clean detections when a dark background
    fragments or drops icons."""

    # 15 bboxes from a real failed Nexus calibration: top row has a 41px-wide
    # fragment at ~2552 and a missing slot at ~2612.
    _NEXUS_RAW = [
        (2348, 870, 2404, 926), (2414, 870, 2470, 926), (2480, 870, 2536, 926),
        (2552, 870, 2593, 926), (2678, 870, 2734, 926),
        (2348, 936, 2404, 992), (2414, 936, 2470, 992),
        (2429, 1007, 2459, 1037), (2465, 1007, 2495, 1037),
        (2501, 1007, 2531, 1037), (2537, 1007, 2567, 1037),
        (2573, 1007, 2603, 1037), (2609, 1007, 2639, 1037),
        (2645, 1007, 2675, 1037),
    ]

    def test_recovers_fragment_and_missing_icon(self):
        top = _cluster_rows(_reconstruct_grid(self._NEXUS_RAW))[0]
        assert len(top) == 6  # 5 (one a fragment) + 1 missing → 6 clean cells
        widths = [b[2] - b[0] for b in top]
        assert all(w == widths[0] for w in widths)  # uniform width
        centers = [(b[0] + b[2]) / 2 for b in top]
        pitches = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        assert all(abs(p - pitches[0]) <= 1 for p in pitches)  # even spacing

    def test_clean_grid_unchanged(self):
        # A perfectly-detected 1x4 grid must pass through with the same cells.
        clean = [(x, 0, x + 50, 50) for x in (0, 60, 120, 180)]
        out = _reconstruct_grid(clean)
        assert [(b[0] + b[2]) / 2 for b in out] == [25, 85, 145, 205]

    def test_sparse_row_not_overbuilt(self):
        # Only 2 icons with a big gap → too few to trust a grid; keep as-is
        # rather than inventing phantom cells between them.
        out = _reconstruct_grid([(0, 0, 50, 50), (300, 0, 350, 50)])
        assert len(out) == 2
