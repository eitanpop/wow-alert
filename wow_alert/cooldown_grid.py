"""Find individual cooldown icon bboxes within a confirmed cooldown-region crop.

The LLM is reliable at locating the rough cooldown-manager region but
imprecise at per-icon bbox detection — its bboxes routinely cut across
the seams between adjacent icons. This module replaces per-icon LLM
detection with OpenCV connected-components analysis inside the user-
confirmed region.

Algorithm
---------
1. Convert the crop to HSV; keep the Value (brightness) channel.
2. Threshold to find DARK pixels — these are the icon borders/frames
   (and the narrow gaps between adjacent icons). Background game-world
   pixels are typically too bright to be caught.
3. Dilate the dark mask with a 3×3 kernel so the borders form a
   continuous grid that separates icon interiors.
4. Invert: connected white regions are now individual icons (plus
   whatever bright game-world area sits outside the cooldown
   manager, but the user's confirmed region should crop that out).
5. `cv2.connectedComponentsWithStats` to enumerate each region.
6. Filter each region by:
     - aspect ratio close to square (icons in WoW are square; allow some
       slack for borders nibbling slightly off one side)
     - dimension bounds — too-small are inner-icon sub-components or
       digit overlays; too-large are non-icon UI panels.
     - area minimum — drops noise like single-character text inside an
       icon that gets separated by the border-dilation.
     - containment check — skip any region whose bbox is fully inside
       another region's bbox (handles the inner-element-of-an-icon case).
7. Sort the surviving bboxes into grid order: clustered into rows by
   y-center, then left-to-right within each row.

Returns coordinates in CROP-LOCAL space; callers add the crop origin
to translate back to source-frame coordinates.

Why brightness-based, not saturation-based: WoW's native cooldown
manager overlays icons directly on the game world rather than on a
dark backing panel. The world is itself saturated, so saturation
thresholding can't separate icons from background. Icon BORDERS, on
the other hand, are consistently dark — that's the discriminating
feature.
"""
from __future__ import annotations

import logging
import statistics

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Darkness threshold on the HSV Value channel. Pixels at or below this
# value are treated as "icon border" / "gap between icons". Tuned
# against real WoW screenshots — icon frames are usually <40, while
# even shadowy areas of the game world rarely dip below 60.
_DARK_THRESHOLD = 50

# Dilation kernel for the dark mask. 3x3 + 1 iteration is enough to
# bridge the 1-2 px wide borders into a continuous grid that fully
# separates adjacent icons.
_DILATE_KERNEL = 3

# Per-icon dimension bounds (pixels). Cooldown icons in WoW UIs range
# roughly 20-180 px depending on UI scale and the addon's settings.
_MIN_ICON_PX = 20
_MAX_ICON_PX = 200

# Aspect ratio bounds. Icons are square; some slack accounts for
# threshold edges nibbling slightly off one side.
_MIN_ASPECT = 0.7
_MAX_ASPECT = 1.4

# Minimum component area in pixels. Below this we're picking up digit
# overlays, single-character glyphs, or other sub-icon artifacts.
_MIN_AREA = 300

# Fraction of a bbox's area that must lie inside another (larger) bbox
# for the smaller one to be dropped as a sub-component. 0.7 is loose
# enough to catch sub-elements that drift past their parent's edge by
# a few pixels at higher resolutions; tight enough that genuinely-
# adjacent icons (which only ever touch corner-to-corner) aren't merged.
_OVERLAP_CONTAINED = 0.7

# Grid reconstruction. The cooldown manager is a regular grid, but a dark
# dungeon background (or dark icon art) fragments some icons into slivers or
# drops them. Within a row, the cleanly-detected icons of the dominant size
# define the grid pitch; the row is then rebuilt at that pitch, filling gaps
# and replacing fragments. A row is only reconstructed when at least this many
# clean icons agree on a size (fewer = not enough to trust a grid).
_MIN_CLEAN_FOR_RECON = 3
# A detected icon counts as "clean" (full-size, not a fragment) when its width
# is within this fraction of the row's median width.
_SIZE_TOL = 0.2


def find_icon_bboxes(crop_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect square icon bboxes within `crop_bgr`.

    Returns crop-local coordinates `(x1, y1, x2, y2)` in left-to-right,
    top-to-bottom grid order.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []
    if crop_bgr.shape[0] < _MIN_ICON_PX or crop_bgr.shape[1] < _MIN_ICON_PX:
        return []

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    val = hsv[:, :, 2]

    # Build the dark-border mask, dilate to form a continuous separator
    # grid, then invert so icon interiors become connected components.
    dark = (val < _DARK_THRESHOLD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_DILATE_KERNEL, _DILATE_KERNEL),
    )
    dark = cv2.dilate(dark, kernel, iterations=1)
    icon_mask = cv2.bitwise_not(dark)

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        icon_mask, connectivity=8,
    )
    # stats columns: [x, y, w, h, area]. Skip label 0 (the background of
    # the connectedComponents call, which inverted to be the dark grid).
    candidates: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < _MIN_ICON_PX or h < _MIN_ICON_PX:
            continue
        if w > _MAX_ICON_PX or h > _MAX_ICON_PX:
            continue
        if area < _MIN_AREA:
            continue
        aspect = w / h
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            continue
        candidates.append((x, y, x + w, y + h))

    # Containment filter: drop any candidate whose bbox is fully inside
    # another candidate's bbox. Catches the "sub-element inside an icon
    # became its own connected component" case (e.g., the dark shield
    # symbol inside a Blessing of Protection icon).
    candidates = _drop_contained(candidates)
    bboxes = _reconstruct_grid(candidates)
    logger.info(
        "cooldown_grid: %d icons after grid reconstruction (%d raw) in %dx%d region",
        len(bboxes), len(candidates), crop_bgr.shape[1], crop_bgr.shape[0],
    )
    return bboxes


def _drop_contained(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Remove any bbox whose area is mostly inside a larger bbox.

    `_OVERLAP_CONTAINED` of 0.7 catches the typical case — a sub-element
    of a complex icon (e.g. the dark shield symbol inside a Blessing of
    Protection icon) shows up as a separate connected component that
    sits >=70% inside the parent icon's bbox. At higher resolutions
    this sub-component sometimes extends a few pixels past the parent
    on one side, so the stricter "fully inside" check would leave it
    in. The 0.7 ratio leaves room for that drift without merging
    genuinely-adjacent icons.
    """
    out: list[tuple[int, int, int, int]] = []
    for i, a in enumerate(bboxes):
        a_area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        dropped = False
        for j, b in enumerate(bboxes):
            if i == j:
                continue
            b_area = (b[2] - b[0]) * (b[3] - b[1])
            if b_area <= a_area:
                # Only consider larger bboxes as potential "parents".
                continue
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / a_area >= _OVERLAP_CONTAINED:
                dropped = True
                break
        if not dropped:
            out.append(a)
    return out


def _cluster_rows(
    bboxes: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int, int, int]]]:
    """Group bboxes into rows by y-center.

    Two bboxes share a row when their y-centers are within half the average
    icon height. Each returned row is sorted left-to-right; rows are in
    top-to-bottom order.
    """
    if not bboxes:
        return []
    sorted_by_y = sorted(bboxes, key=lambda b: (b[1] + b[3]) / 2)
    avg_height = sum(b[3] - b[1] for b in sorted_by_y) / len(sorted_by_y)
    row_tolerance = avg_height / 2

    rows: list[list[tuple[int, int, int, int]]] = [[sorted_by_y[0]]]
    for b in sorted_by_y[1:]:
        y_center = (b[1] + b[3]) / 2
        last_row = rows[-1]
        last_row_center = (
            sum((bb[1] + bb[3]) / 2 for bb in last_row) / len(last_row)
        )
        if abs(y_center - last_row_center) <= row_tolerance:
            last_row.append(b)
        else:
            rows.append([b])
    return [sorted(row, key=lambda b: b[0]) for row in rows]


def _reconstruct_grid(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Rebuild each row onto its regular grid, filling fragments and gaps.

    Returns bboxes in grid order (rows top-to-bottom, left-to-right).
    """
    out: list[tuple[int, int, int, int]] = []
    for row in _cluster_rows(bboxes):
        out.extend(_reconstruct_row(row))
    return out


def _reconstruct_row(
    row: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Snap one row of detections onto an even grid.

    The clean (dominant-width) icons set the pitch — the smallest gap between
    adjacent icon centers is the true grid pitch, robust to missing icons in
    between. Cells are then laid from the first to the last clean center. When
    a row has too few clean icons to trust a grid, it's returned unchanged."""
    if len(row) < 2:
        return row
    median_w = statistics.median(b[2] - b[0] for b in row)
    clean = [b for b in row if abs((b[2] - b[0]) - median_w) <= _SIZE_TOL * median_w]
    if len(clean) < _MIN_CLEAN_FOR_RECON:
        return row

    centers = sorted((b[0] + b[2]) / 2 for b in clean)
    gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    gaps = [g for g in gaps if g > 1]
    if not gaps:
        return row
    pitch = min(gaps)  # adjacent icons sit one pitch apart; gaps over missing
    # icons are multiples of it, so the minimum is the true pitch.

    w = statistics.median(b[2] - b[0] for b in clean)
    h = statistics.median(b[3] - b[1] for b in clean)
    yc = statistics.median((b[1] + b[3]) / 2 for b in clean)
    first, last = centers[0], centers[-1]
    n = round((last - first) / pitch) + 1

    cells: list[tuple[int, int, int, int]] = []
    for i in range(n):
        cx = first + i * pitch
        cells.append((
            int(round(cx - w / 2)), int(round(yc - h / 2)),
            int(round(cx + w / 2)), int(round(yc + h / 2)),
        ))
    return cells
