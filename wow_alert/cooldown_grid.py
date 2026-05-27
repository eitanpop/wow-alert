"""Find individual cooldown icon bboxes within a confirmed cooldown-region crop.

The LLM is reliable at locating the rough cooldown-manager region but
imprecise at per-icon bbox detection — its bboxes routinely cut across
the seams between adjacent icons. This module replaces per-icon LLM
detection with OpenCV contour detection inside the user-confirmed
region.

Algorithm
---------
1. Convert the crop to HSV; keep only the saturation channel. WoW spell
   icon art is high-saturation; the game world / cooldown-manager
   background between icons is low-saturation.
2. Threshold the saturation channel to a binary mask.
3. Morphological close to fill small gaps within icons (e.g., dark
   regions inside a mostly-bright icon).
4. `cv2.findContours` on the resulting mask.
5. Filter each contour's bounding rect by:
     - aspect ratio close to square (icons in WoW are square; allow some
       slack for partial threshold misses)
     - dimension bounds — too-small contours are spell-name text or
       digit overlays; too-large are non-icon UI panels.
6. Sort the surviving bboxes into grid order: clustered into rows by
   y-center, then left-to-right within each row.

Returns coordinates in CROP-LOCAL space; callers add the crop origin
to translate back to source-frame coordinates.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# HSV saturation cutoff. Pixels at or above this saturation level are
# considered "icon art". 30 catches all but the most washed-out icon
# pixels; tune up if game-world background bleeds in, down if dim
# icons get cropped.
_SAT_THRESHOLD = 30

# Per-icon dimension bounds (pixels). Cooldown icons in WoW UIs range
# roughly 28-128 px depending on UI scale and the addon's settings. The
# bounds here are generous on both ends; the aspect-ratio filter does
# more of the work.
_MIN_ICON_PX = 22
_MAX_ICON_PX = 220

# Aspect ratio bounds. Icons are square; some slack accounts for
# threshold edges nibbling slightly off one side.
_MIN_ASPECT = 0.7
_MAX_ASPECT = 1.4

# Morphological close kernel size (px). Fills small gaps inside an
# icon (e.g., dark stripes from a frame overlay) so the contour stays
# in one piece.
_CLOSE_KERNEL = 5


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
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, _SAT_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (_CLOSE_KERNEL, _CLOSE_KERNEL),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    bboxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < _MIN_ICON_PX or h < _MIN_ICON_PX:
            continue
        if w > _MAX_ICON_PX or h > _MAX_ICON_PX:
            continue
        aspect = w / h
        if aspect < _MIN_ASPECT or aspect > _MAX_ASPECT:
            continue
        bboxes.append((x, y, x + w, y + h))

    bboxes = _sort_grid(bboxes)
    logger.info(
        "cooldown_grid: %d icons detected in %dx%d region",
        len(bboxes), crop_bgr.shape[1], crop_bgr.shape[0],
    )
    return bboxes


def _sort_grid(
    bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Cluster bboxes into rows by y-center, then sort each row by x.

    Two bboxes belong to the same row when their y-centers are within
    half the average icon height. This lets a single row of slightly
    mis-aligned icons cluster correctly.
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

    out: list[tuple[int, int, int, int]] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b[0]))
    return out
