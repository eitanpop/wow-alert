"""Template-match a cropped cooldown-icon bbox against the local icon DB.

The icon DB is a flat directory of `<spell_id>.png` files (the user-data
icons dir) — one PNG per WoW spell ID that any loaded class library
references. Calibration
hands each detected icon's bbox to `IconMatcher.match(crop)`; the
matcher returns the best-scoring spell_id (or None if every reference
scores below threshold).

Why this lives next to calibration rather than inside it: same matcher
is reused by the spec-inference step (count matches per class library
to pick the active spec) and could be reused later for live-stream
icon ID if we ever need it.

Robustness choices:

- **Interior crop.** ElvUI / custom WeakAuras draw a colored border
  around each icon. The matcher crops the inner 70% of both reference
  and live image before comparing, so border chrome doesn't drag the
  score down.

- **Multi-scale.** The reference PNGs are typically 64×64; the live
  bbox might be anywhere from 32 to 128 pixels depending on UI scale.
  For each candidate spell_id we resize the reference to the live crop's
  size and score; that's the published score. Reverse-direction
  resizing (resize live → reference) is also tried; whichever is
  higher wins. Avoids losing matches just because the live icon is
  smaller than 64×64.

- **Grayscale TM_CCOEFF_NORMED.** Both sides are desaturated to
  luminance before correlating, so the score keys on icon *structure*
  rather than color. TM_CCOEFF_NORMED already tolerates uniform
  brightness shifts; doing it on grayscale additionally makes the match
  invariant to the *saturation* shifts a dark/colored dungeon background
  imparts — the same reference matches the same icon whether it's
  captured in a bright town or a dark void zone, so no per-dungeon or
  per-client reference tuning is needed. Trade-off: two icons with
  identical luminance structure but different colors become harder to
  tell apart — acceptable because the matcher is already restricted to a
  single spec's icon set, which is structurally distinct. The match
  score is in [-1, 1]; ~0.7+ is a confident hit, ~0.4-0.7 uncertain.

- **Threshold default 0.7.** Tuned by eye against synthetic + real
  crops in the unit tests. Tightening it (0.8+) trades recall for
  precision — fewer false positives, but small UI-scale variations or
  partial occlusions start missing. Loosening (0.5) catches more at
  the cost of confusing visually-similar icons.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Fraction of the bbox kept after center-cropping. 0.7 = drop the outer
# 15% on each side. Tuned to skip ElvUI / WeakAura border chrome that
# differs between the reference PNG and the live screen capture.
_INTERIOR_FRACTION = 0.7


class IconMatcher:
    def __init__(
        self,
        icon_dir: Path,
        threshold: float = 0.7,
        allowed_spell_ids: set[int] | None = None,
    ):
        """Match crops against the local icon DB.

        `allowed_spell_ids` restricts the loaded reference set — used
        for the class-restricted second pass so e.g. mistweaver
        calibrations don't match against paladin-labeled icons.
        Default (None) loads every PNG in `icon_dir`.
        """
        self.icon_dir = Path(icon_dir)
        self.threshold = threshold
        self._allowed = (
            set(allowed_spell_ids) if allowed_spell_ids is not None else None
        )
        # spell_id -> reference BGR ndarray, already center-cropped.
        self._references: dict[int, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.icon_dir.exists():
            logger.warning(
                "Icon directory %s does not exist; matcher will reject every "
                "lookup. Run `python -m wow_alert.tools.fetch_icons` to "
                "populate it.",
                self.icon_dir,
            )
            return
        for path in sorted(self.icon_dir.glob("*.png")):
            try:
                spell_id = int(path.stem)
            except ValueError:
                logger.debug("Skipping non-numeric icon filename: %s", path)
                continue
            if self._allowed is not None and spell_id not in self._allowed:
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Could not decode icon at %s; skipping", path)
                continue
            self._references[spell_id] = _interior_crop(img)
        scope = "all" if self._allowed is None else f"restricted to {len(self._allowed)} spells"
        logger.info(
            "IconMatcher loaded %d reference icons from %s (%s)",
            len(self._references), self.icon_dir, scope,
        )

    def __len__(self) -> int:
        return len(self._references)

    def spell_ids(self) -> list[int]:
        return list(self._references.keys())

    def match(self, crop: np.ndarray) -> tuple[int | None, float, bool]:
        """Return `(closest_spell_id, score, passed_threshold)` for `crop`.

        `closest_spell_id` is the best-scoring reference regardless of
        threshold (or None when no references are loaded). `passed`
        tells you whether the score cleared `self.threshold`. Callers
        that want a thresholded result use `closest if passed else None`.

        This shape gives the diagnostic path "closest miss was X at
        0.55" without losing the closest-id information when the score
        falls below threshold.

        Live crops from non-square LLM bboxes (common — WoW cells often
        include a keybind label or cooldown timer below the actual icon
        art) are center-cropped to the smaller dimension first, so the
        comparison runs on icon-art-only pixels.
        """
        if crop is None or crop.size == 0 or not self._references:
            return None, 0.0, False
        live = _interior_crop(_center_square(crop))
        if live.size == 0:
            return None, 0.0, False

        best_id: int | None = None
        best_score = -1.0
        for spell_id, reference in self._references.items():
            score = _score(live, reference)
            if score > best_score:
                best_score = score
                best_id = spell_id

        passed = best_id is not None and best_score >= self.threshold
        return best_id, best_score, passed


def _center_square(img: np.ndarray) -> np.ndarray:
    """Center-crop `img` to the largest square it contains.

    LLM-reported icon bboxes routinely include UI chrome above or below
    the actual icon — a keybind label, a cooldown-timer text, a stack-
    count overlay. Cropping to the centered square of the smaller
    dimension drops that chrome before the interior-crop step looks at
    the icon art.
    """
    h, w = img.shape[:2]
    if h == w or h < 2 or w < 2:
        return img
    side = min(h, w)
    y_off = (h - side) // 2
    x_off = (w - side) // 2
    return img[y_off:y_off + side, x_off:x_off + side]


def _interior_crop(img: np.ndarray) -> np.ndarray:
    """Center-crop to `_INTERIOR_FRACTION` of each dimension.

    Both live and reference images go through this so the comparison
    happens on the icon art, not on the surrounding UI chrome.
    """
    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return img
    keep_h = max(1, int(h * _INTERIOR_FRACTION))
    keep_w = max(1, int(w * _INTERIOR_FRACTION))
    y_off = (h - keep_h) // 2
    x_off = (w - keep_w) // 2
    return img[y_off:y_off + keep_h, x_off:x_off + keep_w]


def _score(live: np.ndarray, reference: np.ndarray) -> float:
    """Two-direction normalized-correlation score.

    Resizes the reference to the live crop's dimensions and scores;
    also resizes the live to the reference dimensions and scores. Takes
    the higher of the two. Robust to either direction of UI scale.
    """
    lh, lw = live.shape[:2]
    rh, rw = reference.shape[:2]
    if lh < 4 or lw < 4 or rh < 4 or rw < 4:
        return -1.0
    s1 = _correlate(live, _resize(reference, lw, lh))
    s2 = _correlate(_resize(live, rw, rh), reference)
    return max(s1, s2)


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    # Down-scaling uses INTER_AREA (best for shrinking), up-scaling uses
    # INTER_CUBIC (smoother on the small icon art than INTER_LINEAR).
    interp = cv2.INTER_AREA if (w < img.shape[1] or h < img.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def _correlate(a: np.ndarray, b: np.ndarray) -> float:
    """Single-call normalized cross-correlation, returns a scalar in
    [-1, 1]. Both arrays must already be the same shape.

    Both are desaturated to grayscale first so the correlation depends on
    luminance structure, not color — that's what makes a match survive the
    saturation/lighting shift a dark dungeon background imparts."""
    if a.shape != b.shape:
        return -1.0
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) if b.ndim == 3 else b
    result = cv2.matchTemplate(a_gray, b_gray, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])
