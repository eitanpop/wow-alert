"""Polls the calibrated cooldown-icon regions and reports availability.

A `QTimer` on the watcher's dedicated QThread fires every `_POLL_MS`
milliseconds. Each tick grabs the worker's most recent frame, samples
each calibrated icon bbox, and pushes the resulting `dict[int, bool]`
(keyed by the icon's spell_id, populated at calibration time by the
icon matcher) into the worker via `PipelineWorker.set_cooldowns`. The
worker reads it when building a RuleDecisionContext; rule priorities
with class-action filters skip actions whose entry is True.

Icons whose spell_id is None (unidentified — no high-confidence match
in the local icon DB) are skipped: there's nothing to key the dict
entry against, so the rule engine treats those actions as always
available.

Threading
---------
The watcher runs on its own QThread (created by `MainWindow`) so the
per-tick OCR work doesn't block UI repaints. `start()`, `stop()`, and
`set_icons()` are signal-routed: callers from any thread emit a request
signal that lands as a slot call on the watcher's own thread, so the
QTimer is always started/stopped from the thread it actually fires on.

Detection heuristic
-------------------
Three-band saturation:

  1. `sat < _SAT_ON_CD_CEILING` → grey icon, on cooldown. No OCR.
  2. `_SAT_ON_CD_CEILING <= sat <= _SAT_OCR_CEILING` → ambiguous zone
     where an active-phase timer might be overlaid on a still-saturated
     icon. Run the center-text OCR check; digit-dominant text in the
     center → on cooldown.
  3. `sat > _SAT_OCR_CEILING` → bright icon, definitively available. No
     OCR. This band is where most idle icons sit; skipping OCR here
     keeps CPU bounded even with a dozen calibrated icons.

The center crop deliberately excludes the corners so corner-anchored
stack-count overlays (the "2" / "5" badges) don't confuse the timer
detector.
"""
from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from wow_alert.calibration import CooldownIcon
from wow_alert.pipeline import PipelineWorker

logger = logging.getLogger(__name__)


class CooldownWatcher(QObject):
    """Background-thread sampler: reads frames, writes availability."""

    # Signals that route start/stop/set_icons across threads so the QTimer
    # only gets started/stopped on the watcher's own thread (Qt warns
    # otherwise). Callers emit; the matching slots run with AutoConnection.
    _start_request = Signal()
    _stop_request = Signal()
    _icons_update = Signal(object)

    # Cooldown state changes at the pace of the player's GCD (~1s), so we
    # don't need fast sampling. 2 FPS is more than enough and trivially
    # cheap compared with the YOLO pipeline.
    _POLL_MS = 500

    # Saturation upper bound for the "definitely on cooldown" band — mean
    # saturation below this reads as a fully-greyed icon and skips OCR.
    _SAT_ON_CD_CEILING = 60.0

    # Saturation upper bound for the "ambiguous, might be active-phase"
    # band. Anything above is bright enough that there's no realistic
    # chance a centered timer overlay is darkening the icon — skip OCR.
    # The ambiguous band [_SAT_ON_CD_CEILING, _SAT_OCR_CEILING] is the only
    # one that pays the OCR cost; in practice that's ~1-2 icons per tick.
    _SAT_OCR_CEILING = 95.0

    # Smallest evaluable crop. Anything tinier than this fails fast as
    # "available" rather than silently confirming an unusable spell.
    _MIN_REGION = 4

    # Center-crop fraction for the OCR pass. 0.5 = inner 50 %. WoW puts the
    # cooldown timer in the center of the icon; stack counts (which we want
    # to ignore) sit in corners, so dropping the outer ring filters them.
    _CENTER_CROP_FRACTION = 0.5

    def __init__(
        self,
        worker: PipelineWorker,
        icons: Iterable[CooldownIcon] | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._worker = worker
        self._icons: list[CooldownIcon] = list(icons) if icons else []
        self._timer = QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._tick)
        # AutoConnection routes signals across the watcher's QThread
        # boundary as queued slot calls; same-thread emits stay direct.
        self._start_request.connect(self._on_start)
        self._stop_request.connect(self._on_stop)
        self._icons_update.connect(self._on_icons_update)

    def set_icons(self, icons: Iterable[CooldownIcon]) -> None:
        """Swap the polled icon set. Thread-safe — routed to the watcher's
        own thread before mutating internal state."""
        self._icons_update.emit(list(icons))

    def start(self) -> None:
        """Thread-safe start. Routed to the watcher's own thread so the
        QTimer fires from the thread it lives on."""
        self._start_request.emit()

    def stop(self) -> None:
        """Thread-safe stop. Routed to the watcher's own thread."""
        self._stop_request.emit()

    @Slot()
    def _on_start(self) -> None:
        logger.info("CooldownWatcher started")
        self._timer.start()

    @Slot()
    def _on_stop(self) -> None:
        self._timer.stop()
        logger.info("CooldownWatcher stopped")

    @Slot(object)
    def _on_icons_update(self, icons: list[CooldownIcon]) -> None:
        self._icons = list(icons)
        logger.info("Cooldown watcher polling %d icons", len(self._icons))

    # ---- internals ----

    def _tick(self) -> None:
        if not self._icons:
            return
        frame = self._worker.latest_frame()
        if frame is None:
            return
        ocr = getattr(self._worker, "ocr", None)
        cooldowns: dict[int, bool] = {}
        for icon in self._icons:
            if icon.spell_id is None:
                continue
            cooldowns[icon.spell_id] = self._on_cooldown(frame, icon.bbox, ocr)
        # `set_cooldowns` is a `@Slot` on the worker; the assignment inside
        # is a single dict reference store (atomic under the GIL) so a
        # direct cross-thread call is safe without a queued connection.
        self._worker.set_cooldowns(cooldowns)

    def _on_cooldown(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        ocr,
    ) -> bool:
        """True when the icon's bbox reads as on cooldown.

        Pass 1: HSV saturation. Clearly-grey icons return True immediately
        with no OCR call. Pass 2: for icons that look saturated, OCR the
        center crop to catch the "active phase" case where the icon stays
        colorful but a timer number is overlaid in the middle (Aura
        Mastery while the buff is up, Sac while ticking, etc.).

        Returns False when the crop is too small to evaluate — better to
        miss a recommendation than to silently confirm an unusable spell.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 - x1 < self._MIN_REGION or y2 - y1 < self._MIN_REGION:
            return False
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_sat = float(hsv[:, :, 1].mean())
        if avg_sat < self._SAT_ON_CD_CEILING:
            return True
        if avg_sat > self._SAT_OCR_CEILING:
            return False
        if ocr is not None and self._center_has_timer_number(crop, ocr):
            return True
        return False

    @classmethod
    def _center_has_timer_number(cls, crop: np.ndarray, ocr) -> bool:
        """OCR the icon's center and decide whether the text reads as a
        cooldown timer ("30", "1.5", "1m"). Stack counts in corners are
        excluded by the center crop. Defensive against OCR failure — a
        raised exception just reads as "no timer", so a bad frame doesn't
        crash the watcher.
        """
        h, w = crop.shape[:2]
        keep_h = max(8, int(h * cls._CENTER_CROP_FRACTION))
        keep_w = max(8, int(w * cls._CENTER_CROP_FRACTION))
        y_off = (h - keep_h) // 2
        x_off = (w - keep_w) // 2
        center = crop[y_off:y_off + keep_h, x_off:x_off + keep_w]
        try:
            results = ocr.read(center)
        except Exception:
            logger.debug("OCR raised during cooldown-timer check", exc_info=True)
            return False
        for entry in results:
            # OcrEngine.read returns [(text, conf, x_left, x_right), ...].
            if len(entry) < 2:
                continue
            text = str(entry[0]).strip()
            if not text:
                continue
            # Quick filters: anything dominated by digits (with optional
            # "m" / "." for "1m" / "1.5") reads as a timer.
            digits = sum(c.isdigit() for c in text)
            letters_excluding_m = sum(
                c.isalpha() and c.lower() != "m" for c in text
            )
            if digits >= 1 and digits >= letters_excluding_m:
                return True
        return False
