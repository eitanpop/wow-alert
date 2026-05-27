"""Polls the calibrated cooldown-icon regions and reports availability.

A `QTimer` on the UI thread fires every `_POLL_MS` milliseconds. Each
tick grabs the worker's most recent frame, samples each calibrated icon
bbox, and pushes the resulting `dict[str, bool]` into the worker via
`PipelineWorker.set_cooldowns`. The worker reads it when building a
RuleDecisionContext; rule priorities with class-action filters skip
actions whose entry is True.

Why UI-thread and not its own QThread: the watcher samples at ~2 FPS
and each tick does a small HSV mean. Spinning up a dedicated QThread
buys nothing (the work is well under one frame budget on a UI thread
already loaded with paint events) and complicates cross-thread state
ownership.

Detection heuristic
-------------------
Cooldown icons in WoW (default UI, WeakAura) darken and desaturate when
used — the recharge sweep + grayscale overlay reduces both brightness
and saturation. Conversely, an available icon is full-color.

We work in HSV and take the *mean saturation* of the icon's bbox. An
average at or above `_SAT_THRESHOLD` reads as available (False — not on
cooldown); below it reads as on cooldown (True). Saturation alone is
more discriminating than brightness because some icon art is naturally
dark when off cooldown — but no icon is naturally gray.

Binary state is what the rule engine actually needs (the only check is
`is this action on cooldown right now?`). A true seconds-remaining
reading would require OCR of the small overlay text and is left for
later if it turns out to be useful.
"""
from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QTimer

from wow_alert.calibration import CooldownIcon
from wow_alert.pipeline import PipelineWorker

logger = logging.getLogger(__name__)


class CooldownWatcher(QObject):
    """UI-thread sampler: reads frames, writes availability."""

    # Cooldown state changes at the pace of the player's GCD (~1s), so we
    # don't need fast sampling. 2 FPS is more than enough and trivially
    # cheap compared with the YOLO pipeline.
    _POLL_MS = 500

    # Mean HSV-saturation cutoff. Tuned by eye on the default WoW UI.
    # Raise to be stricter ("only fully bright icons read as available");
    # lower to be more permissive.
    _SAT_THRESHOLD = 60.0

    # Smallest evaluable crop — below this we just bail and assume
    # available, since the LLM probably mislocated the icon bbox.
    _MIN_REGION = 4

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

    def set_icons(self, icons: Iterable[CooldownIcon]) -> None:
        """Swap the polled icons. Safe to call any time; the next tick
        will use the new set. Called from the UI thread after a fresh
        calibration is accepted."""
        self._icons = list(icons)
        logger.info("Cooldown watcher polling %d icons", len(self._icons))

    def start(self) -> None:
        logger.info("CooldownWatcher started")
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        logger.info("CooldownWatcher stopped")

    # ---- internals ----

    def _tick(self) -> None:
        if not self._icons:
            return
        frame = self._worker.latest_frame()
        if frame is None:
            return
        cooldowns: dict[str, bool] = {}
        for icon in self._icons:
            cooldowns[icon.action] = self._on_cooldown(frame, icon.bbox)
        # Pushed through a Qt slot so the worker (which lives on its own
        # thread) sees the dict assigned atomically from its own event
        # loop. Auto-detected connection type queues the call across
        # threads when needed.
        self._worker.set_cooldowns(cooldowns)

    @classmethod
    def _on_cooldown(
        cls, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> bool:
        """True when the icon's bbox reads as on cooldown.

        Clamps the bbox into the frame and returns False (available) when
        the crop is too small to evaluate — the alternative (returning True
        for a degenerate bbox) would silently suppress alerts.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 - x1 < cls._MIN_REGION or y2 - y1 < cls._MIN_REGION:
            return False
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        avg_sat = float(hsv[:, :, 1].mean())
        return avg_sat < cls._SAT_THRESHOLD
