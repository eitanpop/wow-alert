"""Per-frame pipeline worker.

Drives one tick of: capture -> detect -> tracker -> (OCR + parse + rules + alert
on new tracks only) -> signal emission. Lives on a background QThread.

Why "new tracks only" matters: OCR is the slowest stage by an order of magnitude.
A cast bar typically persists across many frames; OCR'ing the same bar every
frame would dominate the per-tick budget for no new information. The tracker
classifies each detection as `new` or `continuing` by IoU against the previous
frame. New tracks pay the OCR cost exactly once; continuing tracks emit a
lightweight signal and skip OCR/rules/alert entirely.

Threading: the worker is event-driven, not a blocking `while` loop. Each tick
schedules the next via `QTimer.singleShot(0, self._tick)`. Between ticks, Qt's
event loop on the worker thread dispatches queued slot calls (set_paused, stop,
set_confidence) from the UI thread. A blocking loop would starve those signals
and make pause/stop unresponsive.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from wow_alert.capture import WindowCapture
from wow_alert.cast_bar import make_cast_event, tokens_from_ocr_output
from wow_alert.dedupe import CastDeduper, Disposition
from wow_alert.events import (
    Alert,
    AlertPlayer,
    Detector,
    OcrEngine,
    Recommendation,
    RuleDecisionContext,
    ScreenContext,
)
from wow_alert.rules import RuleEngine
from wow_alert.tracker import CastBarTracker, Track

logger = logging.getLogger(__name__)


@dataclass
class PipelineDeps:
    capture: WindowCapture
    detector: Detector
    tracker: CastBarTracker
    ocr: OcrEngine
    deduper: CastDeduper
    rule_engine: RuleEngine
    alert_player: AlertPlayer


class PipelineWorker(QObject):
    frame_ready = Signal(np.ndarray, list)   # (frame_bgr, detections)
    cast_event = Signal(object)               # CastEvent — raw, every track
    continuing_track = Signal(int)            # track_id
    alert = Signal(object)                    # Alert — triggers TTS
    error = Signal(str, str)                  # (stage, message)
    worker_message = Signal(str, str)         # (level, text) — narrative log
    stopped = Signal()

    # Idle delay between ticks when paused or the window is missing; keeps the
    # CPU quiet without blocking signal delivery.
    _IDLE_DELAY_MS = 50
    _WINDOW_RETRY_MS = 500

    def __init__(
        self,
        deps: PipelineDeps,
        target_fps: int = 10,
        preview_enabled: bool = True,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._deps = deps
        self._running = False
        self._paused = False
        self._preview_enabled = preview_enabled
        self._context = ScreenContext()
        # Most recently captured frame, updated each tick. Exposed via
        # latest_frame() so the calibration flow can grab a current frame
        # without subscribing to frame_ready (which is gated by preview).
        # Cross-thread read is safe (numpy array assignment is atomic in
        # CPython); consumers may see one tick behind, which is fine for
        # calibration purposes.
        self._latest_frame: np.ndarray | None = None
        # Minimum wall time per tick. Cast bars last 1-10 s, so even 5 FPS
        # catches them; the default 10 FPS leaves headroom for jitter without
        # starving the GPU/game render thread.
        self._min_tick_ms = max(1, int(1000 / max(1, target_fps)))

    @Slot(float)
    def set_confidence(self, value: float) -> None:
        self._deps.detector.set_confidence(value)

    @Slot(bool)
    def set_paused(self, value: bool) -> None:
        self._paused = value

    def latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame, or None before the first
        successful capture. Callable from any thread."""
        return self._latest_frame

    def update_calibration_context(
        self,
        roster: list[str],
        dungeon: str | None,
        roles: dict[str, str],
    ) -> None:
        """Push fresh calibration data into the screen context.

        Called from the main thread when a new calibration is applied
        (startup-load and after the user accepts a fresh calibration).
        The worker thread reads these fields when building a
        RuleDecisionContext for each new cast; Python's GIL is enough
        coherence here since updates are rare and the reads use a single
        attribute access per field.
        """
        self._context.roster = list(roster)
        self._context.dungeon = dungeon
        self._context.roles = dict(roles)

    @Slot(bool)
    def set_preview_enabled(self, value: bool) -> None:
        """Toggle the per-tick `frame_ready` emit.

        When off, the pipeline still captures and detects normally, but skips
        the cross-thread ndarray copy that the live-preview pane requires.
        Worth tens of MB/s of memcpy on borderless 1440p captures.
        """
        self._preview_enabled = value

    @Slot()
    def run(self) -> None:
        """Start the event-driven pipeline. Called once when the thread starts."""
        if self._running:
            return
        self._running = True
        self._deps.capture.refresh_region()
        logger.info("PipelineWorker started")
        QTimer.singleShot(0, self._tick)

    @Slot()
    def stop(self) -> None:
        self._running = False

    @Slot()
    def _tick(self) -> None:
        if not self._running:
            logger.info("PipelineWorker stopping")
            self.stopped.emit()
            return

        if self._paused:
            QTimer.singleShot(self._IDLE_DELAY_MS, self._tick)
            return

        tick_start = time.monotonic()
        delay_ms = 0
        try:
            delay_ms = self._process_one_frame()
        except Exception as exc:  # pragma: no cover — defensive only
            logger.exception("Unexpected error in pipeline tick")
            self.error.emit("pipeline", str(exc))
            delay_ms = self._IDLE_DELAY_MS

        if delay_ms == 0:
            elapsed_ms = int((time.monotonic() - tick_start) * 1000)
            delay_ms = max(0, self._min_tick_ms - elapsed_ms)

        if self._running:
            QTimer.singleShot(delay_ms, self._tick)

    # ---- per-frame stages ----

    def _process_one_frame(self) -> int:
        try:
            frame = self._deps.capture.grab()
        except Exception as exc:
            self.error.emit("capture", str(exc))
            return self._WINDOW_RETRY_MS

        if frame is None:
            self.error.emit("capture", "window not found")
            return self._WINDOW_RETRY_MS

        self._latest_frame = frame

        try:
            detections = self._deps.detector.detect(frame)
        except Exception as exc:
            self.error.emit("detect", str(exc))
            detections = []

        try:
            update = self._deps.tracker.update(detections)
        except Exception as exc:
            self.error.emit("tracker", str(exc))
            if self._preview_enabled:
                self.frame_ready.emit(frame, detections)
            return 0

        if self._preview_enabled:
            self.frame_ready.emit(frame, detections)

        # Note: we intentionally do NOT emit `continuing_track` per tick — at
        # the capped FPS that's still many signals per second per cast bar,
        # and the UI log gets unreadable. The signal is kept on the class so
        # callers that want it can still wire one up.

        for track in update.new:
            self._process_new_track(frame, track)

        return 0

    def _process_new_track(self, frame: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = track.bbox
        h, w = frame.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            return
        crop = frame[y1c:y2c, x1c:x2c]

        try:
            ocr_out = self._deps.ocr.read(crop)
        except Exception as exc:
            self.error.emit("ocr", str(exc))
            return

        crop_width = crop.shape[1]
        tokens = tokens_from_ocr_output(ocr_out, crop_width=crop_width)
        cast = make_cast_event(tokens, track.bbox, track.track_id, crop_width=crop_width)
        # If OCR produced no usable spell text, the bbox is almost certainly
        # a false positive from the detector (a buff icon, an HP plate) or
        # an OCR miss on a real cast bar. Either way the downstream stages
        # have nothing to act on and emitting the event just spams the log.
        if not cast.spell:
            return

        # Raw event goes out for any consumer that wants the unfiltered stream.
        # The narrative log gets its own DEBUG line below for legibility.
        self.cast_event.emit(cast)
        raw_desc = self._describe(cast)
        self.worker_message.emit("DEBUG", f"cast: {raw_desc}")

        outcome = self._deps.deduper.process(cast)

        if outcome.disposition is Disposition.MATCHED_NEW:
            spell = outcome.canonical_spell
            # Swap raw OCR target for the roster-canonical name when the
            # deduper resolved one. Rule engine + alert message + log line
            # all see the canonical target this way.
            if outcome.canonical_target is not None and outcome.canonical_target != cast.target:
                cast = replace(cast, target=outcome.canonical_target)
            self.worker_message.emit("DEBUG", f"matched in DB: {spell.name}")
            self.worker_message.emit(
                "LOG",
                f"registered: {spell.name}{self._target_suffix(cast)} "
                f"→ rule engine (ttl={outcome.ttl_s:.1f}s)",
            )
            # Build the explicit decision context the engine expects. Most
            # fields stay default until Phase F populates cooldowns and
            # Phase E filters available_counters; the engine doesn't read
            # them yet either.
            decision_ctx = RuleDecisionContext(
                spell=spell,
                cast=cast,
                canonical_target=outcome.canonical_target,
                dungeon=self._context.dungeon,
                player_class=self._context.player_class,
                player_spec=self._context.player_spec,
                cooldowns=dict(self._context.cooldowns),
                roster=list(self._context.roster),
                roles=dict(self._context.roles),
            )
            try:
                output = self._deps.rule_engine.decide(decision_ctx)
            except Exception as exc:
                self.error.emit("rules", str(exc))
                return
            if isinstance(output, Alert):
                try:
                    self._deps.alert_player.play(output.phrase)
                except Exception as exc:
                    self.error.emit("audio", str(exc))
                self.alert.emit(output)
            elif isinstance(output, Recommendation):
                # Phase E will wire up Recommendation playback (token-concat
                # WAV: action + target). For now, surface it as a LOG so
                # the operator sees the engine made the recommendation
                # decision even before audio is hooked up.
                self.worker_message.emit(
                    "LOG", f"RECOMMEND: {output.message}"
                )

        elif outcome.disposition is Disposition.MATCHED_DUPLICATE:
            spell = outcome.canonical_spell
            # Use canonical target in the skip log too, so the message
            # matches the registered/alert line for the same cast.
            display_target = outcome.canonical_target or cast.target
            target_suffix = f" on {display_target}" if display_target else ""
            self.worker_message.emit(
                "DEBUG",
                f"skipping {spell.name}{target_suffix}: cached (in DB)",
            )

        elif outcome.disposition is Disposition.UNMATCHED_NEW:
            self.worker_message.emit(
                "DEBUG", f"unmatched: {raw_desc} (not in spells.yaml)"
            )
            self.worker_message.emit(
                "LOG",
                f"registered: {raw_desc} (no DB match, ttl={outcome.ttl_s:.1f}s)",
            )
            # No rule engine call — no spell match means no alert is possible.

        elif outcome.disposition is Disposition.UNMATCHED_DUPLICATE:
            self.worker_message.emit(
                "DEBUG",
                f"skipping {raw_desc}: fuzzy-matched recent (no DB)",
            )

    @staticmethod
    def _target_suffix(cast) -> str:
        return f" on {cast.target}" if cast.target else ""

    @classmethod
    def _describe(cls, cast) -> str:
        duration = f" ({cast.duration:.1f}s)" if cast.duration is not None else ""
        return f"{cast.spell}{cls._target_suffix(cast)}{duration}"
