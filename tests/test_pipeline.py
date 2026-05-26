"""Unit tests for PipelineWorker.

Scope: the per-stage contract expressed by the worker's signals and downstream
calls, exercised by constructing the worker with mocked dependencies and
invoking individual methods directly. No Qt event loop is spun up.

Contract under test:
  - A new track triggers OCR exactly once.
  - The deduper decides what happens next via the disposition enum:
    * MATCHED_NEW → rule engine runs, alerts emit, narrative logs fire.
    * MATCHED_DUPLICATE → rule engine skipped, only a "skipping" debug line.
    * UNMATCHED_NEW → rule engine skipped (no spell match possible), but
      a registered LOG line fires.
    * UNMATCHED_DUPLICATE → rule engine skipped, "skipping" debug line.

End-to-end behavior (real capture + real YOLO + real OCR) is verified
manually by running the application, not from these tests.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from wow_alert.dedupe import DedupeOutcome, Disposition
from wow_alert.events import Alert, Detection, Severity, Spell
from wow_alert.pipeline import PipelineDeps, PipelineWorker
from wow_alert.tracker import CastBarTracker


def make_frame(w: int = 200, h: int = 60) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_deps(outcome: DedupeOutcome | None = None):
    capture = MagicMock()
    detector = MagicMock()
    tracker = CastBarTracker()
    ocr = MagicMock()
    # (text, conf, x_left, x_right) within a 100px-wide crop (bbox 10..110).
    ocr.read.return_value = [
        ("Polymorph", 0.95, 2.0, 30.0),
        ("John", 0.9, 60.0, 80.0),
        ("3.0", 0.92, 85.0, 99.0),
    ]
    deduper = MagicMock()
    if outcome is None:
        outcome = DedupeOutcome(
            disposition=Disposition.MATCHED_NEW,
            canonical_spell=Spell(id="poly", name="Polymorph",
                                  severity=Severity.DANGER, duration=3.0),
            canonical_target=None,
            ttl_s=3.0,
        )
    deduper.process.return_value = outcome
    rule_engine = MagicMock()
    rule_engine.evaluate.return_value = [
        Alert(severity=Severity.DANGER, phrase="DANGER",
              message="Polymorph on John (3.0s)"),
    ]
    alert_player = MagicMock()
    return PipelineDeps(
        capture=capture, detector=detector, tracker=tracker,
        ocr=ocr, deduper=deduper, rule_engine=rule_engine,
        alert_player=alert_player,
    ), detector, ocr, deduper, rule_engine, alert_player


def collect_messages(worker: PipelineWorker) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    worker.worker_message.connect(lambda level, text: out.append((level, text)))
    return out


def detection_at_origin() -> Detection:
    return Detection(class_name="cast_bar", confidence=0.9, bbox=(10, 10, 110, 40))


def test_matched_new_runs_full_downstream():
    deps, _detector, ocr, deduper, rule_engine, alert_player = make_deps()
    worker = PipelineWorker(deps)
    messages = collect_messages(worker)
    alerts: list[Alert] = []
    worker.alert.connect(alerts.append)

    frame = make_frame()
    update = deps.tracker.update([detection_at_origin()])
    worker._process_new_track(frame, update.new[0])

    ocr.read.assert_called_once()
    deduper.process.assert_called_once()
    rule_engine.evaluate.assert_called_once()
    alert_player.play.assert_called_once_with("DANGER")
    assert len(alerts) == 1

    levels = [lvl for lvl, _ in messages]
    texts = " | ".join(t for _, t in messages)
    assert "DEBUG" in levels
    assert "LOG" in levels
    assert "matched in DB: Polymorph" in texts
    assert "registered: Polymorph" in texts


def test_matched_duplicate_skips_rule_engine():
    outcome = DedupeOutcome(
        disposition=Disposition.MATCHED_DUPLICATE,
        canonical_spell=Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER, duration=3.0),
        canonical_target=None,
        ttl_s=0.0,
    )
    deps, _detector, _ocr, _deduper, rule_engine, alert_player = make_deps(outcome=outcome)
    worker = PipelineWorker(deps)
    messages = collect_messages(worker)

    frame = make_frame()
    update = deps.tracker.update([detection_at_origin()])
    worker._process_new_track(frame, update.new[0])

    rule_engine.evaluate.assert_not_called()
    alert_player.play.assert_not_called()
    texts = " | ".join(t for _, t in messages)
    assert "skipping" in texts
    assert "Polymorph" in texts
    # No LOG-level message — the user wanted skips kept to DEBUG.
    assert "LOG" not in [lvl for lvl, _ in messages]


def test_unmatched_new_registers_without_rule_engine():
    outcome = DedupeOutcome(
        disposition=Disposition.UNMATCHED_NEW,
        canonical_spell=None,
        canonical_target=None,
        ttl_s=3.0,
    )
    deps, _detector, _ocr, _deduper, rule_engine, alert_player = make_deps(outcome=outcome)
    worker = PipelineWorker(deps)
    messages = collect_messages(worker)

    frame = make_frame()
    update = deps.tracker.update([detection_at_origin()])
    worker._process_new_track(frame, update.new[0])

    rule_engine.evaluate.assert_not_called()
    alert_player.play.assert_not_called()
    levels = [lvl for lvl, _ in messages]
    texts = " | ".join(t for _, t in messages)
    assert "LOG" in levels   # the registered: line still fires
    assert "registered" in texts
    assert "no DB match" in texts


def test_unmatched_duplicate_skips_silently_at_log_level():
    outcome = DedupeOutcome(
        disposition=Disposition.UNMATCHED_DUPLICATE,
        canonical_spell=None,
        canonical_target=None,
        ttl_s=0.0,
    )
    deps, _detector, _ocr, _deduper, rule_engine, alert_player = make_deps(outcome=outcome)
    worker = PipelineWorker(deps)
    messages = collect_messages(worker)

    frame = make_frame()
    update = deps.tracker.update([detection_at_origin()])
    worker._process_new_track(frame, update.new[0])

    rule_engine.evaluate.assert_not_called()
    alert_player.play.assert_not_called()
    levels = [lvl for lvl, _ in messages]
    assert "LOG" not in levels  # skipped: only DEBUG noise


def test_continuing_track_does_not_call_ocr():
    deps, _detector, ocr, _deduper, rule_engine, alert_player = make_deps()
    worker = PipelineWorker(deps)
    continuing_ids: list[int] = []
    worker.continuing_track.connect(continuing_ids.append)

    detection = detection_at_origin()
    deps.tracker.update([detection])
    update2 = deps.tracker.update([detection])
    assert len(update2.continuing) == 1
    assert update2.new == []

    for track in update2.continuing:
        worker.continuing_track.emit(track.track_id)

    ocr.read.assert_not_called()
    rule_engine.evaluate.assert_not_called()
    alert_player.play.assert_not_called()
    assert continuing_ids == [1]
