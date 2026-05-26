"""Tests for the IoU-based cast-bar tracker."""
from __future__ import annotations

from wow_alert.events import Detection
from wow_alert.tracker import CastBarTracker, iou


def det(x1, y1, x2, y2, conf=0.9):
    return Detection(class_name="cast_bar", confidence=conf, bbox=(x1, y1, x2, y2))


class TestIou:
    def test_identical(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint(self):
        assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0

    def test_half_overlap(self):
        score = iou((0, 0, 10, 10), (5, 0, 15, 10))
        # intersection = 50, union = 150
        assert abs(score - 50 / 150) < 1e-6


class TestTracker:
    def test_first_frame_all_new(self):
        tracker = CastBarTracker()
        update = tracker.update([det(0, 0, 10, 10), det(100, 0, 110, 10)])
        assert len(update.new) == 2
        assert update.continuing == []
        assert update.dropped_ids == []

    def test_continuing_track(self):
        tracker = CastBarTracker()
        tracker.update([det(0, 0, 10, 10)])
        update = tracker.update([det(1, 0, 11, 10)])  # tiny jitter
        assert update.new == []
        assert len(update.continuing) == 1
        assert update.continuing[0].track_id == 1

    def test_new_track_after_existing(self):
        tracker = CastBarTracker()
        tracker.update([det(0, 0, 10, 10)])
        update = tracker.update([det(0, 0, 10, 10), det(200, 0, 210, 10)])
        assert len(update.new) == 1
        assert update.new[0].track_id == 2
        assert len(update.continuing) == 1

    def test_one_frame_dropout_recovers(self):
        tracker = CastBarTracker(max_missed_frames=2)
        tracker.update([det(0, 0, 10, 10)])         # tick 1: track #1 born
        tracker.update([])                          # tick 2: missed once
        update = tracker.update([det(0, 0, 10, 10)])  # tick 3: same bbox returns
        # Tracker should have continued track #1 (not created #2) since it was still alive.
        assert len(update.continuing) == 1
        assert update.continuing[0].track_id == 1
        assert update.new == []

    def test_track_dropped_after_too_many_misses(self):
        tracker = CastBarTracker(max_missed_frames=1)
        tracker.update([det(0, 0, 10, 10)])
        tracker.update([])
        update = tracker.update([])
        assert 1 in update.dropped_ids
