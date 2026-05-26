"""Cast-bar tracker. Matches new detections to existing tracks via IoU.

Each track is uniquely identified by `track_id`. The tracker classifies each
frame's detections into:

  - new tracks  -> downstream pipeline runs OCR + parse + rules + alert once
  - continuing  -> downstream is skipped (already alerted); a single log line emits

Tracks that haven't been seen for `max_missed_frames` ticks are dropped. That
handles brief one-frame dropout (YOLO occasionally missing a detection) without
spawning a new track id.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wow_alert.events import BBox, Detection


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: BBox
    detection: Detection
    missed: int = 0


@dataclass
class TrackUpdate:
    new: list[Track] = field(default_factory=list)
    continuing: list[Track] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)


class CastBarTracker:
    def __init__(self, iou_threshold: float = 0.5, max_missed_frames: int = 2):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> TrackUpdate:
        update = TrackUpdate()
        matched_existing_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        for tid, track in self._tracks.items():
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_detection_indices:
                    continue
                score = iou(track.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                track.bbox = det.bbox
                track.detection = det
                track.missed = 0
                matched_existing_ids.add(tid)
                matched_detection_indices.add(best_idx)
                update.continuing.append(track)

        for tid in list(self._tracks.keys()):
            if tid in matched_existing_ids:
                continue
            self._tracks[tid].missed += 1
            if self._tracks[tid].missed > self.max_missed_frames:
                update.dropped_ids.append(tid)
                del self._tracks[tid]

        for i, det in enumerate(detections):
            if i in matched_detection_indices:
                continue
            track = Track(track_id=self._next_id, bbox=det.bbox, detection=det)
            self._next_id += 1
            self._tracks[track.track_id] = track
            update.new.append(track)

        return update

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
