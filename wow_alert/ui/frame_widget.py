"""Live annotated frame view.

Displays BGR numpy frames inside a QLabel. The frame is letterboxed (centered
with black bars) to fit whatever size the parent window currently has, so the
captured-window aspect ratio is preserved at every size the user drags to.
"""
from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from wow_alert.events import Detection


def fit_to_window(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox `frame` into a (target_w, target_h) canvas, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return frame
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            out, label, (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out


class FrameWidget(QLabel):
    """Displays BGR numpy frames. Always letterboxes to its current size so the
    captured-window aspect ratio is preserved when the user resizes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet("background-color: #111; color: #8a8d92; font-size: 16px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._last_frame: np.ndarray | None = None
        self._last_detections: list[Detection] = []

    def set_placeholder(self, text: str) -> None:
        """Guidance shown in the empty preview area before any frame arrives
        (e.g. 'pick a dungeon to start'). Replaced by the live frame once one
        is rendered; shown again only while no frame is present."""
        if self._last_frame is None:
            self.setText(text)

    @Slot(np.ndarray, list)
    def update_frame(self, frame: np.ndarray, detections: list[Detection]) -> None:
        self._last_frame = frame
        self._last_detections = detections
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._last_frame is None:
            return
        annotated = draw_detections(self._last_frame, self._last_detections)
        target_w = max(1, self.width())
        target_h = max(1, self.height())
        fitted = fit_to_window(annotated, target_w, target_h)

        # BGR -> RGB for Qt
        rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        bytes_per_line = w * 3
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(image.copy()))
