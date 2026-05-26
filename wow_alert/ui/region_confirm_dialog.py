"""Region confirmation dialog.

Shown between calibration's Pass 1 (locate) and Pass 2/3 (read). Displays
the captured screenshot with two color-coded overlays — green for the
party frame region, orange for the cooldown manager — that the user can
drag to reposition or grab by a corner handle to resize.

Why this exists: the LLM's pass-1 region detection is approximate. On
ultrawide / high-resolution / small-UI-scale setups it tends to be off,
which makes pass-2/3 crops miss the actual UI elements. A few seconds of
human drag-and-drop gives pass 2/3 guaranteed-correct crops to read from.

If the LLM returned None for a region, the editor places a centered
400x300 default box that the user adjusts.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Names used as keys in the editor's region dict.
_PARTY = "party"
_COOLDOWN = "cooldown"

# Per-region color (drawn outline + corner handle fill).
_REGION_COLORS = {
    _PARTY: QColor(80, 200, 100),         # green
    _COOLDOWN: QColor(255, 165, 0),       # orange
}

_HANDLE_PX = 10
_MIN_REGION_PX = 20
_DEFAULT_REGION_W = 400
_DEFAULT_REGION_H = 300


class _RegionEditor(QWidget):
    """Custom widget: shows an image, lets the user drag/resize overlay
    rectangles. Rectangles are stored in source-image coordinates so all
    coordinate-space conversion happens here, not in the dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(900, 480)
        self.setMouseTracking(True)
        self._image_bgr: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        # Display state, recomputed on resize. _scale = displayed_dim / image_dim.
        self._scale = 1.0
        self._offset = (0, 0)
        self._displayed_size = (0, 0)
        # Region state — stored as (x1, y1, x2, y2) in image coords.
        self._regions: dict[str, tuple[int, int, int, int]] = {}
        # Active drag: (region_name, mode in {move,tl,tr,br,bl}, start_widget_pos)
        # plus the region's rect at drag start so deltas are absolute.
        self._drag: tuple[str, str, QPoint] | None = None
        self._drag_origin_rect: tuple[int, int, int, int] | None = None

    # ---- public API ----

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._image_bgr = image_bgr
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._recompute_display()
        self.update()

    def set_region(self, name: str, bbox: tuple[int, int, int, int] | None) -> None:
        if bbox is None:
            bbox = self._default_region()
        # Normalize so x2>x1, y2>y1; clamp to image bounds.
        self._regions[name] = self._clamp(self._normalize(bbox))
        self.update()

    def get_region(self, name: str) -> tuple[int, int, int, int] | None:
        return self._regions.get(name)

    # ---- Qt event overrides ----

    def resizeEvent(self, event) -> None:
        self._recompute_display()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        if self._pixmap is None:
            return
        ox, oy = self._offset
        dw, dh = self._displayed_size
        painter.drawPixmap(
            ox, oy, self._pixmap.scaled(
                dw, dh,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ),
        )
        for name, bbox in self._regions.items():
            color = _REGION_COLORS.get(name, QColor(255, 255, 255))
            display_rect = self._image_rect_to_widget(bbox)
            # Outline
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(display_rect)
            # Corner handles
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            for cx, cy in self._corner_points(display_rect):
                painter.drawRect(
                    cx - _HANDLE_PX // 2, cy - _HANDLE_PX // 2,
                    _HANDLE_PX, _HANDLE_PX,
                )
            # Label inside the rectangle
            painter.setPen(color)
            painter.drawText(
                display_rect.adjusted(6, 4, -6, -4),
                int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft),
                name.upper(),
            )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        hit = self._hit_test(pos)
        if hit is None:
            return
        name, mode = hit
        self._drag = (name, mode, pos)
        self._drag_origin_rect = self._regions[name]

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._drag is None:
            self.setCursor(self._cursor_for(pos))
            return
        name, mode, start = self._drag
        assert self._drag_origin_rect is not None
        ox1, oy1, ox2, oy2 = self._drag_origin_rect
        # Delta in image coordinates.
        if self._scale <= 0:
            return
        dx = int(round((pos.x() - start.x()) / self._scale))
        dy = int(round((pos.y() - start.y()) / self._scale))
        if mode == "move":
            new = (ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy)
        elif mode == "tl":
            new = (ox1 + dx, oy1 + dy, ox2, oy2)
        elif mode == "tr":
            new = (ox1, oy1 + dy, ox2 + dx, oy2)
        elif mode == "br":
            new = (ox1, oy1, ox2 + dx, oy2 + dy)
        elif mode == "bl":
            new = (ox1 + dx, oy1, ox2, oy2 + dy)
        else:
            return
        self._regions[name] = self._clamp(self._normalize(new))
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
        self._drag_origin_rect = None
        self.setCursor(self._cursor_for(event.position().toPoint()))

    # ---- internals ----

    def _default_region(self) -> tuple[int, int, int, int]:
        if self._image_bgr is None:
            return (0, 0, _DEFAULT_REGION_W, _DEFAULT_REGION_H)
        h, w = self._image_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        rw = min(_DEFAULT_REGION_W, w - 20)
        rh = min(_DEFAULT_REGION_H, h - 20)
        return (cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2)

    def _normalize(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def _clamp(self, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if self._image_bgr is None:
            return bbox
        h, w = self._image_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        # Enforce minimum size, then clamp inside the image.
        if x2 - x1 < _MIN_REGION_PX:
            x2 = x1 + _MIN_REGION_PX
        if y2 - y1 < _MIN_REGION_PX:
            y2 = y1 + _MIN_REGION_PX
        # Shift if past the right/bottom edge.
        if x2 > w:
            x2 = w
            x1 = max(0, x2 - _MIN_REGION_PX) if x2 - x1 < _MIN_REGION_PX else max(0, x1)
        if y2 > h:
            y2 = h
            y1 = max(0, y2 - _MIN_REGION_PX) if y2 - y1 < _MIN_REGION_PX else max(0, y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        return (x1, y1, x2, y2)

    def _recompute_display(self) -> None:
        if self._pixmap is None:
            self._scale = 1.0
            self._offset = (0, 0)
            self._displayed_size = (0, 0)
            return
        ww, wh = max(1, self.width()), max(1, self.height())
        sx = ww / self._pixmap.width()
        sy = wh / self._pixmap.height()
        self._scale = min(sx, sy)
        dw = int(self._pixmap.width() * self._scale)
        dh = int(self._pixmap.height() * self._scale)
        self._displayed_size = (dw, dh)
        self._offset = ((ww - dw) // 2, (wh - dh) // 2)

    def _image_rect_to_widget(self, bbox: tuple[int, int, int, int]) -> QRect:
        ox, oy = self._offset
        x1, y1, x2, y2 = bbox
        return QRect(
            ox + int(round(x1 * self._scale)),
            oy + int(round(y1 * self._scale)),
            int(round((x2 - x1) * self._scale)),
            int(round((y2 - y1) * self._scale)),
        )

    @staticmethod
    def _corner_points(rect: QRect) -> list[tuple[int, int]]:
        # Order: TL, TR, BR, BL — matches mode encoding in mousePressEvent.
        return [
            (rect.left(), rect.top()),
            (rect.left() + rect.width(), rect.top()),
            (rect.left() + rect.width(), rect.top() + rect.height()),
            (rect.left(), rect.top() + rect.height()),
        ]

    def _hit_test(self, pos: QPoint) -> tuple[str, str] | None:
        for name, bbox in self._regions.items():
            rect = self._image_rect_to_widget(bbox)
            for idx, (cx, cy) in enumerate(self._corner_points(rect)):
                if abs(pos.x() - cx) <= _HANDLE_PX and abs(pos.y() - cy) <= _HANDLE_PX:
                    return (name, ["tl", "tr", "br", "bl"][idx])
            if rect.contains(pos):
                return (name, "move")
        return None

    def _cursor_for(self, pos: QPoint) -> Qt.CursorShape:
        hit = self._hit_test(pos)
        if hit is None:
            return Qt.CursorShape.ArrowCursor
        _, mode = hit
        if mode in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if mode in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        return Qt.CursorShape.SizeAllCursor


class RegionConfirmDialog(QDialog):
    """Modal dialog wrapping a `_RegionEditor`.

    Returns the user-confirmed regions and dungeon name via the
    `result_regions()` method. The dialog itself doesn't touch
    `Calibration` or invoke the LLM — its only job is collecting
    user-confirmed bounding boxes + a dungeon string.
    """

    def __init__(
        self,
        image_bgr: np.ndarray,
        party_region: tuple[int, int, int, int] | None,
        cooldown_region: tuple[int, int, int, int] | None,
        dungeon_name: str | None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Confirm calibration regions")
        self.resize(1100, 720)

        self._editor = _RegionEditor()
        self._editor.set_image(image_bgr)
        self._editor.set_region(_PARTY, party_region)
        self._editor.set_region(_COOLDOWN, cooldown_region)

        instructions = QLabel(
            "Drag a rectangle to move it. Drag a corner handle to resize.\n"
            "Green = party frames. Orange = cooldown manager. "
            "Make each rectangle a tight fit around the corresponding UI element."
        )
        instructions.setWordWrap(True)

        form = QFormLayout()
        self._dungeon_edit = QLineEdit(dungeon_name or "")
        self._dungeon_edit.setPlaceholderText("e.g. Mists of Tirna Scithe")
        form.addRow("Dungeon:", self._dungeon_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(instructions)
        layout.addWidget(self._editor, stretch=1)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def result_regions(
        self,
    ) -> tuple[
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
        str | None,
    ]:
        return (
            self._editor.get_region(_PARTY),
            self._editor.get_region(_COOLDOWN),
            self._dungeon_edit.text().strip() or None,
        )
