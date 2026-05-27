"""Generic 'run any callable on a QThread, emit signals on completion'.

Used by the calibration flow to run blocking LLM calls without freezing
the UI. Kept Qt-thread-shaped (lives on a QThread you create) rather
than `QtConcurrent.run` so the callable can be any Python function,
including ones that talk to non-Qt SDKs.
"""
from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot

from wow_alert.calibration import CalibrationError

logger = logging.getLogger(__name__)


class BackgroundRunner(QObject):
    """Move this to a QThread, connect `thread.started → run`, then
    listen on `completed` / `failed` for the result."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[], object], parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn()
        except CalibrationError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover — defensive only
            logger.exception("Background task failed")
            self.failed.emit(f"Unexpected error: {exc}")
            return
        self.completed.emit(result)
