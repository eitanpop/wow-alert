"""Main application window: frame view + controls + log pane.

Owns the pipeline worker and its QThread, and wires the worker's signals to the
appropriate widgets. Controls (confidence slider, pause toggle, alerts toggle)
are kept inline in this module rather than split into their own widget — at
this scale a separate file would obscure rather than clarify the wiring.
"""
from __future__ import annotations

import logging

import numpy as np
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from wow_alert.audio import PyttsxWinsoundAlertPlayer
from wow_alert.events import Alert
from wow_alert.pipeline import PipelineWorker
from wow_alert.ui.frame_widget import FrameWidget
from wow_alert.ui.log_widget import LogWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        worker: PipelineWorker,
        alert_player: PyttsxWinsoundAlertPlayer,
        show_preview: bool = True,
    ):
        super().__init__()
        self.setWindowTitle("wow-alert — cast bar awareness")
        self.resize(1280, 900)

        self._worker = worker
        self._alert_player = alert_player

        self.frame_widget = FrameWidget()
        self.log_widget = LogWidget()
        self._build_controls(show_preview=show_preview)

        # Hide the preview pane when disabled at startup. The worker also
        # skips emitting frame_ready when preview is off, so toggling this
        # off saves both the UI render cost and a cross-thread frame copy.
        self.frame_widget.setVisible(show_preview)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addWidget(self.frame_widget, stretch=4)
        layout.addLayout(self._controls_layout, stretch=0)
        layout.addWidget(self.log_widget, stretch=2)
        self.setCentralWidget(central)

        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.stopped.connect(self._thread.quit)

        self._worker.frame_ready.connect(self.frame_widget.update_frame)
        self._worker.alert.connect(self._on_alert)
        self._worker.error.connect(self._on_error)
        self._worker.worker_message.connect(self._on_worker_message)

    def _build_controls(self, show_preview: bool) -> None:
        self._controls_layout = QHBoxLayout()
        self._controls_layout.setSpacing(12)

        conf_label = QLabel("Confidence:")
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(5, 95)
        self._conf_slider.setValue(40)
        self._conf_slider.setFixedWidth(180)
        self._conf_value = QLabel("0.40")
        self._conf_slider.valueChanged.connect(self._on_conf_slider)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggle)

        self._alerts_cb = QCheckBox("Alerts on")
        self._alerts_cb.setChecked(True)
        self._alerts_cb.toggled.connect(self._on_alerts_toggle)

        self._preview_cb = QCheckBox("Preview")
        self._preview_cb.setChecked(show_preview)
        self._preview_cb.toggled.connect(self._on_preview_toggle)

        self._debug_cb = QCheckBox("Debug")
        self._debug_cb.setChecked(True)  # default on per current iteration
        self._debug_cb.toggled.connect(self._on_debug_toggle)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)

        self._controls_layout.addWidget(conf_label)
        self._controls_layout.addWidget(self._conf_slider)
        self._controls_layout.addWidget(self._conf_value)
        self._controls_layout.addSpacing(20)
        self._controls_layout.addWidget(self._pause_btn)
        self._controls_layout.addWidget(self._alerts_cb)
        self._controls_layout.addWidget(self._preview_cb)
        self._controls_layout.addWidget(self._debug_cb)
        self._controls_layout.addStretch(1)
        self._controls_layout.addWidget(self._clear_btn)

    def start(self) -> None:
        self._thread.start()
        self.log_widget.info("worker started")

    def closeEvent(self, event) -> None:
        self._worker.stop()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)

    @Slot(int)
    def _on_conf_slider(self, value: int) -> None:
        conf = value / 100.0
        self._conf_value.setText(f"{conf:.2f}")
        self._worker.set_confidence(conf)

    @Slot(bool)
    def _on_pause_toggle(self, checked: bool) -> None:
        self._worker.set_paused(checked)
        self._pause_btn.setText("Resume" if checked else "Pause")
        self.log_widget.info("paused" if checked else "resumed")

    @Slot(bool)
    def _on_alerts_toggle(self, checked: bool) -> None:
        self._alert_player.set_muted(not checked)
        self.log_widget.info(f"alerts {'on' if checked else 'off'}")

    @Slot(bool)
    def _on_preview_toggle(self, checked: bool) -> None:
        self.frame_widget.setVisible(checked)
        self._worker.set_preview_enabled(checked)
        self.log_widget.info(f"preview {'on' if checked else 'off'}")

    @Slot(bool)
    def _on_debug_toggle(self, checked: bool) -> None:
        self.log_widget.set_show_debug(checked)
        self.log_widget.info(f"debug {'on' if checked else 'off'}")

    @Slot()
    def _on_clear_clicked(self) -> None:
        self.log_widget.clear()

    @Slot(object)
    def _on_alert(self, alert: Alert) -> None:
        # alert.severity is a (str, Enum); its default __str__ is "Severity.DANGER".
        # Use .value for the bare "danger" we want to show.
        self.log_widget.info(f"ALERT [{alert.severity.value}]: {alert.message}")

    @Slot(str, str)
    def _on_error(self, stage: str, message: str) -> None:
        self.log_widget.error(f"[{stage}]: {message}")

    @Slot(str, str)
    def _on_worker_message(self, level: str, message: str) -> None:
        # Narrative stream from the pipeline (matched/unmatched/skipping/etc).
        # The worker chose the level; we just route by it.
        self.log_widget.log(message, level=level)
