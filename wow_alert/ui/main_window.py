"""Main application window: frame view + controls + log pane.

Owns the pipeline worker and its QThread, and wires the worker's signals to the
appropriate widgets. Controls (confidence slider, pause toggle, alerts toggle)
are kept inline in this module rather than split into their own widget — at
this scale a separate file would obscure rather than clarify the wiring.
"""
from __future__ import annotations

import logging
from datetime import datetime

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from typing import Callable

from wow_alert.audio import EdgeTtsAlertPlayer, PyttsxWinsoundAlertPlayer
from wow_alert.calibration import (
    Calibration,
    CooldownIcon,
    calibrate_read,
    load_calibration,
    save_calibration,
)
from wow_alert.class_library import infer_class_spec, load_class_actions
from wow_alert.config import REPO_ROOT
from wow_alert.cooldown_watcher import CooldownWatcher
from wow_alert.dungeon_loader import list_dungeon_names, slugify
from wow_alert.events import Alert
from wow_alert.icon_matcher import IconMatcher
from wow_alert.paths import CALIBRATION_ARTIFACTS_DIR, CALIBRATION_PATH, ICONS_DIR
from wow_alert.pipeline import PipelineWorker
from wow_alert.ui._background_runner import BackgroundRunner
from wow_alert.ui.calibration_dialog import CalibrationDialog
from wow_alert.ui.frame_widget import FrameWidget
from wow_alert.ui.icon_label_dialog import IconLabelDialog
from wow_alert.ui.log_widget import LogWidget
from wow_alert.ui.region_confirm_dialog import RegionConfirmDialog

logger = logging.getLogger(__name__)

# Curated edge-tts English neural voices for the dropdown. Not exhaustive —
# `edge-tts --list-voices` shows the full set, and a persisted voice outside
# this list is added to the dropdown at runtime so it stays selectable.
_EDGE_VOICES = [
    ("Aria (US, female)", "en-US-AriaNeural"),
    ("Jenny (US, female)", "en-US-JennyNeural"),
    ("Guy (US, male)", "en-US-GuyNeural"),
    ("Christopher (US, male)", "en-US-ChristopherNeural"),
    ("Sonia (UK, female)", "en-GB-SoniaNeural"),
    ("Ryan (UK, male)", "en-GB-RyanNeural"),
    ("Natasha (AU, female)", "en-AU-NatashaNeural"),
    ("William (AU, male)", "en-AU-WilliamNeural"),
]


class MainWindow(QMainWindow):
    def __init__(
        self,
        worker: PipelineWorker,
        alert_player: PyttsxWinsoundAlertPlayer,
        show_preview: bool = True,
        on_calibration_apply: Callable[[Calibration], None] | None = None,
        on_dungeon_select: Callable[[str | None], None] | None = None,
        on_clear_calibration: Callable[[str | None], None] | None = None,
        cooldown_watcher: CooldownWatcher | None = None,
    ):
        super().__init__()
        self.setWindowTitle("wow-alert — cast bar awareness")
        self.resize(1280, 900)

        self._worker = worker
        self._alert_player = alert_player
        self._on_calibration_apply = on_calibration_apply
        # Loads a dungeon's spells + prerenders its phrases — the callouts
        # path, usable without calibrating. Run on a background thread on
        # change since the prerender hits the network (edge-tts).
        self._on_dungeon_select = on_dungeon_select
        # Resets the recommendation layer (class/roster/roles/player) to
        # nothing, keeping the dungeon callouts.
        self._on_clear_calibration = on_clear_calibration
        self._cooldown_watcher = cooldown_watcher

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

        # Cooldown watcher lives on the UI thread (QTimer-driven). It
        # reads the worker's latest_frame and pushes a dict[str, bool]
        # to the worker via set_cooldowns. Lifecycle is start() / stop()
        # alongside the pipeline thread.

        self._worker.frame_ready.connect(self.frame_widget.update_frame)
        self._worker.alert.connect(self._on_alert)
        self._worker.error.connect(self._on_error)
        self._worker.worker_message.connect(self._on_worker_message)

        # Calibration plumbing. The QThread is created lazily for each
        # phase, so app startup pays no cost when calibration is never used.
        self._calibration_thread: QThread | None = None
        self._calibration_runner: BackgroundRunner | None = None
        self._calibration: Calibration | None = None
        # Voice re-render plumbing (Apply button). Separate from calibration
        # so a voice render doesn't block / interfere with calibrate state.
        self._voice_thread: QThread | None = None
        self._voice_runner: BackgroundRunner | None = None
        # Dungeon-load plumbing (top-level picker). Its own thread refs so a
        # dungeon swap + prerender doesn't tangle with calibrate/voice state.
        self._dungeon_thread: QThread | None = None
        self._dungeon_runner: BackgroundRunner | None = None
        # Per-run state for the two-phase flow (frame stays valid across the
        # region-confirm dialog so pass-2/3 uses the same image pass-1 saw).
        self._calibration_frame: np.ndarray | None = None
        # Diagnostic info from the icon matcher, keyed by cooldown_icon
        # index. Populated by _resolve_icons_and_spec, consumed by the
        # icon-labeling dialog so it can show side-by-side references and
        # pre-select the closest match even when below threshold.
        self._last_match_diagnostics: list[dict] = []

        # Status bar shows the current calibration target ("Calibrated for:
        # John, Mary, Tank…") so the user can confirm they're configured for
        # the right party at a glance.
        self._calibration_status = QLabel("Not calibrated")
        self.statusBar().addPermanentWidget(self._calibration_status)

        # Auto-load any prior calibration without re-running the LLM. The
        # user explicitly recalibrates when they want fresh data.
        existing = load_calibration(CALIBRATION_PATH)
        if existing is not None:
            self._apply_calibration(existing, persist=False, log_to_pane=False)
        # Initialize the dungeon picker: reflect the calibrated dungeon if one
        # loaded, else load the last-picked dungeon so callouts work with no
        # calibration at all.
        self._init_dungeon_selection()

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

        # Suggestions toggle, persisted across sessions via QSettings. Off =
        # the engine skips all cooldown recommendations and every cast just
        # plays its alert phrase.
        suggestions_on = QSettings("wow-alert", "wow-alert").value(
            "suggestions_enabled", True, type=bool
        )
        self._suggestions_cb = QCheckBox("Suggestions")
        self._suggestions_cb.setToolTip(
            "Recommend cooldowns (Sac, Aura Mastery, …). Off = just play the "
            "spell's alert phrase, no 'press this' callouts."
        )
        self._suggestions_cb.setChecked(suggestions_on)
        self._suggestions_cb.toggled.connect(self._on_suggestions_toggle)
        # Apply the saved state now: setChecked() emits no signal when the
        # initial value matches the box's default, so push it explicitly or a
        # saved "off" wouldn't take effect (the engine defaults to enabled).
        self._worker.set_suggestions_enabled(suggestions_on)

        self._voice_combo = self._build_voice_combo()
        self._dungeon_combo = self._build_dungeon_combo()

        self._debug_cb = QCheckBox("Debug")
        self._debug_cb.setChecked(True)
        self._debug_cb.toggled.connect(self._on_debug_toggle)

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)

        self._clear_cal_btn = QPushButton("Clear calibration")
        self._clear_cal_btn.setToolTip(
            "Forget the saved calibration and drop cooldown recommendations. "
            "Your dungeon callouts stay. Recalibrate anytime."
        )
        self._clear_cal_btn.clicked.connect(self._on_clear_calibration_clicked)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)

        self._controls_layout.addWidget(QLabel("Dungeon:"))
        self._controls_layout.addWidget(self._dungeon_combo)
        self._controls_layout.addSpacing(20)
        self._controls_layout.addWidget(conf_label)
        self._controls_layout.addWidget(self._conf_slider)
        self._controls_layout.addWidget(self._conf_value)
        self._controls_layout.addSpacing(20)
        self._controls_layout.addWidget(self._pause_btn)
        self._controls_layout.addWidget(self._alerts_cb)
        self._controls_layout.addWidget(self._suggestions_cb)
        self._controls_layout.addWidget(self._preview_cb)
        self._controls_layout.addWidget(self._debug_cb)
        self._apply_voice_btn = QPushButton("Apply")
        self._apply_voice_btn.setToolTip(
            "Render the selected voice and switch to it (takes a few seconds)."
        )
        self._apply_voice_btn.setEnabled(self._voice_combo.isEnabled())
        self._apply_voice_btn.clicked.connect(self._on_apply_voice)
        self._controls_layout.addWidget(QLabel("Voice:"))
        self._controls_layout.addWidget(self._voice_combo)
        self._controls_layout.addWidget(self._apply_voice_btn)
        self._controls_layout.addWidget(self._calibrate_btn)
        self._controls_layout.addWidget(self._clear_cal_btn)
        self._controls_layout.addStretch(1)
        self._controls_layout.addWidget(self._clear_btn)

    def _build_voice_combo(self) -> QComboBox:
        """Dropdown of edge-tts voices. Disabled when the active player isn't
        the edge backend. Persists the choice to QSettings; the new voice goes
        live on the next prerender (Calibrate / startup auto-apply), so the
        current session keeps the prior voice until then — no audio gap."""
        combo = QComboBox()
        is_edge = isinstance(self._alert_player, EdgeTtsAlertPlayer)
        default_voice = self._alert_player.voice if is_edge else "en-US-AriaNeural"
        current = QSettings("wow-alert", "wow-alert").value(
            "tts_voice", default_voice, type=str
        )
        voices = list(_EDGE_VOICES)
        if current not in [v for _, v in voices]:
            voices.append((current, current))  # keep a non-listed saved voice selectable
        for label, voice_id in voices:
            combo.addItem(label, userData=voice_id)
        idx = combo.findData(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        if not is_edge:
            combo.setEnabled(False)
            combo.setToolTip("Set tts_engine: edge in your config to use neural voices.")
            return combo
        combo.setToolTip(
            "edge-tts neural voice. Applies on the next Calibrate; current "
            "audio keeps the prior voice until then."
        )
        # Persisted pref differs from how cli built the player → repoint now so
        # the startup auto-apply prerenders the saved voice. Runtime changes go
        # through the Apply button, not on every dropdown change.
        if current != self._alert_player.voice:
            self._alert_player.set_voice(current)
        return combo

    def _build_dungeon_combo(self) -> QComboBox:
        """Top-level dungeon picker. Selecting one loads its spell DB and
        prerenders its callout phrases — the whole callouts path, no
        calibration needed. Populated from the authored dungeons so a pick
        always resolves to a real file. Selection/loading is wired up in
        _init_dungeon_selection (after any saved calibration loads)."""
        combo = QComboBox()
        combo.setToolTip(
            "Pick a dungeon to get cast-bar callouts. Calibrate (optional) "
            "layers cooldown recommendations on top."
        )
        combo.addItem("(none)", userData=None)
        for name in list_dungeon_names(REPO_ROOT / "config"):
            combo.addItem(name, userData=name)
        combo.currentIndexChanged.connect(self._on_dungeon_changed)
        return combo

    def _init_dungeon_selection(self) -> None:
        """Set the picker's initial value once startup state is known. A
        calibrated dungeon is reflected without reloading (calibration already
        loaded it); otherwise the last-picked dungeon is loaded so callouts
        work with no calibration."""
        settings = QSettings("wow-alert", "wow-alert")
        cal_dungeon = self._calibration.dungeon_name if self._calibration else None
        if cal_dungeon:
            self._select_dungeon_in_combo(cal_dungeon, block=True)
            settings.setValue("dungeon", cal_dungeon)
            return
        saved = settings.value("dungeon", "", type=str)
        if saved:
            # Not blocked → fires _on_dungeon_changed → background load.
            self._select_dungeon_in_combo(saved, block=False)

    def _select_dungeon_in_combo(self, name: str, *, block: bool) -> bool:
        """Select the combo entry whose name slugs to `name`. With block=True
        the change signal is suppressed (no reload). Returns whether a match
        was found."""
        target = slugify(name)
        for i in range(self._dungeon_combo.count()):
            data = self._dungeon_combo.itemData(i)
            if data and slugify(data) == target:
                if block:
                    self._dungeon_combo.blockSignals(True)
                    self._dungeon_combo.setCurrentIndex(i)
                    self._dungeon_combo.blockSignals(False)
                else:
                    self._dungeon_combo.setCurrentIndex(i)
                return True
        return False

    def start(self) -> None:
        self._thread.start()
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.start()
        self.log_widget.info("worker started")

    def closeEvent(self, event) -> None:
        self._worker.stop()
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.stop()
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
    def _on_suggestions_toggle(self, checked: bool) -> None:
        self._worker.set_suggestions_enabled(checked)
        QSettings("wow-alert", "wow-alert").setValue("suggestions_enabled", checked)
        self.log_widget.info(
            f"suggestions {'on' if checked else 'off'}"
            + ("" if checked else " — alert phrases only, no cooldown recs")
        )

    @Slot()
    def _on_apply_voice(self) -> None:
        """Render the dropdown's selected voice in the background and switch to
        it. The current voice's clips keep playing until the render finishes,
        so there's no silent gap; only an explicit click triggers the work."""
        if not isinstance(self._alert_player, EdgeTtsAlertPlayer):
            return
        if self._voice_thread is not None:
            self.log_widget.info("voice render already in progress")
            return
        voice = self._voice_combo.currentData()
        if not voice:
            return
        QSettings("wow-alert", "wow-alert").setValue("tts_voice", voice)
        self._alert_player.set_voice(voice)
        phrases = self._alert_player.known_phrases()
        self._apply_voice_btn.setEnabled(False)
        self._voice_combo.setEnabled(False)
        self.log_widget.info(f"rendering voice {voice}… ({len(phrases)} clips)")

        thread = QThread(self)
        runner = BackgroundRunner(lambda: self._alert_player.prerender(phrases))
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.completed.connect(self._on_voice_render_done)
        runner.failed.connect(self._on_voice_render_failed)
        runner.completed.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(self._clear_voice_refs)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._voice_thread = thread
        self._voice_runner = runner
        thread.start()

    @Slot(object)
    def _on_voice_render_done(self, _result) -> None:
        self._apply_voice_btn.setEnabled(True)
        self._voice_combo.setEnabled(True)
        self.log_widget.info(f"voice ready: {self._alert_player.voice}")

    @Slot(str)
    def _on_voice_render_failed(self, message: str) -> None:
        self._apply_voice_btn.setEnabled(True)
        self._voice_combo.setEnabled(True)
        self.log_widget.error(
            f"voice render failed: {message} — keeping the prior clips"
        )

    @Slot()
    def _clear_voice_refs(self) -> None:
        self._voice_thread = None
        self._voice_runner = None

    @Slot(int)
    def _on_dungeon_changed(self, index: int) -> None:
        """Load the picked dungeon (spells + phrase prerender) on a background
        thread so the network prerender doesn't freeze the UI. Persists the
        choice; the current spell set keeps working until the swap lands."""
        if self._on_dungeon_select is None:
            return
        if self._dungeon_thread is not None:
            self.log_widget.info("dungeon load already in progress")
            return
        dungeon = self._dungeon_combo.currentData()
        QSettings("wow-alert", "wow-alert").setValue("dungeon", dungeon or "")
        self._dungeon_combo.setEnabled(False)
        self.log_widget.info(f"loading dungeon {dungeon or '(none)'}…")

        thread = QThread(self)
        runner = BackgroundRunner(lambda: self._on_dungeon_select(dungeon))
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.completed.connect(self._on_dungeon_load_done)
        runner.failed.connect(self._on_dungeon_load_failed)
        runner.completed.connect(thread.quit)
        runner.failed.connect(thread.quit)
        thread.finished.connect(self._clear_dungeon_refs)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._dungeon_thread = thread
        self._dungeon_runner = runner
        thread.start()

    @Slot(object)
    def _on_dungeon_load_done(self, _result) -> None:
        self._dungeon_combo.setEnabled(True)
        dungeon = self._dungeon_combo.currentData()
        self.log_widget.info(f"dungeon ready: {dungeon or '(none)'}")

    @Slot(str)
    def _on_dungeon_load_failed(self, message: str) -> None:
        self._dungeon_combo.setEnabled(True)
        self.log_widget.error(f"dungeon load failed: {message}")

    @Slot()
    def _clear_dungeon_refs(self) -> None:
        self._dungeon_thread = None
        self._dungeon_runner = None

    @Slot(bool)
    def _on_debug_toggle(self, checked: bool) -> None:
        self.log_widget.set_show_debug(checked)
        self.log_widget.info(f"debug {'on' if checked else 'off'}")

    @Slot()
    def _on_clear_clicked(self) -> None:
        self.log_widget.clear()

    @Slot()
    def _on_clear_calibration_clicked(self) -> None:
        """Forget the saved calibration + drop recommendations, keeping the
        dungeon callouts. Confirmed because it deletes saved data (though
        it's fully recoverable by recalibrating)."""
        reply = QMessageBox.question(
            self,
            "Clear calibration",
            "Forget the saved calibration and drop cooldown recommendations?\n\n"
            "Your dungeon callouts stay. You can recalibrate anytime.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            CALIBRATION_PATH.unlink(missing_ok=True)
        except OSError as exc:
            self.log_widget.error(f"couldn't delete saved calibration file: {exc}")
        self._calibration = None
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.set_icons([])
        if self._on_clear_calibration is not None:
            self._on_clear_calibration(self._dungeon_combo.currentData())
        self._calibration_status.setText("Not calibrated")
        self.log_widget.info("calibration cleared — callouts only (dungeon kept)")

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

    # ---- calibration ----

    @Slot()
    def _on_calibrate_clicked(self) -> None:
        """Open the region-confirm dialog so the user can draw the
        party-frame and cooldown-manager regions, then kick off Pass 2
        (party-name read) against the confirmed regions.

        No LLM-driven region locate: the previous Pass 1 call placed
        regions unreliably (often top-left of an ultrawide source far
        from the actual UI), so the user redrew them every time anyway.
        Dropping the call saves an API call + ~5 s of latency.
        """
        if self._calibration_thread is not None:
            self.log_widget.info("calibration already in progress")
            return

        frame = self._worker.latest_frame()
        if frame is None:
            self.log_widget.error(
                "no frame available yet — wait for the worker to capture one"
            )
            return

        self._calibrate_btn.setEnabled(False)
        self._calibration_frame = frame
        self._open_region_confirm(frame)

    def _open_region_confirm(self, frame) -> None:
        """Show the region-confirm dialog. Defaults come from the
        previously-saved calibration (so on a re-calibrate the user
        just confirms with minor tweaks); first-ever calibration falls
        through to the editor's centered-rectangle default."""
        party_region = self._derive_region_from_prior("party")
        cooldown_region = self._derive_region_from_prior("cooldown")
        if party_region or cooldown_region:
            self._calibration_status.setText("Calibrating… confirm regions")
            self.log_widget.info(
                "calibrating: regions pre-filled from your previous "
                "calibration — confirm or adjust"
            )
        else:
            self._calibration_status.setText("Calibrating… draw regions")
            self.log_widget.info(
                "calibrating: draw the party + cooldown regions"
            )
        dialog = RegionConfirmDialog(
            image_bgr=frame,
            party_region=party_region,
            cooldown_region=cooldown_region,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.log_widget.info("calibration cancelled at region confirmation")
            self._finish_calibration_flow(success=False)
            return

        party_region, cooldown_region = dialog.result_regions()
        # The dungeon comes from the top-level picker — the single source.
        dungeon_name = self._dungeon_combo.currentData()
        self._calibration_status.setText("Calibrating… reading party + icons")
        self.log_widget.info("calibrating: reading party names + finding icons")
        self._start_runner(
            lambda: calibrate_read(
                frame,
                party_region=party_region,
                cooldown_region=cooldown_region,
                dungeon_name=dungeon_name,
                prior_notes="",
            ),
            on_completed=self._on_read_completed,
        )

    @Slot(object)
    def _on_read_completed(self, cal: Calibration) -> None:
        """Pass 2/3 done — match icons against the local DB, auto-detect
        the class+spec from the matches, open the edit dialog, then the
        icon-labeling dialog so the user can refine the matcher's
        references against their own client rendering."""
        frame = self._calibration_frame
        if frame is not None:
            cal = self._resolve_icons_and_spec(cal, frame)
            dialog = CalibrationDialog(cal, frame, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                cal = dialog.result_calibration()
            else:
                self.log_widget.info("calibration discarded — keeping previous")
                self._finish_calibration_flow(success=False)
                return
            cal = self._label_icons(cal, frame)
        self._apply_calibration(cal, persist=True, log_to_pane=True)
        self._finish_calibration_flow(success=True)

    def _label_icons(self, cal: Calibration, frame) -> Calibration:
        """Open the icon-labeling dialog if there's a class library to
        label against. Cancel keeps the calibration as-is (no reference
        files written); accept writes per-icon PNGs and returns a
        Calibration with the user-confirmed spell_ids."""
        if not cal.player_class or not cal.player_spec:
            self.log_widget.info(
                "icon labeling skipped — no class/spec confirmed"
            )
            return cal
        if not cal.cooldown_icons:
            return cal
        actions = load_class_actions(
            REPO_ROOT / "config", cal.player_class, cal.player_spec,
        )
        if not actions:
            self.log_widget.info(
                "icon labeling skipped — class library empty for "
                f"{cal.player_class}/{cal.player_spec}"
            )
            return cal
        icon_dir = ICONS_DIR
        dialog = IconLabelDialog(
            cal, frame, actions, icon_dir,
            diagnostics=self._last_match_diagnostics,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.log_widget.info(
                "icon labeling cancelled — keeping existing icon references"
            )
            return cal
        labeled_cal, written = dialog.apply_labels()
        self.log_widget.info(
            f"icon labels applied: {written} reference PNG(s) written to "
            f"{ICONS_DIR}"
        )
        return labeled_cal

    def _derive_region_from_prior(
        self, kind: str,
    ) -> tuple[int, int, int, int] | None:
        """Compute a bounding box around all party_members (or all
        cooldown_icons) from the previously-saved calibration. Used as
        the starting region for a re-calibrate so the user doesn't
        have to redraw from scratch."""
        if self._calibration is None:
            return None
        if kind == "party":
            items = self._calibration.party_members
        elif kind == "cooldown":
            items = self._calibration.cooldown_icons
        else:
            return None
        if not items:
            return None
        xs: list[int] = []
        ys: list[int] = []
        for item in items:
            x1, y1, x2, y2 = item.bbox
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        # 10 px padding around the cluster so the user has a little
        # slack when adjusting.
        return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

    def _resolve_icons_and_spec(
        self, cal: Calibration, frame,
    ) -> Calibration:
        """Run the icon matcher on the calibrated bboxes and override the
        Calibration's player_class / player_spec with whatever class
        library best fits the matched icons.

        Surfaces both pieces of state in the log pane and saves debug
        artifacts (per-icon crops + manifest) under
        `%LOCALAPPDATA%\\wow-alert\\calibration_artifacts\\<timestamp>`
        so you can inspect what the LLM bboxes captured when matching
        underperforms.
        """
        icon_dir = ICONS_DIR
        matcher = IconMatcher(icon_dir)
        if len(matcher) == 0:
            self.log_widget.error(
                f"no icons in {icon_dir} — cooldown tracking will be "
                "disabled. Run: python -m wow_alert.tools.fetch_icons"
            )
            return cal

        # Pass 1: match against the full icon DB to figure out which
        # class+spec the player is on.
        per_icon = self._match_with_diagnostics(cal, frame, matcher)
        matched_ids = {ic.spell_id for ic, _ in per_icon if ic.spell_id is not None}
        cls, spec, count = infer_class_spec(REPO_ROOT / "config", matched_ids)

        # Pass 2: re-match restricted to just the detected class's
        # spell IDs. Stops cross-class false references — e.g., paladin
        # icons labeled in a previous session showing up as closest for
        # a monk character's icons.
        if cls and spec:
            actions = load_class_actions(REPO_ROOT / "config", cls, spec)
            allowed = {a.spell_id for a in actions}
            restricted = IconMatcher(icon_dir, allowed_spell_ids=allowed)
            if len(restricted) > 0:
                per_icon = self._match_with_diagnostics(cal, frame, restricted)
                matcher = restricted
            cal = cal.model_copy(update={"player_class": cls, "player_spec": spec})
            self.log_widget.info(
                f"class auto-detect: {cls}/{spec} ({count} matches in pass 1; "
                f"re-matched against {len(restricted)} {cls}/{spec} refs)"
            )
        else:
            self.log_widget.error(
                "could not auto-detect class+spec from icons — pick "
                "manually in the next dialog"
            )

        cal = cal.model_copy(
            update={
                "cooldown_icons": [ic for ic, _ in per_icon],
            }
        )
        # Stash diagnostics for the labeling dialog. Indexed parallel to
        # cal.cooldown_icons.
        self._last_match_diagnostics = [d for _, d in per_icon]
        artifact_dir = self._save_calibration_artifacts(frame, per_icon, matcher)
        if artifact_dir is not None:
            self.log_widget.info(f"saved calibration artifacts → {artifact_dir}")

        matched_ids = {ic.spell_id for ic, _ in per_icon if ic.spell_id is not None}
        total = len(per_icon)
        self.log_widget.info(
            f"icon matcher: {len(matched_ids)}/{total} icons identified"
        )
        return cal

    def _match_with_diagnostics(self, cal: Calibration, frame, matcher: IconMatcher):
        """Run the matcher per icon and emit a UI log line for each.

        Returns a list of `(updated_CooldownIcon, diagnostic_dict)`. The
        diagnostic carries the bbox-crop image, closest spell_id, score,
        and passed flag so the artifact-saving step doesn't have to re-run
        the matcher.
        """
        h, w = frame.shape[:2]
        out = []
        for idx, icon in enumerate(cal.cooldown_icons):
            x1, y1, x2, y2 = icon.bbox
            x1c, y1c = max(0, min(x1, w)), max(0, min(y1, h))
            x2c, y2c = max(0, min(x2, w)), max(0, min(y2, h))
            if x2c - x1c < 4 or y2c - y1c < 4:
                self.log_widget.error(
                    f"  icon #{idx} at {icon.bbox}: degenerate bbox"
                )
                out.append((icon, {"crop": None, "closest": None, "score": 0.0, "passed": False}))
                continue
            crop = frame[y1c:y2c, x1c:x2c]
            closest, score, passed = matcher.match(crop)
            diag = {"crop": crop, "closest": closest, "score": score, "passed": passed}
            if passed:
                self.log_widget.info(
                    f"  icon #{idx} at {icon.bbox} → spell_id {closest} (score {score:.2f}) MATCH"
                )
                out.append((CooldownIcon(bbox=icon.bbox, spell_id=closest), diag))
            else:
                closest_str = str(closest) if closest is not None else "—"
                self.log_widget.error(
                    f"  icon #{idx} at {icon.bbox}: closest={closest_str} "
                    f"score={score:.2f} (threshold {matcher.threshold:.2f}) BELOW"
                )
                out.append((icon, diag))
        return out

    def _save_calibration_artifacts(self, frame, per_icon, matcher: IconMatcher):
        """Write per-icon crops + a manifest text file under
        `CALIBRATION_ARTIFACTS_DIR/<timestamp>/`. Returns the directory
        path on success or None on failure (silent — diagnostics
        shouldn't break calibration)."""
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            artifact_dir = CALIBRATION_ARTIFACTS_DIR / stamp
            artifact_dir.mkdir(parents=True, exist_ok=True)
            # Full frame so you can verify Pass 1's region detection.
            cv2.imwrite(str(artifact_dir / "frame.png"), frame)
            # Per-icon: crop + reference of the closest match (when known).
            manifest_lines = [
                f"# Calibration debug ({stamp})",
                f"# Matcher threshold: {matcher.threshold:.2f}",
                "# idx  bbox                                   closest_spell_id  score  passed  crop_file",
            ]
            for idx, (icon, diag) in enumerate(per_icon):
                crop = diag["crop"]
                closest = diag["closest"]
                score = diag["score"]
                passed = diag["passed"]
                crop_name = f"icon_{idx:02d}_bbox.png"
                if crop is not None:
                    cv2.imwrite(str(artifact_dir / crop_name), crop)
                if closest is not None:
                    ref_src = ICONS_DIR / f"{closest}.png"
                    if ref_src.exists():
                        ref_dst = artifact_dir / f"icon_{idx:02d}_closest_{closest}.png"
                        ref_dst.write_bytes(ref_src.read_bytes())
                manifest_lines.append(
                    f"{idx:>3d}  {str(icon.bbox):<38}  "
                    f"{str(closest):<16}  {score:5.2f}  {str(passed):<6}  {crop_name}"
                )
            (artifact_dir / "manifest.txt").write_text(
                "\n".join(manifest_lines), encoding="utf-8",
            )
            return artifact_dir
        except Exception as exc:
            logger.warning("Failed to save calibration artifacts: %s", exc)
            return None

    def _log_per_action_match_state(
        self, cal: Calibration, matched_ids: set[int],
    ) -> None:
        """Emit one log line per class-library action: tracked vs untracked.

        Tracked actions (spell_id matched on the player's bar) get an
        info line. Untracked actions get an error line — they won't be
        recommended; rules referencing them will fall through to the
        spell's default phrase. Helps the user see at a glance which
        abilities the engine can reason about for this character.
        """
        if not cal.player_class or not cal.player_spec:
            return
        actions = load_class_actions(
            REPO_ROOT / "config", cal.player_class, cal.player_spec,
        )
        if not actions:
            return
        for a in actions:
            if a.spell_id in matched_ids:
                self.log_widget.info(
                    f"  tracked: {a.id} (spell_id={a.spell_id})"
                )
            else:
                self.log_widget.error(
                    f"  UNTRACKED: {a.id} (spell_id={a.spell_id}) — "
                    f"rules using this will fall through to spell default"
                )

    @Slot(str)
    def _on_calibration_failed(self, message: str) -> None:
        self.log_widget.error(f"calibration failed: {message}")
        self._finish_calibration_flow(success=False)

    # ---- calibration flow helpers ----

    def _start_runner(
        self,
        fn: Callable[[], object],
        *,
        on_completed: Callable,
    ) -> None:
        """Spawn a QThread + `BackgroundRunner` to run `fn`, wire signals."""
        thread = QThread(self)
        runner = BackgroundRunner(fn)
        runner.moveToThread(thread)
        thread.started.connect(runner.run)
        runner.completed.connect(on_completed)
        runner.failed.connect(self._on_calibration_failed)
        runner.completed.connect(thread.quit)
        runner.failed.connect(thread.quit)
        # Clear refs as soon as the event loop exits, so the next phase
        # (or a re-click after completion) doesn't trip on stale state.
        thread.finished.connect(self._clear_calibration_refs)
        thread.finished.connect(runner.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._calibration_thread = thread
        self._calibration_runner = runner
        thread.start()

    @Slot()
    def _clear_calibration_refs(self) -> None:
        self._calibration_thread = None
        self._calibration_runner = None

    def _abort_calibration(self, reason: str) -> None:
        self.log_widget.error(f"calibration aborted: {reason}")
        self._finish_calibration_flow(success=False)

    def _finish_calibration_flow(self, *, success: bool) -> None:
        """End-of-flow cleanup, success or not. Resets per-run state and
        re-enables the Calibrate button."""
        self._calibration_frame = None
        if not success:
            self._calibration_status.setText(
                self._format_calibration_status(self._calibration)
            )
        self._calibrate_btn.setEnabled(True)

    def _apply_calibration(
        self,
        cal: Calibration,
        *,
        persist: bool,
        log_to_pane: bool,
    ) -> None:
        """Apply a calibration to the running app: push the roster into
        downstream components (deduper, spell DB), update status, optionally
        save to disk, optionally announce in the log pane."""
        self._calibration = cal
        # Push the cooldown icon set to the watcher first so the dict is
        # populated by the time the rule engine starts evaluating against
        # the new spell set below.
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.set_icons(cal.cooldown_icons)
        if self._on_calibration_apply is not None:
            self._on_calibration_apply(cal)
        self._calibration_status.setText(self._format_calibration_status(cal))
        if persist:
            try:
                save_calibration(cal, CALIBRATION_PATH)
            except Exception as exc:
                self.log_widget.error(f"failed to save calibration: {exc}")
        if log_to_pane:
            roster = ", ".join(cal.roster()) or "(no party members detected)"
            dungeon_str = f" dungeon={cal.dungeon_name!r}" if cal.dungeon_name else ""
            you_str = f" you={cal.player_name!r}" if cal.player_name else " you=(name unset)"
            self.log_widget.info(
                f"calibrated:{dungeon_str}{you_str} {len(cal.party_members)} party members "
                f"[{roster}], {len(cal.cooldown_icons)} cooldown icons"
            )
            if cal.notes:
                self.log_widget.info(f"calibration notes: {cal.notes}")
            # Per-action tracked/untracked. Runs here (not in
            # _resolve_icons_and_spec) so it fires regardless of whether
            # auto-detect succeeded — the dialog may have set the spec
            # manually after we failed to infer it.
            matched_ids = {
                ic.spell_id for ic in cal.cooldown_icons if ic.spell_id is not None
            }
            self._log_per_action_match_state(cal, matched_ids)

    @staticmethod
    def _format_calibration_status(cal: Calibration | None) -> str:
        if cal is None:
            return "Not calibrated"
        names = cal.roster()
        ts = cal.calibrated_at.strftime("%H:%M:%S")
        prefix = f"[{cal.dungeon_name}] " if cal.dungeon_name else ""
        if not names:
            return f"{prefix}Calibrated (no party detected) at {ts}"
        joined = ", ".join(names)
        return f"{prefix}Calibrated for: {joined} ({len(names)} members) at {ts}"
