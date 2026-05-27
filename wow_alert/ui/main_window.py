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
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from typing import Callable

from wow_alert.audio import PyttsxWinsoundAlertPlayer
from wow_alert.calibration import (
    Calibration,
    CooldownIcon,
    LocateResult,
    calibrate_locate,
    calibrate_read,
    load_calibration,
    save_calibration,
)
from wow_alert.class_library import infer_class_spec, load_class_actions
from wow_alert.config import REPO_ROOT
from wow_alert.cooldown_watcher import CooldownWatcher
from wow_alert.events import Alert
from wow_alert.icon_matcher import IconMatcher
from wow_alert.paths import CALIBRATION_ARTIFACTS_DIR, CALIBRATION_PATH
from wow_alert.pipeline import PipelineWorker
from wow_alert.ui._background_runner import BackgroundRunner
from wow_alert.ui.calibration_dialog import CalibrationDialog
from wow_alert.ui.frame_widget import FrameWidget
from wow_alert.ui.icon_label_dialog import IconLabelDialog
from wow_alert.ui.log_widget import LogWidget
from wow_alert.ui.region_confirm_dialog import RegionConfirmDialog

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        worker: PipelineWorker,
        alert_player: PyttsxWinsoundAlertPlayer,
        show_preview: bool = True,
        on_calibration_apply: Callable[[Calibration], None] | None = None,
        cooldown_watcher: CooldownWatcher | None = None,
    ):
        super().__init__()
        self.setWindowTitle("wow-alert — cast bar awareness")
        self.resize(1280, 900)

        self._worker = worker
        self._alert_player = alert_player
        self._on_calibration_apply = on_calibration_apply
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
        # Per-run state for the two-phase flow (frame stays valid across the
        # region-confirm dialog so pass-2/3 uses the same image pass-1 saw).
        self._calibration_frame: np.ndarray | None = None
        self._calibration_locate: LocateResult | None = None
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
        self._debug_cb.setChecked(True)
        self._debug_cb.toggled.connect(self._on_debug_toggle)

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)

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
        self._controls_layout.addWidget(self._calibrate_btn)
        self._controls_layout.addStretch(1)
        self._controls_layout.addWidget(self._clear_btn)

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

    # ---- calibration ----

    @Slot()
    def _on_calibrate_clicked(self) -> None:
        """Kick off Pass 1 (LLM region detection).

        On completion, a region-confirm dialog opens so the user can adjust
        the bounding boxes before Pass 2/3 reads the contents. This makes
        the system robust to LLM mis-localization on unusual aspect ratios.
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
        self._calibration_status.setText("Calibrating… (1/2: locating regions)")
        self.log_widget.info("calibrating (pass 1: locating regions)")
        self._calibration_frame = frame
        self._start_runner(
            lambda: calibrate_locate(frame),
            on_completed=self._on_locate_completed,
        )

    @Slot(object)
    def _on_locate_completed(self, locate_result: LocateResult) -> None:
        """Pass 1 done — show the region-confirm dialog, then kick off
        Pass 2/3 against the user-confirmed regions."""
        self._calibration_locate = locate_result
        frame = self._calibration_frame
        if frame is None:
            self._abort_calibration("internal: frame missing between phases")
            return

        dialog = RegionConfirmDialog(
            image_bgr=frame,
            party_region=locate_result.party_region,
            cooldown_region=locate_result.cooldown_region,
            dungeon_name=locate_result.dungeon_name,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.log_widget.info("calibration cancelled at region confirmation")
            self._finish_calibration_flow(success=False)
            return

        party_region, cooldown_region, dungeon_name = dialog.result_regions()
        self._calibration_status.setText("Calibrating… (2/2: reading contents)")
        self.log_widget.info("calibrating (pass 2/3: reading confirmed regions)")
        prior_notes = locate_result.notes
        self._start_runner(
            lambda: calibrate_read(
                frame,
                party_region=party_region,
                cooldown_region=cooldown_region,
                dungeon_name=dungeon_name,
                prior_notes=prior_notes,
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
        icon_dir = REPO_ROOT / "config" / "icons"
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
            f"config/icons/"
        )
        return labeled_cal

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
        icon_dir = REPO_ROOT / "config" / "icons"
        matcher = IconMatcher(icon_dir)
        if len(matcher) == 0:
            self.log_widget.error(
                "no icons in config/icons/ — cooldown tracking will be "
                "disabled. Run: python -m wow_alert.tools.fetch_icons"
            )
            return cal

        per_icon = self._match_with_diagnostics(cal, frame, matcher)
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

        cls, spec, count = infer_class_spec(REPO_ROOT / "config", matched_ids)
        if cls and spec:
            self.log_widget.info(
                f"class auto-detect: {cls}/{spec} ({count} icon matches)"
            )
            cal = cal.model_copy(update={"player_class": cls, "player_spec": spec})
        else:
            self.log_widget.error(
                "could not auto-detect class+spec from icons — pick "
                "manually in the next dialog"
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
                    ref_src = REPO_ROOT / "config" / "icons" / f"{closest}.png"
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
        self._calibration_locate = None
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
            self.log_widget.info(
                f"calibrated:{dungeon_str} {len(cal.party_members)} party members "
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
