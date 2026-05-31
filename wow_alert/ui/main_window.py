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
    calibrate_read,
    load_calibration,
    save_calibration,
)
from wow_alert.class_library import load_class_actions
from wow_alert.cooldown_watcher import CooldownWatcher
from wow_alert.dungeon_loader import list_dungeon_names, slugify
from wow_alert.events import Alert
from wow_alert.icon_matcher import IconMatcher
from wow_alert.paths import (
    CALIBRATION_ARTIFACTS_DIR,
    CALIBRATION_PATH,
    ICONS_DIR,
    calibration_path_for,
)
from wow_alert.pipeline import PipelineWorker
from wow_alert.ui._background_runner import BackgroundRunner
from wow_alert.ui.calibration_dialog import RosterDialog
from wow_alert.ui.config_overrides_dialog import ConfigOverridesDialog
from wow_alert.ui.tag_suggestions_dialog import TagSuggestionsDialog
from wow_alert.ui.frame_widget import FrameWidget
from wow_alert.ui.log_widget import LogWidget
from wow_alert.ui.region_confirm_dialog import RegionConfirmDialog
from wow_alert.ui.theme import make_separator

logger = logging.getLogger(__name__)


def _display_token(token: str) -> str:
    """'death_knight' -> 'Death Knight'. Used by the Character picker."""
    return " ".join(p.capitalize() for p in token.split("_"))

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

        # The cooldown watcher runs on its own QThread. With 10+ calibrated
        # icons the per-tick OCR center-text check is expensive enough that
        # leaving it on the UI thread blocked repaints. Lifecycle is
        # `start()` / `stop()` (signal-routed) alongside the pipeline
        # thread; `set_icons()` is also signal-routed for thread safety.
        if self._cooldown_watcher is not None:
            self._cd_thread: QThread | None = QThread(self)
            self._cooldown_watcher.moveToThread(self._cd_thread)
        else:
            self._cd_thread = None

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
        # Per-run state: holds the frame across the region-confirm dialog so
        # calibrate_read uses the same image the user just confirmed against.
        self._calibration_frame: np.ndarray | None = None

        # Status bar shows the current calibration target ("Calibrated for:
        # John, Mary, Tank…") so the user can confirm they're configured for
        # the right party at a glance.
        self._calibration_status = QLabel("Not calibrated")
        self.statusBar().addPermanentWidget(self._calibration_status)

        # Auto-load the last-active character's calibration. Falls back to
        # the legacy single-file calibration (migrating it on first save) if
        # no per-spec file exists yet.
        existing = self._load_active_calibration()
        if existing is not None:
            self._apply_calibration(existing, persist=False, log_to_pane=False)
        # Initialize the dungeon picker: reflect the calibrated dungeon if one
        # loaded, else load the last-picked dungeon so callouts work with no
        # calibration at all.
        self._init_dungeon_selection()
        self._init_character_selection()
        self._update_status()
        self._refresh_button_state()

    def _load_active_calibration(self) -> Calibration | None:
        """Load the calibration for the QSettings-remembered character.

        Three-step fallback: per-spec file → legacy single-file (migrated on
        save) → None. The legacy fallback lets users keep their existing
        calibration after the per-spec change ships.
        """
        settings = QSettings("wow-alert", "wow-alert")
        cls = settings.value("player_class", "", type=str) or None
        spec = settings.value("player_spec", "", type=str) or None
        if cls and spec:
            cal = load_calibration(calibration_path_for(cls, spec))
            if cal is not None:
                return cal
        # Legacy single-file fallback for first launch after the per-spec change.
        return load_calibration(CALIBRATION_PATH)

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

        self._suggestions_categories_btn = QPushButton("Suggestions…")
        self._suggestions_categories_btn.setToolTip(
            "Pick which mechanic categories get a 'press this' callout. "
            "Unchecked categories play just the plain alert."
        )
        self._suggestions_categories_btn.clicked.connect(
            self._on_suggestions_categories_clicked,
        )
        # Apply persisted per-tag selection on startup so a saved subset
        # actually takes effect against the worker's rule engine.
        self._apply_persisted_enabled_tags()

        self._configure_btn = QPushButton("Configure…")
        self._configure_btn.setToolTip(
            "Customize bundled dungeon / class / tag files. Per-file "
            "Customize + Revert; bundled updates reach you for anything "
            "not customized."
        )
        self._configure_btn.clicked.connect(self._on_configure_clicked)

        self._voice_combo = self._build_voice_combo()
        self._dungeon_combo = self._build_dungeon_combo()
        self._class_combo, self._spec_combo = self._build_character_combos()

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

        self._roster_btn = QPushButton("Roster")
        self._roster_btn.setToolTip(
            "Edit the per-run roster. 'Load party members' reads names from "
            "the saved party region — no re-calibration needed."
        )
        self._roster_btn.clicked.connect(self._on_roster_clicked)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._on_clear_clicked)

        self._apply_voice_btn = QPushButton("Apply")
        self._apply_voice_btn.setToolTip(
            "Render the selected voice and switch to it (takes a few seconds)."
        )
        self._apply_voice_btn.setEnabled(self._voice_combo.isEnabled())
        self._apply_voice_btn.clicked.connect(self._on_apply_voice)

        # Grouped left→right by workflow, with thin rules between groups:
        #   Dungeon (callouts) │ Calibration (recommendations) │ Run │ Audio
        # Diagnostics (confidence / preview / debug / clear log) get pushed to
        # the far right so the primary workflow reads first.
        char_label = QLabel("Character:")
        char_label.setToolTip(
            "Class and spec. Switching loads that character's saved "
            "calibration (cooldown bar + party region)."
        )
        self._class_combo.setMinimumWidth(120)
        self._spec_combo.setMinimumWidth(110)
        self._add_control_group(char_label, self._class_combo, self._spec_combo)
        self._controls_layout.addWidget(make_separator())
        dungeon_label = QLabel("Dungeon:")
        dungeon_label.setToolTip(
            "Pick a dungeon to get cast-bar callouts — no calibration needed."
        )
        self._dungeon_combo.setMinimumWidth(180)
        self._add_control_group(dungeon_label, self._dungeon_combo, self._roster_btn)
        self._controls_layout.addWidget(make_separator())
        self._add_control_group(
            self._calibrate_btn, self._clear_cal_btn,
            self._suggestions_cb, self._suggestions_categories_btn,
            self._configure_btn,
        )
        self._controls_layout.addWidget(make_separator())
        self._add_control_group(self._pause_btn, self._alerts_cb)
        self._controls_layout.addWidget(make_separator())
        self._add_control_group(QLabel("Voice:"), self._voice_combo, self._apply_voice_btn)
        self._controls_layout.addStretch(1)
        self._add_control_group(conf_label, self._conf_slider, self._conf_value)
        self._controls_layout.addWidget(make_separator())
        self._add_control_group(self._preview_cb, self._debug_cb, self._clear_btn)

    def _add_control_group(self, *widgets) -> None:
        """Add a run of related widgets to the control bar in order."""
        for w in widgets:
            self._controls_layout.addWidget(w)

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
        for name in list_dungeon_names():
            combo.addItem(name, userData=name)
        combo.currentIndexChanged.connect(self._on_dungeon_changed)
        return combo

    def _build_character_combos(self) -> tuple[QComboBox, QComboBox]:
        """Top-level Class + Spec dropdowns.

        Only the class→spec repopulation is wired here; the
        `_on_character_changed` signal is connected later, after the combos
        are assigned to `self` and the initial value is restored. Otherwise
        the addItem/setCurrentIndex chain runs before `self._class_combo`
        exists and the slot crashes.
        """
        from wow_alert.calibration import WOW_CLASSES, WOW_SPECS

        class_combo = QComboBox()
        class_combo.addItem("(none)", userData=None)
        for cls in WOW_CLASSES:
            class_combo.addItem(_display_token(cls), userData=cls)
        spec_combo = QComboBox()

        def repopulate_spec(*_args) -> None:
            cls = class_combo.currentData()
            spec_combo.clear()
            spec_combo.addItem("(none)", userData=None)
            for spec in WOW_SPECS.get(cls, []):
                spec_combo.addItem(_display_token(spec), userData=spec)

        class_combo.currentIndexChanged.connect(repopulate_spec)
        repopulate_spec()
        return class_combo, spec_combo

    def _init_character_selection(self) -> None:
        """Restore the saved character on startup, then wire the
        on-change handler. Signals are blocked during init so a stale
        selection doesn't trigger a reload of the calibration that was
        just loaded."""
        settings = QSettings("wow-alert", "wow-alert")
        cls = (
            (self._calibration.player_class if self._calibration else None)
            or settings.value("player_class", "", type=str) or None
        )
        spec = (
            (self._calibration.player_spec if self._calibration else None)
            or settings.value("player_spec", "", type=str) or None
        )
        self._class_combo.blockSignals(True)
        self._spec_combo.blockSignals(True)
        try:
            if cls:
                idx = self._class_combo.findData(cls)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
            # Spec combo's contents depend on class — repopulate once now
            # with signals blocked, then pre-select the spec.
            from wow_alert.calibration import WOW_SPECS
            self._spec_combo.clear()
            self._spec_combo.addItem("(none)", userData=None)
            for s in WOW_SPECS.get(cls, []):
                self._spec_combo.addItem(_display_token(s), userData=s)
            if spec:
                idx = self._spec_combo.findData(spec)
                if idx >= 0:
                    self._spec_combo.setCurrentIndex(idx)
        finally:
            self._class_combo.blockSignals(False)
            self._spec_combo.blockSignals(False)
        # Connect now, after the initial restore. Earlier wiring would have
        # fired `_on_character_changed` during widget construction (before
        # `self._class_combo` even exists), crashing startup.
        self._class_combo.currentIndexChanged.connect(self._on_character_changed)
        self._spec_combo.currentIndexChanged.connect(self._on_character_changed)
        # Spec combo also greys itself when no class is picked; keep that in sync.
        self._class_combo.currentIndexChanged.connect(
            lambda _idx: self._refresh_button_state()
        )

    @Slot()
    def _on_character_changed(self) -> None:
        """Class or spec dropdown changed: persist + load that calibration."""
        cls = self._class_combo.currentData()
        spec = self._spec_combo.currentData()
        settings = QSettings("wow-alert", "wow-alert")
        settings.setValue("player_class", cls or "")
        settings.setValue("player_spec", spec or "")
        if not cls or not spec:
            return
        cal = load_calibration(calibration_path_for(cls, spec))
        if cal is None:
            self.log_widget.info(
                f"no calibration for {cls}/{spec} yet — click Calibrate to set "
                "regions and detect icons"
            )
            # Drop the prior calibration so the engine doesn't carry stale icons
            # from the previous character into the new one.
            empty = Calibration(player_class=cls, player_spec=spec)
            self._apply_calibration(empty, persist=False, log_to_pane=False)
            return
        self.log_widget.info(f"loaded calibration for {cls}/{spec}")
        self._apply_calibration(cal, persist=False, log_to_pane=True)

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
        if self._cd_thread is not None:
            self._cd_thread.start()
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.start()
        self.log_widget.info("worker started")

    def closeEvent(self, event) -> None:
        self._worker.stop()
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.stop()
        self._thread.quit()
        self._thread.wait(2000)
        if self._cd_thread is not None:
            self._cd_thread.quit()
            self._cd_thread.wait(2000)
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
        self._update_status()
        self.log_widget.info(
            f"suggestions {'on' if checked else 'off'}"
            + ("" if checked else " — alert phrases only, no cooldown recs")
        )

    @Slot()
    def _on_suggestions_categories_clicked(self) -> None:
        """Open the per-tag suggestion dialog. On accept, push the new
        subset to the rule engine and persist the names to QSettings."""
        tag_rules = self._worker.rule_engine.tag_rules
        if not tag_rules.precedence:
            QMessageBox.information(
                self, "No tag rules loaded",
                "tag_rules.yaml is empty — there are no tag categories to "
                "configure. Add tags to the config or load a dungeon first.",
            )
            return
        settings = QSettings("wow-alert", "wow-alert")
        stored = settings.value("enabled_tags", None)
        if stored is None:
            enabled: set[str] | None = None
        else:
            # QSettings round-trips lists as Python lists on Windows; coerce.
            enabled = set(stored) if isinstance(stored, (list, tuple)) else None
        dialog = TagSuggestionsDialog(tag_rules, enabled, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_enabled = dialog.result_enabled_tags()
        self._worker.rule_engine.set_enabled_tags(new_enabled)
        settings.setValue("enabled_tags", sorted(new_enabled))
        skipped = sorted(set(tag_rules.precedence) - new_enabled)
        if skipped:
            self.log_widget.info(
                f"tag suggestions: enabled {len(new_enabled)} of "
                f"{len(tag_rules.precedence)} categories; off: {', '.join(skipped)}"
            )
        else:
            self.log_widget.info("tag suggestions: all categories enabled")

    @Slot()
    def _on_configure_clicked(self) -> None:
        """Open the Configuration overrides dialog. Changes to override
        files don't take effect until the next dungeon/class load — show
        a hint reminding the user to restart after editing if they want
        the changes live immediately."""
        dialog = ConfigOverridesDialog(parent=self)
        dialog.exec()
        self.log_widget.info(
            "config overrides: restart the app (or re-pick the dungeon / "
            "spec) to pick up edits"
        )

    def _apply_persisted_enabled_tags(self) -> None:
        """Restore the saved per-tag subset on startup. Runs after the
        rule engine has been wired (the worker holds it) so the engine
        sees the user's filter from tick 1."""
        settings = QSettings("wow-alert", "wow-alert")
        stored = settings.value("enabled_tags", None)
        if stored is None:
            return  # default: all enabled
        if not isinstance(stored, (list, tuple)):
            return
        self._worker.rule_engine.set_enabled_tags(set(stored))

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
        # Keep the in-memory calibration's dungeon_name aligned with the
        # picker. Otherwise the auto-Roster flow that opens after this load
        # carries the OLD dungeon through to apply_calibration, which then
        # reloads the old dungeon's spells right back over the new one.
        if self._calibration is not None and self._calibration.dungeon_name != dungeon:
            self._calibration = self._calibration.model_copy(
                update={"dungeon_name": dungeon},
            )
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
        self._update_status()
        self.log_widget.info(f"dungeon ready: {dungeon or '(none)'}")
        # Pug flow: new dungeon = new group. Clear the prior roster and pop
        # the Roster dialog so the user can capture this run's group with
        # one click of "Load party members". Skipped when no dungeon is
        # selected (user picked "(none)") or no party region is calibrated
        # (no point opening a Roster dialog that can't OCR anything).
        if dungeon and self._calibration and self._calibration.party_region:
            self._calibration = self._calibration.model_copy(update={
                "party_members": [], "player_name": None,
            })
            self._on_roster_clicked()

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
    def _on_roster_clicked(self) -> None:
        """Open the Roster dialog so the user can refresh + edit team
        members for this dungeon run. Opens with the saved calibration as
        backing — even when there's no UI calibration yet, an empty Roster
        dialog still lets the user type names manually."""
        cal = self._calibration or Calibration()
        dialog = RosterDialog(
            cal,
            frame_provider=self._worker.latest_frame,
            ocr=self._worker.ocr,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_cal = dialog.result_calibration()
        self._apply_calibration(new_cal, persist=True, log_to_pane=True)

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
        # Delete the active per-spec file (or the legacy single file if no
        # class/spec is set). Best-effort — a stale file is harmless.
        cls = self._class_combo.currentData()
        spec = self._spec_combo.currentData()
        for path in {calibration_path_for(cls, spec), CALIBRATION_PATH}:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self.log_widget.error(f"couldn't delete {path}: {exc}")
        self._calibration = None
        if self._cooldown_watcher is not None:
            self._cooldown_watcher.set_icons([])
        if self._on_clear_calibration is not None:
            self._on_clear_calibration(self._dungeon_combo.currentData())
        self._update_status()
        self._refresh_button_state()
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
        party-frame and cooldown-manager regions, then read party names
        and template-match cooldown icons against the confirmed regions.
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
        prior_class = self._calibration.player_class if self._calibration else None
        prior_spec = self._calibration.player_spec if self._calibration else None
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
            player_class=prior_class,
            player_spec=prior_spec,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.log_widget.info("calibration cancelled at region confirmation")
            self._finish_calibration_flow(success=False)
            return

        party_region, cooldown_region = dialog.result_regions()
        # The top-level Character picker is the primary source for class+spec.
        # The region dialog's class/spec controls override it for the (rare)
        # case of calibrating for a different character.
        dialog_cls, dialog_spec = dialog.result_class_spec()
        player_class = dialog_cls or self._class_combo.currentData()
        player_spec = dialog_spec or self._spec_combo.currentData()
        # The dungeon comes from the top-level picker — the single source.
        dungeon_name = self._dungeon_combo.currentData()
        # Build a class-restricted matcher so calibrate_read's template-match
        # path only slides icons that could be on this character's bar.
        matcher = self._build_calibration_matcher(player_class, player_spec)
        self._calibration_status.setText("Calibrating… reading party + icons")
        self.log_widget.info("calibrating: reading party names + finding icons")
        self._start_runner(
            lambda: calibrate_read(
                frame,
                party_region=party_region,
                cooldown_region=cooldown_region,
                matcher=matcher,
                dungeon_name=dungeon_name,
                player_class=player_class,
                player_spec=player_spec,
                prior_notes="",
            ),
            on_completed=self._on_read_completed,
        )

    def _build_calibration_matcher(
        self, player_class: str | None, player_spec: str | None,
    ) -> IconMatcher | None:
        """Build a class-restricted IconMatcher for `calibrate_read`.

        Returns None when class/spec aren't set — calibrate_read then skips
        the cooldown region (no icons until the user picks a class/spec and
        recalibrates). When the class library can't be loaded, fall back to
        the full icon DB so we at least get something useful.
        """
        if not player_class or not player_spec:
            return None
        actions = load_class_actions(player_class, player_spec)
        allowed = {a.spell_id for a in actions} if actions else None
        matcher = IconMatcher(ICONS_DIR, allowed_spell_ids=allowed)
        if len(matcher) == 0:
            self.log_widget.error(
                f"no icons in {ICONS_DIR} matching {player_class}/{player_spec} — "
                "cooldown tracking will be empty. Run: "
                "python -m wow_alert.tools.fetch_icons"
            )
            return None
        return matcher

    @Slot(object)
    def _on_read_completed(self, cal: Calibration) -> None:
        """Calibrate read done — class+spec, regions, and template-matched
        icons are saved. Roster is edited separately via the Roster button
        (lighter, runs per-dungeon)."""
        frame = self._calibration_frame
        if frame is not None:
            self._save_calibration_artifacts(frame, cal)
        # Carry over the prior roster so a re-calibrate doesn't wipe it. The
        # roster is per-run, not per-UI-calibration.
        if self._calibration is not None:
            cal = cal.model_copy(update={
                "party_members": self._calibration.party_members,
                "player_name": self._calibration.player_name,
            })
        self._apply_calibration(cal, persist=True, log_to_pane=True)
        self._finish_calibration_flow(success=True)

    def _save_calibration_artifacts(self, frame, cal: Calibration):
        """Write per-icon crops + a manifest + an annotated overview under
        `CALIBRATION_ARTIFACTS_DIR/<timestamp>/`. The overview lets the user
        eyeball whether all icons on their cooldown manager got matched, or
        whether some are present but didn't clear the threshold. Best-effort
        — a failure here doesn't break calibration."""
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            artifact_dir = CALIBRATION_ARTIFACTS_DIR / stamp
            artifact_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(artifact_dir / "frame.png"), frame)
            h, w = frame.shape[:2]
            # Annotated overview: the user-drawn cooldown region (orange) and
            # each matched icon's bbox (green) with its spell_id label.
            overview = frame.copy()
            if cal.cooldown_region:
                rx1, ry1, rx2, ry2 = cal.cooldown_region
                cv2.rectangle(overview, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            for icon in cal.cooldown_icons:
                x1, y1, x2, y2 = icon.bbox
                cv2.rectangle(overview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    overview, str(icon.spell_id),
                    (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
            # Crop the overview to the cooldown region (+ a margin) so the
            # user doesn't have to scroll a full ultrawide capture to inspect.
            if cal.cooldown_region:
                rx1, ry1, rx2, ry2 = cal.cooldown_region
                pad = 40
                ox1 = max(0, rx1 - pad)
                oy1 = max(0, ry1 - pad)
                ox2 = min(w, rx2 + pad)
                oy2 = min(h, ry2 + pad)
                cv2.imwrite(
                    str(artifact_dir / "overview_cooldown.png"),
                    overview[oy1:oy2, ox1:ox2],
                )
            cv2.imwrite(str(artifact_dir / "overview_full.png"), overview)
            lines = [
                f"# Calibration debug ({stamp})",
                f"# {len(cal.cooldown_icons)} icons matched",
                "# idx  bbox                                   spell_id  crop_file",
            ]
            for idx, icon in enumerate(cal.cooldown_icons):
                x1, y1, x2, y2 = icon.bbox
                x1c, y1c = max(0, min(x1, w)), max(0, min(y1, h))
                x2c, y2c = max(0, min(x2, w)), max(0, min(y2, h))
                crop_name = f"icon_{idx:02d}_bbox.png"
                if x2c - x1c > 0 and y2c - y1c > 0:
                    cv2.imwrite(
                        str(artifact_dir / crop_name), frame[y1c:y2c, x1c:x2c],
                    )
                lines.append(
                    f"{idx:>3d}  {str(icon.bbox):<38}  "
                    f"{icon.spell_id!s:<8}  {crop_name}"
                )
            (artifact_dir / "manifest.txt").write_text(
                "\n".join(lines), encoding="utf-8",
            )
            self.log_widget.info(f"saved calibration artifacts → {artifact_dir}")
        except Exception as exc:
            logger.warning("Failed to save calibration artifacts: %s", exc)

    def _derive_region_from_prior(
        self, kind: str,
    ) -> tuple[int, int, int, int] | None:
        """Default region for a re-calibrate.

        Source order: this character's saved calibration → another spec on
        the same class → any other character's calibration. UI layout
        rarely changes between specs on the same character, so inheriting
        sibling regions saves the user from redrawing every time.
        """
        # Prefer the explicit region field (newer format), fall back to a
        # bounding box around the existing items (older format).
        def _from_cal(cal: Calibration) -> tuple[int, int, int, int] | None:
            if kind == "party":
                if cal.party_region:
                    return cal.party_region
                items = cal.party_members
            elif kind == "cooldown":
                if cal.cooldown_region:
                    return cal.cooldown_region
                items = cal.cooldown_icons
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
            return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

        if self._calibration is not None:
            r = _from_cal(self._calibration)
            if r is not None:
                return r
        # Sibling-spec fallback: scan saved calibrations for the same class.
        cls = self._class_combo.currentData()
        if cls:
            from wow_alert.calibration import WOW_SPECS
            for sibling_spec in WOW_SPECS.get(cls, []):
                cal = load_calibration(calibration_path_for(cls, sibling_spec))
                if cal is None:
                    continue
                r = _from_cal(cal)
                if r is not None:
                    return r
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
            cal.player_class, cal.player_spec,
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

    def _finish_calibration_flow(self, *, success: bool) -> None:
        """End-of-flow cleanup, success or not. Resets per-run state and
        re-enables the Calibrate button."""
        self._calibration_frame = None
        if not success:
            self._update_status()
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
        self._update_status()
        if persist:
            # Save to the per-spec path so each character has its own file.
            # Falls back to the legacy single-file path when class/spec aren't
            # set (rare, only first-time-no-character calibrations).
            target = calibration_path_for(cal.player_class, cal.player_spec)
            try:
                save_calibration(cal, target)
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
            # Per-action tracked/untracked list — what the engine can reason
            # about for this character vs what will fall through to the
            # spell's default phrase.
            matched_ids = {
                ic.spell_id for ic in cal.cooldown_icons if ic.spell_id is not None
            }
            self._log_per_action_match_state(cal, matched_ids)
        self._refresh_button_state()

    def _refresh_button_state(self) -> None:
        """Enable/disable buttons based on what's actually available.

        Roster requires a party_region (otherwise "Load party members" has
        nothing to OCR against). Clear-calibration only makes sense when
        there's a calibration to clear. Spec dropdown is greyed when no
        class is picked. Calibrate stays on always — it's how you bootstrap
        the rest. Run after every state change (init, apply, clear,
        character switch) so the affordance is always current.
        """
        has_class_and_spec = bool(
            self._class_combo.currentData() and self._spec_combo.currentData()
        )
        has_party_region = bool(
            self._calibration and self._calibration.party_region
        )
        has_any_calibration = self._calibration is not None and bool(
            self._calibration.party_region
            or self._calibration.cooldown_region
            or self._calibration.cooldown_icons
            or self._calibration.party_members
        )

        # Roster needs a party_region (the saved bbox to OCR within).
        self._roster_btn.setEnabled(has_party_region)
        if not has_class_and_spec:
            self._roster_btn.setToolTip("Pick Class + Spec, then Calibrate first.")
        elif not has_party_region:
            self._roster_btn.setToolTip(
                "Calibrate first so we know where the party frames are."
            )
        else:
            self._roster_btn.setToolTip(
                "Edit the per-run roster. 'Load party members' reads names "
                "from the saved party region — no re-calibration needed."
            )

        # Clear calibration only when there's something to clear.
        self._clear_cal_btn.setEnabled(has_any_calibration)
        if not has_any_calibration:
            self._clear_cal_btn.setToolTip(
                "No calibration loaded — nothing to clear."
            )
        else:
            self._clear_cal_btn.setToolTip(
                "Forget the saved calibration and drop cooldown "
                "recommendations. Your dungeon callouts stay. Recalibrate "
                "anytime."
            )

        # Spec dropdown is meaningless without a class picked.
        self._spec_combo.setEnabled(bool(self._class_combo.currentData()))

    def _update_status(self) -> None:
        """Refresh the status bar + preview placeholder to show the current
        mode at a glance: which dungeon's callouts are active and whether
        cooldown recommendations are on. Also the empty-state guidance when no
        dungeon is picked yet."""
        dungeon = self._dungeon_combo.currentData() or (
            self._calibration.dungeon_name if self._calibration else None
        )
        if not dungeon:
            self._calibration_status.setText("No dungeon selected — pick one to start  ▸")
            self.frame_widget.set_placeholder(
                "Pick a dungeon above to start getting callouts.\n"
                "Optional: click Calibrate to add cooldown recommendations."
            )
            return

        cal = self._calibration
        recs_on = bool(
            cal and cal.player_class and cal.player_spec
            and self._suggestions_cb.isChecked()
        )
        if recs_on:
            spec = f"{cal.player_spec} {cal.player_class}".replace("_", " ").title()
            n = len(cal.roster())
            rec_text = f"Recommendations: {spec}" + (f"  ({n} party)" if n else "")
        elif not self._suggestions_cb.isChecked():
            rec_text = "Recommendations: off (Suggestions unchecked)"
        else:
            rec_text = "Recommendations: off — Calibrate to enable"
        self._calibration_status.setText(f"Callouts: {dungeon}      ·      {rec_text}")
        self.frame_widget.set_placeholder(f"Watching for cast bars in {dungeon}…")
