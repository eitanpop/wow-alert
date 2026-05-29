"""Entry point: parse args, load config, wire dependencies, launch the app."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load .env into os.environ before anything else so downstream code (the
# Anthropic SDK, anything reading env vars) sees the values without the user
# having to `export` them in their shell. Silent no-op if .env is absent.
from dotenv import load_dotenv

load_dotenv()

from PySide6.QtWidgets import QApplication

from wow_alert.audio import EdgeTtsAlertPlayer, PyttsxWinsoundAlertPlayer
from wow_alert.capture import WindowCapture
from wow_alert.config import REPO_ROOT, AppConfig, load_config
from wow_alert.calibration import Calibration
from wow_alert.class_library import load_class_actions
from wow_alert.cooldown_watcher import CooldownWatcher
from wow_alert.dedupe import CastDeduper
from wow_alert.detector import YoloDetector
from wow_alert.dungeon_loader import load_dungeon_config
from wow_alert.ocr import RapidOcrEngine
from wow_alert.paths import ensure_user_data_dirs
from wow_alert.pipeline import PipelineDeps, PipelineWorker
from wow_alert.rules import RuleEngine, YamlSpellDb
from wow_alert.tag_rules import load_tag_rules
from wow_alert.tracker import CastBarTracker
from wow_alert.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time WoW cast-bar awareness app.")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "app.yaml",
                   help="Path to app.yaml")
    p.add_argument("--model", dest="model_path", default=None)
    p.add_argument("--window-title", dest="window_title", default=None)
    p.add_argument("--confidence", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--target-fps", dest="target_fps", type=int, default=None,
                   help="Cap pipeline tick rate (frames/sec). Default 10.")
    p.add_argument("--dungeon", default=None,
                   help="Load only spells that match this dungeon (or have no dungeon set).")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def build_app(config: AppConfig) -> tuple[QApplication, MainWindow]:
    # Pass [] so QApplication doesn't try to parse wow-alert's own CLI flags.
    app = QApplication([])

    capture = WindowCapture(
        window_title=config.window_title,
        refresh_interval=config.window_refresh_interval,
    )
    detector = YoloDetector(
        model_path=config.model_path,
        confidence=config.confidence,
        imgsz=config.imgsz,
    )
    tracker = CastBarTracker(
        max_missed_frames=max(2, round(config.lost_track_tolerance_s * config.target_fps)),
    )
    ocr = RapidOcrEngine(thread_cap=config.ocr_thread_cap)
    # Initial spell/rule load. No dungeon at startup — the apply_calibration
    # callback below reloads with the active dungeon once calibration is
    # available (from disk on startup, or from a fresh Calibrate click).
    config_dir = REPO_ROOT / "config"
    initial_spells, initial_rules = load_dungeon_config(
        config_dir, dungeon_name=config.dungeon,
    )
    spell_db = YamlSpellDb(
        initial_spells,
        fuzzy_threshold=config.fuzzy_threshold,
    )
    rule_engine = RuleEngine()
    rule_engine.set_rules(initial_rules)
    rule_engine.set_tag_rules(load_tag_rules(config_dir))
    # Class actions are loaded per calibration, not at startup — see
    # apply_calibration below. Start empty so rule_engine.decide() falls
    # back to spell-default Alerts before the first calibration.
    deduper = CastDeduper(
        default_ttl_s=config.alert_dedupe_default_ttl_s,
        max_matched_ttl_s=config.alert_dedupe_max_ttl_s,
        max_unmatched_ttl_s=config.alert_dedupe_unmatched_max_ttl_s,
    )
    if config.tts_engine == "edge":
        alert_player = EdgeTtsAlertPlayer(
            cache_dir=config.tts_cache_dir, voice=config.tts_voice,
        )
    else:
        alert_player = PyttsxWinsoundAlertPlayer(cache_dir=config.tts_cache_dir)

    # Prerender what we have at startup — typically just the bare minimum
    # since class actions and dungeon-specific phrases are loaded by
    # apply_calibration. apply_calibration tops up the cache after each
    # calibration accept; prerender skips files already on disk.
    phrases = set(spell_db.all_phrases())
    phrases.discard("")
    if not phrases:
        phrases = {"DANGER"}
    phrases = sorted(phrases)
    logger.info("Prerendering TTS phrases: %s", phrases)
    alert_player.prerender(phrases)

    deps = PipelineDeps(
        capture=capture,
        detector=detector,
        tracker=tracker,
        ocr=ocr,
        spell_db=spell_db,
        deduper=deduper,
        rule_engine=rule_engine,
        alert_player=alert_player,
    )
    worker = PipelineWorker(
        deps,
        target_fps=config.target_fps,
        preview_enabled=config.show_preview,
    )
    def load_dungeon(dungeon_name: str | None) -> None:
        """Load a dungeon's spells + rules and prerender its callout phrases.

        This is the whole callouts path and needs no calibration — pick a
        dungeon and cast-bar alerts work. Cooldown recommendations layer on
        top via `apply_calibration` once the player calibrates (class library,
        cooldown icons, roster). Safe to call repeatedly / off the UI thread;
        prerender skips clips already on disk.
        """
        spells, rules = load_dungeon_config(config_dir, dungeon_name=dungeon_name)
        spell_db.replace_spells(spells)
        rule_engine.set_rules(rules)
        worker.set_dungeon(dungeon_name)

        needed: set[str] = set()
        needed.update(spell_db.all_phrases())   # spell default phrases
        needed.update(s.name for s in spells)   # spell names (prefix)
        needed.update(rule_engine.all_phrases())  # literal rule phrases
        needed.discard("")
        if needed:
            logger.info(
                "Prerendering %d dungeon phrase(s) for %r",
                len(needed), dungeon_name,
            )
            alert_player.prerender(sorted(needed))

    def apply_calibration(cal: Calibration) -> None:
        """Layer cooldown recommendations onto the active dungeon.

        Loads the dungeon (via `load_dungeon`), then pushes the calibration's
        class+spec, roster, roles, and player name into the rule engine,
        deduper, and pipeline context, and prerenders the extra TTS clips
        recommendations need (action labels, roster names). MainWindow handles
        the cooldown watcher separately.
        """
        roster = cal.roster()
        dungeon = cal.dungeon_name
        roles = cal.roles_by_name()

        # When either class or spec is null, `load_class_actions` returns
        # an empty list — priorities that bind a ClassAction fail, and
        # decide() falls back to spell-default Alerts.
        class_actions = load_class_actions(
            config_dir, cal.player_class, cal.player_spec,
        )
        rule_engine.set_class_actions(class_actions)

        # Surface actions whose spell_id wasn't matched to any icon on the
        # player's cooldown bar. The rule engine is fail-closed on missing
        # entries: untracked actions are treated as on cooldown and won't
        # be recommended. Rules referencing them fall through to the
        # spell's default phrase. The warning makes the gap visible so the
        # user can either add the ability to their cooldown manager and
        # recalibrate, or remove the action from the class library.
        matched_ids = {
            ic.spell_id for ic in cal.cooldown_icons if ic.spell_id is not None
        }
        unmatched = [a for a in class_actions if a.spell_id not in matched_ids]
        for action in unmatched:
            logger.warning(
                "Action %r (spell_id=%d) has no matching cooldown icon — "
                "rules that would bind it will fall through to the spell's "
                "default phrase. Add it to your cooldown manager and "
                "recalibrate to enable tracking.",
                action.id, action.spell_id,
            )

        # Include the player's own name in the target-matching roster so a
        # cast on the player passes the target gate and canonicalizes to that
        # name — that's what lets `target_is_self` rules fire. The roster sent
        # to the engine context stays the party (the player carries no role).
        match_roster = list(roster)
        if cal.player_name and cal.player_name not in match_roster:
            match_roster.append(cal.player_name)
        spell_db.set_roster(match_roster)
        deduper.set_roster(match_roster)

        load_dungeon(dungeon)  # spells, rules, dungeon context, dungeon phrases
        worker.update_calibration_context(roster, dungeon, roles, cal.player_name)

        # Prerender the extra clips recommendations add on top of the dungeon
        # phrases: class-action labels and roster names. The pipeline stitches
        # these at playback into callouts like "Arcane Salvo Sac Captain
        # Garrick". prerender skips clips already on disk.
        needed = {a.label for a in class_actions} | set(roster)
        needed.discard("")
        if needed:
            logger.info("Catching up TTS cache for %d recommendation clip(s)", len(needed))
            alert_player.prerender(sorted(needed))

    def clear_calibration(dungeon_name: str | None) -> None:
        """Drop the recommendation layer — class actions, roster, roles, and
        player name — back to nothing, while keeping the active dungeon's
        callouts. The UI pairs this with deleting the saved calibration file.
        Cheap (no prerender): the dungeon's phrases are already cached."""
        rule_engine.set_class_actions([])
        spell_db.set_roster([])
        deduper.set_roster([])
        worker.update_calibration_context([], dungeon_name, {}, None)
        logger.info("Calibration cleared; callouts-only for dungeon %r", dungeon_name)

    # CooldownWatcher reads frames from the worker and pushes cooldown
    # availability back via worker.set_cooldowns. MainWindow starts/stops it.
    cooldown_watcher = CooldownWatcher(worker)

    window = MainWindow(
        worker,
        alert_player,
        show_preview=config.show_preview,
        on_calibration_apply=apply_calibration,
        on_dungeon_select=load_dungeon,
        on_clear_calibration=clear_calibration,
        cooldown_watcher=cooldown_watcher,
    )
    window.start()
    return app, window


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_user_data_dirs()

    overrides = {
        "model_path": args.model_path,
        "window_title": args.window_title,
        "confidence": args.confidence,
        "imgsz": args.imgsz,
        "target_fps": args.target_fps,
        "dungeon": args.dungeon,
    }
    config = load_config(args.config, overrides=overrides)

    app, window = build_app(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
