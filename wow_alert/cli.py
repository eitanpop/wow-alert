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

from wow_alert.audio import PyttsxWinsoundAlertPlayer
from wow_alert.capture import WindowCapture
from wow_alert.config import REPO_ROOT, AppConfig, load_config
from wow_alert.dedupe import CastDeduper
from wow_alert.detector import YoloDetector
from wow_alert.dungeon_loader import load_dungeon_config
from wow_alert.ocr import RapidOcrEngine
from wow_alert.paths import ensure_user_data_dirs
from wow_alert.pipeline import PipelineDeps, PipelineWorker
from wow_alert.rules import RuleEngine, YamlSpellDb, apply_counter_filter
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
    p.add_argument("--class", dest="player_class", default=None,
                   help="Player class; filters counter actions to those this class can perform.")
    p.add_argument("--spec", dest="player_spec", default=None,
                   help="Player spec; pairs with --class to filter counter actions.")
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
        apply_counter_filter(initial_spells, config.player_class, config.player_spec),
        fuzzy_threshold=config.fuzzy_threshold,
    )
    rule_engine = RuleEngine(spell_db)
    rule_engine.set_rules(initial_rules)
    deduper = CastDeduper(
        spell_db=spell_db,
        default_ttl_s=config.alert_dedupe_default_ttl_s,
        max_matched_ttl_s=config.alert_dedupe_max_ttl_s,
        max_unmatched_ttl_s=config.alert_dedupe_unmatched_max_ttl_s,
    )
    alert_player = PyttsxWinsoundAlertPlayer(cache_dir=config.tts_cache_dir)

    phrases = spell_db.all_phrases()
    if not phrases:
        phrases = ["DANGER"]
    logger.info("Prerendering TTS phrases: %s", phrases)
    alert_player.prerender(phrases)

    deps = PipelineDeps(
        capture=capture,
        detector=detector,
        tracker=tracker,
        ocr=ocr,
        deduper=deduper,
        rule_engine=rule_engine,
        alert_player=alert_player,
    )
    worker = PipelineWorker(
        deps,
        target_fps=config.target_fps,
        preview_enabled=config.show_preview,
    )
    def apply_calibration(roster: list[str], dungeon: str | None) -> None:
        """Push a fresh calibration into the components that use it.

        Wired from MainWindow's calibration flow: called once at startup
        when an existing calibration is loaded, and again whenever the
        user accepts a new calibration. Roster goes to both spell_db and
        deduper; dungeon triggers a reload from `config/dungeons/` so
        only that dungeon's spells (plus globals) are active.
        """
        spell_db.set_roster(roster)
        deduper.set_roster(roster)
        spells, rules = load_dungeon_config(config_dir, dungeon_name=dungeon)
        spell_db.replace_spells(
            apply_counter_filter(spells, config.player_class, config.player_spec)
        )
        rule_engine.set_rules(rules)

    window = MainWindow(
        worker,
        alert_player,
        show_preview=config.show_preview,
        on_calibration_apply=apply_calibration,
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
        "player_class": args.player_class,
        "player_spec": args.player_spec,
    }
    config = load_config(args.config, overrides=overrides)

    app, window = build_app(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
