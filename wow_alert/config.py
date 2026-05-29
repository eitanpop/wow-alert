"""Application configuration: YAML on disk, pydantic model in memory, CLI overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from wow_alert.paths import TTS_CACHE_DIR


REPO_ROOT = Path(__file__).resolve().parent.parent


class AppConfig(BaseModel):
    """Runtime configuration loaded from YAML, env vars, and CLI overrides."""

    window_title: str = Field(
        default="World of Warcraft",
        description="Exact title of the window to capture.",
    )
    model_path: Path = Field(
        description="Filesystem path to YOLO weights (.pt or .engine).",
    )
    confidence: float = Field(
        default=0.4,
        description="Minimum YOLO detection confidence for a box to be kept.",
    )
    imgsz: int = Field(
        default=1280,
        ge=64,
        description=(
            "YOLO inference resolution (long side, pixels). Must match the "
            "value the model was trained at to avoid distribution shift."
        ),
    )
    window_refresh_interval: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Seconds between re-checks of the captured window's screen "
            "position and size. Larger = less overhead; smaller = faster "
            "reaction to the window being moved or resized."
        ),
    )
    target_fps: int = Field(
        default=10,
        ge=1,
        le=120,
        description=(
            "Upper bound on pipeline tick rate. Cast bars last 1-10 s, so 10 "
            "FPS is plenty to catch them; capping leaves GPU headroom for "
            "the game itself. Raise carefully — this stack will happily peg "
            "the GPU if uncapped."
        ),
    )
    lost_track_tolerance_s: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "How long an existing track survives without a detection before "
            "being dropped. Tuned in seconds so it scales correctly with "
            "`target_fps` — dropped frame count is computed as "
            "round(lost_track_tolerance_s * target_fps). Raise if the model "
            "frequently blips out on the same cast bar."
        ),
    )
    alert_dedupe_default_ttl_s: float = Field(
        default=5.0,
        description=(
            "When a cast event has no parsed duration, suppress repeats of "
            "the same (spell_id, target) for this many seconds."
        ),
    )
    alert_dedupe_max_ttl_s: float = Field(
        default=10.0,
        description=(
            "Hard upper bound on the dedupe TTL for casts matched in the "
            "spell DB. The matched path trusts `Spell.duration`, but this "
            "guards against a DB entry with a nonsensical duration or an "
            "OCR'd duration that survived the parser. 10s matches the "
            "unmatched cap — most WoW casts complete inside that window, "
            "and re-alerting is preferable to missing alerts when a spell "
            "casts twice in quick succession."
        ),
    )
    alert_dedupe_unmatched_max_ttl_s: float = Field(
        default=10.0,
        description=(
            "Hard upper bound on the dedupe TTL for casts that don't match "
            "any entry in the spell DB. The TTL source is the OCR'd duration "
            "(unreliable), so the cap is intentionally short — better to "
            "re-register the same cast than to suppress a genuinely new "
            "one for too long."
        ),
    )
    ocr_thread_cap: int = Field(
        default=2,
        description=(
            "Maximum CPU threads ONNX Runtime can use for RapidOCR. The "
            "default of 2 leaves room for the game; bump it if OCR latency "
            "is unacceptable and CPU headroom exists."
        ),
    )
    show_preview: bool = Field(
        default=True,
        description=(
            "Whether the live annotated-frame preview pane is shown at "
            "startup. Turning it off skips the per-tick cross-thread frame "
            "copy that the preview requires. Runtime-toggleable from the UI."
        ),
    )

    tts_engine: str = Field(
        default="edge",
        description=(
            "TTS backend for alert audio. 'edge' = Microsoft Edge neural "
            "voices (edge-tts) — natural-sounding, needs internet at "
            "prerender time. 'pyttsx' = offline Windows SAPI voice (robotic "
            "but no network). Falls back to spell phrases either way."
        ),
    )
    tts_voice: str = Field(
        default="en-US-AriaNeural",
        description=(
            "edge-tts voice name (ignored when tts_engine='pyttsx'). Examples: "
            "en-US-AriaNeural, en-US-GuyNeural, en-GB-SoniaNeural. Full list: "
            "`edge-tts --list-voices`."
        ),
    )

    tts_cache_dir: Path = Field(
        default_factory=lambda: TTS_CACHE_DIR,
        description=(
            "Directory for TTS-pre-rendered phrase WAVs. Defaults to the OS "
            "user-data directory (%LOCALAPPDATA%\\wow-alert\\tts_cache on "
            "Windows) — generated artifacts don't belong in the project "
            "tree. Override only if you have a specific reason. Safe to "
            "delete the contents to force re-rendering."
        ),
    )

    fuzzy_threshold: int = Field(
        default=85,
        description=(
            "rapidfuzz token_set_ratio threshold (0-100). Applied to both "
            "spell-name and target-name lookups."
        ),
    )

    dungeon: str | None = Field(
        default=None,
        description=(
            "Pre-calibration dungeon hint. The active calibration's "
            "dungeon_name supersedes this once one is loaded."
        ),
    )

    @field_validator("model_path", "tts_cache_dir", mode="before")
    @classmethod
    def _to_path(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v


def load_config(yaml_path: Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load AppConfig from YAML, then apply env vars and explicit overrides.

    Precedence (low to high): YAML file, env vars, explicit overrides (CLI).
    """
    raw: dict[str, Any] = {}
    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    env_map = {
        "WOW_ALERT_MODEL": "model_path",
        "WOW_ALERT_WINDOW_TITLE": "window_title",
    }
    for env_key, field_name in env_map.items():
        if env_key in os.environ:
            raw[field_name] = os.environ[env_key]

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    return AppConfig.model_validate(raw)
