"""Filesystem layout for the app.

Two zones, deliberately separate:

  Authored content — lives in the project tree, git-tracked. Hand-edited by
  the user or maintained as code:
      config/app.yaml
      config/spells.yaml

  Generated content — lives in the OS user-data directory (NOT in the
  project tree, never git-tracked). Created and overwritten by the runtime;
  safe to delete to force regeneration:
      <USER_DATA>/wow-alert/tts_cache/        TTS-rendered phrase WAVs
      <USER_DATA>/wow-alert/calibration.yaml  LLM-derived screen layout (iter 2+)

On Windows, USER_DATA is `%LOCALAPPDATA%`. Resolved via `platformdirs` so
the same code works on other OSes if/when we go cross-platform.
"""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir


APP_NAME = "wow-alert"

USER_DATA_DIR = Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
TTS_CACHE_DIR = USER_DATA_DIR / "tts_cache"
CALIBRATION_PATH = USER_DATA_DIR / "calibration.yaml"
CALIBRATION_ARTIFACTS_DIR = USER_DATA_DIR / "calibration_artifacts"


def ensure_user_data_dirs() -> None:
    """Create the user-data directory tree if it doesn't exist.

    Safe to call repeatedly; only creates missing paths.
    """
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
