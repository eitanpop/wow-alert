"""Filesystem layout for the app.

Three zones, deliberately separate:

  Bundled defaults — ship inside the Python package. Read-only at runtime.
  Updated by upgrading the app:
      wow_alert/_defaults/tag_rules.yaml
      wow_alert/_defaults/dungeons/*.yaml
      wow_alert/_defaults/classes/<class>/<spec>.yaml

  User overrides — under the OS user-data dir. Mirror the bundled tree;
  any file present here replaces the same-named bundled file. New files
  here (e.g. a homebrew dungeon) layer on top of bundled content. Survives
  app upgrades; users own what they put here:
      <USER_DATA>/wow-alert/config/tag_rules.yaml
      <USER_DATA>/wow-alert/config/dungeons/*.yaml
      <USER_DATA>/wow-alert/config/classes/<class>/<spec>.yaml

  Generated content — runtime cache + per-character state. Safe to delete
  to force regeneration:
      <USER_DATA>/wow-alert/tts_cache/         TTS-rendered phrase WAVs
      <USER_DATA>/wow-alert/calibration_*.yaml UI calibration per spec
      <USER_DATA>/wow-alert/icons/             cooldown-matcher reference PNGs
      <USER_DATA>/wow-alert/calibration_artifacts/  debug captures

On Windows, USER_DATA is `%LOCALAPPDATA%`. Resolved via `platformdirs` so
the same code works on other OSes.
"""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir


APP_NAME = "wow-alert"

USER_DATA_DIR = Path(user_data_dir(APP_NAME, appauthor=False, roaming=False))
TTS_CACHE_DIR = USER_DATA_DIR / "tts_cache"
# Legacy single-file calibration. Kept for one-time migration into the
# per-spec format; new saves go to `calibration_<class>_<spec>.yaml`.
CALIBRATION_PATH = USER_DATA_DIR / "calibration.yaml"
CALIBRATION_ARTIFACTS_DIR = USER_DATA_DIR / "calibration_artifacts"
ICONS_DIR = USER_DATA_DIR / "icons"

# Where the user puts override files. Mirrors the bundled `_defaults` tree.
USER_CONFIG_DIR = USER_DATA_DIR / "config"


def defaults_config_dir() -> Path:
    """The bundled `_defaults` directory inside the wow_alert package.

    Resolves the same way for source checkouts and installed wheels
    because both flavors extract the package to a real filesystem path
    that `Path(module.__file__).parent` can address. No zipimport
    juggling needed for our distribution targets (PyInstaller, wheel).
    """
    import wow_alert  # avoids a circular import at module load time
    return Path(wow_alert.__file__).parent / "_defaults"


def calibration_path_for(
    player_class: str | None,
    player_spec: str | None,
) -> Path:
    """Per-spec calibration file path.

    Each `<class>_<spec>` gets its own file because the cooldown bar and
    abilities differ per spec — switching to a new spec loads a different
    calibration. `None` for either argument falls back to a sentinel
    "unknown" file so the app never crashes when class/spec isn't set.
    """
    if not player_class or not player_spec:
        return USER_DATA_DIR / "calibration_unknown.yaml"
    return USER_DATA_DIR / f"calibration_{player_class}_{player_spec}.yaml"


def ensure_user_data_dirs() -> None:
    """Create the user-data directory tree if it doesn't exist.

    Also scaffolds the user config override tree (`<USER_DATA>/wow-alert/
    config/dungeons/`, `<USER_DATA>/wow-alert/config/classes/<class>/`)
    so users see the structure to drop their files into, and drops a
    README explaining the override semantics on first launch. Safe to
    call repeatedly; only creates missing paths.
    """
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (USER_CONFIG_DIR / "dungeons").mkdir(parents=True, exist_ok=True)
    (USER_CONFIG_DIR / "classes").mkdir(parents=True, exist_ok=True)
    # Mirror the bundled `_defaults/classes/<class>/` subdirs so users
    # see where to drop spec yamls. Best-effort — failure here is a UX
    # nicety, not a runtime requirement, so don't bubble it up.
    defaults_classes = defaults_config_dir() / "classes"
    if defaults_classes.exists():
        for class_dir in defaults_classes.iterdir():
            if class_dir.is_dir():
                (USER_CONFIG_DIR / "classes" / class_dir.name).mkdir(
                    parents=True, exist_ok=True,
                )
    readme = USER_CONFIG_DIR / "README.txt"
    if not readme.exists():
        readme.write_text(_USER_CONFIG_README, encoding="utf-8")


_USER_CONFIG_README = """\
wow-alert user configuration overrides
======================================

Drop YAML files in this tree to override what ships with the app. The
loader checks here first for each file, then falls back to the bundled
defaults inside the package. New files (a homebrew dungeon, a custom
spec) layer on top of the bundled content; same-named files replace
the bundled version entirely.

Directory map:

  tag_rules.yaml          replaces bundled tag→priority table
  dungeons/<slug>.yaml    replaces or adds to bundled dungeon list
  classes/<class>/<spec>.yaml
                          replaces or adds to bundled class spec actions

Semantics are file-level, not entry-level. To tweak one rule inside a
dungeon you copy the whole file, edit, and the loader picks it up. The
trade-off: app updates that change the bundled version of a file you've
overridden won't reach your copy. You own what you put here.

Restart the app to pick up changes.
"""
