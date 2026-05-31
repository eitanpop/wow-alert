"""Load per-dungeon spell + rule definitions.

Layout under `config/dungeons/`:
  - `_global.yaml` — spells/rules that apply regardless of dungeon.
  - `<slug>.yaml` — one file per dungeon. The loader picks the file whose
    slug matches the active dungeon name (from calibration).

Each file is:

    dungeon: "Windrunner Spire"   # display name; omit for _global.yaml
    spells:
      - id: ...
        name: ...
        severity: danger
        ...
    rules: []                     # see wow_alert/rule_schema.py:Rule

The slug for a dungeon name is the lowercase string with non-alphanumeric
characters collapsed to underscores: "Mists of Tirna Scithe" →
"mists_of_tirna_scithe". The display name lives inside the YAML header so
filename slugging is presentation-only — display names round-trip exactly
from calibration to UI.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

from wow_alert.events import Spell

logger = logging.getLogger(__name__)


_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Minimum rapidfuzz score (0-100) to accept a fuzzy dungeon-slug match.
# High enough that an unrelated name fails (→ no dungeon, warned) but loose
# enough to absorb a typo or OCR slip ("nexus_point_xenis" → the real file).
_DUNGEON_MATCH_CUTOFF = 80


def slugify(name: str) -> str:
    """Convert a dungeon display name to a filesystem-safe slug.

    Idempotent (slugifying a slug yields the same string). Used to map
    `calibration.dungeon_name` to the on-disk filename.
    """
    return _SLUG_RE.sub("_", name.lower()).strip("_")


def _layered_dungeons_dirs() -> list[Path]:
    """Bundled + user dungeons dirs, in load order (defaults, then user).

    User entries take precedence per filename inside the union; this just
    returns where to look, not how to merge.
    """
    from wow_alert.paths import USER_CONFIG_DIR, defaults_config_dir
    return [defaults_config_dir() / "dungeons", USER_CONFIG_DIR / "dungeons"]


def _union_yaml_files(dirs: list[Path]) -> dict[str, Path]:
    """Map `<slug>.yaml` → effective on-disk path, later dirs winning.

    Walks each existing dir in order; a same-named file in a later dir
    replaces the earlier one. Used so user overrides shadow bundled
    defaults file-for-file while new files at either layer still load.
    """
    out: dict[str, Path] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.glob("*.yaml"):
            out[p.name] = p
    return out


def list_dungeon_names() -> list[str]:
    """Display names of all authored dungeons across bundled + user dirs.

    Returns the sorted set of `dungeon:` header values.
    """
    files = _union_yaml_files(_layered_dungeons_dirs())
    names: list[str] = []
    for name, p in files.items():
        if Path(name).stem == "_global":
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("Could not read dungeon name from %s", p)
            continue
        title = raw.get("dungeon")
        if title:
            names.append(title)
    return sorted(names)


def _resolve_dungeon_slug(slug: str, files: dict[str, Path]) -> str | None:
    """Map a (possibly misspelled / OCR'd) slug to an on-disk dungeon file.

    Works against the unioned filename → path map. Exact match wins;
    otherwise fuzzy-match against the available stems so a typo still
    loads the right dungeon. Returns the matched slug, or None when
    nothing is close enough.
    """
    if f"{slug}.yaml" in files:
        return slug
    choices = [Path(name).stem for name in files if Path(name).stem != "_global"]
    if not choices:
        return None
    hit = process.extractOne(
        slug, choices, scorer=fuzz.ratio, score_cutoff=_DUNGEON_MATCH_CUTOFF,
    )
    if hit is None:
        return None
    matched, score, _ = hit
    logger.info(
        "Dungeon slug %r fuzzy-matched to %r (score %.0f)", slug, matched, score,
    )
    return matched


class DungeonFile(BaseModel):
    """One per-dungeon (or _global) yaml file's parsed shape."""

    dungeon: str | None = None
    spells: list[Spell] = Field(default_factory=list)
    # Rules are passed through as opaque dicts; `RuleEngine.set_rules`
    # validates each against the `Rule` model and skips malformed entries
    # with a warning. Keeping them untyped here lets the loader survive
    # rule-schema typos without taking the whole dungeon file down.
    rules: list[dict] = Field(default_factory=list)


def load_dungeon_config(
    _legacy_config_dir: Path | None = None,
    dungeon_name: str | None = None,
) -> tuple[list[Spell], list[dict]]:
    """Load the active-dungeon's spells + rules plus globals.

    Returns `(spells, rules)`. Looks across bundled defaults + user
    overrides; user files replace same-named bundled files. Each loaded
    file logs whether it came from the bundled or user source so users
    can see what's actually in effect.

    `_legacy_config_dir` is accepted but ignored.
    """
    files = _union_yaml_files(_layered_dungeons_dirs())
    spells: list[Spell] = []
    rules: list[dict] = []

    if not files:
        logger.warning(
            "No dungeon yaml files found in bundled defaults or user "
            "override dir — spell DB will be empty.",
        )
        return spells, rules

    global_path = files.get("_global.yaml")
    if global_path is not None:
        cfg = _load_file(global_path)
        spells.extend(cfg.spells)
        rules.extend(cfg.rules)
        logger.info(
            "Loaded %d global spells / %d global rules from %s (%s)",
            len(cfg.spells), len(cfg.rules), global_path.name,
            _source_label(global_path),
        )

    if dungeon_name:
        slug = slugify(dungeon_name)
        matched = _resolve_dungeon_slug(slug, files)
        if matched is not None:
            path = files[f"{matched}.yaml"]
            cfg = _load_file(path)
            spells.extend(cfg.spells)
            rules.extend(cfg.rules)
            logger.info(
                "Loaded %d spells / %d rules for dungeon %r from %s.yaml (%s)",
                len(cfg.spells), len(cfg.rules), dungeon_name, matched,
                _source_label(path),
            )
        else:
            available = sorted(
                Path(name).stem for name in files
                if Path(name).stem != "_global"
            )
            logger.warning(
                "No dungeon file matched %r (slug %r) — NO spell alerts will "
                "load, only globals. Check the dungeon name. Available: %s",
                dungeon_name, slug, available,
            )

    return spells, rules


def _source_label(path: Path) -> str:
    """Tag a loaded file with whether it came from user overrides or
    bundled defaults, for INFO logs."""
    from wow_alert.paths import USER_CONFIG_DIR
    try:
        path.relative_to(USER_CONFIG_DIR)
        return "user override"
    except ValueError:
        return "bundled"


def _load_file(path: Path) -> DungeonFile:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if isinstance(raw, list):
        # Allow a per-dungeon file to be just a spell list (no rules,
        # no header). Cheap shorthand for files with no dungeon-level
        # metadata to author.
        return DungeonFile(spells=[Spell.model_validate(e) for e in raw])
    return DungeonFile.model_validate(raw)
