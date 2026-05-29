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


def list_dungeon_names(config_dir: Path) -> list[str]:
    """Display names of all authored dungeons (each file's `dungeon:` header),
    sorted. Used to populate the calibration picker so names can't be
    mistyped — a picked name always slugs to a real file."""
    dungeons_dir = config_dir / "dungeons"
    if not dungeons_dir.exists():
        return []
    names: list[str] = []
    for p in sorted(dungeons_dir.glob("*.yaml")):
        if p.stem == "_global":
            continue
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            logger.warning("Could not read dungeon name from %s", p.name)
            continue
        name = raw.get("dungeon")
        if name:
            names.append(name)
    return sorted(names)


def _resolve_dungeon_slug(slug: str, dungeons_dir: Path) -> str | None:
    """Map a (possibly misspelled / OCR'd) slug to an on-disk dungeon file.

    Exact match wins; otherwise fuzzy-match against the available dungeon
    filenames so a typo still loads the right dungeon. Returns the matched
    slug, or None when nothing is close enough.
    """
    if (dungeons_dir / f"{slug}.yaml").exists():
        return slug
    choices = [p.stem for p in dungeons_dir.glob("*.yaml") if p.stem != "_global"]
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
    config_dir: Path,
    dungeon_name: str | None,
) -> tuple[list[Spell], list[dict]]:
    """Load the active-dungeon's spells + rules plus globals.

    Returns `(spells, rules)`. Logs a warning and returns empty lists if
    `config/dungeons/` doesn't exist — that's a misconfiguration; the
    rest of the app still runs but matches nothing.
    """
    dungeons_dir = config_dir / "dungeons"
    spells: list[Spell] = []
    rules: list[dict] = []

    if not dungeons_dir.exists():
        logger.warning(
            "No config/dungeons/ directory at %s — spell DB will be empty.",
            dungeons_dir,
        )
        return spells, rules

    global_path = dungeons_dir / "_global.yaml"
    if global_path.exists():
        cfg = _load_file(global_path)
        spells.extend(cfg.spells)
        rules.extend(cfg.rules)
        logger.info(
            "Loaded %d global spells / %d global rules from %s",
            len(cfg.spells), len(cfg.rules), global_path.name,
        )

    if dungeon_name:
        slug = slugify(dungeon_name)
        matched = _resolve_dungeon_slug(slug, dungeons_dir)
        if matched is not None:
            cfg = _load_file(dungeons_dir / f"{matched}.yaml")
            spells.extend(cfg.spells)
            rules.extend(cfg.rules)
            logger.info(
                "Loaded %d spells / %d rules for dungeon %r from %s.yaml",
                len(cfg.spells), len(cfg.rules), dungeon_name, matched,
            )
        else:
            available = sorted(
                p.stem for p in dungeons_dir.glob("*.yaml") if p.stem != "_global"
            )
            logger.warning(
                "No dungeon file matched %r (slug %r) — NO spell alerts will "
                "load, only globals. Check the dungeon name. Available: %s",
                dungeon_name, slug, available,
            )

    return spells, rules


def _load_file(path: Path) -> DungeonFile:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if isinstance(raw, list):
        # Allow a per-dungeon file to be just a spell list (no rules,
        # no header). Cheap shorthand for files with no dungeon-level
        # metadata to author.
        return DungeonFile(spells=[Spell.model_validate(e) for e in raw])
    return DungeonFile.model_validate(raw)
