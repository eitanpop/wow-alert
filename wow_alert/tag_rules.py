"""Global tag → priority mapping for the rule engine.

A matched cast carries mechanic `tags` (e.g. 'interrupt', 'big_damage_single',
'big_damage_party'). Instead of every dungeon hand-writing a near-identical
rule per spell, the behavior lives once here: each tag maps to an ordered
list of `Priority` objects, and `priorities_for()` concatenates the chains
of a cast's tags — in a fixed precedence order — into a single list the
engine walks like any rule.

Loaded from `config/tag_rules.yaml`. See that file for the authored table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from wow_alert.rule_schema import Priority

logger = logging.getLogger(__name__)


class TagRules(BaseModel):
    """The tag→priorities table plus the precedence order for multi-tag casts."""

    precedence: list[str] = Field(default_factory=list)
    tags: dict[str, list[Priority]] = Field(default_factory=dict)

    def priorities_for(self, tags: list[str]) -> list[Priority]:
        """Ordered priority list for a cast carrying `tags`.

        Tags are visited in `precedence` order (a tag not listed in
        precedence is visited last, after the known ones, in its given
        order). Each tag contributes its priority chain; the chains are
        concatenated. Unknown tags (not in the table) contribute nothing.
        """
        rank = {name: i for i, name in enumerate(self.precedence)}
        ordered = sorted(tags, key=lambda t: rank.get(t, len(rank)))
        out: list[Priority] = []
        for tag in ordered:
            out.extend(self.tags.get(tag, []))
        return out


def load_tag_rules(_legacy_config_dir: Path | None = None) -> TagRules:
    """Load `tag_rules.yaml`, user override winning over bundled default.

    `_legacy_config_dir` is accepted but ignored — resolution goes
    through `paths`:

      1. `<USER_DATA>/wow-alert/config/tag_rules.yaml` (if present)
      2. `wow_alert/_defaults/tag_rules.yaml` (bundled)
      3. Empty table (with warning) if both are missing

    A loaded user override emits an INFO log so users can confirm which
    version they're running.
    """
    from wow_alert.paths import USER_CONFIG_DIR, defaults_config_dir

    user_path = USER_CONFIG_DIR / "tag_rules.yaml"
    default_path = defaults_config_dir() / "tag_rules.yaml"
    if user_path.exists():
        path = user_path
        source = "user override"
    elif default_path.exists():
        path = default_path
        source = "bundled default"
    else:
        logger.warning(
            "tag_rules.yaml not found at %s or %s — tag-driven "
            "recommendations disabled; casts will fall through to "
            "their default phrases.",
            user_path, default_path,
        )
        return TagRules()
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    table = TagRules.model_validate(raw)
    logger.info(
        "Loaded tag rules (%s): %d tags, precedence=%s",
        source, len(table.tags), table.precedence,
    )
    return table
