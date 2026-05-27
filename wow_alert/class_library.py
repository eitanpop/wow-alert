"""Per-class+spec action catalog.

The class library decouples *what the spell wants* (a category of response —
defensive, heal, dispel, ...) from *what tools the current player has* (BoP,
Cocoon, Ironbark, ...). One file per class+spec under
`config/classes/<class>/<spec>.yaml`. Authored once, reused across every
dungeon.

Example (`config/classes/paladin/holy.yaml`):

    class: paladin
    spec: holy

    actions:
      - id: bop
        label: "BOP"                 # what the alert says aloud
        category: defensive          # broad action type
        scope: single_target         # who it affects
        tags: [aggro_dropping]       # nuance — rules can filter on this
        cooldown_icon: bop           # links to a cooldown_icon.action from calibration

      - id: devotion_aura
        label: "Devotion Aura"
        category: defensive
        scope: party_wide
        cooldown_icon: devotion_aura

Three axes describe an action:
  - `category` — controlled vocab: `defensive`, `heal`, `dispel`,
    `interrupt`, `cc`, `stop`. Validated against ALLOWED_CATEGORIES.
  - `scope` — controlled vocab: `self`, `single_target`, `party_wide`,
    `raid_wide`. Validated against ALLOWED_SCOPES.
  - `tags` — free-form list of strings; rules use `has_tag` / `lacks_tag`
    predicates to filter. Add tags as you author rules that need them.

`cooldown_icon` is the join key to calibration: the OpenCV cooldown
watcher reports availability keyed by the icon's action name, and rule
priorities with class-action filters skip actions whose cooldown is
non-zero.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


ALLOWED_CATEGORIES = {"defensive", "heal", "dispel", "interrupt", "cc", "stop"}
ALLOWED_SCOPES = {"self", "single_target", "party_wide", "raid_wide"}


class ClassAction(BaseModel):
    """One ability the player can use against an incoming cast."""

    id: str = Field(description="Stable identifier; referenced by rule 'do' fields.")
    label: str = Field(description="Short TTS-friendly name. Spoken when the action fires.")
    category: str = Field(description="One of ALLOWED_CATEGORIES.")
    scope: str = Field(description="One of ALLOWED_SCOPES.")
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags rules can filter on (e.g. 'aggro_dropping').",
    )
    cooldown_icon: str = Field(
        description=(
            "Joins to a CooldownIcon.action from calibration. The cooldown "
            "watcher writes availability under this key; priorities with "
            "class-action filters skip actions whose value is non-zero."
        ),
    )

    @field_validator("category")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        if v not in ALLOWED_CATEGORIES:
            raise ValueError(
                f"category must be one of {sorted(ALLOWED_CATEGORIES)}, got {v!r}"
            )
        return v

    @field_validator("scope")
    @classmethod
    def _validate_scope(cls, v: str) -> str:
        if v not in ALLOWED_SCOPES:
            raise ValueError(
                f"scope must be one of {sorted(ALLOWED_SCOPES)}, got {v!r}"
            )
        return v


class ClassActions(BaseModel):
    """File-shaped wrapper. The `class` field is reserved in Python, so the
    YAML key `class` maps to the model attribute `character_class`."""

    character_class: str = Field(alias="class")
    spec: str
    actions: list[ClassAction] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


def load_class_actions(
    config_dir: Path,
    player_class: str | None,
    player_spec: str | None,
) -> list[ClassAction]:
    """Load `config/classes/<player_class>/<player_spec>.yaml`.

    Returns an empty list when player_class/spec aren't set or the file
    doesn't exist (with a logged warning) — the rule engine will fall
    back to default Alerts in that case, which is correct behavior for
    "we don't know what tools you have".
    """
    if not player_class or not player_spec:
        logger.info(
            "No player_class/spec configured — class library not loaded; "
            "rule priorities with class-action filters will not match."
        )
        return []
    path = config_dir / "classes" / player_class / f"{player_spec}.yaml"
    if not path.exists():
        logger.warning(
            "Class library file not found at %s — no actions available for %s/%s",
            path, player_class, player_spec,
        )
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ClassActions.model_validate(raw)
    logger.info(
        "Loaded %d actions for %s/%s from %s",
        len(cfg.actions), cfg.character_class, cfg.spec, path,
    )
    return cfg.actions
