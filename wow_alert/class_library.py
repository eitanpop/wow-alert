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
      - id: blessing_of_protection
        label: "BOP"                 # what the alert says aloud
        category: defensive          # broad action type
        scope: single_target         # who it affects
        tags: [aggro_dropping]       # nuance — rules can filter on this
        spell_id: 1022               # canonical WoW spell ID; joins to icon DB

      - id: devotion_aura
        label: "Devotion Aura"
        category: defensive
        scope: party_wide
        spell_id: 465

Three axes describe an action:
  - `category` — controlled vocab: `defensive`, `heal`, `dispel`,
    `interrupt`, `cc`, `stop`, `utility`. `utility` is for abilities that
    are NOT damage mitigation — snare/root breaks like Blessing of Freedom
    or Tiger's Lust — so damage-mitigation rules (`category: defensive`)
    don't bind them by accident. Each value validated against
    ALLOWED_CATEGORIES. May be a single value or a list when one ability
    serves several roles (e.g. Revival is both a party heal and a mass
    dispel: `category: [heal, dispel]`). A rule's `category` filter
    matches if it is among the action's categories.
  - `scope` — controlled vocab: `self`, `single_target`, `party_wide`,
    `raid_wide`. Validated against ALLOWED_SCOPES.
  - `tags` — free-form list of strings; rules use `has_tag` / `lacks_tag`
    predicates to filter. Add tags as you author rules that need them.

`spell_id` is the canonical WoW spell ID. It joins this action to the
icon database (one reference PNG per spell_id, in the user-data icons
dir; used by the calibration matcher to identify icons on the player's
cooldown manager) and to the
cooldown availability dict (rule engine skips actions whose spell_id is
currently on cooldown).
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


ALLOWED_CATEGORIES = {"defensive", "heal", "dispel", "interrupt", "cc", "stop", "utility"}
ALLOWED_SCOPES = {"self", "single_target", "party_wide", "raid_wide"}


class ClassAction(BaseModel):
    """One ability the player can use against an incoming cast."""

    id: str = Field(description="Stable identifier; referenced by rule 'do' fields.")
    label: str = Field(description="Short TTS-friendly name. Spoken when the action fires.")
    category: list[str] = Field(
        description=(
            "One or more of ALLOWED_CATEGORIES. Accepts a single string in "
            "YAML (normalized to a one-element list). A rule's category "
            "filter matches when it is among these."
        ),
    )
    scope: str = Field(description="One of ALLOWED_SCOPES.")
    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags rules can filter on (e.g. 'aggro_dropping').",
    )
    spell_id: int = Field(
        description=(
            "Canonical WoW spell ID. Joins this action to its reference icon "
            "PNG in the user-data icons dir (used by the calibration matcher) "
            "and to the cooldown availability dict (rule engine skips "
            "actions whose spell_id is on cooldown)."
        ),
    )

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, v: str | list[str]) -> list[str]:
        cats = [v] if isinstance(v, str) else list(v)
        bad = [c for c in cats if c not in ALLOWED_CATEGORIES]
        if bad:
            raise ValueError(
                f"category values must each be one of {sorted(ALLOWED_CATEGORIES)}, "
                f"got {bad!r}"
            )
        return cats

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


def _resolve_class_spec_path(
    player_class: str, player_spec: str,
) -> Path | None:
    """User override path if present, else bundled default, else None."""
    from wow_alert.paths import USER_CONFIG_DIR, defaults_config_dir
    rel = Path("classes") / player_class / f"{player_spec}.yaml"
    user = USER_CONFIG_DIR / rel
    if user.exists():
        return user
    bundled = defaults_config_dir() / rel
    if bundled.exists():
        return bundled
    return None


def _layered_class_spec_paths() -> dict[tuple[str, str], Path]:
    """Map `(class, spec)` → effective path, walking bundled + user dirs.

    User overrides take precedence per `<class>/<spec>.yaml` filename.
    Brand-new specs/classes the user adds also show up; this is what the
    inference walk needs to scan to consider every available library.
    """
    from wow_alert.paths import USER_CONFIG_DIR, defaults_config_dir
    out: dict[tuple[str, str], Path] = {}
    for root in (defaults_config_dir() / "classes", USER_CONFIG_DIR / "classes"):
        if not root.exists():
            continue
        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue
            for spec_path in class_dir.glob("*.yaml"):
                out[(class_dir.name, spec_path.stem)] = spec_path
    return out


def load_class_actions(
    player_class: str | None,
    player_spec: str | None,
) -> list[ClassAction]:
    """Load the class+spec actions, user override winning over bundled.

    Returns an empty list when player_class/spec aren't set or the file
    doesn't exist (with a logged warning) — the rule engine then falls
    back to default Alerts.
    """
    if not player_class or not player_spec:
        logger.info(
            "No player_class/spec configured — class library not loaded; "
            "rule priorities with class-action filters will not match."
        )
        return []
    path = _resolve_class_spec_path(player_class, player_spec)
    if path is None:
        logger.warning(
            "Class library file not found for %s/%s — no actions available",
            player_class, player_spec,
        )
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ClassActions.model_validate(raw)
    from wow_alert.paths import USER_CONFIG_DIR
    try:
        path.relative_to(USER_CONFIG_DIR)
        source = "user override"
    except ValueError:
        source = "bundled"
    logger.info(
        "Loaded %d actions for %s/%s from %s (%s)",
        len(cfg.actions), cfg.character_class, cfg.spec, path.name, source,
    )
    return cfg.actions
