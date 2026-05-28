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


def infer_class_spec(
    config_dir: Path,
    matched_spell_ids: set[int],
) -> tuple[str | None, str | None, int]:
    """Pick the class+spec whose library covers the most matched icons.

    Walks every `config/classes/<class>/<spec>.yaml`, counts how many of
    that file's spell IDs appear in `matched_spell_ids`, and returns the
    winner. Tie-broken by alphabetical (class, spec) order — stable
    across runs but otherwise arbitrary.

    Returns `(class, spec, match_count)` or `(None, None, 0)` if no
    class library matched any icon. The match count is exposed so the
    caller can log "matched 7/8 of paladin/holy.yaml → loading Holy".
    """
    classes_dir = config_dir / "classes"
    if not classes_dir.exists() or not matched_spell_ids:
        return None, None, 0
    best: tuple[int, str, str] | None = None
    for class_dir in sorted(classes_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for spec_path in sorted(class_dir.glob("*.yaml")):
            try:
                with spec_path.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                cfg = ClassActions.model_validate(raw)
            except Exception:
                logger.warning(
                    "Skipping malformed class library %s during inference",
                    spec_path,
                )
                continue
            library_ids = {a.spell_id for a in cfg.actions}
            count = len(library_ids & matched_spell_ids)
            if count == 0:
                continue
            if best is None or count > best[0]:
                best = (count, cfg.character_class, cfg.spec)
    if best is None:
        return None, None, 0
    return best[1], best[2], best[0]


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
