"""Pydantic models for the per-dungeon rule schema.

YAML shape:

    rules:
      - on_cast:
          spell_id: windrunner_spire_arcane_salvo
          target_role: tank             # optional; filters which casts to match
        priorities:
          # Priorities are evaluated top to bottom. The first one whose
          # conditions all pass fires. Setting any class-action filter
          # (category / scope / has_tag / lacks_tag) binds the first
          # available action from the player's class library; the priority
          # fails if no such action is up. Cast filters (target_role /
          # target_present) gate independently. A priority with no
          # condition fields is the catch-all.
          - category: defensive
            scope: single_target
            lacks_tag: aggro_dropping
            say: "{action.label} {target}"
            do:  "{action.id}"          # optional; producing this -> Recommendation

          - category: defensive
            scope: party_wide
            say: "{action.label}"

          - say: "Tank Buster on {target}"

Templating tokens supported in `say` / `do` / `phrase`:
  {target}         canonical roster name, or raw OCR target, or ""
  {spell}          canonical spell.name
  {duration}       parsed duration like "3.0s" or "" when absent
  {action.label}   the matched ClassAction's label (set when any
                    class-action filter is present and matched)
  {action.id}      the matched ClassAction's id
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Priority(BaseModel):
    """One branch within a rule. Evaluated in order; first match wins.

    Two kinds of fields gate firing:

      - Class-action filters: category / scope / has_tag / lacks_tag.
        When any is set, the engine walks the class library in file
        order and binds the first action whose attributes satisfy all
        set filters and whose cooldown is ready. The priority fails if
        no such action is available — the rule walker moves to the next
        priority. The bound action becomes available to templates as
        {action.label} and {action.id}.

      - Cast filters: target_role / lacks_target_role / target_is_self /
        target_present / school. These check the cast itself and fail the
        priority if not satisfied. lacks_target_role only passes when the
        target's role is known and differs — an unknown role fails it, so an
        aggro-dropping external is never offered to a target that might be
        the tank. target_is_self compares the target to the configured
        player name; with no player name set it reads as "not self", so
        self-targeted priorities stay dormant until a name is configured.

    All set fields must pass (AND). A priority with no condition fields
    is the catch-all.
    """

    # Class-action filters.
    category: str | None = None
    scope: str | None = None
    has_tag: str | None = None
    lacks_tag: str | None = None

    # Cast filters.
    target_role: str | None = None       # require the target to play this role
    lacks_target_role: str | None = None  # require the target to NOT play this role
    target_is_self: bool | None = None   # require the target to be (true) / not be
                                         # (false) the player's own character
    target_present: bool | None = None   # require / forbid canonical_target
    school: str | None = None            # match the cast's school, e.g. "physical"

    # Output.
    say: str = Field(
        description=(
            "Templated string shown as the alert message. Supports "
            "{target}, {spell}, {duration}, {action.label}, {action.id}."
        ),
    )
    do: str | None = Field(
        default=None,
        description=(
            "Optional templated action id. When set, the engine emits a "
            "Recommendation (a 'go do this' callout) instead of a generic "
            "Alert. Leave unset for pure-informational priorities."
        ),
    )
    phrase: str | None = Field(
        default=None,
        description=(
            "Optional TTS phrase override. When set, this string (after "
            "templating) is the audible callout — useful for spell-specific "
            "guidance like 'Break Shield' that doesn't match any single "
            "action's label. Custom phrases get prerendered alongside "
            "spell defaults and action labels. When unset, Recommendation "
            "playback uses the matched action's label."
        ),
    )
    phrase_prefix: str | None = Field(
        default=None,
        description=(
            "Templated phrase played BEFORE the main phrase so the "
            "player hears CONTEXT first ('Arcane Salvo Devotion Aura' "
            "instead of just 'Devotion Aura'). Default behavior when "
            "this is unset: the spell's canonical name is used as the "
            "prefix. Set to the empty string to explicitly disable the "
            "prefix; set to a different template to override. Spell "
            "names are auto-included in the prerender set so the "
            "default works out of the box."
        ),
    )

    def has_action_filter(self) -> bool:
        """True if any class-action filter is set — i.e. this priority
        wants to bind a ClassAction before firing."""
        return any(
            f is not None
            for f in (self.category, self.scope, self.has_tag, self.lacks_tag)
        )


class OnCast(BaseModel):
    """Selects which casts a rule applies to."""

    spell_id: str = Field(description="The canonical spell id from spells:.")
    target_role: str | None = Field(
        default=None,
        description=(
            "When set, the rule only matches casts whose canonical target's "
            "role equals this value. More-specific rules (with target_role "
            "set) win over generic rules at decide time."
        ),
    )


class Rule(BaseModel):
    on_cast: OnCast
    priorities: list[Priority] = Field(default_factory=list)
