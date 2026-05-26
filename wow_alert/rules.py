"""Spell database + rule engine.

`YamlSpellDb` filters at load time by dungeon and `(class, spec)` so only relevant
entries enter the matcher. Fuzzy matching uses rapidfuzz's `token_set_ratio` on
both the spell and target text. Either side missing the threshold returns None,
which the rule engine treats as "skip" — fail-closed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import yaml
from rapidfuzz import fuzz, process

from wow_alert.events import (
    Alert,
    Recommendation,
    RuleDecisionContext,
    RuleOutput,
    ScreenContext,
    Severity,
    Spell,
)

logger = logging.getLogger(__name__)


class YamlSpellDb:
    # Built-in: WoW's "Interrupted!" cast-bar text appears after any successful
    # kick. OCR truncation across this string varies wildly ("Inter",
    # "rrupted", "Interrupted on Austin", "rrupted [Captai") so we recognize
    # it via partial-ratio fuzzy matching rather than relying on user-
    # authored aliases. Severity IGNORE — the cast was already stopped, no
    # alert needed — but registration still fires so the operator sees
    # confirmation in the log and dedupe suppresses repeat OCR hits on the
    # same interrupt animation.
    #
    # User-authored spells in spells.yaml take priority; this only matches
    # when nothing else does.
    _INTERRUPTED_BUILTIN = Spell(
        id="_builtin_interrupted",
        name="Interrupted",
        severity=Severity.IGNORE,
        phrase="DANGER",
        duration=2.0,  # the on-screen "Interrupted!" lingers ~2s
    )

    def __init__(
        self,
        spells: Iterable[Spell],
        fuzzy_threshold: int = 85,
        roster: Iterable[str] | None = None,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self._roster: list[str] = []
        self._roster_lower: list[str] = []
        self.set_roster(roster or [])

        # `_spells` and the lookup indexes are rebuilt by `replace_spells`,
        # which is also how calibration swaps the active dungeon's spell
        # set in. The constructor is just the first call.
        self._spells: list[Spell] = []
        self._name_index: dict[str, Spell] = {}
        self._all_names: list[str] = []
        self.replace_spells(spells)

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        player_class: str | None = None,
        player_spec: str | None = None,
        fuzzy_threshold: int = 85,
        roster: Iterable[str] | None = None,
    ) -> "YamlSpellDb":
        """Load a flat list-of-spells YAML file.

        Kept for compatibility with the original schema; new code should
        use `wow_alert.dungeon_loader.load_dungeon_config` + `replace_spells`
        to load from `config/dungeons/`. Class/spec filtering on counters
        applies regardless of which loader you use.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or []
        spells = [Spell.model_validate(entry) for entry in raw]
        return cls(
            apply_counter_filter(spells, player_class, player_spec),
            fuzzy_threshold=fuzzy_threshold,
            roster=roster,
        )

    def all_phrases(self) -> list[str]:
        return sorted({s.phrase for s in self._spells})

    def set_roster(self, names: Iterable[str]) -> None:
        """Update the roster used by `_target_ok`'s fuzzy-match check.

        Live-mutable so calibration can refresh the roster mid-session
        without an app restart.
        """
        self._roster = list(names)
        self._roster_lower = [n.lower() for n in self._roster]

    def replace_spells(self, spells: Iterable[Spell]) -> None:
        """Swap in a new active spell set and rebuild lookup indexes.

        Called once at construction (with the initial set) and again
        whenever the dungeon-loader produces a fresh list — e.g., after
        the user accepts a new calibration that changes the active
        dungeon. Roster is unaffected.
        """
        self._spells = list(spells)
        index: dict[str, Spell] = {}
        for spell in self._spells:
            index[spell.name.lower()] = spell
            for alias in spell.aliases:
                index[alias.lower()] = spell
        self._name_index = index
        self._all_names = list(index.keys())
        logger.info("Spell DB now holds %d spells", len(self._spells))

    def lookup(self, spell_text: str, target_text: str | None) -> Spell | None:
        spell = self._lookup_spell(spell_text)
        if spell is None:
            return None
        if not self._target_ok(target_text):
            return None
        return spell

    # ---- internals ----

    def _lookup_spell(self, spell_text: str) -> Spell | None:
        if not spell_text:
            return None
        lower = spell_text.lower()

        # User-defined entries first.
        exact = self._name_index.get(lower)
        if exact is not None:
            return exact
        if self._all_names:
            match = process.extractOne(
                lower,
                self._all_names,
                scorer=fuzz.token_set_ratio,
                score_cutoff=self.fuzzy_threshold,
            )
            if match is not None:
                matched_name, _score, _idx = match
                return self._name_index[matched_name]

        # Built-in fallback: catch "Interrupted!" in any of its OCR-mangled
        # forms. Three checks, from cheap to permissive:
        #   1. Clean substring — covers "Interrupted on X", "Interrupting".
        #   2. `partial_ratio` — fuzzy substring, covers small OCR edits.
        #   3. First-token fuzzy — covers heavy truncation like "Inter on
        #      stin" where "rupted" is gone entirely but "Inter" survives.
        if "interrupt" in lower or "rrupt" in lower:
            return self._INTERRUPTED_BUILTIN
        if fuzz.partial_ratio("interrupted", lower) >= 75:
            return self._INTERRUPTED_BUILTIN
        first_token = lower.split(maxsplit=1)[0] if lower else ""
        if first_token and fuzz.ratio("interrupted", first_token) >= 60:
            return self._INTERRUPTED_BUILTIN

        return None

    def _target_ok(self, target_text: str | None) -> bool:
        """Fuzzy check on an OCR'd target name against the configured roster.

        Three cases:
        - No target detected by OCR → pass. Many cast bars have no target field
          visible; the spell match alone is sufficient.
        - Target detected, no roster configured → pass. Without a known set of
          valid target names there is nothing to validate against, so any
          non-empty target string is accepted.
        - Target detected, roster configured → require fuzzy match against the
          roster at the configured threshold. A miss fails the lookup, so the
          rule engine produces no output. This is the fail-closed path that
          prevents false alerts when OCR garbles a target name into something
          unrelated.
        """
        if not target_text:
            return True
        if not self._roster_lower:
            return True
        match = process.extractOne(
            target_text.lower(),
            self._roster_lower,
            scorer=fuzz.token_set_ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        return match is not None


def apply_counter_filter(
    spells: Iterable[Spell],
    player_class: str | None,
    player_spec: str | None,
) -> list[Spell]:
    """Prune each spell's `counters` list to entries matching the given
    class+spec. No-op when either is None (load every counter; rule engine
    will sort them out at decide-time)."""
    spells = list(spells)
    if player_class is None or player_spec is None:
        return spells
    for spell in spells:
        spell.counters = [
            c for c in spell.counters
            if c.character_class == player_class and c.spec == player_spec
        ]
    return spells


class RuleEngine:
    """Decides what to do for one matched, deduped cast.

    `decide(RuleDecisionContext)` is the primary entry point and a pure
    function — no spell-DB lookup, no dedupe, no temporal state. Upstream
    (pipeline + deduper) does all the resolution and hands the engine a
    fully-populated context. The engine then returns a single
    `RuleOutput` (Alert, Recommendation) or `None` to suppress.

    This shape is intentional: it makes the engine the place where game
    policy lives — "if a counter is available, prefer Recommendation",
    "if dungeon is X and severity is info, suppress", etc. — without
    those policies needing to know anything about how spells get matched
    or how dedupe works.

    `evaluate(ScreenContext)` is kept as a compatibility shim for callers
    that haven't migrated to the explicit context yet (and for tests that
    exercise lookup + alert in one shot).
    """

    def __init__(self, spell_db: YamlSpellDb):
        self.spell_db = spell_db
        # Authored per-dungeon rules from `dungeons/*.yaml`. Stored as
        # opaque dicts for now — `decide()` doesn't read them yet, but
        # plumbing them through means you can start authoring rules and
        # not lose any when Phase E begins interpreting them.
        self._rules: list[dict] = []

    def set_rules(self, rules: list[dict]) -> None:
        """Replace the active rule set (e.g., on dungeon change). No-op
        on `decide()` until Phase E wires up rule interpretation."""
        self._rules = list(rules)
        logger.info("Rule engine now holds %d rules", len(self._rules))

    def decide(self, ctx: RuleDecisionContext) -> RuleOutput | None:
        """Pure policy: given a matched cast and full context, return the
        single output to emit (or None to suppress).

        Iteration-1 logic is deliberately thin:
          - severity IGNORE → suppress
          - any available_counters → Recommendation (using the first one;
            Phase E will introduce priority/selection logic)
          - otherwise → Alert with severity/phrase from the spell

        Tweak this method to add policy; everything upstream is unchanged.
        """
        spell = ctx.spell
        if spell.severity == Severity.IGNORE:
            return None

        target_str = f" on {ctx.cast.target}" if ctx.cast.target else ""
        duration_str = (
            f" ({ctx.cast.duration:.1f}s)" if ctx.cast.duration is not None else ""
        )
        message = f"{spell.name}{target_str}{duration_str}"

        if ctx.available_counters:
            counter = ctx.available_counters[0]
            return Recommendation(
                action=counter.action,
                target=ctx.canonical_target or ctx.cast.target or "",
                phrase=counter.action,  # Phase D's WAV concat will combine action+target
                message=f"{counter.action}{target_str} (for {spell.name})",
            )

        return Alert(
            severity=spell.severity,
            phrase=spell.phrase,
            message=message,
        )

    def evaluate(self, ctx: ScreenContext) -> list[RuleOutput]:
        """Compatibility shim. Looks up each cast event in the DB, builds a
        `RuleDecisionContext`, and dispatches to `decide()`. New callers
        should build the context directly — this path duplicates lookup
        work the deduper already did.
        """
        outputs: list[RuleOutput] = []
        for event in ctx.cast_events:
            spell = self.spell_db.lookup(event.spell, event.target)
            if spell is None:
                continue
            decision_ctx = RuleDecisionContext(
                spell=spell,
                cast=event,
                canonical_target=event.target,
                roster=list(ctx.roster),
                dungeon=ctx.dungeon,
                player_class=ctx.player_class,
                player_spec=ctx.player_spec,
                cooldowns=dict(ctx.cooldowns),
            )
            output = self.decide(decision_ctx)
            if output is not None:
                outputs.append(output)
        return outputs
