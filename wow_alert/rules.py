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

from wow_alert.events import Alert, RuleOutput, ScreenContext, Severity, Spell

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
        self._spells: list[Spell] = list(spells)
        self.fuzzy_threshold = fuzzy_threshold
        self._roster: list[str] = list(roster) if roster else []
        self._roster_lower: list[str] = [name.lower() for name in self._roster]

        # Flat name-to-spell index, including aliases.
        self._name_index: dict[str, Spell] = {}
        for spell in self._spells:
            self._name_index[spell.name.lower()] = spell
            for alias in spell.aliases:
                self._name_index[alias.lower()] = spell
        self._all_names: list[str] = list(self._name_index.keys())

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        dungeon: str | None = None,
        player_class: str | None = None,
        player_spec: str | None = None,
        fuzzy_threshold: int = 85,
        roster: Iterable[str] | None = None,
    ) -> "YamlSpellDb":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or []
        spells = [Spell.model_validate(entry) for entry in raw]
        filtered = [
            s for s in spells
            if s.dungeon is None or dungeon is None or s.dungeon == dungeon
        ]
        if player_class is not None and player_spec is not None:
            for spell in filtered:
                spell.counters = [
                    c for c in spell.counters
                    if c.character_class == player_class and c.spec == player_spec
                ]
        logger.info(
            "Loaded %d spells (filtered from %d) — dungeon=%r class=%r spec=%r",
            len(filtered), len(spells), dungeon, player_class, player_spec,
        )
        return cls(filtered, fuzzy_threshold=fuzzy_threshold, roster=roster)

    def all_phrases(self) -> list[str]:
        return sorted({s.phrase for s in self._spells})

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


class RuleEngine:
    """Maps cast events to alerts.

    Pure mapping — no dedupe lives here. Upstream is responsible for handing
    over only the cast events that should be alerted on (see `CastDeduper`).
    Keeping the engine pure makes it easier to extend with non-temporal rules
    (e.g., class-specific counter recommendations) without entangling them
    with dedup state.
    """

    def __init__(self, spell_db: YamlSpellDb):
        self.spell_db = spell_db

    def evaluate(self, ctx: ScreenContext) -> list[RuleOutput]:
        outputs: list[RuleOutput] = []
        for event in ctx.cast_events:
            spell = self.spell_db.lookup(event.spell, event.target)
            if spell is None:
                continue
            if spell.severity == Severity.IGNORE:
                continue
            target_str = f" on {event.target}" if event.target else ""
            duration_str = f" ({event.duration:.1f}s)" if event.duration is not None else ""
            outputs.append(
                Alert(
                    severity=spell.severity,
                    phrase=spell.phrase,
                    message=f"{spell.name}{target_str}{duration_str}",
                )
            )
        return outputs
