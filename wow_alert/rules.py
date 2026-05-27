"""Spell DB + rule engine.

`YamlSpellDb` is the lookup catalog: fuzzy name → canonical `Spell`. It
holds the active per-dungeon set; `replace_spells` swaps the spell set
when calibration changes the active dungeon.

`RuleEngine.decide(RuleDecisionContext)` is the policy layer. It does
no DB lookup, no dedupe, no temporal state — given a fully-populated
decision context it returns one `RuleOutput` (Alert | Recommendation)
or None to suppress. Rule data and class-action data live as engine
attributes; decide() injects them into the context if the caller didn't.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Iterable

from rapidfuzz import fuzz

from wow_alert.class_library import ClassAction
from wow_alert.events import (
    Alert,
    Recommendation,
    RuleDecisionContext,
    RuleOutput,
    Severity,
    Spell,
)
from wow_alert.rule_schema import (
    Priority,
    Rule,
)

logger = logging.getLogger(__name__)


class YamlSpellDb:
    """In-memory spell catalog. Fuzzy name → canonical Spell."""

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
        logger.info("Spell DB holds %d spells", len(self._spells))

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

        # Exact (canonical name + every alias).
        exact = self._name_index.get(lower)
        if exact is not None:
            return exact

        # Fuzzy match across all names. Use max(token_set, partial_ratio):
        #   - token_set_ratio handles spell names with reordered/extra
        #     tokens ("Holy Light" vs "Light Holy").
        #   - partial_ratio handles OCR-merged tokens ("SpiritBolt" vs
        #     "Spirit Bolt") where there's no whole-token overlap but a
        #     close substring is present.
        # Whichever scorer goes higher wins; the cutoff (fuzzy_threshold,
        # default 85) is the same for both.
        best_score = 0
        best_name: str | None = None
        for candidate in self._all_names:
            score = max(
                fuzz.token_set_ratio(lower, candidate),
                fuzz.partial_ratio(lower, candidate),
            )
            if score > best_score:
                best_score = score
                best_name = candidate
        if best_name is not None and best_score >= self.fuzzy_threshold:
            return self._name_index[best_name]

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

    # Target-match threshold is *below* `fuzzy_threshold` (which gates spell-
    # name matching) because OCR jitter on short single-token names is severe:
    # "AustinH" vs "Austin Huxworth" only scores ~57% on token_set_ratio,
    # well below the 85 we use for multi-token spell names. The deduper's
    # `_canonical_target` uses 70 with `max(token_set, partial_ratio)`; we
    # use the same scoring + threshold here so both layers agree on what
    # counts as "this OCR'd target is a roster member".
    _TARGET_MATCH_THRESHOLD = 70

    def _target_ok(self, target_text: str | None) -> bool:
        """Fuzzy check on an OCR'd target name against the configured roster.

        Three cases:
        - No target detected by OCR → pass.
        - Target detected, no roster configured → pass (nothing to validate).
        - Target detected, roster configured → require either a token_set
          OR a partial_ratio fuzzy match against any roster entry at
          `_TARGET_MATCH_THRESHOLD`. Either signal alone is enough — OCR
          can either merge tokens ("MeredyH" → no token overlap) or drop
          tokens ("Austin Hux" → partial substring); both should count.
        """
        if not target_text:
            return True
        if not self._roster_lower:
            return True
        target_lower = target_text.lower()
        for candidate in self._roster_lower:
            score = max(
                fuzz.token_set_ratio(target_lower, candidate),
                fuzz.partial_ratio(target_lower, candidate),
            )
            if score >= self._TARGET_MATCH_THRESHOLD:
                return True
        return False


@dataclass(frozen=True, slots=True)
class _MatchResult:
    """Predicate evaluation result. `bindings` carries data the rule
    template can reference, like the matched ClassAction for {action.label}."""

    matched: bool
    bindings: dict = field(default_factory=dict)


class RuleEngine:
    """Decides what to do for one matched, deduped cast.

    `decide(RuleDecisionContext)` is a pure function — no spell DB lookup,
    no dedupe, no temporal state. Rule data and class actions are engine
    attributes (set_rules, set_class_actions); decide() injects them into
    the context if the caller didn't supply them, so production code can
    pass a minimal context and tests can pass an explicit, fully-populated
    one.
    """

    def __init__(self):
        self._rules: list[Rule] = []
        self._class_actions: list[ClassAction] = []

    def set_rules(self, raw_rules: list[dict]) -> None:
        """Parse and store rules. Bad entries are skipped with a warning
        rather than failing the load — a typo'd rule shouldn't take the
        whole app down."""
        parsed: list[Rule] = []
        for raw in raw_rules:
            try:
                parsed.append(Rule.model_validate(raw))
            except Exception as exc:
                logger.warning("Skipping malformed rule %r: %s", raw, exc)
        self._rules = parsed
        logger.info("Rule engine holds %d rules", len(self._rules))

    def set_class_actions(self, actions: list[ClassAction]) -> None:
        """Set the player's available actions. Read by priorities that
        carry class-action filters (category / scope / has_tag /
        lacks_tag); affects every subsequent decide() call."""
        self._class_actions = list(actions)
        logger.info(
            "Rule engine has %d class actions", len(self._class_actions),
        )

    def all_phrases(self) -> list[str]:
        """Custom TTS phrases authored in rule priorities (e.g. 'Break
        Shield'). Used by the prerender pass so any priority.phrase that
        references a literal string gets a cached WAV before playback.

        Phrases that contain template tokens like {target} are skipped —
        those are resolved at decide time, and the components they expand
        to (action labels, roster names) are prerendered separately.
        """
        phrases: set[str] = set()
        for rule in self._rules:
            for prio in rule.priorities:
                if prio.phrase and "{" not in prio.phrase:
                    phrases.add(prio.phrase)
        return sorted(phrases)

    def decide(self, ctx: RuleDecisionContext) -> RuleOutput | None:
        """One decision per cast — returns the single output to emit, or
        None to suppress."""
        spell = ctx.spell
        if spell.severity == Severity.IGNORE:
            return None

        # Inject engine-held state when the caller didn't supply it.
        # Frozen dataclass → use `replace` to substitute.
        if not ctx.class_actions and self._class_actions:
            ctx = replace(ctx, class_actions=self._class_actions)

        # Rules first, most-specific (with target_role filter) winning.
        for rule in self._matching_rules(ctx):
            for prio in rule.priorities:
                result = self._eval_priority(prio, ctx)
                if result.matched:
                    return self._build_output(prio, ctx, spell, result.bindings)

        # No rule fired — fall back to the spell's default Alert.
        target_str = f" on {ctx.cast.target}" if ctx.cast.target else ""
        duration_str = (
            f" ({ctx.cast.duration:.1f}s)" if ctx.cast.duration is not None else ""
        )
        return Alert(
            severity=spell.severity,
            phrase=spell.phrase,
            message=f"{spell.name}{target_str}{duration_str}",
        )

    # ---- rule walker internals ----

    def _matching_rules(self, ctx: RuleDecisionContext) -> list[Rule]:
        matching: list[Rule] = []
        for rule in self._rules:
            if rule.on_cast.spell_id != ctx.spell.id:
                continue
            if rule.on_cast.target_role:
                actual = ctx.roles.get(ctx.canonical_target or "")
                if actual != rule.on_cast.target_role:
                    continue
            matching.append(rule)
        # Sort more-specific (more `on_cast` filters set) first.
        matching.sort(key=lambda r: -self._specificity(r))
        return matching

    @staticmethod
    def _specificity(rule: Rule) -> int:
        s = 0
        if rule.on_cast.target_role:
            s += 1
        return s

    def _eval_priority(
        self, prio: Priority, ctx: RuleDecisionContext
    ) -> _MatchResult:
        """AND-conjunction of every set condition on the priority.

        Two stages:
          1. Class-action binding (skipped when no class-action filter
             is set). Walks the player's library in file order and picks
             the first action satisfying category / scope / has_tag /
             lacks_tag whose cooldown is ready. An action is considered
             on cooldown when ctx.cooldowns[action.spell_id] is True
             OR when the spell_id is missing from the dict — fail-closed
             so untracked abilities don't get recommended. The startup
             warning lists actions in that state.
             A failure here fails the priority outright; the bound
             action is exposed to templates as {action.label} /
             {action.id}.
          2. Cast filters (target_role, target_present). Each is
             checked only when set.
        """
        bindings: dict = {}

        if prio.has_action_filter():
            action = self._find_ready_action(prio, ctx)
            if action is None:
                return _MatchResult(False)
            bindings["action"] = action

        if prio.target_role is not None:
            if not ctx.canonical_target:
                return _MatchResult(False)
            actual = ctx.roles.get(ctx.canonical_target)
            if actual is None or actual != prio.target_role:
                return _MatchResult(False)

        if prio.target_present is not None:
            has_target = bool(ctx.canonical_target or ctx.cast.target)
            if has_target != prio.target_present:
                return _MatchResult(False)

        return _MatchResult(True, bindings)

    @staticmethod
    def _find_ready_action(
        prio: Priority, ctx: RuleDecisionContext
    ) -> ClassAction | None:
        for action in ctx.class_actions:
            if prio.category and action.category != prio.category:
                continue
            if prio.scope and action.scope != prio.scope:
                continue
            if prio.has_tag and prio.has_tag not in action.tags:
                continue
            if prio.lacks_tag and prio.lacks_tag in action.tags:
                continue
            # Fail-closed on missing entries: if the cooldown watcher
            # hasn't reported availability for this spell_id (either
            # because the icon isn't on the player's bar or the matcher
            # couldn't identify it), treat as unavailable. The startup
            # warning surfaces which actions are in this state so the
            # user can fix their setup. Better to miss a recommendation
            # than to confidently recommend an unusable spell.
            if ctx.cooldowns.get(action.spell_id, True):
                continue
            return action
        return None

    @classmethod
    def _build_output(
        cls,
        prio: Priority,
        ctx: RuleDecisionContext,
        spell: Spell,
        bindings: dict,
    ) -> RuleOutput:
        """Build an Alert or Recommendation from a fired priority.

        When `do` is set, the priority is "go take this action" → emit
        Recommendation. The TTS phrase comes from `prio.phrase` if the
        priority overrides it (e.g. "Break Shield"); otherwise the
        bound action's label is used so the prerendered action clip
        plays. The pipeline's Recommendation handler stitches the
        target name onto it at playback. When `do` is unset, the
        priority is "just inform" → emit Alert with the spell's default
        phrase as the TTS key; the rendered `say` shows in the log.
        """
        rendered = cls._render(prio.say, ctx, bindings)
        if prio.do is not None:
            do_value = cls._render(prio.do, ctx, bindings)
            if prio.phrase is not None:
                phrase = cls._render(prio.phrase, ctx, bindings)
            else:
                action = bindings.get("action")
                phrase = action.label if action is not None else do_value
            return Recommendation(
                action=do_value,
                target=ctx.canonical_target or ctx.cast.target or "",
                phrase=phrase,
                message=rendered,
            )
        return Alert(
            severity=spell.severity,
            phrase=spell.phrase,
            message=rendered,
        )

    @staticmethod
    def _render(
        template: str, ctx: RuleDecisionContext, bindings: dict
    ) -> str:
        """Render templating tokens. See rule_schema.py for the supported set.

        Collapses runs of whitespace and trims edges so a template like
        `"{action.label} {target}"` with an empty target renders to
        `"BOP"` instead of `"BOP "`.
        """
        target = ctx.canonical_target or ctx.cast.target or ""
        duration = (
            f"{ctx.cast.duration:.1f}s" if ctx.cast.duration is not None else ""
        )
        s = template
        s = s.replace("{target}", target)
        s = s.replace("{spell}", ctx.spell.name)
        s = s.replace("{duration}", duration)
        action = bindings.get("action")
        if action is not None:
            s = s.replace("{action.label}", action.label)
            s = s.replace("{action.id}", action.id)
        return " ".join(s.split())
