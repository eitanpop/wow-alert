"""Tests for the spell DB + rule engine.

The rule engine is a pure map from a fully-populated RuleDecisionContext
to a single RuleOutput; the spell-DB lookup and dedupe happen upstream
(see tests/test_dedupe.py). All tests construct an explicit
RuleDecisionContext and call `decide()` directly.
"""
from __future__ import annotations

from wow_alert.class_library import ClassAction
from wow_alert.events import (
    Alert,
    CastEvent,
    Recommendation,
    RuleDecisionContext,
    Severity,
    Spell,
)
from wow_alert.rules import RuleEngine, YamlSpellDb


def make_db(*spells: Spell, fuzzy_threshold: int = 85, roster=None) -> YamlSpellDb:
    return YamlSpellDb(spells, fuzzy_threshold=fuzzy_threshold, roster=roster)


class TestSpellDbLookup:
    def test_exact_match(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        spell = db.lookup("Polymorph", None)
        assert spell is not None
        assert spell.id == "poly"

    def test_alias_match(self):
        db = make_db(Spell(id="vig", name="Vigilant Defense",
                           aliases=["Vigliant Defense"], severity=Severity.DANGER))
        spell = db.lookup("Vigliant Defense", None)
        assert spell is not None
        assert spell.id == "vig"

    def test_case_insensitive(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        assert db.lookup("polymorph", None) is not None
        assert db.lookup("POLYMORPH", None) is not None

    def test_fuzzy_match_passes(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        assert db.lookup("Polymorpf", None) is not None

    def test_fuzzy_match_below_threshold_returns_none(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    fuzzy_threshold=99)
        assert db.lookup("xy", None) is None

    def test_unknown_spell_returns_none(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        assert db.lookup("Frostbolt", None) is None

    def test_empty_spell_text_returns_none(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        assert db.lookup("", None) is None


class TestTargetFuzzyMatch:
    """Without a roster the target check is permissive; with one it's fuzzy + fail-closed."""

    def test_no_roster_accepts_any_target(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        assert db.lookup("Polymorph", "Anyone") is not None

    def test_no_target_with_roster_is_accepted(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"])
        assert db.lookup("Polymorph", None) is not None

    def test_roster_member_passes(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"])
        assert db.lookup("Polymorph", "John") is not None

    def test_roster_member_fuzzy_passes(self):
        # OCR typo on a single-token short name doesn't pass token_set_ratio
        # at the default 85 threshold (computed ratio is ~75). The deduper's
        # canonical_target path uses a lower target threshold + partial_ratio
        # for this case; here we just demonstrate that fuzzy matching works
        # when the threshold is set to a value typical for short names.
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"], fuzzy_threshold=70)
        assert db.lookup("Polymorph", "Jhon") is not None

    def test_non_roster_target_skips_fail_closed(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"], fuzzy_threshold=85)
        assert db.lookup("Polymorph", "Zxywuv") is None


class TestFuzzyDoesNotConfuse:
    """Negative-side characterization for the spell-name fuzzy threshold.

    Each pair is two distinct WoW-flavored spells that share enough surface
    structure to be plausible confusions. None of them should resolve to
    the wrong canonical name at the default threshold. If any of these
    starts matching the wrong spell, the threshold tuning has drifted.
    """

    def _lookup_with(self, canonical: str, ocr: str):
        db = make_db(Spell(id="X", name=canonical, severity=Severity.DANGER))
        return db.lookup(ocr, None)

    def test_holy_light_not_holy_fire(self):
        assert self._lookup_with("Holy Light", "Holy Fire") is None

    def test_shadow_bolt_not_spirit_bolt(self):
        assert self._lookup_with("Shadow Bolt", "Spirit Bolt") is None

    def test_fireball_not_frostbolt(self):
        assert self._lookup_with("Fireball", "Frostbolt") is None

    def test_dark_command_not_dark_simulacrum(self):
        assert self._lookup_with("Dark Command", "Dark Simulacrum") is None

    def test_mass_dispel_not_mass_resurrection(self):
        assert self._lookup_with("Mass Dispel", "Mass Resurrection") is None

    def test_disjoint_roster_target_rejected(self):
        # With a roster configured, the target must match SOMETHING; bogus
        # text falls through fail-closed.
        db = make_db(
            Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
            roster=["Captain Garrick", "Meredy Huntswell"],
        )
        assert db.lookup("Polymorph", "ZXY") is None

    def test_different_roster_member_short_name_rejected(self):
        # "Garrick" and "Meredy" share zero meaningful structure; with a
        # roster gate the lookup fails closed.
        db = make_db(
            Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
            roster=["Garrick"],
        )
        assert db.lookup("Polymorph", "Meredy") is None


# ---- decide() rule-walker tests ----


def cast(spell="Polymorph", target=None, duration=None):
    return CastEvent(
        spell=spell, target=target, duration=duration,
        bbox=(0, 0, 100, 30), track_id=1,
    )


def spell(id_="poly", name="Polymorph", severity=Severity.DANGER, phrase="DANGER"):
    return Spell(id=id_, name=name, severity=severity, phrase=phrase)


_NEXT_SPELL_ID = [1000]


def action(id_, label=None, category="defensive", scope="single_target",
           tags=None, spell_id=None):
    if spell_id is None:
        _NEXT_SPELL_ID[0] += 1
        spell_id = _NEXT_SPELL_ID[0]
    return ClassAction(
        id=id_,
        label=label or id_.upper(),
        category=category,
        scope=scope,
        tags=tags or [],
        spell_id=spell_id,
    )


def engine_with_rules(*rules):
    eng = RuleEngine()
    eng.set_rules(list(rules))
    return eng


def all_available(*actions) -> dict[int, bool]:
    """Mark every action's spell_id as available. Mirrors the cooldown
    watcher's 'all clear' state — the rule engine is fail-closed on
    missing keys, so tests have to explicitly say 'this is up'."""
    return {a.spell_id: False for a in actions}


class TestDecideFallback:
    """No rules → fall back to the spell's default Alert."""

    def test_no_rules_emits_default_alert(self):
        eng = RuleEngine()
        ctx = RuleDecisionContext(spell=spell(), cast=cast(target="John", duration=3.0))
        out = eng.decide(ctx)
        assert isinstance(out, Alert)
        assert out.phrase == "DANGER"
        assert out.message == "Polymorph on John (3.0s)"

    def test_severity_ignore_returns_none(self):
        eng = RuleEngine()
        ctx = RuleDecisionContext(
            spell=spell(severity=Severity.IGNORE), cast=cast(),
        )
        assert eng.decide(ctx) is None

    def test_rule_for_other_spell_does_not_fire(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "other"},
            "priorities": [{"say": "Should not fire"}],
        })
        ctx = RuleDecisionContext(spell=spell(), cast=cast(target="John"))
        out = eng.decide(ctx)
        assert isinstance(out, Alert)
        assert out.message == "Polymorph on John"  # fallback


class TestClassActionFilters:
    """Class-action filters bind the first available action from the
    player's library; the priority fails when none is available."""

    def test_simple_match_emits_recommendation(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label} {target}",
                "do": "{action.id}",
            }],
        })
        actions = [action("bop", label="BOP")]
        eng.set_class_actions(actions)
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John",
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "bop"
        assert out.phrase == "BOP"
        assert out.message == "BOP John"

    def test_lacks_tag_excludes_action(self):
        # BoP has aggro_dropping → first priority skips it; second matches Sac.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "lacks_tag": "aggro_dropping",
                "say": "{action.label} {target}",
                "do": "{action.id}",
            }],
        })
        actions = [
            action("bop", label="BOP", tags=["aggro_dropping"]),
            action("sac", label="Sac"),
        ]
        eng.set_class_actions(actions)
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John",
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "sac"

    def test_on_cooldown_skips_action(self):
        # First action on cooldown → engine skips to next available one.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label}",
                "do": "{action.id}",
            }],
        })
        eng.set_class_actions([
            action("bop", label="BOP", spell_id=1022),
            action("sac", label="Sac", spell_id=6940),
        ])
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(),
            cooldowns={1022: True, 6940: False},  # BoP on CD, Sac up
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "sac"  # bop was on cooldown

    def test_missing_cooldown_entry_is_fail_closed(self):
        # Untracked spell_id (no entry in cooldowns dict) is treated as
        # on cooldown — never recommended. The priority then fails;
        # in the absence of a catch-all priority the engine falls
        # through to the spell-default Alert.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label}",
                "do": "{action.id}",
            }],
        })
        eng.set_class_actions([action("bop", label="BOP", spell_id=1022)])
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(),
            cooldowns={},  # nothing tracked
        )
        out = eng.decide(ctx)
        assert isinstance(out, Alert)
        assert out.phrase == "DANGER"  # spell-default phrase

    def test_priority_falls_through_to_catchall_when_no_action_available(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [
                {
                    "category": "defensive",
                    "say": "{action.label} {target}",
                    "do": "{action.id}",
                },
                {"say": "Tank Buster on {target}"},   # catch-all
            ],
        })
        # No actions at all → first priority can't match.
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John",
        )
        out = eng.decide(ctx)
        assert isinstance(out, Alert)  # no `do` → Alert
        assert out.message == "Tank Buster on John"


class TestTargetRoleSpecificity:
    """Rules with a target_role filter beat generic rules."""

    def test_target_role_filters_match(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly", "target_role": "tank"},
            "priorities": [{"say": "Tank-specific callout"}],
        })
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John",
            roles={"John": "tank"},
        )
        out = eng.decide(ctx)
        assert out.message == "Tank-specific callout"

    def test_target_role_filter_misses_other_role(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly", "target_role": "tank"},
            "priorities": [{"say": "Tank-specific callout"}],
        })
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="Mary"),
            canonical_target="Mary",
            roles={"Mary": "dps"},  # not the tank
        )
        out = eng.decide(ctx)
        # Falls through to spell default
        assert out.message == "Polymorph on Mary"

    def test_specific_rule_wins_over_generic(self):
        # Both rules match Mary's role check (the generic has no
        # target_role filter, so it matches any). The specific one (tank)
        # wins for John because it's more specific.
        eng = engine_with_rules(
            {
                "on_cast": {"spell_id": "poly"},
                "priorities": [{"say": "Generic callout for {target}"}],
            },
            {
                "on_cast": {"spell_id": "poly", "target_role": "tank"},
                "priorities": [{"say": "Tank callout for {target}"}],
            },
        )
        ctx_tank = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John", roles={"John": "tank"},
        )
        ctx_dps = RuleDecisionContext(
            spell=spell(), cast=cast(target="Mary"),
            canonical_target="Mary", roles={"Mary": "dps"},
        )
        assert eng.decide(ctx_tank).message == "Tank callout for John"
        assert eng.decide(ctx_dps).message == "Generic callout for Mary"

    def test_target_role_unknown_does_not_fire_role_rule(self):
        # John exists in roster but his role wasn't identified — rules
        # keyed on role should not fire (fail-closed).
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly", "target_role": "tank"},
            "priorities": [{"say": "Should not fire"}],
        })
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"),
            canonical_target="John",
            roles={},  # role unknown
        )
        out = eng.decide(ctx)
        # Falls through to spell default
        assert out.message == "Polymorph on John"


class TestComposedScenario:
    """The BoP / BoSac / Aura / catch-all scenario the user described."""

    def test_full_chain(self):
        eng = engine_with_rules({
            "on_cast": {"spell_id": "arcane_salvo", "target_role": "tank"},
            "priorities": [
                {
                    "category": "defensive",
                    "scope": "single_target",
                    "lacks_tag": "aggro_dropping",
                    "say": "{action.label} {target}",
                    "do": "{action.id}",
                },
                {
                    "category": "defensive",
                    "scope": "party_wide",
                    "say": "{action.label}",
                    "do": "{action.id}",
                },
                {"say": "Tank Buster on {target}"},
            ],
        })
        actions = [
            action("bop", label="BOP", scope="single_target",
                   tags=["aggro_dropping"], spell_id=1022),
            action("sac", label="Sac", scope="single_target", spell_id=6940),
            action("aura", label="Devo Aura", scope="party_wide", spell_id=465),
        ]
        eng.set_class_actions(actions)

        arcane_salvo = Spell(
            id="arcane_salvo", name="Arcane Salvo",
            severity=Severity.DANGER, phrase="TANK BUSTER",
        )

        # All defensives available → BoSac wins (BoP excluded by tag).
        ctx = RuleDecisionContext(
            spell=arcane_salvo, cast=cast(spell="Arcane Salvo", target="John"),
            canonical_target="John", roles={"John": "tank"},
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "sac"
        assert out.message == "Sac John"

        # Sac on cooldown → party-wide Aura wins.
        ctx2 = RuleDecisionContext(
            spell=arcane_salvo, cast=cast(spell="Arcane Salvo", target="John"),
            canonical_target="John", roles={"John": "tank"},
            cooldowns={**all_available(*actions), 6940: True},  # Sac on CD
        )
        out = eng.decide(ctx2)
        assert isinstance(out, Recommendation)
        assert out.action == "aura"
        assert out.message == "Devo Aura"

        # All defensives down → catch-all Alert.
        ctx3 = RuleDecisionContext(
            spell=arcane_salvo, cast=cast(spell="Arcane Salvo", target="John"),
            canonical_target="John", roles={"John": "tank"},
            cooldowns={**all_available(*actions), 6940: True, 465: True},
        )
        out = eng.decide(ctx3)
        assert isinstance(out, Alert)
        assert out.message == "Tank Buster on John"
