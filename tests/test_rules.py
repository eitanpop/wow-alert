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

    def test_short_ocr_fragment_does_not_match(self):
        # 1-3 char OCR noise must not match: partial_ratio scores such
        # fragments ~100 against any name containing those letters
        # ("t" -> "Spiri[t] Bol[t]", "er" -> "Disp[er]sal"). Regression for
        # stray-letter false alerts during play.
        db = make_db(
            Spell(id="sb", name="Spirit Bolt", severity=Severity.DANGER),
            Spell(id="sd", name="Spore Dispersal", severity=Severity.DANGER),
        )
        assert db.lookup("T", None) is None
        assert db.lookup("er", None) is None
        assert db.lookup("Sr", None) is None

    def test_short_exact_name_still_matches(self):
        # The guard only blocks fuzzy; a real short name still matches exactly.
        db = make_db(Spell(id="hex", name="Hex", severity=Severity.DANGER))
        assert db.lookup("Hex", None) is not None


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


def spell(id_="poly", name="Polymorph", severity=Severity.DANGER, phrase="DANGER",
          tags=None):
    return Spell(id=id_, name=name, severity=severity, phrase=phrase, tags=tags or [])


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

    def test_utility_action_not_bound_by_defensive_rule(self):
        # Freedom-style utility (snare break) must NOT satisfy a tank-buster
        # defensive rule, even when the real external is on cooldown.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "scope": "single_target",
                "lacks_tag": "aggro_dropping",
                "say": "{action.label} {target}",
                "do": "{action.id}",
            }],
        })
        sac = action("sac", label="Sac", spell_id=6940)
        freedom = action("freedom", label="Freedom", category="utility",
                         scope="single_target", tags=["snare_break"], spell_id=1044)
        eng.set_class_actions([sac, freedom])
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"), canonical_target="John",
            cooldowns={6940: True, 1044: False},  # Sac on CD, Freedom up
        )
        out = eng.decide(ctx)
        # Freedom is utility, not defensive → no bind → fall through to the
        # spell-default Alert, NOT a Freedom recommendation.
        assert isinstance(out, Alert)

    def test_multi_category_action_binds_for_each_category(self):
        # Revival-style action: category [heal, dispel] must bind both a
        # heal rule and a dispel rule.
        revival = action("revival", label="Revival",
                         category=["heal", "dispel"], scope="party_wide")
        for cat in ("heal", "dispel"):
            eng = engine_with_rules({
                "on_cast": {"spell_id": "poly"},
                "priorities": [{
                    "category": cat,
                    "scope": "party_wide",
                    "say": "{action.label}",
                    "do": "{action.id}",
                }],
            })
            eng.set_class_actions([revival])
            ctx = RuleDecisionContext(
                spell=spell(), cast=cast(),
                cooldowns=all_available(revival),
            )
            out = eng.decide(ctx)
            assert isinstance(out, Recommendation), cat
            assert out.action == "revival", cat

    def test_single_target_dispel_prefers_detox_over_mass_dispel(self):
        # Detox (single-target) is listed before Revival so a no-scope
        # magic-dispel rule binds the cheap targeted dispel, not the
        # raid cooldown — even though Revival also carries category dispel.
        detox = action("detox", label="Detox", category="dispel",
                       scope="single_target", tags=["magic"])
        revival = action("revival", label="Revival",
                         category=["heal", "dispel"], scope="party_wide",
                         tags=["magic", "poison", "disease"])
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "dispel",
                "has_tag": "magic",
                "say": "{action.label} {target}",
                "do": "{action.id}",
            }],
        })
        eng.set_class_actions([detox, revival])  # detox first
        ctx = RuleDecisionContext(
            spell=spell(), cast=cast(target="John"), canonical_target="John",
            cooldowns=all_available(detox, revival),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "detox"

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


class TestPhrasePrefix:
    """`phrase_prefix` defaults to the spell name so the player hears
    context before the action ('Polymorph BOP' instead of 'BOP').
    Authors override per-rule when they want a different prefix or
    no prefix at all."""

    def test_default_prefix_is_spell_name(self):
        # No phrase_prefix set → default is ctx.spell.name.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label} {target}",
                "do": "{action.id}",
            }],
        })
        actions = [action("bop", label="BOP", spell_id=1022)]
        eng.set_class_actions(actions)
        ctx = RuleDecisionContext(
            spell=spell(),
            cast=cast(target="John"),
            canonical_target="John",
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.phrase_prefix == "Polymorph"

    def test_explicit_empty_prefix_disables(self):
        # phrase_prefix: "" explicitly disables the spell-name prefix.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label}",
                "do": "{action.id}",
                "phrase_prefix": "",
            }],
        })
        actions = [action("bop", label="BOP", spell_id=1022)]
        eng.set_class_actions(actions)
        ctx = RuleDecisionContext(
            spell=spell(),
            cast=cast(target="John"),
            canonical_target="John",
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.phrase_prefix == ""

    def test_custom_prefix_template_renders(self):
        # A custom phrase_prefix template (e.g. "URGENT {spell}") renders.
        eng = engine_with_rules({
            "on_cast": {"spell_id": "poly"},
            "priorities": [{
                "category": "defensive",
                "say": "{action.label}",
                "do": "{action.id}",
                "phrase_prefix": "URGENT {spell}",
            }],
        })
        actions = [action("bop", label="BOP", spell_id=1022)]
        eng.set_class_actions(actions)
        ctx = RuleDecisionContext(
            spell=spell(),
            cast=cast(target="John"),
            canonical_target="John",
            cooldowns=all_available(*actions),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.phrase_prefix == "URGENT Polymorph"


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


class TestTagResolution:
    """The tag table drives the recommendation when no per-spell rule exists,
    and a per-spell rule overrides the tags (the escape hatch)."""

    def _engine(self, *actions):
        from wow_alert.config import REPO_ROOT
        from wow_alert.tag_rules import load_tag_rules
        eng = RuleEngine()
        eng.set_tag_rules(load_tag_rules(REPO_ROOT / "config"))
        eng.set_class_actions(list(actions))
        return eng

    def test_tags_drive_recommendation_when_no_rule(self):
        sac = action("sac", label="Sac", spell_id=6940)
        eng = self._engine(sac)
        sp = spell(id_="tb", name="Tank Hit", tags=["big_damage_single"])
        ctx = RuleDecisionContext(
            spell=sp, cast=cast(target="John"), canonical_target="John",
            cooldowns=all_available(sac),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "sac"

    def test_per_spell_rule_overrides_tags(self):
        # Carries big_damage_single (→ defensive) AND an explicit heal rule. Rule wins.
        sac = action("sac", label="Sac", category="defensive",
                     scope="single_target", spell_id=6940)
        loh = action("loh", label="Lay on Hands", category="heal",
                     scope="single_target", spell_id=633)
        eng = self._engine(sac, loh)
        eng.set_rules([{
            "on_cast": {"spell_id": "tb"},
            "priorities": [{"category": "heal", "scope": "single_target",
                            "say": "{action.label} {target}", "do": "{action.id}"}],
        }])
        sp = spell(id_="tb", name="Tank Hit", tags=["big_damage_single"])
        ctx = RuleDecisionContext(
            spell=sp, cast=cast(target="John"), canonical_target="John",
            cooldowns=all_available(sac, loh),
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "loh"  # the rule's heal, not the tag's defensive

    def test_big_damage_falls_to_wings_when_party_dr_down(self):
        aura = action("aura", label="Aura Mastery", category="defensive",
                      scope="party_wide", spell_id=31821)
        wings = action("wings", label="Avenging Wrath", category="heal",
                       scope="self", spell_id=31884)
        eng = self._engine(aura, wings)
        sp = spell(id_="big", name="Big Hit", tags=["big_damage_party"])
        ctx = RuleDecisionContext(
            spell=sp, cast=cast(),
            cooldowns={31821: True, 31884: False},  # Aura Mastery on CD, wings up
        )
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        # Wings sits right under the party wall, so it fires before Divine Toll.
        assert out.action == "wings"

    def test_big_damage_party_falls_to_personal_dr_last_resort(self):
        aura = action("aura", label="Aura", category="defensive",
                      scope="party_wide", spell_id=31821)
        wings = action("wings", label="Wings", category="heal",
                       scope="self", spell_id=31884)
        toll = action("toll", label="Divine Toll", category="heal",
                      scope="party_wide", spell_id=375576)
        bubble = action("bubble", label="Bubble", category="defensive",
                        scope="self", tags=["full_immunity"], spell_id=642)
        dp = action("dp", label="Divine Protection", category="defensive",
                    scope="self", spell_id=498)
        eng = self._engine(aura, wings, toll, bubble, dp)
        sp = spell(id_="g", name="Group Hit", phrase="DEF", tags=["big_damage_party"])
        cds = all_available(aura, wings, toll, bubble, dp)
        for i in (31821, 31884, 375576):  # Aura, wings, Divine Toll all down
            cds[i] = True
        out = eng.decide(RuleDecisionContext(spell=sp, cast=cast(), cooldowns=cds))
        assert isinstance(out, Recommendation)
        assert out.action == "dp"  # personal DR last resort, not the immunity Bubble

    def test_suggestions_disabled_skips_recommendation(self):
        sac = action("sac", label="Sac", spell_id=6940)
        eng = self._engine(sac)
        eng.set_suggestions_enabled(False)
        sp = spell(id_="tb", name="Tank Hit", phrase="TANK BUSTER",
                   tags=["big_damage_single"])
        ctx = RuleDecisionContext(
            spell=sp, cast=cast(target="John"), canonical_target="John",
            cooldowns=all_available(sac))
        out = eng.decide(ctx)
        # Would normally recommend Sac; with suggestions off, just the phrase.
        assert isinstance(out, Alert)
        assert out.phrase == "TANK BUSTER"
        # Flipping it back on restores the recommendation.
        eng.set_suggestions_enabled(True)
        out = eng.decide(ctx)
        assert isinstance(out, Recommendation)
        assert out.action == "sac"

    def test_dodge_tag_falls_through_to_default_alert(self):
        eng = self._engine(action("sac", spell_id=6940))
        sp = spell(id_="d", name="Swirly", phrase="MOVE", tags=["dodge"])
        ctx = RuleDecisionContext(spell=sp, cast=cast(),
                                  cooldowns=all_available(action("sac", spell_id=6940)))
        out = eng.decide(ctx)
        assert isinstance(out, Alert)
        assert out.phrase == "MOVE"

    def _big_damage_single_setup(self):
        sac = action("sac", label="Sac", spell_id=6940)
        bop = action("bop", label="BoP", tags=["aggro_dropping"], spell_id=1022)
        aura = action("aura", label="Aura", category="defensive",
                      scope="party_wide", spell_id=31821)
        eng = self._engine(sac, bop, aura)
        sp = spell(id_="tb", name="Smash", tags=["big_damage_single"])
        roles = {"Tank": "tank", "Dps": "dps"}

        def rec(target, on_cd):
            cds = all_available(sac, bop, aura)
            for i in on_cd:
                cds[i] = True
            return eng.decide(RuleDecisionContext(
                spell=sp, cast=cast(target=target), canonical_target=target,
                roster=list(roles), roles=roles, cooldowns=cds))
        return rec

    def test_big_damage_single_excludes_aggro_dropping_on_tank(self):
        rec = self._big_damage_single_setup()
        # Sac up → Sac for anyone (step 1).
        assert rec("Tank", []).action == "sac"
        assert rec("Dps", []).action == "sac"
        # Sac down, target is the tank → BoP (aggro-dropping) excluded → Aura.
        assert rec("Tank", [6940]).action == "aura"
        # Sac down, target is a DPS → BoP is allowed.
        assert rec("Dps", [6940]).action == "bop"

    def test_lacks_target_role_fails_closed_on_unknown_role(self):
        sac = action("sac", label="Sac", spell_id=6940)
        bop = action("bop", label="BoP", tags=["aggro_dropping"], spell_id=1022)
        aura = action("aura", label="Aura", category="defensive",
                      scope="party_wide", spell_id=31821)
        eng = self._engine(sac, bop, aura)
        sp = spell(id_="tb", name="Smash", tags=["big_damage_single"])
        cds = all_available(sac, bop, aura)
        cds[6940] = True  # Sac down
        # Role unknown → BoP step fails closed → Aura, never a possible-tank BoP.
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Mystery"), canonical_target="Mystery",
            roster=["Mystery"], roles={}, cooldowns=cds))
        assert isinstance(out, Recommendation)
        assert out.action == "aura"

    def test_self_target_recommends_personal_defensive_not_immunity(self):
        # Bubble listed first to prove lacks_tag: full_immunity skips it.
        bubble = action("bubble", label="Bubble", category="defensive",
                        scope="self", tags=["full_immunity"], spell_id=642)
        dp = action("dp", label="Divine Protection", category="defensive",
                    scope="self", spell_id=498)
        sac = action("sac", label="Sac", category="defensive",
                     scope="single_target", spell_id=6940)
        eng = self._engine(bubble, dp, sac)
        sp = spell(id_="b", name="Smash", tags=["big_damage_single"])
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Me"), canonical_target="Me",
            player_name="Me", roster=["Me"], cooldowns=all_available(bubble, dp, sac)))
        assert isinstance(out, Recommendation)
        assert out.action == "dp"  # personal DR, not the full-immunity Bubble

    def test_other_target_gets_external_not_self_defensive(self):
        dp = action("dp", label="Divine Protection", category="defensive",
                    scope="self", spell_id=498)
        sac = action("sac", label="Sac", category="defensive",
                     scope="single_target", spell_id=6940)
        eng = self._engine(dp, sac)
        sp = spell(id_="b", name="Smash", tags=["big_damage_single"])
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Tank"), canonical_target="Tank",
            player_name="Me", roster=["Me", "Tank"], roles={"Tank": "tank"},
            cooldowns=all_available(dp, sac)))
        assert isinstance(out, Recommendation)
        assert out.action == "sac"  # external on the teammate, never a self CD

    def test_self_step_dormant_without_player_name(self):
        dp = action("dp", label="Divine Protection", category="defensive",
                    scope="self", spell_id=498)
        sac = action("sac", label="Sac", category="defensive",
                     scope="single_target", spell_id=6940)
        eng = self._engine(dp, sac)
        sp = spell(id_="b", name="Smash", tags=["big_damage_single"])
        # Cast targets "Me" but no player name configured → not treated as
        # self, so the external path fires exactly as before the feature.
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Me"), canonical_target="Me",
            cooldowns=all_available(dp, sac)))
        assert isinstance(out, Recommendation)
        assert out.action == "sac"

    def test_cc_tag_recommends_a_cc_ability(self):
        hoj = action("hoj", label="Hammer", category="cc",
                     scope="single_target", spell_id=853)
        eng = self._engine(hoj)
        sp = spell(id_="addcast", name="Bad Cast", tags=["cc"])
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(), cooldowns=all_available(hoj)))
        assert isinstance(out, Recommendation)
        assert out.action == "hoj"

    def test_dispel_tag_binds_matching_school_only(self):
        cleanse = action("cleanse", label="Cleanse", category="dispel",
                         scope="single_target", tags=["magic", "poison", "disease"],
                         spell_id=4987)
        eng = self._engine(cleanse)
        # Magic debuff → Cleanse (carries the magic subtype).
        magic = spell(id_="m", name="Hex", phrase="DISPEL", tags=["dispel_magic"])
        out = eng.decide(RuleDecisionContext(
            spell=magic, cast=cast(target="Ann"), canonical_target="Ann",
            cooldowns=all_available(cleanse)))
        assert isinstance(out, Recommendation) and out.action == "cleanse"
        # Curse → Cleanse lacks the curse subtype → no dispel, just the warning.
        curse = spell(id_="c", name="Hex", phrase="DISPEL", tags=["dispel_curse"])
        out = eng.decide(RuleDecisionContext(
            spell=curse, cast=cast(target="Ann"), canonical_target="Ann",
            cooldowns=all_available(cleanse)))
        assert isinstance(out, Alert) and out.phrase == "DISPEL"

    def test_snare_tag_binds_snare_break_utility(self):
        freedom = action("freedom", label="Freedom", category="utility",
                         scope="single_target", tags=["snare_break"], spell_id=1044)
        eng = self._engine(freedom)
        sp = spell(id_="s", name="Web", phrase="SNARED", tags=["snare"])
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Ann"), canonical_target="Ann",
            cooldowns=all_available(freedom)))
        assert isinstance(out, Recommendation) and out.action == "freedom"
        # Freedom down → nothing else breaks it → warning.
        out = eng.decide(RuleDecisionContext(
            spell=sp, cast=cast(target="Ann"), canonical_target="Ann",
            cooldowns={1044: True}))
        assert isinstance(out, Alert) and out.phrase == "SNARED"

    def test_bleed_uses_physical_immunity_on_nontank_only(self):
        bop = action("bop", label="BoP",
                     tags=["aggro_dropping", "physical_immunity"], spell_id=1022)
        sac = action("sac", label="Sac", spell_id=6940)  # DR, not an immunity
        aura = action("aura", label="Aura", category="defensive",
                      scope="party_wide", spell_id=31821)
        eng = self._engine(bop, sac, aura)
        sp = spell(id_="bl", name="Rend", phrase="BLEED", tags=["bleed"])
        roles = {"Tank": "tank", "Dps": "dps"}

        def out(target, on_cd=()):
            cds = all_available(bop, sac, aura)
            for i in on_cd:
                cds[i] = True
            return eng.decide(RuleDecisionContext(
                spell=sp, cast=cast(target=target), canonical_target=target,
                roster=list(roles), roles=roles, cooldowns=cds))
        # Bleed on a DPS → BoP (physical immunity actually clears it).
        assert out("Dps").action == "bop"
        # Bleed on the tank → BoP excluded; no DR fallback → just the warning.
        assert isinstance(out("Tank"), Alert)
        assert out("Tank").phrase == "BLEED"
        # BoP down → no immunity, no DR fallback → warning (DR can't clear a bleed).
        assert isinstance(out("Dps", on_cd=[1022]), Alert)
