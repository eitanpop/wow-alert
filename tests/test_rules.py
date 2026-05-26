"""Tests for the spell DB + rule engine.

The rule engine is a pure map from CastEvents → Alerts; dedup lives upstream
in `CastDeduper` (see tests/test_dedupe.py).
"""
from __future__ import annotations

from wow_alert.events import Alert, CastEvent, ScreenContext, Severity, Spell
from wow_alert.rules import RuleEngine, YamlSpellDb


def make_event(spell: str, target: str | None = None, duration: float | None = None) -> CastEvent:
    return CastEvent(
        spell=spell, target=target, duration=duration,
        bbox=(0, 0, 100, 30), track_id=1,
    )


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
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"])
        assert db.lookup("Polymorph", "Jhon") is not None  # OCR typo

    def test_non_roster_target_skips_fail_closed(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER),
                    roster=["John", "Mary"], fuzzy_threshold=85)
        assert db.lookup("Polymorph", "Zxywuv") is None


class TestRuleEngine:
    def test_alert_for_danger_match(self):
        db = make_db(Spell(id="poly", name="Polymorph",
                           severity=Severity.DANGER, phrase="DANGER"))
        engine = RuleEngine(db)
        ctx = ScreenContext(cast_events=[make_event("Polymorph", "John", 3.0)])
        outputs = engine.evaluate(ctx)
        assert len(outputs) == 1
        assert isinstance(outputs[0], Alert)
        assert outputs[0].severity == Severity.DANGER
        assert outputs[0].phrase == "DANGER"
        assert "John" in outputs[0].message

    def test_severity_ignore_skips(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.IGNORE))
        engine = RuleEngine(db)
        ctx = ScreenContext(cast_events=[make_event("Polymorph")])
        assert engine.evaluate(ctx) == []

    def test_unknown_spell_no_output(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        engine = RuleEngine(db)
        ctx = ScreenContext(cast_events=[make_event("UnknownSpell")])
        assert engine.evaluate(ctx) == []

    def test_message_no_target_no_duration(self):
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        engine = RuleEngine(db)
        ctx = ScreenContext(cast_events=[make_event("Polymorph")])
        outputs = engine.evaluate(ctx)
        assert outputs[0].message == "Polymorph"

    def test_duplicate_calls_alert_each_time(self):
        # No dedup here — the engine is pure mapping. Upstream is expected
        # to dedupe before handing events over.
        db = make_db(Spell(id="poly", name="Polymorph", severity=Severity.DANGER))
        engine = RuleEngine(db)
        ctx = ScreenContext(cast_events=[make_event("Polymorph", "John", 3.0)])
        assert len(engine.evaluate(ctx)) == 1
        assert len(engine.evaluate(ctx)) == 1
