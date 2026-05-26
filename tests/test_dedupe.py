"""Tests for the two-path cast-event deduper."""
from __future__ import annotations

from wow_alert.dedupe import CastDeduper, Disposition
from wow_alert.events import CastEvent, Severity, Spell
from wow_alert.rules import YamlSpellDb


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def evt(spell: str, target: str | None = None, duration: float | None = None) -> CastEvent:
    return CastEvent(
        spell=spell, target=target, duration=duration,
        bbox=(0, 0, 100, 30), track_id=1,
    )


def db(*spells: Spell) -> YamlSpellDb:
    return YamlSpellDb(spells)


class TestMatchedPath:
    def test_first_call_is_matched_new_and_uses_db_duration(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER, duration=4.0)),
            clock=clock,
        )
        outcome = d.process(evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_NEW
        assert outcome.canonical_spell.id == "poly"
        assert outcome.ttl_s == 4.0  # from DB, not from event

    def test_second_call_within_ttl_is_matched_duplicate(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER, duration=4.0)),
            clock=clock,
        )
        d.process(evt("Polymorph", "John"))
        clock.advance(1.0)
        outcome = d.process(evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_DUPLICATE
        assert outcome.canonical_spell.id == "poly"
        assert outcome.ttl_s == 0.0  # no new TTL applied

    def test_after_ttl_re_registers(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER, duration=4.0)),
            clock=clock,
        )
        d.process(evt("Polymorph", "John"))
        clock.advance(4.5)
        outcome = d.process(evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_NEW

    def test_canonical_id_dedupes_jittered_text(self):
        # Different OCR transcriptions, same canonical spell → matched_duplicate
        # because the matched cache key uses spell.id, not raw text.
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="sb", name="Spirit Bolt",
                              aliases=["SpiritBolt"],
                              severity=Severity.DANGER, duration=2.5)),
            clock=clock,
        )
        assert d.process(evt("Spirit Bolt", "Ota")).disposition is Disposition.MATCHED_NEW
        clock.advance(0.5)
        assert d.process(evt("SpiritBolt", "Ota")).disposition is Disposition.MATCHED_DUPLICATE

    def test_null_db_duration_falls_back_to_event_duration(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER)),  # no duration in DB
            clock=clock,
        )
        outcome = d.process(evt("Polymorph", "John", duration=3.0))
        assert outcome.disposition is Disposition.MATCHED_NEW
        assert outcome.ttl_s == 3.0  # event duration since DB is null

    def test_matched_ttl_capped(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(Spell(id="poly", name="Polymorph",
                              severity=Severity.DANGER, duration=120.0)),
            clock=clock,
            max_matched_ttl_s=30.0,
        )
        outcome = d.process(evt("Polymorph", "John"))
        assert outcome.ttl_s == 30.0


class TestUnmatchedPath:
    def test_first_call_is_unmatched_new(self):
        clock = FakeClock()
        d = CastDeduper(spell_db=db(), clock=clock)
        outcome = d.process(evt("Mystery Cast", "X", duration=3.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW
        assert outcome.canonical_spell is None
        assert outcome.ttl_s == 3.0

    def test_fuzzy_jittered_unmatched_dedupes(self):
        clock = FakeClock()
        d = CastDeduper(spell_db=db(), clock=clock)
        d.process(evt("Arcane Salvo", "Tank", 4.0))
        clock.advance(1.0)
        outcome = d.process(evt("ArcaneSalvo", "Tank", 4.0))
        assert outcome.disposition is Disposition.UNMATCHED_DUPLICATE

    def test_truly_different_unmatched_registers(self):
        clock = FakeClock()
        d = CastDeduper(spell_db=db(), clock=clock)
        d.process(evt("Arcane Salvo", "Tank", 4.0))
        clock.advance(0.1)
        outcome = d.process(evt("Spirit Bolt", "Tank", 2.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW

    def test_unmatched_ttl_capped_short(self):
        # OCR produced a 233 s "duration"; unmatched cap should clamp it.
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(),
            clock=clock,
            max_unmatched_ttl_s=10.0,
        )
        outcome = d.process(evt("Some Spell", "X", duration=233.0))
        assert outcome.ttl_s == 10.0

    def test_unmatched_no_duration_uses_default(self):
        clock = FakeClock()
        d = CastDeduper(
            spell_db=db(),
            clock=clock,
            default_ttl_s=5.0,
        )
        outcome = d.process(evt("Some Spell", "X"))
        assert outcome.ttl_s == 5.0


class TestTargetNullness:
    def test_both_targetless_with_same_unmatched_spell_dedupes(self):
        clock = FakeClock()
        d = CastDeduper(spell_db=db(), clock=clock)
        d.process(evt("Ferocious Pounce", None, 3.0))
        clock.advance(0.5)
        outcome = d.process(evt("Ferocious Pounce", None, 2.5))
        assert outcome.disposition is Disposition.UNMATCHED_DUPLICATE

    def test_targetless_vs_targeted_does_not_dedupe(self):
        clock = FakeClock()
        d = CastDeduper(spell_db=db(), clock=clock)
        d.process(evt("Fire Spit", None, 3.0))
        clock.advance(0.1)
        outcome = d.process(evt("Fire Spit", "Tank", 3.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW
