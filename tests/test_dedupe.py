"""Tests for the two-path cast-event deduper.

The deduper expects the caller to have already looked the cast up in the
spell DB; tests use the `process` helper below to mimic that
lookup-then-dedupe flow.
"""
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


def process(deduper: CastDeduper, sdb: YamlSpellDb, event: CastEvent):
    """Lookup-then-dedupe flow the pipeline performs."""
    spell = sdb.lookup(event.spell, event.target)
    return deduper.process(event, spell)


class TestMatchedPath:
    def test_first_call_is_matched_new_and_uses_db_duration(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=4.0))
        d = CastDeduper(clock=clock)
        outcome = process(d, sdb, evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_NEW
        assert outcome.ttl_s == 4.0  # from DB, not from event

    def test_second_call_within_ttl_is_matched_duplicate(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=4.0))
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Polymorph", "John"))
        clock.advance(1.0)
        outcome = process(d, sdb, evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_DUPLICATE
        assert outcome.ttl_s == 0.0

    def test_after_ttl_re_registers(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=4.0))
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Polymorph", "John"))
        clock.advance(4.5)
        outcome = process(d, sdb, evt("Polymorph", "John"))
        assert outcome.disposition is Disposition.MATCHED_NEW

    def test_canonical_id_dedupes_jittered_text(self):
        # Different OCR transcriptions, same canonical spell → matched_duplicate
        # because the matched cache key uses spell.id, not raw text.
        clock = FakeClock()
        sdb = db(Spell(id="sb", name="Spirit Bolt",
                       aliases=["SpiritBolt"],
                       severity=Severity.DANGER, duration=2.5))
        d = CastDeduper(clock=clock)
        assert process(d, sdb, evt("Spirit Bolt", "Ota")).disposition is Disposition.MATCHED_NEW
        clock.advance(0.5)
        assert process(d, sdb, evt("SpiritBolt", "Ota")).disposition is Disposition.MATCHED_DUPLICATE

    def test_null_db_duration_falls_back_to_event_duration(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER))  # no duration in DB
        d = CastDeduper(clock=clock)
        outcome = process(d, sdb, evt("Polymorph", "John", duration=3.0))
        assert outcome.disposition is Disposition.MATCHED_NEW
        assert outcome.ttl_s == 3.0  # event duration since DB is null

    def test_matched_ttl_capped(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=120.0))
        d = CastDeduper(clock=clock, max_matched_ttl_s=30.0)
        outcome = process(d, sdb, evt("Polymorph", "John"))
        assert outcome.ttl_s == 30.0


class TestUnmatchedPath:
    def test_first_call_is_unmatched_new(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock)
        outcome = process(d, sdb, evt("Mystery Cast", "X", duration=3.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW
        assert outcome.ttl_s == 3.0

    def test_fuzzy_jittered_unmatched_dedupes(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Arcane Salvo", "Tank", 4.0))
        clock.advance(1.0)
        outcome = process(d, sdb, evt("ArcaneSalvo", "Tank", 4.0))
        assert outcome.disposition is Disposition.UNMATCHED_DUPLICATE

    def test_truly_different_unmatched_registers(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Arcane Salvo", "Tank", 4.0))
        clock.advance(0.1)
        outcome = process(d, sdb, evt("Spirit Bolt", "Tank", 2.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW

    def test_unmatched_ttl_capped_short(self):
        # OCR produced a 233 s "duration"; unmatched cap should clamp it.
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock, max_unmatched_ttl_s=10.0)
        outcome = process(d, sdb, evt("Some Spell", "X", duration=233.0))
        assert outcome.ttl_s == 10.0

    def test_unmatched_no_duration_uses_default(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock, default_ttl_s=5.0)
        outcome = process(d, sdb, evt("Some Spell", "X"))
        assert outcome.ttl_s == 5.0


class TestCanonicalTarget:
    """Roster-driven target canonicalization. Jittered OCR variants of the
    same teammate should collapse to one cache entry once a roster is
    configured."""

    def test_no_roster_keeps_raw_target(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=4.0))
        d = CastDeduper(clock=clock)
        outcome = process(d, sdb, evt("Polymorph", "Jhon"))
        assert outcome.canonical_target == "Jhon"

    def test_roster_canonicalizes_jittered_target(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=4.0))
        d = CastDeduper(roster=["John Smith", "Mary"], clock=clock)
        outcome = process(d, sdb, evt("Polymorph", "Jhon Smith"))
        assert outcome.canonical_target == "John Smith"

    def test_canonical_target_collapses_cache_keys(self):
        # "Meredy" / "MeredyH2" / "Meredy H" all resolve to the same
        # roster member → same cache key → one dedup entry.
        clock = FakeClock()
        sdb = db(Spell(id="sb", name="Spirit Bolt",
                       severity=Severity.DANGER, duration=3.0))
        d = CastDeduper(
            roster=["Meredy Huntswell", "Austin Huxworth"],
            clock=clock,
        )
        first = process(d, sdb, evt("Spirit Bolt", "Meredy"))
        assert first.disposition is Disposition.MATCHED_NEW
        clock.advance(0.5)
        second = process(d, sdb, evt("Spirit Bolt", "MeredyH2"))
        assert second.disposition is Disposition.MATCHED_DUPLICATE
        clock.advance(0.5)
        third = process(d, sdb, evt("Spirit Bolt", "Meredy H"))
        assert third.disposition is Disposition.MATCHED_DUPLICATE

    def test_non_roster_target_does_not_canonicalize(self):
        # Target that's nowhere near a roster entry stays as raw text and
        # gets its own cache key — boss tank-swap shouldn't collapse with
        # a teammate's cast.
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=3.0))
        d = CastDeduper(roster=["John", "Mary"], clock=clock)
        outcome = process(d, sdb, evt("Polymorph", "Zxywuv"))
        assert outcome.canonical_target == "Zxywuv"

    def test_set_roster_updates_canonicalization(self):
        clock = FakeClock()
        sdb = db(Spell(id="poly", name="Polymorph",
                       severity=Severity.DANGER, duration=3.0))
        d = CastDeduper(clock=clock)
        # Pre-roster: jittered target stays raw.
        outcome = process(d, sdb, evt("Polymorph", "Jhon Smith"))
        assert outcome.canonical_target == "Jhon Smith"
        # After roster update, a future cast with a similar name canonicalizes.
        d.set_roster(["John Smith"])
        clock.advance(5.0)  # let TTL clear
        outcome2 = process(d, sdb, evt("Polymorph", "Jhon Smith"))
        assert outcome2.canonical_target == "John Smith"


class TestTargetNullness:
    def test_both_targetless_with_same_unmatched_spell_dedupes(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Ferocious Pounce", None, 3.0))
        clock.advance(0.5)
        outcome = process(d, sdb, evt("Ferocious Pounce", None, 2.5))
        assert outcome.disposition is Disposition.UNMATCHED_DUPLICATE

    def test_targetless_vs_targeted_does_not_dedupe(self):
        clock = FakeClock()
        sdb = db()
        d = CastDeduper(clock=clock)
        process(d, sdb, evt("Fire Spit", None, 3.0))
        clock.advance(0.1)
        outcome = process(d, sdb, evt("Fire Spit", "Tank", 3.0))
        assert outcome.disposition is Disposition.UNMATCHED_NEW
