"""Cast-event deduplication, two paths.

Sits between the parser and the rule engine. For each incoming raw cast
event the deduper returns one of four dispositions:

  - MATCHED_NEW       — spell DB has this spell, not yet in cache. Caches it
                        with TTL from `Spell.duration` (authoritative) and
                        sends to the rule engine for alert evaluation.
  - MATCHED_DUPLICATE — spell DB has this spell, already in cache. Skip.
  - UNMATCHED_NEW     — no spell DB entry. Fuzzy-novel against the unmatched
                        cache. Caches with TTL from the OCR'd duration,
                        hard-capped (the OCR can't be trusted). Does NOT go
                        to the rule engine — no spell to match means no
                        alert.
  - UNMATCHED_DUPLICATE — no spell DB entry, fuzzy-matched against a recent
                          unmatched cache entry. Skip.

The two caches are deliberately separate:
- Matched cache is keyed by `(spell.id, target)` — exact, since fuzzy match
  is already done by the spell DB lookup. Canonical id collapses OCR jitter
  ("Spirit Bolt" / "SpiritE Bolt") to one entry automatically.
- Unmatched cache is keyed by raw OCR text + target with fuzzy comparison at
  lookup. Without a canonical id we can't be exact; OCR jitter will produce
  some duplicates and that's the documented trade-off.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from rapidfuzz import fuzz

from wow_alert.events import CastEvent, Spell
from wow_alert.rules import YamlSpellDb


class Disposition(str, Enum):
    MATCHED_NEW = "matched_new"
    MATCHED_DUPLICATE = "matched_duplicate"
    UNMATCHED_NEW = "unmatched_new"
    UNMATCHED_DUPLICATE = "unmatched_duplicate"


@dataclass(frozen=True)
class DedupeOutcome:
    """The deduper's decision about an incoming cast event.

    `disposition` is the executive summary; pipeline code branches on it. The
    other fields supply detail for narrative logging:
      - `canonical_spell` is populated for MATCHED_* (the Spell from the DB).
      - `ttl_s` is the TTL that was applied when registering (0 for
        duplicates so the pipeline can log a faithful "skipped" message
        without inventing a TTL).
    """

    disposition: Disposition
    canonical_spell: Spell | None
    ttl_s: float


@dataclass
class _UnmatchedEntry:
    spell: str
    target: str | None
    expiry: float


class CastDeduper:
    def __init__(
        self,
        spell_db: YamlSpellDb,
        default_ttl_s: float = 5.0,
        max_matched_ttl_s: float = 60.0,
        max_unmatched_ttl_s: float = 10.0,
        spell_fuzzy_threshold: int = 80,
        target_fuzzy_threshold: int = 70,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._spell_db = spell_db
        self._default_ttl_s = default_ttl_s
        self._max_matched_ttl_s = max_matched_ttl_s
        self._max_unmatched_ttl_s = max_unmatched_ttl_s
        self._spell_threshold = spell_fuzzy_threshold
        self._target_threshold = target_fuzzy_threshold
        self._clock = clock
        # Matched cache: (spell.id, target) -> expiry timestamp
        self._matched: dict[tuple[str, str | None], float] = {}
        # Unmatched cache: list of entries fuzzy-compared at lookup
        self._unmatched: list[_UnmatchedEntry] = []

    def process(self, event: CastEvent) -> DedupeOutcome:
        now = self._clock()
        self._prune(now)

        spell = self._spell_db.lookup(event.spell, event.target)

        if spell is not None:
            key = (spell.id, event.target)
            expiry = self._matched.get(key)
            if expiry is not None and now < expiry:
                return DedupeOutcome(Disposition.MATCHED_DUPLICATE, spell, 0.0)
            ttl = self._matched_ttl(spell, event)
            self._matched[key] = now + ttl
            return DedupeOutcome(Disposition.MATCHED_NEW, spell, ttl)

        # Path B: not in spell DB. Fuzzy compare against recent unmatched.
        for entry in self._unmatched:
            if self._fuzzy_match(event.spell, event.target, entry.spell, entry.target):
                return DedupeOutcome(Disposition.UNMATCHED_DUPLICATE, None, 0.0)

        ttl = self._unmatched_ttl(event)
        self._unmatched.append(
            _UnmatchedEntry(spell=event.spell, target=event.target, expiry=now + ttl)
        )
        return DedupeOutcome(Disposition.UNMATCHED_NEW, None, ttl)

    # ---- TTL policy ----

    def _matched_ttl(self, spell: Spell, event: CastEvent) -> float:
        """Authoritative TTL for matched casts comes from the spell DB.

        If `Spell.duration` is null, fall back to the OCR'd duration (it's
        less trustworthy but better than nothing), then to the configured
        default. Hard-capped to guard against either source being garbage.
        """
        if spell.duration is not None:
            ttl = spell.duration
        elif event.duration is not None:
            ttl = event.duration
        else:
            ttl = self._default_ttl_s
        return max(0.0, min(ttl, self._max_matched_ttl_s))

    def _unmatched_ttl(self, event: CastEvent) -> float:
        """Unmatched casts can't trust any single source.

        Use the OCR'd duration if present, otherwise the default, hard-capped
        at `max_unmatched_ttl_s` (default 10s — the user's rule of thumb).
        """
        ttl = event.duration if event.duration is not None else self._default_ttl_s
        return max(0.0, min(ttl, self._max_unmatched_ttl_s))

    # ---- helpers ----

    def _prune(self, now: float) -> None:
        self._matched = {k: e for k, e in self._matched.items() if e > now}
        self._unmatched = [e for e in self._unmatched if e.expiry > now]

    def _fuzzy_match(
        self,
        spell_a: str,
        target_a: str | None,
        spell_b: str,
        target_b: str | None,
    ) -> bool:
        spell_ratio = fuzz.token_set_ratio(spell_a.lower(), spell_b.lower())
        if spell_ratio < self._spell_threshold:
            return False
        if target_a is None and target_b is None:
            return True
        if target_a is None or target_b is None:
            return False
        target_ratio = fuzz.token_set_ratio(target_a.lower(), target_b.lower())
        return target_ratio >= self._target_threshold
