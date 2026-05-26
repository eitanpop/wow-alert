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
    other fields supply detail for narrative logging and downstream use:
      - `canonical_spell` is populated for MATCHED_* (the Spell from the DB).
      - `canonical_target` is populated for MATCHED_* when the raw OCR
        target fuzzy-matched a roster entry. The pipeline rewrites
        `cast.target` to this value before the rule engine sees it, so
        alerts say "BOP John" rather than "BOP Jhon"/"BOP J ohn".
      - `ttl_s` is the TTL that was applied when registering (0 for
        duplicates so the pipeline can log a faithful "skipped" message
        without inventing a TTL).
    """

    disposition: Disposition
    canonical_spell: Spell | None
    canonical_target: str | None
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
        max_matched_ttl_s: float = 10.0,
        max_unmatched_ttl_s: float = 10.0,
        spell_fuzzy_threshold: int = 80,
        target_fuzzy_threshold: int = 70,
        roster: list[str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._spell_db = spell_db
        self._default_ttl_s = default_ttl_s
        self._max_matched_ttl_s = max_matched_ttl_s
        self._max_unmatched_ttl_s = max_unmatched_ttl_s
        self._spell_threshold = spell_fuzzy_threshold
        self._target_threshold = target_fuzzy_threshold
        self._clock = clock
        # Roster is mutable at runtime — calibration can refresh it without
        # an app restart. Stored alongside a lowercased index for fuzzy
        # matching; both are rebuilt by `set_roster`.
        self._roster: list[str] = []
        self._roster_lower: list[str] = []
        self.set_roster(roster or [])
        # Matched cache: (spell.id, canonical_target) -> expiry timestamp.
        # Using canonical_target (the roster-resolved name) rather than raw
        # OCR text is what collapses "Meredy" / "MeredyH2" / "Meredy H" to
        # a single dedupe entry.
        self._matched: dict[tuple[str, str | None], float] = {}
        # Unmatched cache: list of entries fuzzy-compared at lookup
        self._unmatched: list[_UnmatchedEntry] = []

    def set_roster(self, names: list[str]) -> None:
        """Update the roster used for canonical-target resolution.

        Safe to call at any time; affects all subsequent `process` calls.
        Does NOT invalidate the existing matched cache — that's intentional,
        a mid-fight recalibration shouldn't make in-flight casts re-alert.
        """
        self._roster = list(names)
        self._roster_lower = [n.lower() for n in self._roster]

    def process(self, event: CastEvent) -> DedupeOutcome:
        now = self._clock()
        self._prune(now)

        spell = self._spell_db.lookup(event.spell, event.target)

        if spell is not None:
            # Resolve target through roster before caching: jittered OCR
            # variants of the same teammate name should produce the same
            # canonical_target, hence the same cache key.
            canonical_target = self._canonical_target(event.target)
            key = (spell.id, canonical_target)
            expiry = self._matched.get(key)
            if expiry is not None and now < expiry:
                return DedupeOutcome(
                    Disposition.MATCHED_DUPLICATE, spell, canonical_target, 0.0
                )
            ttl = self._matched_ttl(spell, event)
            self._matched[key] = now + ttl
            return DedupeOutcome(
                Disposition.MATCHED_NEW, spell, canonical_target, ttl
            )

        # Path B: not in spell DB. Fuzzy compare against recent unmatched.
        # No canonicalization here — without a DB match we don't trust the
        # target field enough to claim it's a known roster member.
        for entry in self._unmatched:
            if self._fuzzy_match(event.spell, event.target, entry.spell, entry.target):
                return DedupeOutcome(Disposition.UNMATCHED_DUPLICATE, None, None, 0.0)

        ttl = self._unmatched_ttl(event)
        self._unmatched.append(
            _UnmatchedEntry(spell=event.spell, target=event.target, expiry=now + ttl)
        )
        return DedupeOutcome(Disposition.UNMATCHED_NEW, None, None, ttl)

    def _canonical_target(self, target: str | None) -> str | None:
        """Resolve a raw OCR target to a roster entry via fuzzy match.

        Returns the canonical roster name if any entry scores above the
        threshold; otherwise returns `target` unchanged.

        Uses max(token_set_ratio, partial_ratio) rather than either alone:
          - token_set_ratio catches cases where OCR preserved tokens
            ("Meredy H" vs "Meredy Huntswell").
          - partial_ratio catches cases where OCR merged tokens or
            chopped suffixes ("MeredyH2" vs "Meredy Huntswell"), where
            no whole token overlaps but a substring still matches well.

        The fallback (return raw text) is intentional: with no roster the
        cache still works by raw target, which is no worse than the
        pre-Phase-B behavior. With a roster but no fuzzy match, the raw
        text is preserved so a non-roster target (e.g., a boss tank-swap)
        still gets its own cache entry.
        """
        if target is None or not self._roster_lower:
            return target
        target_lower = target.lower()
        best_score = 0.0
        best_idx = -1
        for idx, candidate in enumerate(self._roster_lower):
            score = max(
                fuzz.token_set_ratio(target_lower, candidate),
                fuzz.partial_ratio(target_lower, candidate),
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= self._target_threshold:
            return self._roster[best_idx]
        return target

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
