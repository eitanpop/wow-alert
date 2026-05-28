"""Shared event types and protocols.

Centralized here so detector / ocr / cast_bar / rules / pipeline don't import
each other for type definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field, field_validator


BBox = tuple[int, int, int, int]  # x1, y1, x2, y2


class Severity(str, Enum):
    DANGER = "danger"
    INFO = "info"
    IGNORE = "ignore"


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: BBox


@dataclass(frozen=True)
class CastEvent:
    spell: str
    target: str | None
    duration: float | None
    bbox: BBox
    track_id: int


class RuleOutput:
    """Base type for anything the rule engine produces in response to screen state."""


@dataclass(frozen=True)
class Alert(RuleOutput):
    """A pre-rendered audible alert with a human-readable log message."""

    severity: Severity
    phrase: str          # key that AlertPlayer looks up to play the cached WAV
    message: str         # for log panes / debug output


@dataclass(frozen=True)
class Recommendation(RuleOutput):
    """A class-specific action suggestion (e.g. "cast BOP on John").

    Produced when the rule engine has team roster + cooldown context, not just
    cast events.
    """

    action: str          # e.g. "BOP"
    target: str          # teammate name to act on
    phrase: str          # key that AlertPlayer looks up
    message: str
    # Optional rendered prefix played before `phrase` (e.g. the spell
    # name). Pipeline stitches as `[phrase_prefix?, phrase, target?]`.
    phrase_prefix: str = ""


@dataclass(frozen=True)
class RuleDecisionContext:
    """Everything the rule engine needs to decide what to do about one cast.

    Built upstream by the pipeline after spell-DB lookup and dedupe. The
    engine itself does no DB queries, no dedupe, no temporal state — given
    the same context, `decide()` returns the same answer. This makes the
    engine trivially unit-testable: construct a literal context, call
    decide, assert.

    Only matched casts reach decide() (unmatched ones are filtered upstream),
    so `spell` is always non-null.
    """

    # The matched spell from the DB. Source of truth for severity, phrase,
    # canonical name. Set by the pipeline after the dedupe lookup.
    spell: "Spell"

    # The cast event itself. `cast.target` is the canonical (roster-
    # resolved) name when canonical_target is set, otherwise the raw OCR.
    cast: "CastEvent"

    # Roster-canonical form of the target, or None if no roster match (or
    # no target at all). Redundant with cast.target after pipeline rewrite,
    # but kept as the explicit "this was canonicalized" signal — handy for
    # policy decisions that care about "is this teammate or boss?".
    canonical_target: str | None = None

    # Wider context the engine may consult for non-trivial decisions.
    dungeon: str | None = None
    cooldowns: dict[int, bool] = field(default_factory=dict)
    roster: list[str] = field(default_factory=list)
    # canonical roster name -> "tank" | "healer" | "dps". Members whose
    # role wasn't identified during calibration are absent — a missing
    # entry means "unknown", not "this member isn't a tank/healer/dps".
    roles: dict[str, str] = field(default_factory=dict)
    # The current player's action catalog. Populated by the rule engine if
    # the caller doesn't supply one (engine holds them in shared state for
    # the hot path). Listed here so tests can construct a context with
    # explicit actions; production code can leave the default empty list.
    class_actions: list = field(default_factory=list)


class Spell(BaseModel):
    """A spell entry in the rule engine's lookup table."""

    id: str = Field(
        description=(
            "Stable identifier. Not used for matching; useful for rule engines "
            "that cross-reference spells by id."
        ),
    )
    name: str = Field(
        description=(
            "Canonical display name. Primary fuzzy-match target for OCR'd "
            "spell text."
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Known OCR misreads of `name` (e.g. character-level substitutions). "
            "Searched alongside `name` at lookup time."
        ),
    )
    dungeon: str | None = Field(
        default=None,
        description=(
            "Restricts this spell to a single dungeon. Null means the spell "
            "can match in any context."
        ),
    )
    severity: Severity = Field(
        default=Severity.INFO,
        description=(
            "Drives whether the rule engine produces an Alert and how the UI "
            "presents it."
        ),
    )
    phrase: str = Field(
        default="DANGER",
        description=(
            "AlertPlayer phrase key to play on a danger-severity match. Must "
            "have been prerendered before runtime."
        ),
    )
    duration: float | None = Field(
        default=None,
        description=(
            "Authoritative cast time in seconds. When present, the dedupe "
            "TTL for matched casts uses this rather than OCR-parsed numbers, "
            "which are often garbled. Null means fall back to the OCR'd "
            "duration (capped) or the configured default."
        ),
    )

    # Extended metadata — optional. Consumed by rule engines that have
    # roster/class context; ignored by simpler engines.
    cast_by: list[str] = Field(
        default_factory=list,
        description="Kinds of units that cast this spell (e.g. 'boss', 'mob').",
    )
    school: str | None = Field(
        default=None,
        description="Spell school (e.g. 'physical', 'shadow').",
    )
    interruptible: bool | None = Field(
        default=None,
        description="Whether the cast can be interrupted by an interrupt ability.",
    )
    notes: str | None = Field(
        default=None,
        description="Free-text reminder for human readers.",
    )

    @field_validator("aliases", "cast_by", mode="before")
    @classmethod
    def _scalar_to_list(cls, v: Any) -> Any:
        """Accept a bare string where a list is expected.

        Lets `spells.yaml` use either `cast_by: mob` or `cast_by: [mob, boss]`
        without a confusing pydantic validation error for the common
        single-item case.
        """
        if isinstance(v, str):
            return [v]
        return v


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...
    def set_confidence(self, value: float) -> None: ...


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, crop: np.ndarray) -> list[tuple[str, float, float, float]]:
        """Return one tuple per detected text region: (text, confidence, x_left, x_right).

        x_left/x_right are pixel coordinates within the crop. The parser uses them
        to find the visual gap between the spell text and the target/duration on
        the right side of a cast bar.
        """
        ...


@runtime_checkable
class SpellDb(Protocol):
    def lookup(self, spell_text: str, target_text: str | None) -> Spell | None: ...
    def all_phrases(self) -> list[str]: ...


@runtime_checkable
class AlertPlayer(Protocol):
    def prerender(self, phrases: list[str]) -> None: ...

    def play(self, phrase: str | list[str]) -> None:
        """Play one phrase, or concatenate a list of phrases into one clip.

        The list form stitches prerendered phrase WAVs end-to-end —
        used for action+target callouts like ["BOP", "Captain Garrick"].
        """
        ...


__all__ = [
    "BBox",
    "Severity",
    "Detection",
    "CastEvent",
    "RuleOutput",
    "Alert",
    "Recommendation",
    "RuleDecisionContext",
    "Spell",
    "Detector",
    "OcrEngine",
    "SpellDb",
    "AlertPlayer",
]
