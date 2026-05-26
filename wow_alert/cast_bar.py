"""Parse OCR output into a CastEvent.

Pure functions, no IO, no dependency on a particular OCR engine. The two example
cast bars in `examples/*.png` motivate the shape: spell name on the left, target
name (when present) on the right just before the duration. WoW cast bars
visually separate the spell text from the target/duration with whitespace —
the parser uses that gap to tell "no target, multi-word spell" apart from
"target present".

Pipeline:

  1. Pop the right-most numeric token as `duration`.
  2. If a clear x-gap exists between adjacent remaining tokens (above
     `GAP_FRACTION` of crop width), split there: tokens right of the gap form
     the `target`, tokens left form the `spell`.
  3. If no gap exceeds the threshold, all remaining tokens form the spell and
     `target` is None. This is the targetless-cast path.

The gap heuristic exploits a visual property of cast bars (target/duration is
right-aligned, separated from the spell text by whitespace) rather than a
positional rule like "second-to-last token is target", which fails for
multi-word spells that have no target.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from wow_alert.events import BBox, CastEvent

DURATION_RE = re.compile(r"^\d{1,3}(\.\d{1,2})?$")

# Minimum x-gap to treat as a spell/target separator, as a fraction of crop
# width. Intra-word spacing in cast-bar fonts is typically ~1-3% of crop width;
# the spell→target gap is typically ~10-25%. 6% gives clear margin both ways.
GAP_FRACTION = 0.06


@dataclass(frozen=True)
class OcrToken:
    text: str
    confidence: float
    x_left: float
    x_right: float

    @property
    def x_center(self) -> float:
        return (self.x_left + self.x_right) / 2.0


def _clean(token: str) -> str:
    """Strip stray punctuation that OCR commonly adds around game text."""
    return token.strip().strip(":,;.|*-_")


def _parse_duration(token: str) -> float | None:
    cleaned = _clean(token).replace("s", "").replace("S", "")
    if DURATION_RE.match(cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def parse_tokens(
    tokens: list[OcrToken],
    crop_width: int | None = None,
) -> tuple[str, str | None, float | None]:
    """Return (spell, target_or_None, duration_or_None) from OCR tokens.

    `crop_width` sets the gap threshold for separating spell from target. If
    omitted, the span of the supplied tokens is used as an approximation —
    fine for tests but the pipeline always passes the real crop width.
    Empty token list yields ("", None, None).
    """
    cleaned = [
        OcrToken(
            text=_clean(t.text),
            confidence=t.confidence,
            x_left=t.x_left,
            x_right=t.x_right,
        )
        for t in tokens
        if _clean(t.text)
    ]
    if not cleaned:
        return "", None, None

    cleaned.sort(key=lambda t: t.x_left)

    duration: float | None = None
    dur = _parse_duration(cleaned[-1].text)
    if dur is not None:
        duration = dur
        cleaned = cleaned[:-1]

    if not cleaned:
        return "", None, duration

    if crop_width is None or crop_width <= 0:
        crop_width = max(1, int(cleaned[-1].x_right - cleaned[0].x_left))
    gap_threshold = GAP_FRACTION * crop_width

    split_idx = -1
    max_gap = 0.0
    for i in range(len(cleaned) - 1):
        gap = cleaned[i + 1].x_left - cleaned[i].x_right
        if gap > max_gap:
            max_gap = gap
            split_idx = i

    target: str | None = None
    if split_idx >= 0 and max_gap >= gap_threshold:
        spell_tokens = cleaned[: split_idx + 1]
        target_tokens = cleaned[split_idx + 1 :]
        target = " ".join(t.text for t in target_tokens).strip() or None
    else:
        spell_tokens = cleaned

    spell = " ".join(t.text for t in spell_tokens).strip()
    return spell, target, duration


def make_cast_event(
    tokens: list[OcrToken],
    bbox: BBox,
    track_id: int,
    crop_width: int | None = None,
) -> CastEvent:
    spell, target, duration = parse_tokens(tokens, crop_width=crop_width)
    return CastEvent(
        spell=spell,
        target=target,
        duration=duration,
        bbox=bbox,
        track_id=track_id,
    )


def tokens_from_ocr_output(
    ocr_output: list[tuple[str, float, float, float]],
    crop_width: int,
) -> list[OcrToken]:
    """Adapter: turn the OcrEngine `(text, conf, x_left, x_right)` tuples into
    OcrTokens. `crop_width` is accepted for symmetry with engines that may not
    supply bboxes (none currently do; left in the signature to keep call sites
    stable).
    """
    del crop_width  # unused; kept for caller-side symmetry
    return [
        OcrToken(text=text, confidence=conf, x_left=x_left, x_right=x_right)
        for text, conf, x_left, x_right in ocr_output
    ]
