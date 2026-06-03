"""Tests for the cast-bar parser. Pure functions; no IO."""
from __future__ import annotations

import pytest

from wow_alert.cast_bar import (
    OcrToken,
    make_cast_event,
    parse_tokens,
    tokens_from_ocr_output,
)


CROP_WIDTH = 1000  # arbitrary; tokens below use absolute pixel positions inside this


def tok(text: str, x_left: float, x_right: float, conf: float = 0.9) -> OcrToken:
    return OcrToken(text=text, confidence=conf, x_left=x_left, x_right=x_right)


class TestParseTokens:
    def test_empty(self):
        assert parse_tokens([], crop_width=CROP_WIDTH) == ("", None, None)

    def test_spell_only(self):
        spell, target, duration = parse_tokens(
            [tok("Polymorph", 50, 200)], crop_width=CROP_WIDTH
        )
        assert spell == "Polymorph"
        assert target is None
        assert duration is None

    def test_spell_and_duration(self):
        spell, target, duration = parse_tokens(
            [tok("Polymorph", 50, 200), tok("2.3", 880, 940)],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Polymorph"
        assert target is None
        assert duration == pytest.approx(2.3)

    def test_spell_target_duration(self):
        # Spell on the left, gap, then target + duration on the right.
        spell, target, duration = parse_tokens(
            [
                tok("Polymorph", 30, 200),
                tok("John", 600, 700),
                tok("3.0", 880, 940),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Polymorph"
        assert target == "John"
        assert duration == pytest.approx(3.0)

    def test_multi_word_spell_with_target(self):
        # "Vigilant Defense" as two OCR boxes with a small intra-word gap,
        # then a big gap before the target.
        spell, target, duration = parse_tokens(
            [
                tok("Vigilant", 30, 180),
                tok("Defense", 200, 360),     # intra-word gap = 20px (~2%)
                tok("Tank", 700, 800),        # spell→target gap = 340px (~34%)
                tok("12.0", 870, 950),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Vigilant Defense"
        assert target == "Tank"
        assert duration == pytest.approx(12.0)

    def test_multi_word_spell_no_target(self):
        # Multi-word spell with no target: a positional "last token is the
        # target" rule would misclassify "Defense" as the target. The
        # gap-based parser instead sees the small intra-word gap and groups
        # both words into the spell.
        spell, target, duration = parse_tokens(
            [
                tok("Vigilant", 30, 180),
                tok("Defense", 200, 360),
                tok("3.3", 880, 940),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Vigilant Defense"
        assert target is None
        assert duration == pytest.approx(3.3)

    def test_long_spell_spills_past_center_no_target(self):
        # Spell text reaches past the visual center but is still contiguous
        # (small intra-word gaps). No big gap → no target.
        spell, target, duration = parse_tokens(
            [
                tok("Devastating", 30, 280),
                tok("Shadow", 300, 460),
                tok("Bolt", 480, 580),
                tok("4.1", 880, 940),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Devastating Shadow Bolt"
        assert target is None
        assert duration == pytest.approx(4.1)

    def test_long_spell_with_target(self):
        # Same long spell but a clearly separated target on the right.
        spell, target, duration = parse_tokens(
            [
                tok("Devastating", 30, 280),
                tok("Shadow", 300, 460),
                tok("Bolt", 480, 580),
                tok("Healer", 740, 840),   # gap of 160px (~16%)
                tok("4.1", 880, 940),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Devastating Shadow Bolt"
        assert target == "Healer"
        assert duration == pytest.approx(4.1)

    def test_duration_with_s_suffix(self):
        _, _, duration = parse_tokens(
            [tok("Polymorph", 50, 200), tok("2.3s", 880, 940)],
            crop_width=CROP_WIDTH,
        )
        assert duration == pytest.approx(2.3)

    def test_unparseable_trailing_token_treated_as_target(self):
        # No duration; the trailing non-numeric token sits past the gap, so
        # it becomes the target.
        spell, target, duration = parse_tokens(
            [tok("Polymorph", 30, 200), tok("abc", 700, 800)],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Polymorph"
        assert target == "abc"
        assert duration is None

    def test_numeric_trailing_token_not_treated_as_target(self):
        # A nameplate health/percent number past the gap is not a unit name.
        # It must not become the target, or it would veto an exact spell
        # match in the lookup's roster check.
        spell, target, duration = parse_tokens(
            [tok("Flaming", 30, 110), tok("Updraft", 112, 200), tok("0", 700, 720)],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Flaming Updraft"
        assert target is None
        assert duration is None

    def test_punctuation_stripped(self):
        spell, _, _ = parse_tokens(
            [tok("Polymorph:", 50, 200)], crop_width=CROP_WIDTH
        )
        assert spell == "Polymorph"

    def test_tokens_sorted_by_x(self):
        # Same tokens in scrambled order.
        spell, target, duration = parse_tokens(
            [
                tok("3.0", 880, 940),
                tok("John", 600, 700),
                tok("Polymorph", 30, 200),
            ],
            crop_width=CROP_WIDTH,
        )
        assert spell == "Polymorph"
        assert target == "John"
        assert duration == pytest.approx(3.0)

    def test_gap_threshold_relative_to_crop_width(self):
        # Same absolute pixel gap, doubled crop width → gap now below
        # threshold → treated as contiguous spell text.
        tokens = [
            tok("Foo", 30, 180),
            tok("Bar", 240, 380),       # 60px gap
            tok("2.0", 880, 940),
        ]
        # crop_width=500 → threshold = 30px; 60px gap > threshold → target
        spell, target, _ = parse_tokens(tokens, crop_width=500)
        assert spell == "Foo"
        assert target == "Bar"

        # crop_width=2000 → threshold = 120px; 60px gap < threshold → no target
        spell, target, _ = parse_tokens(tokens, crop_width=2000)
        assert spell == "Foo Bar"
        assert target is None


class TestTokensFromOcrOutput:
    def test_empty(self):
        assert tokens_from_ocr_output([]) == []

    def test_passes_through_bboxes(self):
        out = tokens_from_ocr_output(
            [
                ("A", 0.9, 10.0, 50.0),
                ("B", 0.8, 100.0, 140.0),
            ],
        )
        assert len(out) == 2
        assert out[0].text == "A"
        assert out[0].x_left == 10.0
        assert out[0].x_right == 50.0
        assert out[1].x_left == 100.0
        assert out[1].x_right == 140.0


class TestMakeCastEvent:
    def test_full_pipeline(self):
        tokens = [
            tok("Polymorph", 30, 200),
            tok("John", 600, 700),
            tok("3.0", 880, 940),
        ]
        cast = make_cast_event(
            tokens, bbox=(10, 20, 110, 50), track_id=7, crop_width=CROP_WIDTH
        )
        assert cast.spell == "Polymorph"
        assert cast.target == "John"
        assert cast.duration == pytest.approx(3.0)
        assert cast.track_id == 7
        assert cast.bbox == (10, 20, 110, 50)
