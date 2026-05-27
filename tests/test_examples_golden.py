"""Golden-file regression test against real cast-bar screenshots.

Runs the full OCR + parser stack against the bundled `examples/*.png`
and asserts the canonical (spell, target, duration) for each. Catches
regressions in either the OCR engine (different ONNX model, different
thread limits) or the cast-bar parser (gap heuristic, duration regex).

Marked slow because RapidOCR's first read initializes ONNX, which
typically takes 1-3 s. Skipped when RapidOCR isn't installed so the
core test suite stays green on lean environments.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from wow_alert.cast_bar import make_cast_event, tokens_from_ocr_output

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


pytest.importorskip(
    "rapidocr_onnxruntime",
    reason="RapidOCR not installed; skipping golden-file OCR tests",
)


@pytest.fixture(scope="module")
def ocr():
    from wow_alert.ocr import RapidOcrEngine

    engine = RapidOcrEngine(thread_cap=2)
    engine.warmup()
    return engine


@pytest.mark.slow
class TestGoldenFiles:
    def test_no_target_cast_bar(self, ocr):
        """`Vigilant Defense` mid-cast, no target, ~3.3 s remaining.

        Verifies: multi-word spell, no spell/target gap (the parser
        should NOT spuriously split "Vigilant" and "Defense" into
        spell+target), duration on the right.
        """
        path = EXAMPLES / "cast_bar_no_target.png"
        crop = cv2.imread(str(path))
        assert crop is not None, f"failed to read {path}"

        tokens = tokens_from_ocr_output(ocr.read(crop))
        cast = make_cast_event(
            tokens, bbox=(0, 0, crop.shape[1], crop.shape[0]),
            track_id=1, crop_width=crop.shape[1],
        )

        assert cast.spell.lower() == "vigilant defense", (
            f"OCR/parser produced spell={cast.spell!r}"
        )
        assert cast.target is None, (
            f"expected no target on this cast bar; got {cast.target!r}"
        )
        assert cast.duration is not None
        assert 3.0 <= cast.duration <= 3.5, (
            f"expected duration ~3.3; got {cast.duration!r}"
        )

    def test_interruptible_target_cast_bar(self, ocr):
        """`Rampage` cast on target `Troodon`, ~2.3 s remaining.

        Verifies the gap heuristic correctly separates the spell from
        the right-side target/duration when both are present.
        """
        path = EXAMPLES / "cast_bar_interruptible_target.png"
        crop = cv2.imread(str(path))
        assert crop is not None, f"failed to read {path}"

        tokens = tokens_from_ocr_output(ocr.read(crop))
        cast = make_cast_event(
            tokens, bbox=(0, 0, crop.shape[1], crop.shape[0]),
            track_id=1, crop_width=crop.shape[1],
        )

        assert cast.spell.lower() == "rampage", (
            f"OCR/parser produced spell={cast.spell!r}"
        )
        assert cast.target is not None and "troodon" in cast.target.lower(), (
            f"expected target containing 'Troodon'; got {cast.target!r}"
        )
        assert cast.duration is not None
        assert 2.0 <= cast.duration <= 2.6, (
            f"expected duration ~2.3; got {cast.duration!r}"
        )
