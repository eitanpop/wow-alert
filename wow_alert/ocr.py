"""OCR engine implementations.

This module hosts concrete implementations of the `OcrEngine` protocol defined
in `wow_alert.events`. The protocol exists so the pipeline depends only on the
shape `read(crop) -> [(text, conf, x_left, x_right)]` and the engine can be
swapped without touching the rest of the codebase.

`RapidOcrEngine` uses rapidocr-onnxruntime, chosen for low latency on short
stylized text typical of game UI. ONNX Runtime by default grabs every CPU
core for its thread pools, which starves the game; `thread_cap` constrains it
to a small number (default 2). The cap is applied two ways for belt-and-
suspenders: via environment variables (read by OpenMP / MKL backends at first
use) and via RapidOCR's own `intra_op_num_threads` / `inter_op_num_threads`
kwargs when supported by the installed version.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class RapidOcrEngine:
    """rapidocr-onnxruntime wrapper. Lazy-loads the ONNX session on first call."""

    def __init__(self, thread_cap: int = 2):
        self.thread_cap = max(1, thread_cap)
        self._reader = None

    def warmup(self) -> None:
        """Force the ONNX session to initialize so the first read() doesn't pay it."""
        self._ensure_reader()

    def _ensure_reader(self):
        if self._reader is None:
            # Constrain backend thread pools before rapidocr imports onnxruntime.
            # setdefault preserves any explicit user override from the env.
            os.environ.setdefault("OMP_NUM_THREADS", str(self.thread_cap))
            os.environ.setdefault("MKL_NUM_THREADS", str(self.thread_cap))

            from rapidocr_onnxruntime import RapidOCR

            logger.info("Initializing RapidOCR (thread_cap=%d)", self.thread_cap)
            try:
                self._reader = RapidOCR(
                    intra_op_num_threads=self.thread_cap,
                    inter_op_num_threads=self.thread_cap,
                )
            except TypeError:
                # Older / newer rapidocr versions may not accept these kwargs;
                # fall back to env-var-only control.
                logger.info("RapidOCR doesn't accept thread kwargs; using env vars only")
                self._reader = RapidOCR()
        return self._reader

    def read(self, crop: np.ndarray) -> list[tuple[str, float, float, float]]:
        reader = self._ensure_reader()
        result, _elapsed = reader(crop)
        if not result:
            return []
        out: list[tuple[str, float, float, float]] = []
        for entry in result:
            # rapidocr returns [polygon, text, confidence] per detection, where
            # `polygon` is a 4-point [[x,y], ...] outline of the text region.
            if len(entry) >= 3:
                polygon, text, conf = entry[0], entry[1], entry[2]
                xs = [float(pt[0]) for pt in polygon]
                out.append((str(text), float(conf), min(xs), max(xs)))
        return out

    def read_boxes(
        self, crop: np.ndarray
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Like `read()` but keeps the full bounding box per text region:
        `[(text, conf, (x1, y1, x2, y2))]`, crop-local coordinates. Used by
        calibration to read party-frame names *and* where each sits (to order
        them top-to-bottom)."""
        reader = self._ensure_reader()
        result, _elapsed = reader(crop)
        if not result:
            return []
        out: list[tuple[str, float, tuple[int, int, int, int]]] = []
        for entry in result:
            if len(entry) >= 3:
                polygon, text, conf = entry[0], entry[1], entry[2]
                xs = [float(pt[0]) for pt in polygon]
                ys = [float(pt[1]) for pt in polygon]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                out.append((str(text), float(conf), bbox))
        return out
