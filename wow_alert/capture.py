"""Window capture via DXGI Desktop Duplication.

Uses `bettercam` (a maintained dxcam fork) rather than GDI BitBlt. The reason
is correctness, not just speed: BitBlt forces a GPU→CPU readback on every
grab, which synchronizes with the captured game's render thread and produces
a perceptible stutter. DXGI Desktop Duplication shares the GPU surface with
DWM, so grabbing is essentially free as far as the game is concerned. It is
also the only capture path that works when WoW switches between DXGI present
modes (loading screens, full-screen toggles) — BitBlt fails in that window.

Single-monitor / primary-output assumption: `bettercam.create()` binds to the
default output (the primary monitor). If WoW runs on a secondary monitor,
we'll need an `output_idx` flag — flagged here for future work.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import bettercam
import numpy as np
import win32gui

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureRegion:
    top: int
    left: int
    width: int
    height: int

    def as_dxgi_rect(self) -> tuple[int, int, int, int]:
        """bettercam region: (left, top, right, bottom). right/bottom exclusive."""
        return (self.left, self.top, self.left + self.width, self.top + self.height)


def find_capture_region(window_title: str) -> CaptureRegion | None:
    """Return the client-area capture region for a window matched by exact title."""
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        return None

    left_c, top_c, right_c, bottom_c = win32gui.GetClientRect(hwnd)
    width = right_c - left_c
    height = bottom_c - top_c
    if width <= 0 or height <= 0:
        return None

    left, top = win32gui.ClientToScreen(hwnd, (left_c, top_c))
    return CaptureRegion(top=top, left=left, width=width, height=height)


class WindowCapture:
    """Grabs frames from a window by title via DXGI Desktop Duplication.

    Refreshes the bounding region periodically in case the window has been
    moved or resized.
    """

    def __init__(self, window_title: str, refresh_interval: float = 1.0):
        self.window_title = window_title
        self.refresh_interval = refresh_interval
        self._region: CaptureRegion | None = None
        self._last_refresh = 0.0
        self._camera = bettercam.create(output_color="BGR")
        self._last_frame: np.ndarray | None = None

    def __enter__(self) -> "WindowCapture":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._camera.release()
        except Exception:
            logger.debug("camera release raised; ignoring", exc_info=True)

    def refresh_region(self) -> CaptureRegion | None:
        self._region = find_capture_region(self.window_title)
        self._last_refresh = time.monotonic()
        return self._region

    def region(self) -> CaptureRegion | None:
        return self._region

    def grab(self) -> np.ndarray | None:
        """Return a BGR frame, or None if the window can't be found.

        bettercam.grab returns None when DWM has not produced a new frame
        since the last call — a normal feature of Desktop Duplication, not a
        failure. We cache the last good frame and return it so the pipeline
        keeps detecting/tracking against the most recent capture instead of
        treating "no change" as "window not found".
        """
        now = time.monotonic()
        if self._region is None or (now - self._last_refresh) >= self.refresh_interval:
            self.refresh_region()
        if self._region is None:
            return None

        frame = self._camera.grab(region=self._region.as_dxgi_rect())
        if frame is not None:
            self._last_frame = frame
        return self._last_frame
