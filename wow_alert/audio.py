"""Alert audio: TTS-renders phrases to WAVs ahead of time, plays them with winsound.

Two-phase design isolates expensive synthesis from the latency-sensitive playback
path:

  prerender(phrases)  — slow. Synthesizes each phrase to a cached WAV. Safe to
                        call repeatedly; existing WAVs are not re-rendered.
                        Call this from a setup path, never from a hot loop.
  play(phrase)        — fast. Looks up the cached WAV path and dispatches it via
                        winsound's SND_ASYNC, which returns immediately.
                        A phrase that was never prerendered raises KeyError —
                        that is a wiring bug in the caller.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _phrase_to_filename(phrase: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in phrase).strip("_")
    return f"{safe}.wav" if safe else "_blank.wav"


class PyttsxWinsoundAlertPlayer:
    """TTS-pre-rendered alert player. Windows-only (winsound)."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._phrase_paths: dict[str, Path] = {}
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, value: bool) -> None:
        self._muted = value

    def prerender(self, phrases: list[str]) -> None:
        """Render any not-yet-cached phrases to WAVs.

        A fresh pyttsx3 engine is created per phrase rather than reused across
        the loop. Reusing a single engine deadlocks SAPI on Windows after a
        handful of `save_to_file` + `runAndWait` cycles — the COM-backed
        engine's internal queue gets stuck and the next `runAndWait` never
        returns. Per-phrase init costs a few ms but completes deterministically.
        """
        import pyttsx3

        rendered = 0
        for phrase in phrases:
            target = self.cache_dir / _phrase_to_filename(phrase)
            self._phrase_paths[phrase] = target
            if target.exists():
                continue
            logger.info("Prerendering TTS phrase %r -> %s", phrase, target)
            engine = pyttsx3.init()
            try:
                engine.save_to_file(phrase, str(target))
                engine.runAndWait()
                rendered += 1
            finally:
                try:
                    engine.stop()
                except Exception:
                    logger.debug("pyttsx3 engine.stop() raised; ignoring", exc_info=True)
                del engine
        logger.info(
            "TTS prerender complete (%d new of %d total)", rendered, len(phrases)
        )

    def play(self, phrase: str) -> None:
        if self._muted:
            return
        path = self._phrase_paths.get(phrase)
        if path is None:
            raise KeyError(f"Phrase not prerendered: {phrase!r}")
        if not path.exists():
            raise FileNotFoundError(f"Phrase WAV missing on disk: {path}")
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            logger.warning("Audio playback skipped (non-Windows platform): %s", path)
