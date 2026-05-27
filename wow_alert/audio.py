"""Alert audio: TTS-renders phrases to WAVs ahead of time, plays them with winsound.

Two-phase design isolates expensive synthesis from the latency-sensitive playback
path:

  prerender(phrases)  — slow. Synthesizes each phrase to a cached WAV. Safe to
                        call repeatedly; existing WAVs are not re-rendered.
                        Call this from a setup path, never from a hot loop.
  play(phrase)        — fast. Either:
                          - str: looks up the cached WAV and plays it as-is.
                          - list[str]: stitches the cached WAVs into a
                            single concat clip on disk (cached under a
                            deterministic filename derived from the phrase
                            sequence) and plays that file. "BOP" + "Captain
                            Garrick" plays as a single "BOP Captain Garrick"
                            callout. Disk caching is necessary because
                            winsound's SND_ASYNC mode is incompatible with
                            SND_MEMORY — async playback requires a file path.
                        Phrases never prerendered raise KeyError — that is
                        a wiring bug in the caller.
"""
from __future__ import annotations

import logging
import sys
import wave
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
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 is not installed. Run `poetry install` to pick up "
                "the audio dependencies."
            ) from exc

        rendered = 0
        for phrase in phrases:
            target = self.cache_dir / _phrase_to_filename(phrase)
            self._phrase_paths[phrase] = target
            if target.exists():
                continue
            logger.info("Prerendering TTS phrase %r -> %s", phrase, target)
            try:
                engine = pyttsx3.init()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialize the system TTS engine ({exc}). "
                    f"On Windows, this usually means SAPI isn't available — "
                    f"check Windows speech settings."
                ) from exc
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

    def play(self, phrase: str | list[str]) -> None:
        """Play a single phrase or a concatenated sequence.

        list[str] is the Phase-D path used for action+target callouts.
        Missing entries inside a list (skipped instead of failing) are
        logged but don't abort playback — better to say "BOP" than to
        say nothing because the target name wasn't prerendered.
        """
        if self._muted:
            return
        if isinstance(phrase, list):
            self._play_concat(phrase)
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

    def _play_concat(self, phrases: list[str]) -> None:
        path = self._concat_wav_path(phrases)
        if path is None:
            logger.warning("No clips to concat for %r", phrases)
            return
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            logger.warning(
                "Audio playback skipped (non-Windows platform): concat %r", phrases,
            )

    def _concat_wav_path(self, phrases: list[str]) -> Path | None:
        """Return a filesystem path to the concatenated clip for `phrases`,
        building and caching it on disk if it doesn't already exist.

        Caching is keyed by the joined per-phrase filenames so repeated
        callouts like "BOP Captain Garrick" hit the cache after the first
        synthesis. Skipped sub-clips (missing path or missing file) are
        logged but don't abort the build — partial output is more useful
        than silence.
        """
        usable = [p for p in phrases if self._phrase_paths.get(p)
                  and self._phrase_paths[p].exists()]
        for p in phrases:
            if p not in usable:
                logger.warning("Skipping missing clip %r in concat", p)
        if not usable:
            return None

        key = "__".join(
            _phrase_to_filename(p).removesuffix(".wav") for p in usable
        )
        concat_dir = self.cache_dir / "concat"
        concat_dir.mkdir(parents=True, exist_ok=True)
        target = concat_dir / f"{key}.wav"
        if target.exists():
            return target

        with wave.open(str(target), "wb") as writer:
            first = True
            for phrase in usable:
                with wave.open(str(self._phrase_paths[phrase]), "rb") as reader:
                    if first:
                        writer.setparams(reader.getparams())
                        first = False
                    writer.writeframes(reader.readframes(reader.getnframes()))
        return target
