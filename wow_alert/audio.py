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

    def known_phrases(self) -> list[str]:
        """Every phrase prerendered this session. The set to re-render when
        switching voices so the new voice covers everything already in use."""
        return list(self._phrase_paths.keys())

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


class EdgeTtsAlertPlayer(PyttsxWinsoundAlertPlayer):
    """Neural TTS via Microsoft Edge's online voices (edge-tts).

    Overrides only `prerender`: each phrase is synthesized to MP3 with
    edge-tts, decoded to PCM with miniaudio, and written as a WAV at the same
    cache path the base class plays from — so `play` / concat / mute are
    inherited unchanged. Requires internet at prerender time; playback is
    offline from the cache.

    The cache is namespaced per voice (`<cache>/edge/<voice>/`) so switching
    voices — or coming from the pyttsx3 player, which writes the same
    filenames — re-renders cleanly instead of replaying stale clips.
    """

    def __init__(self, cache_dir: Path, voice: str = "en-US-AriaNeural"):
        self._base_cache_dir = Path(cache_dir)
        self._voice = voice
        super().__init__(self._voice_cache_dir(voice))

    @property
    def voice(self) -> str:
        return self._voice

    def _voice_cache_dir(self, voice: str) -> Path:
        safe = "".join(c if c.isalnum() else "_" for c in voice).strip("_") or "default"
        return self._base_cache_dir / "edge" / safe

    def set_voice(self, voice: str) -> None:
        """Switch the voice. Points the cache at the new voice's subdir but
        deliberately leaves the current phrase→WAV map intact, so playback
        keeps using the prior voice's clips until the next `prerender`
        repopulates the map for this voice — no silent gap. The new voice
        therefore goes live after the next prerender (i.e. Calibrate or the
        startup auto-apply)."""
        self._voice = voice
        self.cache_dir = self._voice_cache_dir(voice)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def prerender(self, phrases: list[str]) -> None:
        try:
            import asyncio

            import edge_tts
            import miniaudio
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts / miniaudio not installed. Run `poetry install` to "
                "pick up the neural-TTS dependencies, or set tts_engine to "
                "'pyttsx' in your config to use the offline system voice."
            ) from exc

        rendered = 0
        for phrase in phrases:
            target = self.cache_dir / _phrase_to_filename(phrase)
            self._phrase_paths[phrase] = target
            if target.exists():
                continue
            logger.info("Prerendering (edge-tts %s) %r -> %s", self._voice, phrase, target)
            mp3 = target.with_suffix(".mp3")
            try:
                asyncio.run(edge_tts.Communicate(phrase, self._voice).save(str(mp3)))
                decoded = miniaudio.decode_file(
                    str(mp3), nchannels=1, sample_rate=24000,
                )
                with wave.open(str(target), "wb") as w:
                    w.setnchannels(decoded.nchannels)
                    w.setsampwidth(decoded.sample_width)
                    w.setframerate(decoded.sample_rate)
                    w.writeframes(decoded.samples.tobytes())
                rendered += 1
            except Exception:
                # One bad phrase (network blip, odd text) shouldn't sink the
                # whole prerender — log and move on. Playback later skips or
                # surfaces the missing clip per the base class's handling.
                logger.warning(
                    "edge-tts failed to render %r; skipping", phrase, exc_info=True,
                )
            finally:
                if mp3.exists():
                    try:
                        mp3.unlink()
                    except OSError:
                        logger.debug("could not remove temp mp3 %s", mp3, exc_info=True)
        logger.info(
            "edge-tts prerender complete (%d new of %d total)", rendered, len(phrases),
        )
