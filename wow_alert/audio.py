"""Alert audio: TTS-renders phrases to WAVs ahead of time, plays them with winsound.

Two-phase design isolates expensive synthesis from the latency-sensitive playback
path:

  prerender(phrases)  — slow. Synthesizes each phrase to a cached WAV. Safe to
                        call repeatedly; existing WAVs are not re-rendered.
                        Call this from a setup path, never from a hot loop.
  play(phrase)        — fast. Either:
                          - str: looks up the cached WAV and plays it as-is.
                          - list[str]: stitches the cached WAVs into a single
                            concat WAV in a session-only temp dir and plays
                            that file. The temp dir is wiped on app exit so
                            concats never persist between sessions — the
                            per-roster-per-dungeon combinatorial explosion
                            stays bounded by what's played in one session.
                            winsound's SND_ASYNC needs a file path; we use a
                            temp file rather than memory streaming.
                        Phrases never prerendered raise KeyError — that is
                        a wiring bug in the caller.

Roster-name eviction: when a new roster loads, the previous roster's name
WAVs are deleted via `evict_phrases(names)`. Keeps a pug player's cache
flat over time — only the current group's names live on disk.
"""
from __future__ import annotations

import atexit
import logging
import shutil
import sys
import tempfile
import threading
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
        # Per-phrase coordination so concurrent prerender() callers don't
        # try to render the same phrase twice. The startup catalog warmup
        # runs on a background thread; if a calibration prerender (which
        # asks for a subset of the same phrases) races it on a cold cache,
        # both would otherwise hit edge-tts for the same phrase and decode
        # half-written MP3s. With the dict + lock, the second caller sees
        # the in-flight Event and waits instead of duplicating the call.
        self._rendering: dict[str, threading.Event] = {}
        self._rendering_lock = threading.Lock()
        # Session-only directory for stitched concat clips. Wiped on app
        # exit so concat WAVs never accumulate across sessions — addresses
        # the per-roster-per-dungeon combinatorial explosion that would
        # otherwise pile up GBs over weeks of pug play.
        self._session_temp = Path(tempfile.mkdtemp(prefix="wow_alert_concat_"))
        atexit.register(self._cleanup_session_temp)

    def _cleanup_session_temp(self) -> None:
        try:
            shutil.rmtree(self._session_temp, ignore_errors=True)
        except Exception:
            logger.debug("session-temp cleanup raised", exc_info=True)

    def evict_phrases(self, phrases: list[str]) -> None:
        """Delete cached WAVs for `phrases` from disk + the in-memory map.

        Used by the Roster flow: when "Load party members" runs, the prior
        roster's name clips are evicted before the new names render. Caps
        on-disk size at the current group plus the bounded content catalog.
        Missing files are ignored — calling on a fresh install is safe.
        """
        for phrase in phrases:
            path = self._phrase_paths.pop(phrase, None)
            if path is None:
                path = self.cache_dir / _phrase_to_filename(phrase)
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("evict_phrases: couldn't remove %s: %s", path, exc)

    def _claim_phrase(
        self, phrase: str, force_owner: bool = True,
    ) -> threading.Event | None:
        """Per-phrase render coordination.

        First caller for a phrase gets a fresh Event back, becomes the
        owner, and renders. Concurrent callers see the existing Event and
        get None (with `force_owner=True`) or get the Event back (with
        `force_owner=False`) so they can `event.wait()` for the owner to
        finish.

        The two-call dance is so we don't `wait()` while still holding the
        lock, which would deadlock other phrases.
        """
        with self._rendering_lock:
            existing = self._rendering.get(phrase)
            if existing is not None:
                return None if force_owner else existing
            event = threading.Event()
            self._rendering[phrase] = event
            return event

    def _release_phrase(self, phrase: str) -> None:
        """Mark a phrase rendered (success or failure) and wake waiters."""
        with self._rendering_lock:
            event = self._rendering.pop(phrase, None)
        if event is not None:
            event.set()

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
            # Per-phrase coordination so concurrent prerender() calls don't
            # both kick off SAPI on the same phrase. Same dance as the
            # edge-tts subclass (see _claim_phrase docstring).
            event = self._claim_phrase(phrase)
            if event is None:
                event = self._claim_phrase(phrase, force_owner=False)
                if event is not None:
                    event.wait()
                continue
            logger.info("Prerendering TTS phrase %r -> %s", phrase, target)
            try:
                engine = pyttsx3.init()
            except Exception as exc:
                self._release_phrase(phrase)
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
                self._release_phrase(phrase)
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
        """Build the stitched WAV in this session's temp dir, return its path.

        Within a session we cache by joined-filename key so repeat callouts
        (same spell + same target) skip rebuild. Across sessions there's
        nothing to cache against — the temp dir is wiped on exit. That
        keeps the per-roster-per-dungeon combinatorial pile-up bounded by
        whatever's actually been played in the current session, not by
        cumulative playtime.

        Skipped sub-clips (missing prerender) are logged; partial output is
        better than silence.
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
        target = self._session_temp / f"{key}.wav"
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
            # Coordinate with other threads: if another prerender call is
            # already rendering this phrase, wait for it instead of starting
            # a duplicate edge-tts call that would race the same MP3 path.
            event = self._claim_phrase(phrase)
            if event is None:
                event = self._claim_phrase(phrase, force_owner=False)
                if event is not None:
                    event.wait()
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
                self._release_phrase(phrase)
                if mp3.exists():
                    try:
                        mp3.unlink()
                    except OSError:
                        logger.debug("could not remove temp mp3 %s", mp3, exc_info=True)
        logger.info(
            "edge-tts prerender complete (%d new of %d total)", rendered, len(phrases),
        )
