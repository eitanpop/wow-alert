"""Tests for the alert player. winsound + pyttsx3 are mocked."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wow_alert.audio import PyttsxWinsoundAlertPlayer, _phrase_to_filename


class TestPhraseToFilename:
    def test_alnum_preserved(self):
        assert _phrase_to_filename("DANGER") == "DANGER.wav"

    def test_special_chars_replaced(self):
        assert _phrase_to_filename("BOP JOHN!") == "BOP_JOHN.wav"

    def test_empty(self):
        assert _phrase_to_filename("") == "_blank.wav"


class TestAlertPlayer:
    def test_prerender_writes_wavs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake_pyttsx3 = types.ModuleType("pyttsx3")
        engine = MagicMock()
        # save_to_file should produce a real file so play() finds it.
        def save_to_file(text, dest):
            Path(dest).write_bytes(b"RIFF")  # any bytes; we won't actually play
        engine.save_to_file.side_effect = save_to_file
        fake_pyttsx3.init = MagicMock(return_value=engine)
        monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)

        player = PyttsxWinsoundAlertPlayer(cache_dir=tmp_path)
        player.prerender(["DANGER"])
        assert (tmp_path / "DANGER.wav").exists()
        engine.save_to_file.assert_called_once()
        engine.runAndWait.assert_called_once()

    def test_play_calls_winsound(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Stub pyttsx3 and ensure WAV exists
        fake_pyttsx3 = types.ModuleType("pyttsx3")
        engine = MagicMock()
        engine.save_to_file.side_effect = lambda t, d: Path(d).write_bytes(b"RIFF")
        fake_pyttsx3.init = MagicMock(return_value=engine)
        monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)

        # Stub winsound (only loaded on win32; we force-inject it)
        fake_winsound = types.ModuleType("winsound")
        fake_winsound.PlaySound = MagicMock()
        fake_winsound.SND_FILENAME = 0x20000
        fake_winsound.SND_ASYNC = 0x1
        monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
        monkeypatch.setattr(sys, "platform", "win32")

        player = PyttsxWinsoundAlertPlayer(cache_dir=tmp_path)
        player.prerender(["DANGER"])
        player.play("DANGER")
        fake_winsound.PlaySound.assert_called_once()

    def test_play_missing_phrase_raises(self, tmp_path: Path):
        player = PyttsxWinsoundAlertPlayer(cache_dir=tmp_path)
        with pytest.raises(KeyError):
            player.play("NEVER_PRERENDERED")

    def test_muted_skips_play(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake_pyttsx3 = types.ModuleType("pyttsx3")
        engine = MagicMock()
        engine.save_to_file.side_effect = lambda t, d: Path(d).write_bytes(b"RIFF")
        fake_pyttsx3.init = MagicMock(return_value=engine)
        monkeypatch.setitem(sys.modules, "pyttsx3", fake_pyttsx3)

        fake_winsound = types.ModuleType("winsound")
        fake_winsound.PlaySound = MagicMock()
        fake_winsound.SND_FILENAME = 0x20000
        fake_winsound.SND_ASYNC = 0x1
        monkeypatch.setitem(sys.modules, "winsound", fake_winsound)
        monkeypatch.setattr(sys, "platform", "win32")

        player = PyttsxWinsoundAlertPlayer(cache_dir=tmp_path)
        player.prerender(["DANGER"])
        player.set_muted(True)
        player.play("DANGER")
        fake_winsound.PlaySound.assert_not_called()
