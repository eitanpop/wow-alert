"""Tests for AppConfig loading."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wow_alert.config import AppConfig, load_config


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "app.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


class TestLoadConfig:
    def test_loads_minimum(self, tmp_path: Path):
        cfg_path = write_yaml(tmp_path, {"model_path": "C:/models/best.pt"})
        cfg = load_config(cfg_path)
        assert cfg.model_path == Path("C:/models/best.pt")
        assert cfg.window_title == "World of Warcraft"  # default
        assert cfg.confidence == 0.4

    def test_cli_overrides_yaml(self, tmp_path: Path):
        cfg_path = write_yaml(tmp_path, {
            "model_path": "C:/models/old.pt",
            "confidence": 0.4,
        })
        cfg = load_config(cfg_path, overrides={
            "model_path": "C:/models/new.pt",
            "confidence": 0.6,
        })
        assert cfg.model_path == Path("C:/models/new.pt")
        assert cfg.confidence == 0.6

    def test_cli_none_does_not_clobber_yaml(self, tmp_path: Path):
        cfg_path = write_yaml(tmp_path, {"model_path": "C:/models/x.pt", "confidence": 0.5})
        cfg = load_config(cfg_path, overrides={"confidence": None})
        assert cfg.confidence == 0.5

    def test_env_var_overrides_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg_path = write_yaml(tmp_path, {"model_path": "C:/models/x.pt"})
        monkeypatch.setenv("WOW_ALERT_WINDOW_TITLE", "Other Game")
        cfg = load_config(cfg_path)
        assert cfg.window_title == "Other Game"

    def test_missing_required_field_raises(self, tmp_path: Path):
        cfg_path = write_yaml(tmp_path, {"window_title": "X"})
        with pytest.raises(Exception):
            load_config(cfg_path)

    def test_dungeon_pass_through(self, tmp_path: Path):
        cfg_path = write_yaml(tmp_path, {"model_path": "C:/models/x.pt"})
        cfg = load_config(cfg_path, overrides={"dungeon": "Operation: Floodgate"})
        assert cfg.dungeon == "Operation: Floodgate"
