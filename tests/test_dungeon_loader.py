"""Dungeon-name → file resolution, including typo/OCR tolerance."""
from wow_alert.config import REPO_ROOT
from wow_alert.dungeon_loader import load_dungeon_config, slugify

CONFIG = REPO_ROOT / "config"


def test_exact_name_loads_spells():
    spells, _ = load_dungeon_config(CONFIG, "Nexus-Point Xenas")
    assert any(s.id.startswith("nexus_") for s in spells)


def test_punctuation_and_spacing_dont_matter():
    # Hyphen vs space vs apostrophe all slug the same way.
    a, _ = load_dungeon_config(CONFIG, "Nexus Point Xenas")
    b, _ = load_dungeon_config(CONFIG, "nexus-point  xenas")
    assert {s.id for s in a} == {s.id for s in b}
    assert any(s.id.startswith("nexus_") for s in a)


def test_typo_fuzzy_matches_the_right_dungeon():
    # A misspelling that used to load nothing now resolves to the real file.
    spells, _ = load_dungeon_config(CONFIG, "Nexus Point Xenis")
    assert any(s.id.startswith("nexus_") for s in spells)


def test_unrelated_name_matches_nothing():
    # Far-off text must NOT mis-resolve to a real dungeon — only globals load.
    spells, _ = load_dungeon_config(CONFIG, "Zzqq Totally Made Up Place")
    assert not any(s.id.startswith("nexus_") for s in spells)


def test_slugify_idempotent():
    assert slugify(slugify("Magisters' Terrace")) == slugify("Magisters' Terrace")
