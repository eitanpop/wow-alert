"""Local-only screen calibration.

The user draws two regions (party frames + cooldown manager) in the
region-confirm dialog. This module reads what's inside them:

  Party path — local OCR (rapidocr) reads each party-slot name from a
    tight crop of the party region. Roles aren't OCR-able (they're
    icons), so they're left blank for the user to set in the edit dialog.
  Cooldown path — an `IconMatcher` pre-restricted to the player's
    class+spec slides each known spell icon (`<spell_id>.png` in the
    user-data icons dir) across the cooldown crop. Spells whose peak
    correlation clears the matcher's threshold are returned at the peak
    position. Background colors / textures don't matter — `matchTemplate`
    with `TM_CCOEFF_NORMED` correlates icon-intrinsic pixels.

Bbox bookkeeping: every coordinate in the returned `Calibration` is in
the original full-resolution frame's pixel space, so the cooldown
watcher and icon matcher can use them directly.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# WoW class / spec taxonomy. Keys are the canonical lowercase tokens
# (underscore for multi-word) used as filesystem slugs and dict keys
# throughout the app. Display labels are derived by title-casing the
# tokens — see _display() helpers in the dialog.
WOW_CLASSES: list[str] = [
    "death_knight", "demon_hunter", "druid", "evoker", "hunter",
    "mage", "monk", "paladin", "priest", "rogue", "shaman",
    "warlock", "warrior",
]

WOW_SPECS: dict[str, list[str]] = {
    "death_knight": ["blood", "frost", "unholy"],
    "demon_hunter": ["havoc", "vengeance"],
    "druid":        ["balance", "feral", "guardian", "restoration"],
    "evoker":       ["devastation", "preservation", "augmentation"],
    "hunter":       ["beast_mastery", "marksmanship", "survival"],
    "mage":         ["arcane", "fire", "frost"],
    "monk":         ["brewmaster", "mistweaver", "windwalker"],
    "paladin":      ["holy", "protection", "retribution"],
    "priest":       ["discipline", "holy", "shadow"],
    "rogue":        ["assassination", "outlaw", "subtlety"],
    "shaman":       ["elemental", "enhancement", "restoration"],
    "warlock":      ["affliction", "demonology", "destruction"],
    "warrior":      ["arms", "fury", "protection"],
}

# Padding (fraction of the region's longer edge) added when cropping for
# passes 2/3. The pass-1 bboxes are approximate; padding ensures we don't
# clip the edges of the UI element by accident.
_CROP_PADDING_FRACTION = 0.10


class PartyMember(BaseModel):
    # name is None when the LLM couldn't read the slot with confidence —
    # prevents fabricated names from polluting the roster.
    name: str | None
    # Best-effort role identification from the LLM. None means "unknown,
    # ask the user in the edit dialog". Constrained values when set:
    # "tank", "healer", "dps". Validators coerce common variants.
    role: str | None = None
    bbox: tuple[int, int, int, int]

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if not s:
            return None
        # Accept a few common spellings the LLM might produce.
        if s in ("tank", "healer", "dps"):
            return s
        if s in ("heal", "healing"):
            return "healer"
        if s in ("dd", "damage", "damage dealer", "damage_dealer", "ranged", "melee"):
            return "dps"
        # Unknown value — treat as "couldn't determine" rather than failing
        # the whole calibration on a single bad role string.
        return None


class CooldownIcon(BaseModel):
    """One cooldown-manager icon located during calibration.

    `bbox` comes from the LLM's region read. `spell_id` is populated by
    the IconMatcher after calibration finishes — None means the matcher
    couldn't identify which spell this icon represents (no high-enough
    score against any reference PNG in the icons dir).
    """

    bbox: tuple[int, int, int, int]
    spell_id: int | None = None


class Calibration(BaseModel):
    """The user's saved UI + roster state.

    Two logical groups of fields live in one file because they share the same
    on-disk format and version:

      Slow-changing (UI calibration — set once per character / UI setup):
        `party_region`, `cooldown_region`, `player_class`, `player_spec`,
        `cooldown_icons`. Refreshed only when the user explicitly clicks
        Calibrate.

      Fast-changing (per-run roster + session):
        `party_members`, `player_name`, `dungeon_name`. Updated frequently
        — every dungeon swap or party reshuffle. Edited via the Roster
        dialog; dungeon also via the top-level picker.

    All bboxes are in the original (full-resolution) capture frame's pixel
    coordinates. `party_region` / `cooldown_region` are the user-drawn
    bounding rectangles around the UI elements — kept so the Roster dialog
    can re-OCR party names without forcing the user back through region
    calibration.
    """

    party_region: tuple[int, int, int, int] | None = None
    cooldown_region: tuple[int, int, int, int] | None = None
    party_members: list[PartyMember] = Field(default_factory=list)
    cooldown_icons: list[CooldownIcon] = Field(default_factory=list)
    dungeon_name: str | None = None
    player_class: str | None = None
    player_spec: str | None = None
    # The player's own character name — the roster row marked "Me". Used so
    # rules can tell a cast on the player from one on a teammate.
    player_name: str | None = None
    notes: str = ""
    calibrated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("player_name", mode="before")
    @classmethod
    def _normalize_player_name(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("player_class", mode="before")
    @classmethod
    def _normalize_class(cls, v):
        """Accept either 'paladin' or 'Paladin' / 'Death Knight' etc.;
        normalize to lowercase underscore-token. Reject unknown values
        (return null) rather than failing the whole calibration."""
        if v is None:
            return None
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return s if s in WOW_CLASSES else None

    @field_validator("player_spec", mode="before")
    @classmethod
    def _normalize_spec(cls, v):
        """Normalize but don't cross-validate against player_class — the
        dialog handles that interactively, and the LLM may produce class
        and spec inconsistently. Worst case: invalid combo loads no
        class file; user fixes via the dropdown."""
        if v is None:
            return None
        s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return s if s else None

    def roster(self) -> list[str]:
        """Confidently-read names only. Slots the LLM marked uncertain are
        dropped — better to miss a target match than to canonicalize OCR
        text into a hallucinated teammate name."""
        return [m.name for m in self.party_members if m.name]

    def roles_by_name(self) -> dict[str, str]:
        """Map confidently-read name -> assigned role.

        Only includes members whose role was identified (either by the LLM
        or by the user in the edit dialog). Missing entries mean "role
        unknown" — the rule engine should treat them as unconstrained
        rather than as "not the tank/healer/dps".
        """
        return {
            m.name: m.role for m in self.party_members
            if m.name and m.role
        }


class CalibrationError(RuntimeError):
    """Raised when calibration cannot complete. Message is user-facing."""


# Party-frame names render smaller than cast-bar text; the OCR detector
# down-samples large inputs, so upscale the (already-small) party crop to keep
# the names resolvable.
_PARTY_OCR_UPSCALE = 3


def _read_party_via_ocr(crop: np.ndarray, ocr) -> list[dict]:
    """Read party-member names from a party-frame crop with local OCR.

    Upscales the crop, OCRs it, keeps name-like text (dropping HP bars like
    '1.2M/1.2M' and bare level numbers), strips the game's '...' truncation,
    and returns members ordered top-to-bottom in crop-local coords. Roles
    aren't OCR-able (they're icons), so they're left unset for the user to
    fill in the confirm dialog. Returns [] if `ocr` is None.
    """
    if ocr is None or crop is None or crop.size == 0:
        return []
    up = cv2.resize(
        crop, None, fx=_PARTY_OCR_UPSCALE, fy=_PARTY_OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    members: list[dict] = []
    for text, _conf, (x1, y1, x2, y2) in ocr.read_boxes(up):
        name = text.strip().rstrip(".").strip()
        letters = sum(c.isalpha() for c in name)
        digits = sum(c.isdigit() for c in name)
        if letters < 2 or "/" in name or digits >= letters:
            continue  # HP ('1.2M/1.2M'), level ('3'), other non-name text
        bbox = (
            x1 // _PARTY_OCR_UPSCALE, y1 // _PARTY_OCR_UPSCALE,
            x2 // _PARTY_OCR_UPSCALE, y2 // _PARTY_OCR_UPSCALE,
        )
        members.append({"name": name, "bbox": bbox})
    members.sort(key=lambda m: m["bbox"][1])
    return members


def ocr_party_members(
    image_bgr: np.ndarray,
    party_region: tuple[int, int, int, int],
    ocr,
) -> list[dict]:
    """OCR the party region of a fresh frame to refresh the roster.

    Returns `[{"name": str, "bbox": (x1, y1, x2, y2)}, ...]` in source-frame
    coordinates, ordered top-to-bottom. The Roster dialog calls this when
    the user clicks "Load party members" — no full calibrate_read needed,
    since regions + class/spec + icons don't change between runs.
    """
    crop, crop_origin = _crop_with_padding(image_bgr, party_region)
    ox, oy = crop_origin
    out: list[dict] = []
    for member in _read_party_via_ocr(crop, ocr):
        x1, y1, x2, y2 = member["bbox"]
        out.append({
            "name": member["name"],
            "bbox": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
        })
    return out


def calibrate_read(
    image_bgr: np.ndarray,
    party_region: tuple[int, int, int, int] | None,
    cooldown_region: tuple[int, int, int, int] | None,
    matcher=None,
    dungeon_name: str | None = None,
    player_class: str | None = None,
    player_spec: str | None = None,
    prior_notes: str = "",
) -> Calibration:
    """UI calibration: regions, class/spec, cooldown icons.

    Slides every spell icon in the class-restricted `matcher` across the
    cooldown crop and returns peak positions with spell IDs already
    populated. The party region is saved on the Calibration so the Roster
    dialog can OCR it later — calibrate_read itself does not read names,
    since roster changes per dungeon run while UI calibration only changes
    when the user's UI does.

    `party_region` and `cooldown_region` are in source-frame coords; either
    may be None to skip. `matcher` must be class-restricted; a None matcher
    leaves `cooldown_icons` empty.
    """
    notes: list[str] = []
    if prior_notes:
        notes.append(prior_notes)

    cooldown_icons: list[dict] = []
    if cooldown_region is not None and matcher is not None:
        crop, crop_origin = _crop_with_padding(image_bgr, cooldown_region)
        ox, oy = crop_origin
        matches = matcher.find_in_crop(crop)
        for spell_id, (x1, y1, x2, y2), _score in matches:
            cooldown_icons.append({
                "bbox": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                "spell_id": spell_id,
            })
        notes.append(
            f"cooldowns: template match identified {len(matches)} icons"
        )
    elif cooldown_region is not None:
        notes.append("cooldowns: no matcher provided — region skipped")

    payload = {
        "party_region": party_region,
        "cooldown_region": cooldown_region,
        "cooldown_icons": cooldown_icons,
        "dungeon_name": dungeon_name,
        "player_class": player_class,
        "player_spec": player_spec,
        "notes": " | ".join(notes),
    }
    try:
        cal = Calibration.model_validate(payload)
    except Exception as exc:
        raise CalibrationError(
            f"Could not assemble calibration: {exc}. Payload: {payload!r}"
        ) from exc

    logger.info(
        "calibrate_read: %d cooldown icons (party roster is set separately)",
        len(cal.cooldown_icons),
    )
    return cal


def save_calibration(cal: Calibration, path: Path) -> None:
    """Persist calibration to disk as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cal.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    logger.info("Saved calibration to %s", path)


def load_calibration(path: Path) -> Calibration | None:
    """Load the Calibration from `path`, or None if the file is missing
    or malformed. A malformed file is logged but doesn't raise — the app
    should boot fine without one."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Calibration.model_validate(raw)
    except Exception:
        logger.exception("Failed to load calibration from %s; ignoring", path)
        return None


# ---- internals ----


def _crop_with_padding(
    image_bgr: np.ndarray, region: tuple[int, int, int, int]
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop the frame to `region` with `_CROP_PADDING_FRACTION` padding.

    Returns (crop, (origin_x, origin_y)) where origin is the top-left of
    the crop in source-frame coordinates. Padding guards against the
    pass-1 region bbox clipping the UI element at the edges.
    """
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = region
    pad = int(_CROP_PADDING_FRACTION * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return image_bgr[y1:y2, x1:x2].copy(), (x1, y1)
