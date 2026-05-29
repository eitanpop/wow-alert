"""LLM-driven screen calibration.

Two-pass LLM design (plus one CV step).

  Pass 1 — "locate". Send the full screenshot. Ask Claude to return two
           bboxes only: where the party frames are and where the cooldown
           manager is. No name reading, no icon ID — just region location.
  Pass 2 — "read party". Crop the source frame to the party region (with
           padding) and resend. The crop fills the LLM's visual field, so
           small text reads accurately.
  CV step — "find icons". Within the user-confirmed cooldown region,
           OpenCV contour detection finds per-icon bboxes. Replaces a
           previous LLM call that proved imprecise — it routinely
           returned bboxes that cut across icon seams.

Why CV over LLM for icon localization: WoW spell icons are high-
saturation square sprites on a low-saturation background, which makes
them trivial to isolate with `cv2.findContours` on a saturation-
thresholded mask. The LLM eyeballs positions and lands within a few
pixels of the right spot, but a few pixels is enough to produce half-
and-half crops when icons are tightly packed.

Cost: one LLM call removed. Latency: ~5 s saved.

Bbox bookkeeping: every coordinate in the returned `Calibration` is in
the original full-resolution frame's pixel space, so the OpenCV cooldown
watcher and icon matcher can use them directly.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048


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


_PROMPT_READ_PARTY = """\
This image is a tight crop of a World of Warcraft party/raid frame. Read
each visible teammate slot.

For each slot:
- `name`: transcribe the player name character-by-character from the
  pixels you actually see. If ANY character is unclear, set name to null.
  Do not invent or fill in plausible-looking names.
- `role`: one of "tank", "healer", "dps", or null. Look for visible cues:
  small role icons (shield = tank, cross = healer, sword = dps); class
  icons together with what you know about the spec; party-frame coloring
  conventions. If you can't tell with high confidence, return null — the
  user will fill it in. Do not guess from name alone.
- `bbox`: [x1, y1, x2, y2] in pixels (relative to THIS cropped image,
  top-left origin), encompassing the full slot.

Return an empty list if no slots are legible.

Respond with ONLY this JSON object, no prose, no code fences:
{
  "party_members": [
    {"name": "..." | null, "role": "tank" | "healer" | "dps" | null,
     "bbox": [x1, y1, x2, y2]},
    ...
  ],
  "notes": "any caveats"
}
"""


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
    """One snapshot of what the LLM saw on screen.

    All bboxes are in the original (full-resolution) capture frame's pixel
    coordinates. Pass-1 region detection + pass-2/3 crop+read accumulate
    into the same coordinate space via the unscale-then-offset transforms
    inside `calibrate()`.

    `player_class` and `player_spec` are the player's character config. The
    LLM proposes them from on-screen cues (cooldown manager icons, action
    bar); the user confirms/overrides in the post-calibration dialog.
    Together they pick the file at `config/classes/<class>/<spec>.yaml`
    which gives the rule engine its action library.
    """

    party_members: list[PartyMember] = Field(default_factory=list)
    cooldown_icons: list[CooldownIcon] = Field(default_factory=list)
    dungeon_name: str | None = None
    player_class: str | None = None
    player_spec: str | None = None
    # The player's own character name. Used to detect when a cast targets
    # the player (vs a teammate), so rules can recommend a self defensive.
    # User-entered in the calibration dialog; the LLM doesn't read it.
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
    """Raised when calibration cannot complete (no API key, API error,
    unparseable response, etc.). The message is user-facing."""


def _make_client() -> Any:
    if os.environ.get("ANTHROPIC_API_KEY") is None:
        raise CalibrationError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before running calibration."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise CalibrationError(
            "anthropic SDK is not installed. Run `poetry install`."
        ) from exc
    return anthropic.Anthropic()


def calibrate_read(
    image_bgr: np.ndarray,
    party_region: tuple[int, int, int, int] | None,
    cooldown_region: tuple[int, int, int, int] | None,
    dungeon_name: str | None = None,
    prior_notes: str = "",
) -> Calibration:
    """Read the contents of the user-confirmed regions.

    Pass 2: LLM transcribes party-frame names + roles from a crop of
    the party region. Pass 3 (icons): OpenCV contour detection finds
    individual cooldown-icon bboxes inside the cooldown region.

    `party_region` and `cooldown_region` are in source-frame coords;
    either may be None to skip that pass. `dungeon_name` and
    `prior_notes` flow through unchanged.
    """
    client = _make_client()
    notes: list[str] = []
    if prior_notes:
        notes.append(f"locate: {prior_notes}")

    party_members: list[dict] = []
    if party_region is not None:
        crop, crop_origin = _crop_with_padding(image_bgr, party_region)
        parsed = _call_pass(client, crop, _PROMPT_READ_PARTY)
        scale = parsed.get("_encoding_scale", 1.0)
        for m in parsed.get("party_members", []) or []:
            bbox = _resolve_bbox(m.get("bbox"), scale, offset=crop_origin)
            if bbox is None:
                continue
            party_members.append({
                "name": m.get("name"),
                # The dialog's role dropdown pre-selects from this value
                # so the user usually only has to confirm.
                "role": m.get("role"),
                "bbox": bbox,
            })
        if parsed.get("notes"):
            notes.append(f"party: {parsed['notes']}")

    cooldown_icons: list[dict] = []
    if cooldown_region is not None:
        from wow_alert.cooldown_grid import find_icon_bboxes

        crop, crop_origin = _crop_with_padding(image_bgr, cooldown_region)
        local_bboxes = find_icon_bboxes(crop)
        ox, oy = crop_origin
        for x1, y1, x2, y2 in local_bboxes:
            # Translate crop-local coords back to source-frame coords.
            cooldown_icons.append({
                "bbox": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
            })
        notes.append(
            f"cooldowns: cv2 contour detection found {len(local_bboxes)} icons"
        )

    payload = {
        "party_members": party_members,
        "cooldown_icons": cooldown_icons,
        "dungeon_name": dungeon_name,
        "notes": " | ".join(notes),
    }
    try:
        cal = Calibration.model_validate(payload)
    except Exception as exc:
        raise CalibrationError(
            f"Could not assemble calibration: {exc}. Payload: {payload!r}"
        ) from exc

    logger.info(
        "calibrate_read: %d party members (%d named), %d cooldown icons",
        len(cal.party_members), len(cal.roster()), len(cal.cooldown_icons),
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


# Anthropic enforces a 5 MB per-image hard limit. Stay well below it.
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
# Floor for downscale fallback — going smaller than this hurts text legibility
# more than it helps fit in the byte budget.
_MIN_IMAGE_DIM = 1280
# Minimum size we upscale crops to before sending. Small UI elements (e.g.,
# a 65x115 party-frame crop on a low-UI-scale ultrawide) are pixelated
# postage stamps to the LLM otherwise; upscaling with cubic interpolation
# gives Claude's vision model more tokens to spend on the same content,
# which makes small text legible.
_MIN_UPSCALE_DIM = 1024


def _call_pass(client: Any, image_bgr: np.ndarray, prompt: str) -> dict:
    """Encode, send, parse one LLM round-trip.

    Stuffs the encoding scale into the returned dict under the
    `_encoding_scale` key so the caller can un-scale returned bboxes
    without juggling extra return values per pass.
    """
    import anthropic  # already imported in caller; cheap re-import for typing

    image_b64, media_type, scale = _encode_for_api(image_bgr)
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except anthropic.APIError as exc:
        raise CalibrationError(f"Anthropic API error: {exc}") from exc

    text = _extract_text(response)
    parsed = _parse_json(text)
    parsed["_encoding_scale"] = scale
    return parsed


def _resolve_bbox(
    bbox: Any,
    scale: float,
    offset: tuple[int, int] = (0, 0),
) -> tuple[int, int, int, int] | None:
    """Convert an LLM-returned bbox into source-frame coordinates.

    `scale` is the factor `_encode_for_api` applied before sending — bbox
    coords are in that scaled space, so we divide. `offset` is the (x, y)
    origin of the crop (if any) within the source frame, added after
    unscaling. Returns None if the input is missing or malformed.
    """
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        inv = 1.0 / scale
        x1, y1, x2, y2 = (int(round(c * inv)) for c in bbox)
    except (TypeError, ValueError):
        return None
    ox, oy = offset
    return (x1 + ox, y1 + oy, x2 + ox, y2 + oy)


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


def _encode_for_api(image_bgr: np.ndarray) -> tuple[str, str, float]:
    """Encode BGR ndarray as base64 JPEG that fits Anthropic's 5 MB limit.

    Two-stage sizing:
      - If the input is smaller than `_MIN_UPSCALE_DIM` on the long edge,
        upscale it with cubic interpolation first. Small crops are
        unreadable otherwise — see the constant comment for why.
      - If the encoded JPEG exceeds the 5 MB budget, iteratively downscale
        by 20% until it fits or we hit the floor.

    Returns (base64_string, media_type, scale_factor). `scale_factor` is
    final-image-dim / original-input-dim — callers divide their returned
    bbox coords by this to recover original-input coordinates.
    """
    src_h, src_w = image_bgr.shape[:2]
    long_edge = max(src_h, src_w)

    if long_edge < _MIN_UPSCALE_DIM:
        scale = _MIN_UPSCALE_DIM / long_edge
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        img = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug(
            "Upscaled crop %dx%d -> %dx%d (scale=%.2fx) for LLM legibility",
            src_w, src_h, new_w, new_h, scale,
        )
    else:
        img = image_bgr
        scale = 1.0

    while True:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise CalibrationError("Failed to JPEG-encode the calibration frame.")
        size = len(buf)
        if size <= _MAX_IMAGE_BYTES:
            logger.debug(
                "Encoded calibration frame: %d bytes (%dx%d, scale=%.3f)",
                size, img.shape[1], img.shape[0], scale,
            )
            return (
                base64.standard_b64encode(buf.tobytes()).decode("utf-8"),
                "image/jpeg",
                scale,
            )
        new_w = int(img.shape[1] * 0.8)
        new_h = int(img.shape[0] * 0.8)
        if max(new_w, new_h) < _MIN_IMAGE_DIM:
            raise CalibrationError(
                f"Calibration frame is too large to fit Anthropic's 5 MB "
                f"limit without dropping below {_MIN_IMAGE_DIM}px (text "
                f"becomes unreadable). Try a smaller game window."
            )
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale = scale * 0.8


def _extract_text(response: Any) -> str:
    """Pull the text payload out of an Anthropic Messages response."""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise CalibrationError("Anthropic response contained no text block.")


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json(text: str) -> dict:
    """Robust JSON extraction. Tries raw parse first, falls back to ```json
    fence extraction."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_FENCE_RE.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise CalibrationError(
                f"Calibration response had a code fence but contents weren't "
                f"valid JSON: {exc}"
            ) from exc
    raise CalibrationError(
        f"Calibration response was not JSON. First 500 chars: {text[:500]!r}"
    )
