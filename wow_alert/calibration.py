"""LLM-driven screen calibration.

Three-pass design.

  Pass 1 — "locate". Send the full screenshot. Ask Claude to return two
           bboxes only: where the party frames are and where the cooldown
           manager is. No name reading, no icon ID — just region location.
  Pass 2 — "read party". Crop the source frame to the party region (with
           padding) and resend. The crop fills the LLM's visual field, so
           small text reads accurately.
  Pass 3 — "read cooldowns". Same, for the cooldown manager crop.

Why three passes instead of one: on ultrawide / dual-monitor captures, the
party frame is a tiny fraction of the image. Even at full source resolution,
the LLM can't reliably transcribe text that occupies <5% of the visual
field. Cropping first puts each UI element in its own focused image.

Cost: ~3x a single-pass call, still under $0.05 per calibration with Sonnet
4.6. Latency: ~10-15 s total (passes are serial; could be parallelized
later if needed). Calibration runs rarely so this is fine.

Bbox bookkeeping: every pass returns bboxes in the coordinate space of the
image we sent. Two transforms apply per pass — the encoding scale
(downscale applied by `_encode_for_api` to fit the 5 MB limit) and, for
passes 2/3, the crop origin in source-frame coordinates. After unscaling
and offsetting, every bbox in the returned `Calibration` is in the
original full-resolution frame's pixel space — directly usable by the
OpenCV cooldown watcher.
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


_PROMPT_LOCATE = """\
You are looking at a screenshot of World of Warcraft. Find three things
and report them. Bounding boxes only for the UI regions — no name reading
yet, that comes in a separate pass.

1. Party / raid frames — a column or row of slots showing teammate names
   above HP bars. Default position is top-left; addons (ElvUI etc.) can
   relocate them. Return one bounding box that encompasses ALL the party
   slots together.

2. Cooldown manager — a row or grid of small spell icons that grey out
   when used (often a WeakAura). Return one bounding box that encompasses
   the full set of icons.

3. Dungeon / zone name — usually a few words of large text near the top
   of the screen (minimap area on the right, or near the player frame).
   Read it directly from this image if legible; set null if you can't
   read it confidently. Do not guess.

Bounding boxes are [x1, y1, x2, y2] in pixels, top-left origin, with
x2 > x1 and y2 > y1. Set any field you can't find to null.

Respond with ONLY this JSON object, no prose, no code fences:
{
  "party_region": [x1, y1, x2, y2] | null,
  "cooldown_region": [x1, y1, x2, y2] | null,
  "dungeon_name": "..." | null,
  "notes": "any caveats"
}
"""


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


_PROMPT_READ_COOLDOWNS = """\
This image is a tight crop of a World of Warcraft cooldown-manager UI —
typically a WeakAura showing spell/ability icons that grey out on cooldown.

For each icon:
- `bbox`: [x1, y1, x2, y2] in pixels (relative to THIS cropped image,
  top-left origin), encompassing just the icon square.
- `action`: the spell or ability name if you can identify it from the
  icon art (e.g. "Blessing of Protection", "Purge", "Kick"). If you
  don't recognize the icon, use "unknown". Do not guess.

The icon set itself reveals the player's class (and usually spec) far
better than anything else on screen, so also report:
- `player_class`: one of the canonical lowercase tokens — death_knight,
  demon_hunter, druid, evoker, hunter, mage, monk, paladin, priest,
  rogue, shaman, warlock, warrior. Null if uncertain.
- `player_spec`: the in-class spec token — examples:
    death_knight: blood, frost, unholy
    demon_hunter: havoc, vengeance
    druid:        balance, feral, guardian, restoration
    evoker:       devastation, preservation, augmentation
    hunter:       beast_mastery, marksmanship, survival
    mage:         arcane, fire, frost
    monk:         brewmaster, mistweaver, windwalker
    paladin:      holy, protection, retribution
    priest:       discipline, holy, shadow
    rogue:        assassination, outlaw, subtlety
    shaman:       elemental, enhancement, restoration
    warlock:      affliction, demonology, destruction
    warrior:      arms, fury, protection
  Spec is harder to nail than class — only commit when the icons include
  a spec-defining ability (e.g. Beacon of Light → holy paladin,
  Avenger's Shield → protection paladin, Renewing Mist → mistweaver
  monk). Null otherwise.

Return an empty list if you can't make out any icons.

Respond with ONLY this JSON object, no prose, no code fences:
{
  "cooldown_icons": [{"action": "...", "bbox": [x1, y1, x2, y2]}, ...],
  "player_class": "..." | null,
  "player_spec": "..." | null,
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
    action: str
    bbox: tuple[int, int, int, int]


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
    notes: str = ""
    calibrated_at: datetime = Field(default_factory=datetime.now)

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


class LocateResult(BaseModel):
    """Pass 1 output. Regions are in source-frame coords (or None if the LLM
    couldn't locate the element). `notes` carries any caveats the LLM
    reported."""

    party_region: tuple[int, int, int, int] | None = None
    cooldown_region: tuple[int, int, int, int] | None = None
    dungeon_name: str | None = None
    notes: str = ""


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


def calibrate_locate(image_bgr: np.ndarray) -> LocateResult:
    """Pass 1: find rough regions + dungeon name in a full screenshot.

    Returns a `LocateResult` whose region fields may be None when the LLM
    couldn't locate the element. The caller is expected to show these to
    the user for confirmation/adjustment before invoking `calibrate_read`.
    """
    client = _make_client()
    parsed = _call_pass(client, image_bgr, _PROMPT_LOCATE)
    scale = parsed.get("_encoding_scale", 1.0)
    party_region = _resolve_bbox(parsed.get("party_region"), scale)
    cooldown_region = _resolve_bbox(parsed.get("cooldown_region"), scale)
    dungeon_raw = parsed.get("dungeon_name")
    dungeon_name = dungeon_raw.strip() if isinstance(dungeon_raw, str) else None
    notes = parsed.get("notes") or ""
    logger.info(
        "calibrate_locate: party=%s cooldown=%s dungeon=%r",
        party_region, cooldown_region, dungeon_name,
    )
    return LocateResult(
        party_region=party_region,
        cooldown_region=cooldown_region,
        dungeon_name=dungeon_name,
        notes=notes,
    )


def calibrate_read(
    image_bgr: np.ndarray,
    party_region: tuple[int, int, int, int] | None,
    cooldown_region: tuple[int, int, int, int] | None,
    dungeon_name: str | None = None,
    prior_notes: str = "",
) -> Calibration:
    """Passes 2 and 3: crop to user-confirmed regions and read contents.

    `party_region` and `cooldown_region` must be in source-frame coords (the
    coords `image_bgr` uses). They're typically the user-adjusted output of
    `calibrate_locate`, but can also come from elsewhere (e.g., a re-run
    with hand-typed bboxes). Either may be None — that region is skipped
    and the corresponding output is empty.

    `dungeon_name` and `prior_notes` flow through unchanged; pass through
    what the user confirmed in the region-confirmation step.
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
    player_class: str | None = None
    player_spec: str | None = None
    if cooldown_region is not None:
        crop, crop_origin = _crop_with_padding(image_bgr, cooldown_region)
        parsed = _call_pass(client, crop, _PROMPT_READ_COOLDOWNS)
        scale = parsed.get("_encoding_scale", 1.0)
        for icon in parsed.get("cooldown_icons", []) or []:
            bbox = _resolve_bbox(icon.get("bbox"), scale, offset=crop_origin)
            if bbox is None:
                continue
            cooldown_icons.append({
                "action": icon.get("action") or "unknown",
                "bbox": bbox,
            })
        # Class/spec ride along with the cooldown-icon read: the icon set
        # is the most class-defining signal on screen, and Pass 3 sees it
        # at native size in the crop. Validators on Calibration normalize
        # and reject unknowns.
        player_class = parsed.get("player_class")
        player_spec = parsed.get("player_spec")
        if parsed.get("notes"):
            notes.append(f"cooldowns: {parsed['notes']}")

    payload = {
        "party_members": party_members,
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
        "calibrate_read: %d party members (%d named), %d cooldown icons",
        len(cal.party_members), len(cal.roster()), len(cal.cooldown_icons),
    )
    return cal


def calibrate(image_bgr: np.ndarray) -> Calibration:
    """Convenience: run locate + read without an intermediate confirmation.

    Useful for tests and headless callers. Production UI invokes
    calibrate_locate, shows a region-confirmation dialog, then
    calibrate_read against the confirmed regions.
    """
    locate = calibrate_locate(image_bgr)
    return calibrate_read(
        image_bgr,
        party_region=locate.party_region,
        cooldown_region=locate.cooldown_region,
        dungeon_name=locate.dungeon_name,
        prior_notes=locate.notes,
    )


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
