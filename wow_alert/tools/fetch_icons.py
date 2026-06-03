"""Fetch icon PNGs for every spell_id referenced by a class library.

Usage:
    python -m wow_alert.tools.fetch_icons

Walks every class library the app can see — bundled
`wow_alert/_defaults/classes/<class>/<spec>.yaml` plus any user
overrides under the user-data config dir — collects every `spell_id`
value, and for each one that doesn't already have a `<spell_id>.png` in
the icons dir (the user-data location by default; see --icons-dir),
fetches the icon and writes it.

Source order:

1. **Wowhead tooltip JSON endpoint** (`nether.wowhead.com/tooltip/spell/<id>`).
   Designed for programmatic embed, returns JSON with an `icon` field.
   The icon name is then used to download the canonical large (56×56)
   PNG from Wowhead's CDN at `wow.zamimg.com`.

2. **og:image scrape** (`wowhead.com/spell=<id>`). Fallback for spells
   the tooltip endpoint doesn't return cleanly.

Both attempts use a full browser-shaped header set so Cloudflare's
anti-bot heuristics don't reject the request the way they do for a
bare Python user agent.

Idempotent: existing files are skipped. Safe to re-run after editing a
class library to add new actions.

If both endpoints fail for a given spell_id, the script prints a clear
manual-download instruction with the canonical Wowhead URL — open it
in your browser, right-click the icon, save it into the icons dir as `<id>.png`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

from wow_alert.class_library import ClassActions, _layered_class_spec_paths
from wow_alert.paths import ICONS_DIR

logger = logging.getLogger("fetch_icons")


# Pretend to be a recent Chrome on Windows. Cloudflare's bot-detection
# looks at the full header set, not just the User-Agent.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


_OG_IMAGE_RE = re.compile(
    r'<meta\s+property="og:image"\s+content="([^"]+)"',
    re.IGNORECASE,
)

# Polite delay between requests so we don't trip rate limits on top of
# whatever anti-bot is running. Cheap insurance — the user runs this
# once at setup time and once per class-library edit, so even 0.5s × 25
# icons is only ~12s of total wait.
_INTER_REQUEST_DELAY_S = 0.5


def _http_get(url: str, accept: str | None = None, timeout: float = 20.0) -> bytes:
    headers = dict(_BROWSER_HEADERS)
    if accept is not None:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        encoding = resp.headers.get("Content-Encoding", "")
        if "gzip" in encoding.lower():
            body = gzip.decompress(body)
        # 'br' (Brotli) handled only if the brotli module is present.
        # urllib will return the raw bytes either way; if we asked for
        # br and got it without brotli installed, downstream parse will
        # fail clearly.
        return body


def _icon_name_from_tooltip(spell_id: int) -> str | None:
    """Hit the JSON tooltip endpoint and pull the `icon` field."""
    url = f"https://nether.wowhead.com/tooltip/spell/{spell_id}"
    try:
        body = _http_get(url, accept="application/json")
    except urllib.error.HTTPError as exc:
        logger.debug("tooltip endpoint %d for spell %d: %s", exc.code, spell_id, exc)
        return None
    except urllib.error.URLError as exc:
        logger.debug("tooltip endpoint failed for spell %d: %s", spell_id, exc)
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        logger.debug("tooltip endpoint for spell %d returned non-JSON", spell_id)
        return None
    icon = data.get("icon")
    if not isinstance(icon, str) or not icon:
        return None
    return icon


def _icon_url_from_og_image(spell_id: int) -> str | None:
    """Fallback: scrape the og:image meta tag from the main spell page."""
    url = f"https://www.wowhead.com/spell={spell_id}"
    try:
        html = _http_get(url).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        logger.debug("og:image scrape %d for spell %d: %s", exc.code, spell_id, exc)
        return None
    except urllib.error.URLError as exc:
        logger.debug("og:image scrape failed for spell %d: %s", spell_id, exc)
        return None
    match = _OG_IMAGE_RE.search(html)
    if not match:
        return None
    return match.group(1)


def _fetch_one(spell_id: int, target: Path) -> bool:
    """Try the JSON tooltip first, fall back to og:image, write the PNG."""
    icon_url: str | None = None
    icon_name = _icon_name_from_tooltip(spell_id)
    if icon_name:
        # Wowhead's CDN serves icons at three sizes; "large" is 56x56,
        # plenty for template matching against a live 64x64 crop.
        icon_url = f"https://wow.zamimg.com/images/wow/icons/large/{icon_name}.jpg"
        source = "tooltip"
    else:
        icon_url = _icon_url_from_og_image(spell_id)
        source = "og:image"

    if icon_url is None:
        logger.error(
            "Could not resolve icon URL for spell %d. Manual fallback: "
            "open https://www.wowhead.com/spell=%d in a browser, "
            "right-click the icon, save as %s",
            spell_id, spell_id, target,
        )
        return False

    try:
        data = _http_get(icon_url, accept="image/avif,image/webp,image/*,*/*;q=0.8")
    except urllib.error.URLError as exc:
        logger.error(
            "Resolved icon URL for spell %d (%s) but download failed: %s. "
            "Manual fallback: save %s to %s",
            spell_id, icon_url, exc, icon_url, target,
        )
        return False

    target.write_bytes(data)
    logger.info(
        "spell %d -> %s (%d bytes, via %s)",
        spell_id, target.name, len(data), source,
    )
    return True


def _collect_spell_ids() -> set[int]:
    """Every spell_id across all effective class libraries (bundled + user)."""
    ids: set[int] = set()
    for path in sorted(set(_layered_class_spec_paths().values())):
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            cfg = ClassActions.model_validate(raw)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue
        for action in cfg.actions:
            ids.add(action.spell_id)
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--icons-dir", type=Path, default=ICONS_DIR,
        help="Where to write icon PNGs (default: the user-data icons dir, "
             "%(default)s).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if the icon file already exists.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )

    icon_dir = args.icons_dir
    icon_dir.mkdir(parents=True, exist_ok=True)

    spell_ids = _collect_spell_ids()
    if not spell_ids:
        logger.error(
            "No spell IDs found in any class library. Add a "
            "classes/<class>/<spec>.yaml file first."
        )
        return 1

    logger.info("Found %d spell IDs across class libraries", len(spell_ids))

    fetched = 0
    skipped = 0
    failed: list[int] = []
    for spell_id in sorted(spell_ids):
        target = icon_dir / f"{spell_id}.png"
        if target.exists() and not args.force:
            skipped += 1
            continue
        if _fetch_one(spell_id, target):
            fetched += 1
        else:
            failed.append(spell_id)
        time.sleep(_INTER_REQUEST_DELAY_S)

    logger.info(
        "Done. fetched=%d skipped=%d failed=%d (icons in %s)",
        fetched, skipped, len(failed), icon_dir,
    )
    if failed:
        logger.warning(
            "Failed spell IDs: %s. Open each at https://www.wowhead.com/spell=<id> "
            "in a browser and save the icon manually to %s/<id>.png if you want "
            "those actions tracked.",
            failed, icon_dir,
        )
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
