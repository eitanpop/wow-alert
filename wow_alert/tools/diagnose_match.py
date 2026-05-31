"""Diagnose icon-match results against the saved calibration + a frame.

Re-runs the template matcher with thresholds dropped, ranks every spell in
the class library by peak score, and writes an annotated overlay that
shows every candidate above 0.4 — not just the ones that cleared the
default 0.7 threshold. The output:

    diagnose_match/
      overview_full.png       cooldown crop with every candidate ≥ 0.4
                              drawn (color = score band) + spell_id label
      report.txt              per-spell peak score + scale + position,
                              sorted score-descending

Usage::

    python -m wow_alert.tools.diagnose_match <artifact_dir>

`<artifact_dir>` is the timestamped directory under
``%LOCALAPPDATA%\\wow-alert\\calibration_artifacts\\`` containing the
``frame.png`` you want to re-score. If omitted, the most recent one is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from wow_alert.calibration import load_calibration
from wow_alert.class_library import load_class_actions
from wow_alert.icon_matcher import _INTERIOR_FRACTION, _interior_crop
from wow_alert.paths import CALIBRATION_ARTIFACTS_DIR, CALIBRATION_PATH, ICONS_DIR


_SCALES = (16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96)


def _score_band_color(score: float) -> tuple[int, int, int]:
    """BGR color keyed to score band so the overlay reads at a glance.

    Green ≥0.7 (the live threshold), yellow ≥0.55 (near miss), red <0.55
    (probably the wrong reference or not actually on screen)."""
    if score >= 0.7:
        return (0, 255, 0)
    if score >= 0.55:
        return (0, 255, 255)
    return (0, 0, 255)


def _diagnose(artifact_dir: Path) -> int:
    frame_path = artifact_dir / "frame.png"
    if not frame_path.exists():
        print(f"missing frame.png at {frame_path}", file=sys.stderr)
        return 2
    cal = load_calibration(CALIBRATION_PATH)
    if cal is None:
        print(f"no calibration at {CALIBRATION_PATH}", file=sys.stderr)
        return 2
    if cal.cooldown_region is None:
        print("calibration has no cooldown_region — recalibrate first")
        return 2
    if not cal.player_class or not cal.player_spec:
        print("calibration has no class/spec — recalibrate first")
        return 2

    actions = load_class_actions(cal.player_class, cal.player_spec)
    if not actions:
        print(
            f"empty class library for {cal.player_class}/{cal.player_spec}",
            file=sys.stderr,
        )
        return 2

    frame = cv2.imread(str(frame_path))
    rx1, ry1, rx2, ry2 = cal.cooldown_region
    crop = frame[ry1:ry2, rx1:rx2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Load references for every spell in the class library, using the same
    # interior-crop the live matcher does so scores compare apples-to-apples.
    refs: dict[int, np.ndarray] = {}
    for action in actions:
        sid = action.spell_id
        ref_path = ICONS_DIR / f"{sid}.png"
        if not ref_path.exists():
            continue
        img = cv2.imread(str(ref_path))
        if img is None:
            continue
        refs[sid] = cv2.cvtColor(_interior_crop(img), cv2.COLOR_BGR2GRAY)

    # Per-spell: best (score, position, size) across the scale ladder.
    results: list[dict] = []
    h, w = crop_gray.shape[:2]
    for sid, ref in refs.items():
        best_score = -1.0
        best_pos = (0, 0)
        best_size = ref.shape[1]
        for size in _SCALES:
            if size + 2 > min(h, w):
                continue
            interp = cv2.INTER_AREA if size < ref.shape[1] else cv2.INTER_CUBIC
            tmpl = cv2.resize(ref, (size, size), interpolation=interp)
            res = cv2.matchTemplate(crop_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_pos = max_loc
                best_size = size
        action = next(a for a in actions if a.spell_id == sid)
        results.append({
            "spell_id": sid, "id": action.id, "label": action.label,
            "score": best_score, "x": best_pos[0], "y": best_pos[1],
            "size": best_size,
        })

    # Sort highest score first.
    results.sort(key=lambda r: -r["score"])

    out_dir = artifact_dir / "diagnose_match"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Overlay every candidate above 0.4, color-coded by score band. Match
    # positions are for the icon's *inner* art region (references are
    # interior-cropped to skip border chrome), so expand outward to the
    # full icon bbox before drawing — otherwise the rectangles slice
    # through the borders and look misaligned.
    outer = 1.0 / _INTERIOR_FRACTION
    overlay = crop.copy()
    for r in results:
        if r["score"] < 0.4:
            continue
        color = _score_band_color(r["score"])
        inner_s = r["size"]
        cx = r["x"] + inner_s / 2
        cy = r["y"] + inner_s / 2
        full_s = inner_s * outer
        fx1 = max(0, int(round(cx - full_s / 2)))
        fy1 = max(0, int(round(cy - full_s / 2)))
        fx2 = min(w, int(round(cx + full_s / 2)))
        fy2 = min(h, int(round(cy + full_s / 2)))
        r["full_bbox"] = (fx1, fy1, fx2, fy2)
        cv2.rectangle(overlay, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(
            overlay, f"{r['spell_id']} {r['score']:.2f}",
            (fx1, max(11, fy1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "overview_full.png"), overlay)

    # Side-by-side comparison: at each spell's peak position, take the
    # on-screen crop and put it next to the reference PNG that was assigned
    # to that position. If the two pictures show different abilities, the
    # match is wrong (high score doesn't mean correct identification —
    # generic visual features can drag scores up across the wrong icon).
    tile_h = 64
    tiles: list[np.ndarray] = []
    for r in results:
        ref_path = ICONS_DIR / f"{r['spell_id']}.png"
        if not ref_path.exists():
            continue
        ref_img = cv2.imread(str(ref_path))
        x1, y1, x2, y2 = r.get(
            "full_bbox",
            (r["x"], r["y"],
             r["x"] + r["size"], r["y"] + r["size"]),
        )
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        live = crop[y1:y2, x1:x2]
        live_t = cv2.resize(live, (tile_h, tile_h), interpolation=cv2.INTER_AREA)
        ref_t = cv2.resize(ref_img, (tile_h, tile_h), interpolation=cv2.INTER_AREA)
        # Border color by score band so misses stand out at a glance.
        color = _score_band_color(r["score"])
        cv2.rectangle(live_t, (0, 0), (tile_h - 1, tile_h - 1), color, 2)
        cv2.rectangle(ref_t, (0, 0), (tile_h - 1, tile_h - 1), color, 2)
        # Label strip beneath both tiles.
        label = np.zeros((22, tile_h * 2 + 4, 3), dtype=np.uint8)
        cv2.putText(
            label, f"{r['spell_id']} {r['score']:.2f}", (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
        pair = np.hstack([live_t, np.zeros((tile_h, 4, 3), dtype=np.uint8), ref_t])
        tile = np.vstack([pair, label])
        tiles.append(tile)
    if tiles:
        # Stack tiles into a grid; 4 per row reads at a glance.
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        tile_w = tile_h * 2 + 4
        tile_full_h = tile_h + 22
        grid = np.full(
            (rows * tile_full_h + (rows - 1) * 6, cols * tile_w + (cols - 1) * 6, 3),
            32, dtype=np.uint8,
        )
        for i, t in enumerate(tiles):
            r_i, c_i = divmod(i, cols)
            y_off = r_i * (tile_full_h + 6)
            x_off = c_i * (tile_w + 6)
            grid[y_off:y_off + tile_full_h, x_off:x_off + tile_w] = t
        cv2.imwrite(str(out_dir / "side_by_side.png"), grid)

    lines = [
        f"# Match diagnosis for {artifact_dir}",
        f"# Class/spec: {cal.player_class}/{cal.player_spec}",
        f"# Cooldown crop: {w}x{h} (region {cal.cooldown_region})",
        f"# {len(refs)}/{len(actions)} class actions have reference PNGs",
        "# score band: GREEN ≥0.70 (live match), YELLOW 0.55–0.69 (near miss),",
        "#             RED <0.55 (reference probably wrong or not on screen)",
        "#",
        "# score   spell_id  size  x,y          id",
    ]
    for r in results:
        lines.append(
            f"  {r['score']:5.2f}  {r['spell_id']:<8}  {r['size']:>3}   "
            f"{r['x']:>3},{r['y']:>3}     {r['id']}"
        )
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote: {out_dir}\\overview_full.png")
    print(f"wrote: {out_dir}\\report.txt")
    print()
    print("\n".join(lines[7:]))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        artifact_dir = Path(argv[0])
    else:
        # Most recent timestamped dir under CALIBRATION_ARTIFACTS_DIR.
        candidates = sorted(
            [p for p in CALIBRATION_ARTIFACTS_DIR.glob("*") if p.is_dir()],
            key=lambda p: p.name,
            reverse=True,
        )
        if not candidates:
            print(f"no artifact dirs under {CALIBRATION_ARTIFACTS_DIR}", file=sys.stderr)
            return 2
        artifact_dir = candidates[0]
        print(f"using most-recent artifact dir: {artifact_dir}")
    return _diagnose(artifact_dir)


if __name__ == "__main__":
    sys.exit(main())
