#!/usr/bin/env python3
"""Jev Vision — objects in a dashcam video, detected frame by frame with TypeSafe Jev.

What it does
------------
1. extract  Cut frames out of the source video at a fixed rate (default: the
            source's own rate, so the output cadence matches the input).
2. detect   Ask Jev about each frame. Round one looks at the whole frame; when a
            class looks busy, round two looks at the frame in tiles, so several
            objects of the same class can be found instead of just one.
            Answers are cached as JSON per frame.
3. render   Draw the boxes and a top-right HUD (Jev latency, cumulative cost,
            cumulative detections) onto each frame.
4. encode   Join the rendered frames into a video at the same rate.

Why the stages: detection is the only step that costs money. It runs once and
writes a cache, so re-rendering the video (different colors, thresholds, HUD)
is free.

Notes
-----
* Images and Box questions are TypeSafe's "Danger Zone" features: every request
  that carries an image must include the `TypeSafe-Danger-Zone-Images: true`
  header, and Box questions need Jev 1.14 (`jev-1.14.0` / `jev-preview`).
* Cost is derived from the API's own `usage.input_tokens` on each response, at
  the published rate of $42 per billion input tokens (output tokens are free).
  Nothing is estimated client-side; the token counts come from Jev.
* A Box answer is always four edges, even when the target is absent, and one Box
  question finds one thing. So the code gates every box behind a presence
  probability, asks for distinguishable instances, splits busy frames into tiles
  the way the docs suggest, and drops boxes that overlap a better one."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from self_driving.config import DEFAULT_MODEL, MAX_SIDE, PROJECT_DIR, Config, cache_tag
from self_driving.detection import detect_all
from self_driving.encode import encode
from self_driving.extract import extract_frames
from self_driving.render import render_all
from self_driving.tiles import parse_tile_grids
from self_driving.video import check_disk, parse_fps, parse_timestamp, validate_window


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_DIR / "clip" / "dashcam_23h00-24h00.mp4",
        help="source video; the repo ships a one-minute clip, so the defaults run",
    )
    parser.add_argument("--workdir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output video; default: out/<labels>_<YYYY-MM-DD_HH-MM-SS>.mp4",
    )
    parser.add_argument(
        "--start", default="0:00", help="start timestamp in the source video (MM:SS)"
    )
    parser.add_argument("--end", default="1:00", help="end timestamp in the source video")
    parser.add_argument(
        "--label", default="dashcam", help="name used in the default output filename"
    )
    parser.add_argument(
        "--fps",
        default="source",
        help="'source' (match the video's own rate), a number, or a ratio like 29962/1047",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="frames detected in parallel; 1 keeps them strictly one by one",
    )
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=16,
        help="tiles asked in parallel; the frame, its tiles and every grid "
        "go out together, so this is the width of the fan-out",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=4,
        help="detect with Jev on 1 frame, then track across the next N-1 "
        "(4 = detect 1, track 3); 1 disables tracking",
    )
    parser.add_argument(
        "--track-threshold",
        type=float,
        default=0.5,
        help="minimum template-match quality (NCC) to keep a track alive",
    )
    parser.add_argument(
        "--track-search",
        type=int,
        default=48,
        help="pixels to search around a tracked box each frame",
    )
    parser.add_argument(
        "--track-smooth",
        type=float,
        default=0.6,
        help="how much of a tracked move to apply; lower is steadier",
    )
    parser.add_argument(
        "--detect-blend",
        type=float,
        default=0.35,
        help="how much a fresh detection moves an existing box; "
        "lower is steadier but converges slower",
    )
    parser.add_argument(
        "--presence-threshold",
        type=float,
        default=0.5,
        help="minimum presence probability for a class to be considered visible",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.45,
        help="minimum Box confidence to keep a detection",
    )
    parser.add_argument("--iou-dedupe", type=float, default=0.6, help=argparse.SUPPRESS)
    parser.add_argument(
        "--tile-grids",
        default="4,6,8,12",
        help="tile counts to cut, all asked in parallel ('off' = whole frame only)",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.0,
        help="fraction of a tile shared with its neighbour; the different grids "
        "already cut in different places",
    )
    parser.add_argument(
        "--max-per-tile",
        type=int,
        default=10,
        help="most objects asked for per class per region (frame or tile)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.35,
        help="same-class boxes overlapping more than this: the weaker one goes",
    )
    parser.add_argument(
        "--nms-center",
        type=float,
        default=0.4,
        help="same-class boxes whose centres are this close, in diagonals of the "
        "smaller box, are treated as one (catches loose duplicate boxes)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.35,
        help="small boxes thinner than this ratio are rejected as slivers",
    )
    parser.add_argument(
        "--min-ratio-area",
        type=float,
        default=0.015,
        help="area below which the sliver guard applies, as a fraction of the frame",
    )
    parser.add_argument(
        "--containment",
        type=float,
        default=0.7,
        help="a smaller box this far inside a larger one is dropped as a duplicate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N frames (for tests)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="skip detection, re-render from the cache (re-extracts frames if needed)",
    )
    parser.add_argument(
        "--keep-render",
        action="store_true",
        help="keep the rendered frames after encoding (they are temporary)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="keep the extracted frames after encoding",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env", override=False)

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1

    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    fps = parse_fps(args.fps, args.video)
    tile_grids = parse_tile_grids(args.tile_grids)
    duration = end - start
    if duration <= 0:
        parser.error("--end must be after --start")
    if args.limit:  # a limit shortens the window instead of fighting it
        duration = min(duration, args.limit / float(fps))
    total = round(duration * float(fps))
    if total <= 0:
        parser.error("the window is shorter than one frame")

    workdir = args.workdir / cache_tag(
        fps, tile_grids, args.tile_overlap, args.max_per_tile, max(1, args.detect_every)
    )
    out = args.out or (
        PROJECT_DIR / "out" / f"jev-vision_{args.label}_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"
    )

    cfg = Config(
        video=args.video,
        workdir=workdir,
        out=out,
        start=start,
        duration=duration,
        fps=fps,
        model=args.model,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
        workers=args.workers,
        presence_threshold=args.presence_threshold,
        box_threshold=args.box_threshold,
        tile_grids=tile_grids,
        tile_overlap=args.tile_overlap,
        max_per_tile=args.max_per_tile,
        nms_iou=args.nms_iou,
        nms_center=args.nms_center,
        containment=args.containment,
        min_ratio=args.min_ratio,
        min_ratio_area=args.min_ratio_area,
        tile_workers=args.tile_workers,
        detect_every=max(1, args.detect_every),
        track_threshold=args.track_threshold,
        track_search=args.track_search,
        track_smooth=args.track_smooth,
        detect_blend=args.detect_blend,
        render_only=args.render_only,
    )

    wall = time.perf_counter()
    validate_window(cfg, total)
    check_disk(cfg, total)
    print(f"[cache] {cfg.workdir}")

    if not cfg.render_only:
        have = len(list(cfg.frames_dir.glob("*.jpg"))) if cfg.frames_dir.exists() else 0
        if have < total:
            extract_frames(cfg, total)
        else:
            print(f"[extract] {have} frames already extracted, skipping")
        detect_all(cfg, total)
    else:
        # Re-rendering needs the pixels but never the API; re-cut them if absent.
        have = len(list(cfg.frames_dir.glob("*.jpg"))) if cfg.frames_dir.exists() else 0
        if have < total:
            extract_frames(cfg, total)

    totals = render_all(cfg, total)
    encode(cfg)

    if not args.keep_render:
        for stale in cfg.render_dir.glob("*.jpg"):
            stale.unlink()
        print(f"[clean] removed {total} rendered frames (keep them with --keep-render)")
    if not args.keep_frames:
        for stale in cfg.frames_dir.glob("*.jpg"):
            stale.unlink()
        cfg.frames_dir.rmdir()
        print("[clean] removed the extracted frames (keep them with --keep-frames)")

    print("\n" + "=" * 62)
    print(f"  frames rendered      {totals.frames}  ({float(cfg.fps):.3f} fps, {cfg.fps_expr})")
    if totals.detect_frames:
        print(
            f"  detect / track       {totals.detect_frames} vision (Jev) + "
            f"{totals.track_frames} tracked (1 in {cfg.detect_every})"
        )
    if totals.track_frames:
        print(
            f"  tracking cost        {totals.track_ms / totals.track_frames:.1f} ms per "
            f"tracked frame, {totals.track_ms / 1000:.1f} s total, $0.00"
        )
    print(
        f"  Jev requests         {totals.requests}  "
        f"({totals.requests / max(totals.detect_frames, 1):.2f} per detected frame)"
    )
    if cfg.tile_grids:
        shapes = " + ".join(f"{count}" for count in cfg.tile_grids)
        print(
            f"  tiling               {shapes} = {sum(cfg.tile_grids)} tiles per frame, "
            f"max {cfg.max_per_tile} objects per region"
        )
        print(
            f"  fusion               containment {cfg.containment:.2f}, "
            f"NMS IoU {cfg.nms_iou:.2f}, NMS centre {cfg.nms_center:.2f}"
        )
    else:
        print("  tiling               off (whole frame only)")
    print(f"  detections (total)   {totals.detections}")
    if totals.candidates:
        print(
            f"  raw boxes from Jev   {totals.candidates}  "
            f"({totals.candidates / max(totals.detect_frames, 1):.1f} per detected frame) "
            f"-> fused to {totals.detections} "
            f"({100 * (1 - totals.detections / totals.candidates):.0f}% removed)"
        )
    print(
        f"  detections (mean)    {totals.detections / totals.frames:.2f} per frame"
        if totals.frames
        else ""
    )
    for label, count in sorted(totals.per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {label:<18} {count}")
    print(f"  cost (API usage)     ${totals.cost:.6f}")
    if totals.latency_ms:
        ordered = sorted(totals.latency_ms)
        print(
            f"  jev latency          mean {sum(ordered) / len(ordered):.0f} ms  "
            f"p50 {ordered[len(ordered) // 2]:.0f} ms  max {ordered[-1]:.0f} ms"
        )
    print(f"  output               {cfg.out}")
    print(f"  wall clock           {(time.perf_counter() - wall) / 60:.1f} min")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
