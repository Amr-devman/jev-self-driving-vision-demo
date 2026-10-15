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
  the way the docs suggest, and drops boxes that overlap a better one.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from typesafe_sdk import Box, Image, Noul, TypeSafeClient

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROJECT_DIR = Path(__file__).resolve().parent

# Published TypeSafe pricing: $42 per billion input tokens; output tokens free.
USD_PER_BTOK = 42.0

# Images / Box questions are off by default and need this header.
DANGER_ZONE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}

# Image limits: 256..1568 px per side, so a 1080p frame is scaled to 1568x882.
MAX_SIDE = 1568

# Box questions require Jev 1.14.
DEFAULT_MODEL = "jev-1.14.0"

TILE_COUNTS_DEFAULT = (4, 6, 8, 12)

# Bumped when the cached record layout changes, so old caches are never misread.
CACHE_VERSION = 6


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def grid_layout(count: int) -> tuple[int, int]:
    """(columns, rows) for a tile count, shaped as close to the frame as possible.

    4 -> 2x2, 6 -> 3x2, 8 -> 4x2, 12 -> 4x3. Different shapes cut in different
    places, which is the point of running several of them at once.
    """
    best: tuple[float, int, int] | None = None
    for columns in range(1, count + 1):
        if count % columns:
            continue
        rows = count // columns
        score = abs(math.log((columns / rows) / (16 / 9)))
        if best is None or score < best[0]:
            best = (score, columns, rows)
    assert best is not None
    return best[1], best[2]


def parse_tile_grids(text: str) -> tuple[int, ...]:
    """'4,6,8,12' -> (4, 6, 8, 12); 'off' -> ()."""
    if text.strip().lower() in {"off", "none", "no", ""}:
        return ()
    counts = tuple(int(part) for part in text.replace(" ", "").split(",") if part)
    if any(c < 1 for c in counts):
        raise ValueError("tile counts must be at least 1")
    return counts


def level_weight(tile_area: float, frame_area: float) -> float:
    """How much a level's answer counts, relative to the whole frame.

    More context means a more reliable answer, so the whole frame weighs 1.0 and
    a small tile less. This is the multi-scale idea detectors have used since
    image pyramids: weigh a prediction by the resolution/context it was made at.
    """
    if tile_area >= frame_area:
        return 1.0
    return 0.75 + 0.25 * math.sqrt(tile_area / frame_area)

# The classes this demo looks for. `noun` is how the class is described to Jev;
# `color` is how it is drawn.
CLASSES: dict[str, dict[str, Any]] = {
    "car": {"noun": "car (a passenger vehicle: sedan, SUV, van, hatchback)", "color": (255, 64, 64)},
    "motorcycle": {"noun": "motorcycle, motorbike or scooter", "color": (255, 165, 0)},
    "bus": {"noun": "bus (a large public-transport or coach vehicle)", "color": (120, 230, 120)},
}


def class_color(label: str) -> tuple[int, int, int]:
    return CLASSES.get(label, {}).get("color", (255, 255, 255))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    video: Path
    workdir: Path          # already namespaced by fps + tiling (see cache_tag)
    out: Path
    start: float
    duration: float
    fps: Fraction
    model: str
    max_side: int
    jpeg_quality: int
    workers: int
    presence_threshold: float
    box_threshold: float
    tile_grids: tuple[int, ...]   # tile counts to cut, e.g. (4, 6, 8, 12)
    tile_overlap: float
    max_per_tile: int             # most objects asked for per class per level
    nms_iou: float                # same-class IoU above this suppresses the weaker box
    nms_center: float             # centre distance, in smaller-box diagonals, that also suppresses
    containment: float            # a smaller box this far inside a larger one is dropped
    min_ratio: float              # smallest allowed side ratio for a small box (sliver guard)
    min_ratio_area: float         # area below which the ratio guard applies
    tile_workers: int             # tiles of one frame asked in parallel
    detect_every: int      # detect on 1 frame, track across the next N-1
    track_threshold: float # minimum match quality to keep a track alive
    track_search: int      # search radius in pixels around the previous box
    track_smooth: float    # 0..1, how much of a tracked move to apply (less = smoother)
    detect_blend: float    # 0..1, how much a fresh detection moves an existing track
    render_only: bool

    @property
    def frames_dir(self) -> Path:
        return self.workdir / "frames"

    @property
    def detections_dir(self) -> Path:
        return self.workdir / "detections"

    @property
    def render_dir(self) -> Path:
        return self.workdir / "render"

    @property
    def fps_expr(self) -> str:
        return f"{self.fps.numerator}/{self.fps.denominator}"

    def frame_time(self, index: int) -> float:
        return self.start + index * float(self.fps.denominator) / float(self.fps.numerator)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def parse_timestamp(text: str) -> float:
    """Parse a position in the source video, consistently by shape.

    "1380"     -> 1380 seconds
    "23:00"    -> 23 min 00 s      (MM:SS — the natural reading for a 55-min clip)
    "0:23:00"  -> 23 min 00 s      (HH:MM:SS)

    Mixing the two shapes across --start/--end (e.g. "23:00" with "23:00:01")
    is almost always a mistake, which is why the window is validated against
    the real duration of the video before anything runs.
    """
    if ":" not in text:
        return float(text)
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"unrecognized timestamp: {text!r}")


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:06.3f}"


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def probe_fps(path: Path) -> Fraction:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return Fraction(result.stdout.strip())


def parse_fps(text: str, video: Path | None) -> Fraction:
    """'source' | '30' | '29962/1047' -> an exact rate as a Fraction."""
    text = text.strip()
    if text == "source":
        if video is None:
            raise ValueError("--fps source needs --video")
        return probe_fps(video)
    if "/" in text:
        return Fraction(text)
    return Fraction(text).limit_denominator(10_000)


def cache_tag(fps: Fraction, tile_grids: tuple[int, ...], tile_overlap: float,
              max_per_tile: int, detect_every: int) -> str:
    """A cache is only valid for one frame grid and one question set.

    30 fps and 28.617 fps put different timestamps on frame 900, and different
    question sets produce different answers, so each combination gets its own
    directory instead of silently mixing records. The version moves when the
    record layout changes.
    """
    fps_part = f"fps{fps.numerator}x{fps.denominator}"
    if not tile_grids:
        mode = "notiles"
    else:
        grids = "-".join(str(c) for c in tile_grids)
        mode = f"grids{grids}-ov{tile_overlap:g}-max{max_per_tile}"
    return f"v{CACHE_VERSION}-{fps_part}-{mode}-detect{detect_every}"


def validate_window(cfg: Config, total: int) -> None:
    """Refuse to start unless the requested window exists in the source."""
    source = probe_duration(cfg.video)
    print(f"[window] source is {format_timestamp(source)} long, {cfg.fps} fps")
    print(f"[window] {format_timestamp(cfg.start)} -> {format_timestamp(cfg.start + cfg.duration)} "
          f"({cfg.duration:.1f} s, {total} frames at {float(cfg.fps):.3f} fps)")

    if cfg.start >= source:
        raise SystemExit(
            f"error: --start {cfg.start:.1f}s is past the end of the video ({source:.1f}s).\n"
            f"       Note the shape of timestamps: '23:00' is MM:SS (23 minutes), "
            f"'0:23:00' is HH:MM:SS (23 hours)."
        )
    if cfg.start + cfg.duration > source + 0.05:
        raise SystemExit(
            f"error: window ends at {cfg.start + cfg.duration:.1f}s but the video is only "
            f"{source:.1f}s long.\n"
            f"       Check that --start and --end use the same shape: '23:00' is MM:SS, "
            f"'23:00:01' is HH:MM:SS (23 hours in)."
        )
    if cfg.duration > 30 * 60:
        raise SystemExit(f"error: {cfg.duration / 60:.0f} minutes of video in one run looks wrong; "
                         f"pass a shorter --start/--end window.")
    if total > 20_000:
        raise SystemExit(f"error: {total} frames in one run looks wrong; "
                         f"pass a shorter window or a lower --fps.")


def check_disk(cfg: Config, total: int) -> None:
    """A frame costs roughly 350 KB on disk and a rendered frame roughly 450 KB."""
    needed = total * (350_000 + 450_000) * 1.1
    free = shutil.disk_usage(cfg.workdir.parent).free
    if free < needed:
        raise SystemExit(
            f"error: need about {needed / 1e9:.1f} GB for {total} frames + rendered frames, "
            f"only {free / 1e9:.1f} GB free in {cfg.workdir.parent}."
        )
    print(f"[disk] about {needed / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")


# --------------------------------------------------------------------------- #
# Stage 1 — extract frames
# --------------------------------------------------------------------------- #

def extract_frames(cfg: Config, total: int) -> int:
    """Cut `total` frames starting at cfg.start, on the source's own grid."""
    cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in cfg.frames_dir.glob("*.jpg"):
        stale.unlink()

    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{cfg.start:.3f}",
        "-i", str(cfg.video),
        # A cut of exactly `duration` can lose the final tick; the frames above
        # it are discarded by -frames:v, which is exact.
        "-t", f"{cfg.duration + 1.0:.3f}",
        "-vf", f"fps={cfg.fps_expr}",
        "-frames:v", str(total),
        "-q:v", "3",
        "-start_number", "0",
        str(cfg.frames_dir / "%06d.jpg"),
    ]
    print(f"[extract] {format_timestamp(cfg.start)} +{cfg.duration:.0f}s @ {cfg.fps_expr} -> {total} frames")
    subprocess.run(cmd, check=True)

    n = len(list(cfg.frames_dir.glob("*.jpg")))
    if n != total:
        raise SystemExit(f"error: extraction produced {n} frames, expected {total}; refusing to continue")
    print(f"[extract] {n} frames on disk")
    return n


# --------------------------------------------------------------------------- #
# Stage 2 — ask Jev about each frame
# --------------------------------------------------------------------------- #

def scale_frame(path: Path, max_side: int) -> PILImage.Image:
    with PILImage.open(path) as source:
        frame = source.convert("RGB")
    frame.thumbnail((max_side, max_side))
    return frame


def axis_spans(size: int, count: int, overlap: float) -> list[tuple[int, int, bool, bool]]:
    """`count` spans covering an axis, overlapping by `overlap`.

    Returns (start, end, interior_start, interior_end) per span. An interior
    edge is a tile boundary inside the image: a box touching one is a fragment
    of an object that continues into the neighbouring tile.
    """
    if count == 1:
        return [(0, size, False, False)]
    step = size / count
    span = step * (1 + overlap)
    stride = (size - span) / (count - 1)
    spans = []
    for i in range(count):
        start = i * stride
        spans.append((round(start), round(start + span), i > 0, i < count - 1))
    return spans


def tile_rects(width: int, height: int, grid: tuple[int, int],
               overlap: float) -> list[dict[str, Any]]:
    """Overlapping tile rectangles, left to right then top to bottom."""
    columns, rows = grid
    spans_x = axis_spans(width, columns, overlap)
    spans_y = axis_spans(height, rows, overlap)
    tiles = []
    for (y0, y1, top_interior, bottom_interior) in spans_y:
        for (x0, x1, left_interior, right_interior) in spans_x:
            tiles.append({
                "rect": (x0, y0, x1, y1),
                "interior": {"left": left_interior, "right": right_interior,
                             "top": top_interior, "bottom": bottom_interior},
            })
    return tiles


def touches_interior_edge(inner: dict[str, float], interior: dict[str, bool],
                          epsilon: float = 0.02) -> bool:
    """True when a box runs off an interior tile edge, so it is a fragment."""
    return (
        (interior["left"] and inner["left"] <= epsilon)
        or (interior["right"] and inner["right"] >= 1 - epsilon)
        or (interior["top"] and inner["top"] <= epsilon)
        or (interior["bottom"] and inner["bottom"] >= 1 - epsilon)
    )


def level_questions(classes: list[str], region: str, max_objects: int) -> dict[str, Any]:
    """Presence plus up to `max_objects` ranked boxes per class, for one region.

    A Box question finds one thing, and the questions inside a request are
    answered in parallel, so asking for the 1st..Nth most prominent instance is
    how one request returns several objects of the same class.
    """
    questions: dict[str, Any] = {}
    for name in classes:
        noun = CLASSES[name]["noun"]
        questions[f"{name}_present"] = Noul(instructions=f"Is a {noun} visible in `{region}`?")
        for rank in range(1, max_objects + 1):
            questions[f"{name}_box_{rank}"] = Box(
                instructions=f"the {ordinal(rank)} most prominent {noun} in `{region}`")
    return questions


def answer_box(answer: Any) -> dict[str, float]:
    b = answer.box
    return {"left": b.left, "top": b.top, "right": b.right, "bottom": b.bottom}


def run_request(client: TypeSafeClient, cfg: Config, state: dict[str, Any],
                questions: dict[str, Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    response = client.system_one(
        state=state, questions=questions, model=cfg.model, extra_headers=DANGER_ZONE_HEADERS,
    )
    return response, round((time.perf_counter() - started) * 1000, 1)


def level_request(client: TypeSafeClient, cfg: Config, api_frame: PILImage.Image,
                  level: dict[str, Any], classes: list[str]) -> dict[str, Any]:
    """One region (the frame, or one tile of one grid): ask, then collect boxes."""
    x0, y0, x1, y1 = level["rect"]
    region = level["region"]
    width, height = api_frame.size
    crop = api_frame if region == "frame" else api_frame.crop((x0, y0, x1, y1))

    record: dict[str, Any] = {"level": level["name"], "rect": [x0, y0, x1, y1],
                              "presence": {}, "boxes": [], "input_tokens": 0,
                              "latency_ms": None, "error": None}
    try:
        response, latency = run_request(
            client, cfg, {region: Image.from_pil(crop)},
            level_questions(classes, region, cfg.max_per_tile))
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["latency_ms"] = latency
    record["input_tokens"] = response.usage.input_tokens or 0
    for name, answer in response.answers.items():
        if answer.type == "noul":
            record["presence"][name.removesuffix("_present")] = round(answer.noul, 4)
        elif answer.type == "box" and "_box_" in name:
            label, _, rank = name.partition("_box_")
            inner = answer_box(answer)
            # Region fractions -> whole-frame fractions.
            box = {
                "left": (x0 + inner["left"] * (x1 - x0)) / width,
                "top": (y0 + inner["top"] * (y1 - y0)) / height,
                "right": (x0 + inner["right"] * (x1 - x0)) / width,
                "bottom": (y0 + inner["bottom"] * (y1 - y0)) / height,
            }
            record["boxes"].append({
                "label": label,
                "rank": int(rank),
                "confidence": answer.confidence,
                "box": box,
                "fragment": touches_interior_edge(inner, level["interior"]),
            })
    return record


def level_candidates(cfg: Config, level: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    """Boxes from one region, gated by that region's own presence answers."""
    candidates = []
    for entry in record["boxes"]:
        label = entry["label"]
        if record["presence"].get(label, 0.0) < cfg.presence_threshold:
            continue
        candidates.append({
            "label": label, "confidence": entry["confidence"], "box": entry["box"],
            "source": level["name"], "weight": level["weight"],
            "fragment": entry["fragment"], "rank": entry["rank"],
        })
    return candidates


def detect_frame(cfg: Config, client: TypeSafeClient, frame_path: Path, index: int,
                 tile_pool: ThreadPoolExecutor | None = None) -> dict[str, Any]:
    """Every tiling of one frame, asked in parallel, then fused into boxes."""
    record: dict[str, Any] = {
        "frame": index,
        "time": cfg.frame_time(index),
        "latency_ms": None,
        "model": None,
        "request_id": None,
        "input_tokens": 0,
        "output_tokens": 0,
        "requests": 0,
        "presence": {},
        "levels": [],
        "candidates": [],
        "error": None,
    }

    api_frame = scale_frame(frame_path, cfg.max_side)
    width, height = api_frame.size
    frame_area = float(width * height)
    classes = list(CLASSES)

    levels: list[dict[str, Any]] = [{
        "name": "frame", "region": "frame", "rect": (0, 0, width, height),
        "weight": 1.0, "interior": {"left": False, "right": False, "top": False, "bottom": False},
        "area": frame_area, "count": 1,
    }]
    for count in cfg.tile_grids:
        columns, rows = grid_layout(count)
        for tile in tile_rects(width, height, (columns, rows), cfg.tile_overlap):
            x0, y0, x1, y1 = tile["rect"]
            area = float((x1 - x0) * (y1 - y0))
            levels.append({
                "name": f"grid{count}", "region": "tile", "rect": tile["rect"],
                "weight": level_weight(area, frame_area), "interior": tile["interior"],
                "area": area, "count": count,
            })

    try:
        started = time.perf_counter()
        if tile_pool is None:
            level_records = [level_request(client, cfg, api_frame, level, classes) for level in levels]
        else:
            futures = [tile_pool.submit(level_request, client, cfg, api_frame, level, classes)
                       for level in levels]
            level_records = [f.result() for f in futures]
        # Every region was asked at once, so the frame costs one round trip,
        # not the sum of its regions.
        latency = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # keep the pipeline alive; the frame renders empty
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["latency_ms"] = latency
    record["requests"] = len(levels)
    for level, level_record in zip(levels, level_records):
        record["input_tokens"] += level_record.get("input_tokens") or 0
        record["model"] = record["model"] or None
        if level["name"] == "frame":
            for key, value in level_record["presence"].items():
                if key not in record["presence"]:
                    record["presence"][key] = value
        record["levels"].append({
            "level": level["name"], "rect": list(level["rect"]), "weight": round(level["weight"], 3),
            "presence": level_record["presence"], "boxes": len(level_record["boxes"]),
            "input_tokens": level_record["input_tokens"], "latency_ms": level_record["latency_ms"],
            "error": level_record["error"],
        })
        record["candidates"].extend(level_candidates(cfg, level, level_record))

    return record


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    x1, y1 = max(a["left"], b["left"]), max(a["top"], b["top"])
    x2, y2 = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a["right"] - a["left"]) * (a["bottom"] - a["top"])
    area_b = (b["right"] - b["left"]) * (b["bottom"] - b["top"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box: dict[str, float]) -> float:
    return max(0.0, box["right"] - box["left"]) * max(0.0, box["bottom"] - box["top"])


def intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1, y1 = max(a["left"], b["left"]), max(a["top"], b["top"])
    x2, y2 = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def contains(inner: dict[str, float], outer: dict[str, float], threshold: float) -> bool:
    """How much of `inner` sits inside `outer`: 1.0 means fully contained."""
    area = box_area(inner)
    if area <= 0:
        return False
    return intersection_area(inner, outer) / area >= threshold


def centers_close(a: dict[str, float], b: dict[str, float], factor: float) -> bool:
    """True when two boxes sit on the same thing, judged by their centres.

    IoU alone is not enough for loose boxes: two views of one car can overlap
    well below any usable IoU threshold while their centres coincide. Comparing
    centre distance against the smaller box's diagonal is what face detectors
    add to NMS for exactly this reason.
    """
    aw, ah = a["right"] - a["left"], a["bottom"] - a["top"]
    bw, bh = b["right"] - b["left"], b["bottom"] - b["top"]
    diagonal = min(math.hypot(aw, ah), math.hypot(bw, bh))
    if diagonal <= 0:
        return False
    dx = (a["left"] + a["right"]) / 2 - (b["left"] + b["right"]) / 2
    dy = (a["top"] + a["bottom"]) / 2 - (b["top"] + b["bottom"]) / 2
    return math.hypot(dx, dy) <= factor * diagonal


def fuse_detections(cfg: Config, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Overlay every level's boxes, then reduce them the way detectors do it.

    Candidates arrive from the whole frame and from every tiling, so one object
    is typically described several times: whole, cut in half by a grid edge,
    seen again in a coarser grid. Three passes turn that pile into one box per
    target, per class:

    1. **Score.** Each box gets `confidence x level weight`, so a box seen with
       full context beats the same object seen in a small tile — the multi-scale
       version of "trust the more reliable prediction".
    2. **Containment.** A smaller box that sits mostly inside a larger one is a
       duplicate view or a fragment, and goes. The exception keeps a clearly
       stronger inner box (score > 1.15x the outer one), which is the usual
       score-first behaviour of non-maximum suppression.
    3. **NMS.** Greedy non-maximum suppression: walk boxes best-first and drop
       any later box of the same class that overlaps it beyond `--nms-iou` or
       whose centre sits within `--nms-center` of the smaller box's diagonal.
       Partial overlaps are exactly the case containment cannot decide, and
       keeping the top candidate is what detectors have always done.
    """
    pool: list[dict[str, Any]] = []
    for cand in candidates:
        confidence = cand.get("confidence")
        if confidence is not None and confidence < cfg.box_threshold:
            continue
        box = cand["box"]
        area = box_area(box)
        if area <= 0:
            continue
        # Small, extremely thin boxes are slivers, not objects: detectors reject
        # these with anchor aspect-ratio limits, and it is the one kind of box
        # neither containment nor NMS can rule out, because it overlaps nothing.
        if area < cfg.min_ratio_area:
            width = box["right"] - box["left"]
            height = box["bottom"] - box["top"]
            ratio = min(width / height, height / width) if width > 0 and height > 0 else 0.0
            if ratio < cfg.min_ratio:
                continue
        scored = dict(cand)
        scored["score"] = round((confidence if confidence is not None else 0.5)
                                * cand.get("weight", 1.0), 6)
        pool.append(scored)

    kept: list[dict[str, Any]] = []
    for label in sorted({c["label"] for c in pool}):
        group = sorted((c for c in pool if c["label"] == label), key=lambda c: -c["score"])
        alive = [True] * len(group)

        # containment: drop the smaller of two nested boxes
        for i in range(len(group)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(group)):
                if not alive[j]:
                    continue
                if contains(group[j]["box"], group[i]["box"], cfg.containment):
                    if group[j]["score"] <= 1.15 * group[i]["score"]:
                        alive[j] = False
                elif contains(group[i]["box"], group[j]["box"], cfg.containment):
                    if group[i]["score"] <= 1.15 * group[j]["score"]:
                        alive[i] = False

        # greedy NMS on what is left, best first
        for i in range(len(group)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(group)):
                if not alive[j]:
                    continue
                if iou(group[i]["box"], group[j]["box"]) >= cfg.nms_iou \
                        or centers_close(group[i]["box"], group[j]["box"], cfg.nms_center):
                    alive[j] = False

        kept.extend(g for g, keep in zip(group, alive) if keep)

    return kept


def keyframe_indices(total: int, detect_every: int) -> list[int]:
    """Frames that get a Jev request: one detection, then `detect_every - 1` tracked frames."""
    return list(range(0, total, max(1, detect_every)))


def detect_all(cfg: Config, total: int) -> None:
    cfg.detections_dir.mkdir(parents=True, exist_ok=True)
    keyframes = keyframe_indices(total, cfg.detect_every)
    pending = [i for i in keyframes if not (cfg.detections_dir / f"{i:06d}.json").exists()]
    cached = len(keyframes) - len(pending)
    if cached:
        print(f"[detect] {cached} keyframes already cached, {len(pending)} to do")
    if not pending:
        return

    if cfg.tile_grids:
        shapes = " + ".join(f"{count} ({grid_layout(count)[0]}x{grid_layout(count)[1]})"
                            for count in cfg.tile_grids)
        tile_desc = f"{shapes} = {sum(cfg.tile_grids)} tiles, {cfg.tile_workers} in parallel"
    else:
        tile_desc = "off"
    print(f"[detect] model={cfg.model}  frames={cfg.workers}  tile pass={tile_desc}")
    print(f"[detect] detecting {len(keyframes)} of {total} frames "
          f"(1 in {cfg.detect_every}; the rest are tracked locally, no API cost)")

    # One client per worker thread: the sync client wraps a connection pool,
    # and this keeps each worker's connections independent. threading.local
    # matters here — a plain dict would be shared by every worker.
    local = threading.local()

    def client_for_thread() -> TypeSafeClient:
        client = getattr(local, "client", None)
        if client is None:
            client = TypeSafeClient()
            local.client = client
        return client

    # Tiles of a frame are independent requests, so they go out together. The
    # pool is sized for every frame worker's tiles at once, which keeps the
    # total in flight bounded.
    tile_pool = None
    if cfg.tile_grids and cfg.tile_workers > 1:
        tile_pool = ThreadPoolExecutor(max_workers=cfg.tile_workers * max(1, cfg.workers))

    def work(index: int) -> dict[str, Any]:
        return detect_frame(cfg, client_for_thread(), cfg.frames_dir / f"{index:06d}.jpg",
                            index, tile_pool)

    done = 0
    started = time.perf_counter()
    errors = 0
    try:
        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = {pool.submit(work, i): i for i in pending}
            for future in as_completed(futures):
                record = future.result()
                out_path = cfg.detections_dir / f"{record['frame']:06d}.json"
                out_path.write_text(json.dumps(record))
                done += 1
                if record["error"]:
                    errors += 1
                if done % 25 == 0 or done == len(pending):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"[detect] {done}/{len(pending)}  "
                        f"{rate:.2f} keyframes/s  eta {(len(pending) - done) / rate / 60:.1f} min"
                        if rate else f"[detect] {done}/{len(pending)}",
                        flush=True,
                    )
    finally:
        if tile_pool is not None:
            tile_pool.shutdown(wait=True)
        client = getattr(local, "client", None)
        if client is not None:
            client.close()
    if errors:
        print(f"[detect] {errors} frames failed (recorded in the cache; no detections for them)")


# --------------------------------------------------------------------------- #
# Local tracking (no API): follow a box across the frames between detections
# --------------------------------------------------------------------------- #

def to_gray(frame: PILImage.Image) -> np.ndarray:
    return np.asarray(frame.convert("L"), dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation: 1 is a perfect match, 0 is unrelated."""
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / denom if denom > 1e-6 else 0.0


def _ssd_search(prev: np.ndarray, curr: np.ndarray, x0: int, y0: int,
                w: int, h: int, radius: int) -> tuple[float, int, int] | None:
    """Best integer offset within `radius` by sum of squared differences."""
    height, width = curr.shape
    template = prev[y0:y0 + h, x0:x0 + w]
    if template.size == 0:
        return None
    th, tw = template.shape  # numpy clips slices at the edge; match that shape
    best: tuple[float, int, int] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or xx < 0 or yy + th > height or xx + tw > width:
                continue
            patch = curr[yy:yy + th, xx:xx + tw]
            if patch.shape != template.shape:
                continue
            diff = template - patch
            score = float((diff * diff).sum())
            if best is None or score < best[0]:
                best = (score, dx, dy)
    return best


def track_box(prev_gray: np.ndarray, curr_gray: np.ndarray, box: dict[str, float],
              search: int, min_template: int = 12) -> tuple[dict[str, float] | None, float]:
    """Follow one box into the next frame by template matching.

    A coarse pass on a downscaled grid covers real motion cheaply; a
    full-resolution pass then lands the box within a pixel. The match quality
    (NCC) is returned so callers can drop tracks that lost their object.
    """
    height, width = curr_gray.shape
    x0 = int(round(box["left"] * width))
    y0 = int(round(box["top"] * height))
    x1 = int(round(box["right"] * width))
    y1 = int(round(box["bottom"] * height))
    # Keep the box fully inside the frame: a slice that runs off the edge comes
    # back smaller than asked for, and shapes must agree across both frames.
    x1 = min(width, max(x1, 0))
    y1 = min(height, max(y1, 0))
    x0 = max(0, min(x0, x1))
    y0 = max(0, min(y0, y1))
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return None, 0.0

    scale = int(min(4, max(1, min(w, h) // min_template)))
    dx = dy = 0
    if scale > 1:
        coarse = _ssd_search(prev_gray[::scale, ::scale], curr_gray[::scale, ::scale],
                             x0 // scale, y0 // scale,
                             max(2, w // scale), max(2, h // scale),
                             max(1, search // scale))
        if coarse is None:
            return None, 0.0
        _, cdx, cdy = coarse
        dx, dy = cdx * scale, cdy * scale

    fine = _ssd_search(prev_gray, curr_gray, x0 + dx, y0 + dy, w, h,
                       search if scale == 1 else scale + 2)
    if fine is None:
        return None, 0.0
    _, fdx, fdy = fine
    nx, ny = x0 + dx + fdx, y0 + dy + fdy
    nx = max(0, min(nx, width - w))
    ny = max(0, min(ny, height - h))

    quality = _ncc(prev_gray[y0:y0 + h, x0:x0 + w], curr_gray[ny:ny + h, nx:nx + w])
    return {
        "left": nx / width, "top": ny / height,
        "right": (nx + w) / width, "bottom": (ny + h) / height,
    }, quality


def blend_box(new: dict[str, float], old: dict[str, float], weight_new: float) -> dict[str, float]:
    """Weighted mix of two boxes: weight_new 1.0 means take the new one as is."""
    return {k: weight_new * new[k] + (1 - weight_new) * old[k] for k in new}


def dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float],
                     color: tuple[int, int, int], width: int = 4,
                     dash: int = 20, gap: int = 14) -> None:
    x1, y1, x2, y2 = box
    for start, end, horizontal in ((y1, y2, False), (x1, x2, True)):
        pos = start
        while pos < end:
            stop = min(pos + dash, end)
            if horizontal:
                draw.line((pos, y1, stop, y1), fill=color, width=width)
                draw.line((pos, y2, stop, y2), fill=color, width=width)
            else:
                draw.line((x1, pos, x1, stop), fill=color, width=width)
                draw.line((x2, pos, x2, stop), fill=color, width=width)
            pos = stop + gap


# --------------------------------------------------------------------------- #
# Stage 3 — render boxes + HUD
# --------------------------------------------------------------------------- #

@dataclass
class Totals:
    cost: float = 0.0
    detections: int = 0
    frames: int = 0
    detect_frames: int = 0
    track_frames: int = 0
    track_ms: float = 0.0
    requests: int = 0
    tiled_frames: int = 0
    candidates: int = 0
    latency_ms: list[float] = field(default_factory=list)
    per_class: dict[str, int] = field(default_factory=dict)


def load_records(cfg: Config, total: int) -> list[dict[str, Any]]:
    records = []
    for i in range(total):
        path = cfg.detections_dir / f"{i:06d}.json"
        if path.exists():
            records.append(json.loads(path.read_text()))
    records.sort(key=lambda r: r["frame"])
    return records


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def render_all(cfg: Config, total: int) -> Totals:
    cfg.render_dir.mkdir(parents=True, exist_ok=True)
    for stale in cfg.render_dir.glob("*.jpg"):
        stale.unlink()

    font_label = load_font(30, bold=True)
    font_hud = load_font(34, bold=True)
    font_hud_small = load_font(26, bold=False)
    font_corner = load_font(26, bold=False)

    totals = Totals()
    by_frame = {r["frame"]: r for r in load_records(cfg, total)}
    missing = [i for i in keyframe_indices(total, cfg.detect_every) if i not in by_frame]
    if missing:
        raise SystemExit(f"error: {len(missing)} detection frames have no record "
                         f"(first: {missing[0]}); run the detect stage before rendering")

    prev_gray: np.ndarray | None = None
    tracks: list[dict[str, Any]] = []
    latency_sum = 0.0
    latency_n = 0

    for index in range(total):
        record = by_frame.get(index)
        frame_path = cfg.frames_dir / f"{index:06d}.jpg"
        if not frame_path.exists():
            raise SystemExit(f"error: frame {index} is missing from {cfg.frames_dir}")

        with PILImage.open(frame_path) as source:
            frame = source.convert("RGB")
        gray = to_gray(frame)
        totals.frames += 1

        if record is not None:
            # --- detection frame: ask Jev, then re-seed the tracks ----------
            detections = fuse_detections(cfg, record.get("candidates") or [])
            if cfg.detect_blend < 1.0:
                detections = [
                    dict(det, box=blend_box(det["box"], match["box"], cfg.detect_blend))
                    if (match := next((t for t in tracks
                                       if t["label"] == det["label"]
                                       and iou(det["box"], t["box"]) >= 0.1), None))
                    else det
                    for det in detections
                ]
            tracks = [dict(d) for d in detections]
            totals.detect_frames += 1
            totals.cost += (record.get("input_tokens") or 0) * USD_PER_BTOK / 1e9
            totals.requests += record.get("requests") or 0
            if record.get("tiles"):
                totals.tiled_frames += 1
            totals.candidates += len(record.get("candidates") or [])
            frame_latency = record.get("latency_ms")
            if frame_latency is not None:
                totals.latency_ms.append(frame_latency)
                latency_sum += frame_latency
                latency_n += 1
        else:
            # --- track frame: follow the live boxes locally, no API cost ----
            started = time.perf_counter()
            followed: list[dict[str, Any]] = []
            for track in tracks:
                box, quality = track_box(prev_gray, gray, track["box"], cfg.track_search)
                if box is None or quality < cfg.track_threshold:
                    continue
                followed.append({**track,
                                 "box": blend_box(box, track["box"], cfg.track_smooth),
                                 "source": "track", "quality": round(quality, 3)})
            tracks = followed
            totals.track_frames += 1
            totals.track_ms += (time.perf_counter() - started) * 1000
            frame_latency = None

        detections = tracks
        for det in detections:
            totals.detections += 1
            totals.per_class[det["label"]] = totals.per_class.get(det["label"], 0) + 1

        with PILImage.open(frame_path) as source:
            frame = source.convert("RGB")
        draw = ImageDraw.Draw(frame, "RGBA")
        w, h = frame.size

        # --- detections: solid for Jev answers, dashed for tracked boxes -----
        for det in detections:
            box = det["box"]
            x1, y1 = box["left"] * w, box["top"] * h
            x2, y2 = box["right"] * w, box["bottom"] * h
            color = class_color(det["label"])
            tracked = det.get("source") == "track"
            if tracked:
                dashed_rectangle(draw, (x1, y1, x2, y2), color)
            else:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=4)

            conf = det.get("confidence")
            if tracked:
                text = f"~{det['label']}"
            elif conf is not None:
                text = f"{det['label']} {conf:.2f}"
            else:
                text = det["label"]
            tb = draw.textbbox((0, 0), text, font=font_label)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            pad = 6
            chip_top = max(0.0, y1 - th - 2 * pad - 2)
            draw.rectangle((x1, chip_top, x1 + tw + 2 * pad, chip_top + th + 2 * pad),
                           fill=color if not tracked else (color[0], color[1], color[2], 170))
            draw.text((x1 + pad, chip_top + pad - tb[1]), text, font=font_label, fill=(16, 16, 16))

        # --- HUD, top right -------------------------------------------------
        # Latency is a running average over the frames Jev actually answered:
        # a per-frame number changes 30 times a second and reads as noise, and
        # tracked frames have no Jev latency to report at all.
        running_latency = latency_sum / latency_n if latency_n else None
        lines = [
            ("JEV LATENCY AVG", f"{running_latency:,.0f} ms" if running_latency is not None else "—"),
            ("TOTAL COST", f"${totals.cost:.6f}"),
            ("DETECTIONS", f"{totals.detections}"),
        ]
        line_h = 54
        pad_x, pad_y = 26, 20
        widest = 0
        for label, value in lines:
            lb = draw.textbbox((0, 0), label, font=font_hud_small)
            vb = draw.textbbox((0, 0), value, font=font_hud)
            widest = max(widest, lb[2] - lb[0] + 24 + vb[2] - vb[0])
        panel_w = widest + 2 * pad_x
        panel_h = len(lines) * line_h + 2 * pad_y
        x0, y0 = w - panel_w - 24, 24
        draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=12,
                               fill=(10, 12, 16, 190), outline=(255, 255, 255, 60), width=1)
        for row, (label, value) in enumerate(lines):
            row_y = y0 + pad_y + row * line_h
            draw.text((x0 + pad_x, row_y + 10), label, font=font_hud_small, fill=(150, 160, 175))
            vb = draw.textbbox((0, 0), value, font=font_hud)
            draw.text((x0 + panel_w - pad_x - (vb[2] - vb[0]), row_y), value,
                      font=font_hud, fill=(255, 255, 255))

        # --- corner: where we are in the source ------------------------------
        time_text = format_timestamp(record["time"]) if record is not None else \
            format_timestamp(cfg.frame_time(index))
        corner = (f"frame {index + 1}/{total}   t={time_text}   "
                  f"{float(cfg.fps):.3f} fps   detect 1 / track {cfg.detect_every - 1}")
        cb = draw.textbbox((0, 0), corner, font=font_corner)
        draw.rectangle((24, h - (cb[3] - cb[1]) - 40, 24 + (cb[2] - cb[0]) + 24, h - 16),
                       fill=(10, 12, 16, 190))
        draw.text((36, h - (cb[3] - cb[1]) - 28), corner, font=font_corner, fill=(200, 210, 225))

        frame.save(cfg.render_dir / f"{index:06d}.jpg", "JPEG", quality=cfg.jpeg_quality)
        prev_gray = gray

    return totals


# --------------------------------------------------------------------------- #
# Stage 4 — encode
# --------------------------------------------------------------------------- #

def encode(cfg: Config) -> None:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-framerate", cfg.fps_expr,
        "-i", str(cfg.render_dir / "%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", cfg.fps_expr,
        "-movflags", "+faststart",
        str(cfg.out),
    ]
    print(f"[encode] {cfg.fps_expr} fps ({float(cfg.fps):.3f}) -> {cfg.out}")
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path,
                        default=PROJECT_DIR / "clip" / "dashcam_23h00-24h00.mp4",
                        help="source video; the repo ships a one-minute clip, so the defaults run")
    parser.add_argument("--workdir", type=Path, default=PROJECT_DIR / "work")
    parser.add_argument("--out", type=Path, default=None,
                        help="output video; default: out/<labels>_<YYYY-MM-DD_HH-MM-SS>.mp4")
    parser.add_argument("--start", default="0:00", help="start timestamp in the source video (MM:SS)")
    parser.add_argument("--end", default="1:00", help="end timestamp in the source video")
    parser.add_argument("--label", default="dashcam", help="name used in the default output filename")
    parser.add_argument("--fps", default="source",
                        help="'source' (match the video's own rate), a number, or a ratio like 29962/1047")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=4,
                        help="frames detected in parallel; 1 keeps them strictly one by one")
    parser.add_argument("--tile-workers", type=int, default=16,
                        help="tiles asked in parallel; the frame, its tiles and every grid "
                             "go out together, so this is the width of the fan-out")
    parser.add_argument("--detect-every", type=int, default=4,
                        help="detect with Jev on 1 frame, then track across the next N-1 "
                             "(4 = detect 1, track 3); 1 disables tracking")
    parser.add_argument("--track-threshold", type=float, default=0.5,
                        help="minimum template-match quality (NCC) to keep a track alive")
    parser.add_argument("--track-search", type=int, default=48,
                        help="pixels to search around a tracked box each frame")
    parser.add_argument("--track-smooth", type=float, default=0.6,
                        help="how much of a tracked move to apply; lower is steadier")
    parser.add_argument("--detect-blend", type=float, default=0.35,
                        help="how much a fresh detection moves an existing box; "
                             "lower is steadier but converges slower")
    parser.add_argument("--presence-threshold", type=float, default=0.5,
                        help="minimum presence probability for a class to be considered visible")
    parser.add_argument("--box-threshold", type=float, default=0.45,
                        help="minimum Box confidence to keep a detection")
    parser.add_argument("--iou-dedupe", type=float, default=0.6, help=argparse.SUPPRESS)
    parser.add_argument("--tile-grids", default="4,6,8,12",
                        help="tile counts to cut, all asked in parallel ('off' = whole frame only)")
    parser.add_argument("--tile-overlap", type=float, default=0.0,
                        help="fraction of a tile shared with its neighbour; the different grids "
                             "already cut in different places")
    parser.add_argument("--max-per-tile", type=int, default=10,
                        help="most objects asked for per class per region (frame or tile)")
    parser.add_argument("--nms-iou", type=float, default=0.35,
                        help="same-class boxes overlapping more than this: the weaker one goes")
    parser.add_argument("--nms-center", type=float, default=0.4,
                        help="same-class boxes whose centres are this close, in diagonals of the "
                             "smaller box, are treated as one (catches loose duplicate boxes)")
    parser.add_argument("--min-ratio", type=float, default=0.35,
                        help="small boxes thinner than this ratio are rejected as slivers")
    parser.add_argument("--min-ratio-area", type=float, default=0.015,
                        help="area below which the sliver guard applies, as a fraction of the frame")
    parser.add_argument("--containment", type=float, default=0.7,
                        help="a smaller box this far inside a larger one is dropped as a duplicate")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N frames (for tests)")
    parser.add_argument("--render-only", action="store_true",
                        help="skip detection, re-render from the cache (re-extracts frames if needed)")
    parser.add_argument("--keep-render", action="store_true",
                        help="keep the rendered frames after encoding (they are temporary)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="keep the extracted frames after encoding")
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

    workdir = args.workdir / cache_tag(fps, tile_grids, args.tile_overlap,
                                      args.max_per_tile, max(1, args.detect_every))
    out = args.out or (PROJECT_DIR / "out" /
                       f"jev-vision_{args.label}_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4")

    cfg = Config(
        video=args.video, workdir=workdir, out=out,
        start=start, duration=duration, fps=fps, model=args.model,
        max_side=args.max_side, jpeg_quality=args.jpeg_quality, workers=args.workers,
        presence_threshold=args.presence_threshold, box_threshold=args.box_threshold,
        tile_grids=tile_grids, tile_overlap=args.tile_overlap, max_per_tile=args.max_per_tile,
        nms_iou=args.nms_iou, nms_center=args.nms_center, containment=args.containment,
        min_ratio=args.min_ratio, min_ratio_area=args.min_ratio_area,
        tile_workers=args.tile_workers,
        detect_every=max(1, args.detect_every), track_threshold=args.track_threshold,
        track_search=args.track_search, track_smooth=args.track_smooth,
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
        print(f"  detect / track       {totals.detect_frames} vision (Jev) + {totals.track_frames} tracked "
              f"(1 in {cfg.detect_every})")
    if totals.track_frames:
        print(f"  tracking cost        {totals.track_ms / totals.track_frames:.1f} ms per tracked frame, "
              f"{totals.track_ms / 1000:.1f} s total, $0.00")
    print(f"  Jev requests         {totals.requests}  "
          f"({totals.requests / max(totals.detect_frames, 1):.2f} per detected frame)")
    if cfg.tile_grids:
        shapes = " + ".join(f"{count}" for count in cfg.tile_grids)
        print(f"  tiling               {shapes} = {sum(cfg.tile_grids)} tiles per frame, "
              f"max {cfg.max_per_tile} objects per region")
        print(f"  fusion               containment {cfg.containment:.2f}, "
              f"NMS IoU {cfg.nms_iou:.2f}, NMS centre {cfg.nms_center:.2f}")
    else:
        print("  tiling               off (whole frame only)")
    print(f"  detections (total)   {totals.detections}")
    if totals.candidates:
        print(f"  raw boxes from Jev   {totals.candidates}  "
              f"({totals.candidates / max(totals.detect_frames, 1):.1f} per detected frame) "
              f"-> fused to {totals.detections} "
              f"({100 * (1 - totals.detections / totals.candidates):.0f}% removed)")
    print(f"  detections (mean)    {totals.detections / totals.frames:.2f} per frame" if totals.frames else "")
    for label, count in sorted(totals.per_class.items(), key=lambda kv: -kv[1]):
        print(f"    {label:<18} {count}")
    print(f"  cost (API usage)     ${totals.cost:.6f}")
    if totals.latency_ms:
        ordered = sorted(totals.latency_ms)
        print(f"  jev latency          mean {sum(ordered)/len(ordered):.0f} ms  "
              f"p50 {ordered[len(ordered)//2]:.0f} ms  max {ordered[-1]:.0f} ms")
    print(f"  output               {cfg.out}")
    print(f"  wall clock           {(time.perf_counter() - wall)/60:.1f} min")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
