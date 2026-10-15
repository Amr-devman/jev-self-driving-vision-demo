"""The only stage that costs money.

One request per region: presence questions to gate a class, plus ranked Box
questions asking where the 1st..Nth most prominent instance is. Every region of
a frame goes out in parallel, and every answer is cached as JSON with its token
usage, so an interrupted run resumes and a re-render never re-bills.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from typesafe_sdk import Box, Image, Noul, TypeSafeClient

from .config import CLASSES, DANGER_ZONE_HEADERS, Config
from .tiles import grid_layout, level_weight, ordinal, tile_rects, touches_interior_edge


def scale_frame(path: Path, max_side: int) -> PILImage.Image:
    with PILImage.open(path) as source:
        frame = source.convert("RGB")
    frame.thumbnail((max_side, max_side))
    return frame


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
                instructions=f"the {ordinal(rank)} most prominent {noun} in `{region}`"
            )
    return questions


def answer_box(answer: Any) -> dict[str, float]:
    b = answer.box
    return {"left": b.left, "top": b.top, "right": b.right, "bottom": b.bottom}


def run_request(
    client: TypeSafeClient,
    cfg: Config,
    state: dict[str, Any],
    questions: dict[str, Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    response = client.system_one(
        state=state,
        questions=questions,
        model=cfg.model,
        extra_headers=DANGER_ZONE_HEADERS,
    )
    return response, round((time.perf_counter() - started) * 1000, 1)


def level_request(
    client: TypeSafeClient,
    cfg: Config,
    api_frame: PILImage.Image,
    level: dict[str, Any],
    classes: list[str],
) -> dict[str, Any]:
    """One region (the frame, or one tile of one grid): ask, then collect boxes."""
    x0, y0, x1, y1 = level["rect"]
    region = level["region"]
    width, height = api_frame.size
    crop = api_frame if region == "frame" else api_frame.crop((x0, y0, x1, y1))

    record: dict[str, Any] = {
        "level": level["name"],
        "rect": [x0, y0, x1, y1],
        "presence": {},
        "boxes": [],
        "input_tokens": 0,
        "latency_ms": None,
        "error": None,
    }
    try:
        response, latency = run_request(
            client,
            cfg,
            {region: Image.from_pil(crop)},
            level_questions(classes, region, cfg.max_per_tile),
        )
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
            record["boxes"].append(
                {
                    "label": label,
                    "rank": int(rank),
                    "confidence": answer.confidence,
                    "box": box,
                    "fragment": touches_interior_edge(inner, level["interior"]),
                }
            )
    return record


def level_candidates(
    cfg: Config, level: dict[str, Any], record: dict[str, Any]
) -> list[dict[str, Any]]:
    """Boxes from one region, gated by that region's own presence answers."""
    candidates = []
    for entry in record["boxes"]:
        label = entry["label"]
        if record["presence"].get(label, 0.0) < cfg.presence_threshold:
            continue
        candidates.append(
            {
                "label": label,
                "confidence": entry["confidence"],
                "box": entry["box"],
                "source": level["name"],
                "weight": level["weight"],
                "fragment": entry["fragment"],
                "rank": entry["rank"],
            }
        )
    return candidates


def detect_frame(
    cfg: Config,
    client: TypeSafeClient,
    frame_path: Path,
    index: int,
    tile_pool: ThreadPoolExecutor | None = None,
) -> dict[str, Any]:
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

    levels: list[dict[str, Any]] = [
        {
            "name": "frame",
            "region": "frame",
            "rect": (0, 0, width, height),
            "weight": 1.0,
            "interior": {"left": False, "right": False, "top": False, "bottom": False},
            "area": frame_area,
            "count": 1,
        }
    ]
    for count in cfg.tile_grids:
        columns, rows = grid_layout(count)
        for tile in tile_rects(width, height, (columns, rows), cfg.tile_overlap):
            x0, y0, x1, y1 = tile["rect"]
            area = float((x1 - x0) * (y1 - y0))
            levels.append(
                {
                    "name": f"grid{count}",
                    "region": "tile",
                    "rect": tile["rect"],
                    "weight": level_weight(area, frame_area),
                    "interior": tile["interior"],
                    "area": area,
                    "count": count,
                }
            )

    try:
        started = time.perf_counter()
        if tile_pool is None:
            level_records = [
                level_request(client, cfg, api_frame, level, classes) for level in levels
            ]
        else:
            futures = [
                tile_pool.submit(level_request, client, cfg, api_frame, level, classes)
                for level in levels
            ]
            level_records = [f.result() for f in futures]
        # Every region was asked at once, so the frame costs one round trip,
        # not the sum of its regions.
        latency = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # keep the pipeline alive; the frame renders empty
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["latency_ms"] = latency
    record["requests"] = len(levels)
    # lengths must match: one record per region asked
    for level, level_record in zip(levels, level_records, strict=True):
        record["input_tokens"] += level_record.get("input_tokens") or 0
        record["model"] = record["model"] or None
        if level["name"] == "frame":
            for key, value in level_record["presence"].items():
                if key not in record["presence"]:
                    record["presence"][key] = value
        record["levels"].append(
            {
                "level": level["name"],
                "rect": list(level["rect"]),
                "weight": round(level["weight"], 3),
                "presence": level_record["presence"],
                "boxes": len(level_record["boxes"]),
                "input_tokens": level_record["input_tokens"],
                "latency_ms": level_record["latency_ms"],
                "error": level_record["error"],
            }
        )
        record["candidates"].extend(level_candidates(cfg, level, level_record))

    return record


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
        shapes = " + ".join(
            f"{count} ({grid_layout(count)[0]}x{grid_layout(count)[1]})" for count in cfg.tile_grids
        )
        tile_desc = f"{shapes} = {sum(cfg.tile_grids)} tiles, {cfg.tile_workers} in parallel"
    else:
        tile_desc = "off"
    print(f"[detect] model={cfg.model}  frames={cfg.workers}  tile pass={tile_desc}")
    print(
        f"[detect] detecting {len(keyframes)} of {total} frames "
        f"(1 in {cfg.detect_every}; the rest are tracked locally, no API cost)"
    )

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
        return detect_frame(
            cfg,
            client_for_thread(),
            cfg.frames_dir / f"{index:06d}.jpg",
            index,
            tile_pool,
        )

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
                        if rate
                        else f"[detect] {done}/{len(pending)}",
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
