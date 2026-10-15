"""Drawing the detections and the HUD.

Rendering is where policy gets applied (thresholds, fusion, what counts as a
detection) and where the picture is made. It reads the cache and never touches
the network, which is why changing anything visual is free.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from .config import USD_PER_BTOK, Config, class_color
from .detection import keyframe_indices
from .fusion import fuse_detections
from .geometry import iou
from .tracking import blend_box, to_gray, track_box
from .video import format_timestamp


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
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    width: int = 4,
    dash: int = 20,
    gap: int = 14,
) -> None:
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
        raise SystemExit(
            f"error: {len(missing)} detection frames have no record "
            f"(first: {missing[0]}); run the detect stage before rendering"
        )

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
                    if (
                        match := next(
                            (
                                t
                                for t in tracks
                                if t["label"] == det["label"] and iou(det["box"], t["box"]) >= 0.1
                            ),
                            None,
                        )
                    )
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
                followed.append(
                    {
                        **track,
                        "box": blend_box(box, track["box"], cfg.track_smooth),
                        "source": "track",
                        "quality": round(quality, 3),
                    }
                )
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
            draw.rectangle(
                (x1, chip_top, x1 + tw + 2 * pad, chip_top + th + 2 * pad),
                fill=color if not tracked else (color[0], color[1], color[2], 170),
            )
            draw.text(
                (x1 + pad, chip_top + pad - tb[1]),
                text,
                font=font_label,
                fill=(16, 16, 16),
            )

        # --- HUD, top right -------------------------------------------------
        # Latency is a running average over the frames Jev actually answered:
        # a per-frame number changes 30 times a second and reads as noise, and
        # tracked frames have no Jev latency to report at all.
        running_latency = latency_sum / latency_n if latency_n else None
        lines = [
            (
                "JEV LATENCY AVG",
                f"{running_latency:,.0f} ms" if running_latency is not None else "—",
            ),
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
        draw.rounded_rectangle(
            (x0, y0, x0 + panel_w, y0 + panel_h),
            radius=12,
            fill=(10, 12, 16, 190),
            outline=(255, 255, 255, 60),
            width=1,
        )
        for row, (label, value) in enumerate(lines):
            row_y = y0 + pad_y + row * line_h
            draw.text(
                (x0 + pad_x, row_y + 10),
                label,
                font=font_hud_small,
                fill=(150, 160, 175),
            )
            vb = draw.textbbox((0, 0), value, font=font_hud)
            draw.text(
                (x0 + panel_w - pad_x - (vb[2] - vb[0]), row_y),
                value,
                font=font_hud,
                fill=(255, 255, 255),
            )

        # --- corner: where we are in the source ------------------------------
        time_text = (
            format_timestamp(record["time"])
            if record is not None
            else format_timestamp(cfg.frame_time(index))
        )
        corner = (
            f"frame {index + 1}/{total}   t={time_text}   "
            f"{float(cfg.fps):.3f} fps   detect 1 / track {cfg.detect_every - 1}"
        )
        cb = draw.textbbox((0, 0), corner, font=font_corner)
        draw.rectangle(
            (24, h - (cb[3] - cb[1]) - 40, 24 + (cb[2] - cb[0]) + 24, h - 16),
            fill=(10, 12, 16, 190),
        )
        draw.text(
            (36, h - (cb[3] - cb[1]) - 28),
            corner,
            font=font_corner,
            fill=(200, 210, 225),
        )

        frame.save(cfg.render_dir / f"{index:06d}.jpg", "JPEG", quality=cfg.jpeg_quality)
        prev_gray = gray

    return totals
