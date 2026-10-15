"""Settings, constants and the cache key.

Everything the rest of the package needs to agree on lives here: what the three
classes are and how Jev is told about them, how the request header for the
"danger zone" image features reads, and the Config dataclass the stages take.

Cost model: TypeSafe bills input tokens only ($42 per billion), so USD_PER_BTOK
is the single number that turns a response's `usage` into dollars.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

CACHE_VERSION = 6
USD_PER_BTOK = 42.0
DANGER_ZONE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
MAX_SIDE = 1568
DEFAULT_MODEL = "jev-1.14.0"
CLASSES: dict[str, dict[str, Any]] = {
    "car": {
        "noun": "car (a passenger vehicle: sedan, SUV, van, hatchback)",
        "color": (255, 64, 64),
    },
    "motorcycle": {"noun": "motorcycle, motorbike or scooter", "color": (255, 165, 0)},
    "bus": {
        "noun": "bus (a large public-transport or coach vehicle)",
        "color": (120, 230, 120),
    },
}


def class_color(label: str) -> tuple[int, int, int]:
    return CLASSES.get(label, {}).get("color", (255, 255, 255))


@dataclass
class Config:
    video: Path
    workdir: Path  # already namespaced by fps + tiling (see cache_tag)
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
    tile_grids: tuple[int, ...]  # tile counts to cut, e.g. (4, 6, 8, 12)
    tile_overlap: float
    max_per_tile: int  # most objects asked for per class per level
    nms_iou: float  # same-class IoU above this suppresses the weaker box
    nms_center: float  # centre distance, in smaller-box diagonals, that also suppresses
    containment: float  # a smaller box this far inside a larger one is dropped
    min_ratio: float  # smallest allowed side ratio for a small box (sliver guard)
    min_ratio_area: float  # area below which the ratio guard applies
    tile_workers: int  # tiles of one frame asked in parallel
    detect_every: int  # detect on 1 frame, track across the next N-1
    track_threshold: float  # minimum match quality to keep a track alive
    track_search: int  # search radius in pixels around the previous box
    track_smooth: float  # 0..1, how much of a tracked move to apply (less = smoother)
    detect_blend: float  # 0..1, how much a fresh detection moves an existing track
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


def cache_tag(
    fps: Fraction,
    tile_grids: tuple[int, ...],
    tile_overlap: float,
    max_per_tile: int,
    detect_every: int,
) -> str:
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


# The repository root: this file lives in jev/, so the root is its parent.
PROJECT_DIR = Path(__file__).resolve().parent.parent
