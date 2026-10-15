"""Cutting a frame into regions.

The frame and every tiling of it are just "regions" here: a pixel rectangle plus
the labels each one needs to tell apart an edge of the frame from an edge shared
with a neighbouring tile. `level_weight` is the multi-scale trust factor, used
later to weigh a small tile's answer against the whole frame's.
"""

from __future__ import annotations

import math
from typing import Any

TILE_COUNTS_DEFAULT = (4, 6, 8, 12)


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


def tile_rects(
    width: int, height: int, grid: tuple[int, int], overlap: float
) -> list[dict[str, Any]]:
    """Overlapping tile rectangles, left to right then top to bottom."""
    columns, rows = grid
    spans_x = axis_spans(width, columns, overlap)
    spans_y = axis_spans(height, rows, overlap)
    tiles = []
    for y0, y1, top_interior, bottom_interior in spans_y:
        for x0, x1, left_interior, right_interior in spans_x:
            tiles.append(
                {
                    "rect": (x0, y0, x1, y1),
                    "interior": {
                        "left": left_interior,
                        "right": right_interior,
                        "top": top_interior,
                        "bottom": bottom_interior,
                    },
                }
            )
    return tiles


def touches_interior_edge(
    inner: dict[str, float], interior: dict[str, bool], epsilon: float = 0.02
) -> bool:
    """True when a box runs off an interior tile edge, so it is a fragment."""
    return (
        (interior["left"] and inner["left"] <= epsilon)
        or (interior["right"] and inner["right"] >= 1 - epsilon)
        or (interior["top"] and inner["top"] <= epsilon)
        or (interior["bottom"] and inner["bottom"] >= 1 - epsilon)
    )
