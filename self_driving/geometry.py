"""Box math on fractions of the frame.

Boxes stay in normalized coordinates (0..1) everywhere, so the same numbers work
for the 1568-px copy Jev sees and the 1920-px frame that gets drawn.
"""

from __future__ import annotations

import math


def box_area(box: dict[str, float]) -> float:
    return max(0.0, box["right"] - box["left"]) * max(0.0, box["bottom"] - box["top"])


def intersection_area(a: dict[str, float], b: dict[str, float]) -> float:
    x1, y1 = max(a["left"], b["left"]), max(a["top"], b["top"])
    x2, y2 = min(a["right"], b["right"]), min(a["bottom"], b["bottom"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


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
