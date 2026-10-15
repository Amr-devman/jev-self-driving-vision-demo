"""Turning many overlapping boxes into one box per object.

The paid stage answers the same question many times: the whole frame, four
tilings, ten objects per class each. Fusion is where that pile becomes
detections, and it is deliberately separate from the API calls so thresholds and
rules can be retuned on cached answers without paying again.
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .geometry import box_area, centers_close, contains, iou


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
        scored["score"] = round(
            (confidence if confidence is not None else 0.5) * cand.get("weight", 1.0), 6
        )
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
                elif (
                    contains(group[i]["box"], group[j]["box"], cfg.containment)
                    and group[i]["score"] <= 1.15 * group[j]["score"]
                ):
                    alive[i] = False

        # greedy NMS on what is left, best first
        for i in range(len(group)):
            if not alive[i]:
                continue
            for j in range(i + 1, len(group)):
                if not alive[j]:
                    continue
                if iou(group[i]["box"], group[j]["box"]) >= cfg.nms_iou or centers_close(
                    group[i]["box"], group[j]["box"], cfg.nms_center
                ):
                    alive[j] = False

        kept.extend(g for g, keep in zip(group, alive, strict=True) if keep)

    return kept
