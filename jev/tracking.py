"""Following boxes between detections, with no API call.

Jev answers one frame in four. For the frames in between, the previous box
becomes a template: a coarse pass on a downscaled grid finds it, a
full-resolution pass lands it within a pixel, and normalized cross-correlation
decides whether the object is still there.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image as PILImage


def to_gray(frame: PILImage.Image) -> np.ndarray:
    return np.asarray(frame.convert("L"), dtype=np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation: 1 is a perfect match, 0 is unrelated."""
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / denom if denom > 1e-6 else 0.0


def _ssd_search(
    prev: np.ndarray, curr: np.ndarray, x0: int, y0: int, w: int, h: int, radius: int
) -> tuple[float, int, int] | None:
    """Best integer offset within `radius` by sum of squared differences."""
    height, width = curr.shape
    template = prev[y0 : y0 + h, x0 : x0 + w]
    if template.size == 0:
        return None
    th, tw = template.shape  # numpy clips slices at the edge; match that shape
    best: tuple[float, int, int] | None = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or xx < 0 or yy + th > height or xx + tw > width:
                continue
            patch = curr[yy : yy + th, xx : xx + tw]
            if patch.shape != template.shape:
                continue
            diff = template - patch
            score = float((diff * diff).sum())
            if best is None or score < best[0]:
                best = (score, dx, dy)
    return best


def track_box(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    box: dict[str, float],
    search: int,
    min_template: int = 12,
) -> tuple[dict[str, float] | None, float]:
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
        coarse = _ssd_search(
            prev_gray[::scale, ::scale],
            curr_gray[::scale, ::scale],
            x0 // scale,
            y0 // scale,
            max(2, w // scale),
            max(2, h // scale),
            max(1, search // scale),
        )
        if coarse is None:
            return None, 0.0
        _, cdx, cdy = coarse
        dx, dy = cdx * scale, cdy * scale

    fine = _ssd_search(
        prev_gray,
        curr_gray,
        x0 + dx,
        y0 + dy,
        w,
        h,
        search if scale == 1 else scale + 2,
    )
    if fine is None:
        return None, 0.0
    _, fdx, fdy = fine
    nx, ny = x0 + dx + fdx, y0 + dy + fdy
    nx = max(0, min(nx, width - w))
    ny = max(0, min(ny, height - h))

    quality = _ncc(prev_gray[y0 : y0 + h, x0 : x0 + w], curr_gray[ny : ny + h, nx : nx + w])
    return {
        "left": nx / width,
        "top": ny / height,
        "right": (nx + w) / width,
        "bottom": (ny + h) / height,
    }, quality


def blend_box(new: dict[str, float], old: dict[str, float], weight_new: float) -> dict[str, float]:
    """Weighted mix of two boxes: weight_new 1.0 means take the new one as is."""
    return {k: weight_new * new[k] + (1 - weight_new) * old[k] for k in new}
