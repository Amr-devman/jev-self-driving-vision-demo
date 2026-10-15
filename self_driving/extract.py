"""Cutting frames out of the source video with ffmpeg.

The rate defaults to the video's own (`--fps source`), read with ffprobe as
r_frame_rate, so the output cadence matches the input instead of approximating
it. Frames are cut slightly long and then trimmed with -frames:v, which is
exact: a cut of precisely `duration` can drop the final tick.
"""

from __future__ import annotations

import subprocess

from .config import Config
from .video import format_timestamp


def extract_frames(cfg: Config, total: int) -> int:
    """Cut `total` frames starting at cfg.start, on the source's own grid."""
    cfg.frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in cfg.frames_dir.glob("*.jpg"):
        stale.unlink()

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        f"{cfg.start:.3f}",
        "-i",
        str(cfg.video),
        # A cut of exactly `duration` can lose the final tick; the frames above
        # it are discarded by -frames:v, which is exact.
        "-t",
        f"{cfg.duration + 1.0:.3f}",
        "-vf",
        f"fps={cfg.fps_expr}",
        "-frames:v",
        str(total),
        "-q:v",
        "3",
        "-start_number",
        "0",
        str(cfg.frames_dir / "%06d.jpg"),
    ]
    print(
        f"[extract] {format_timestamp(cfg.start)} +{cfg.duration:.0f}s "
        f"@ {cfg.fps_expr} -> {total} frames"
    )
    subprocess.run(cmd, check=True)

    n = len(list(cfg.frames_dir.glob("*.jpg")))
    if n != total:
        raise SystemExit(
            f"error: extraction produced {n} frames, expected {total}; refusing to continue"
        )
    print(f"[extract] {n} frames on disk")
    return n
