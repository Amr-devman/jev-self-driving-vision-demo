"""Joining the rendered frames into a video."""

from __future__ import annotations

import subprocess

from .config import Config


def encode(cfg: Config) -> None:
    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        cfg.fps_expr,
        "-i",
        str(cfg.render_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-r",
        cfg.fps_expr,
        "-movflags",
        "+faststart",
        str(cfg.out),
    ]
    print(f"[encode] {cfg.fps_expr} fps ({float(cfg.fps):.3f}) -> {cfg.out}")
    subprocess.run(cmd, check=True)
