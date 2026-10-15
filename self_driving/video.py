"""Probing and guarding the source video.

Timestamps are parsed by shape ("23:00" is MM:SS, "0:23:00" is HH:MM:SS), which
is ambiguous if the two ends of a window use different shapes. Rather than guess,
validate_window checks the requested window against the real duration of the
file and refuses to run, so a typo costs one second instead of an hour of
pointless extraction.
"""

from __future__ import annotations

import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .config import Config


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
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def probe_fps(path: Path) -> Fraction:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
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


def validate_window(cfg: Config, total: int) -> None:
    """Refuse to start unless the requested window exists in the source."""
    source = probe_duration(cfg.video)
    print(f"[window] source is {format_timestamp(source)} long, {cfg.fps} fps")
    print(
        f"[window] {format_timestamp(cfg.start)} -> {format_timestamp(cfg.start + cfg.duration)} "
        f"({cfg.duration:.1f} s, {total} frames at {float(cfg.fps):.3f} fps)"
    )

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
        raise SystemExit(
            f"error: {cfg.duration / 60:.0f} minutes of video in one run looks wrong; "
            f"pass a shorter --start/--end window."
        )
    if total > 20_000:
        raise SystemExit(
            f"error: {total} frames in one run looks wrong; pass a shorter window or a lower --fps."
        )


def check_disk(cfg: Config, total: int) -> None:
    """A frame costs roughly 350 KB on disk and a rendered frame roughly 450 KB."""
    needed = total * (350_000 + 450_000) * 1.1
    free = shutil.disk_usage(existing_ancestor(cfg.workdir)).free
    if free < needed:
        raise SystemExit(
            f"error: need about {needed / 1e9:.1f} GB for {total} frames + rendered frames, "
            f"only {free / 1e9:.1f} GB free in {cfg.workdir.parent}."
        )
    print(f"[disk] about {needed / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")


def existing_ancestor(path: Path) -> Path:
    """The closest directory that exists: work/ is created during a run and is
    not in git, so a fresh clone has no work directory at all."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
