"""video_utils.py – thin wrapper around FFmpeg for cutting video segments.

Designed for internal use by `extensions.video_bites_extension`.

All functions are *pure* (raise on failure) and never print PII.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Tuple

__all__ = [
    "cut_video_segment",
]


FFMPEG_BIN = "ffmpeg"  # Assumes ffmpeg is on PATH


def _run(cmd: list[str]) -> None:
    """Run *cmd* (list) with stderr suppressed, raise on non-zero exit."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def cut_video_segment(
    src_video: str | Path,
    dst_video: str | Path,
    *,
    start: float,
    end: float,
    audio_only: bool = False,
    codec_copy: bool = True,
) -> None:
    """Cut the interval [start, end] from *src_video* into *dst_video*.

    Parameters
    ----------
    start / end : seconds (float)
    audio_only  : if True, strip video (`-vn`)
    codec_copy  : if True, use `-c copy` for video path to avoid re-encoding.
    """
    src_video = str(src_video)
    dst_video = str(dst_video)

    if end <= start:
        raise ValueError("end must be greater than start")

    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start}", "-to", f"{end}", "-i", src_video]

    if audio_only:
        cmd += ["-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", dst_video]
    else:
        if codec_copy:
            cmd += ["-c", "copy"]
        cmd.append(dst_video)

    _run(cmd) 