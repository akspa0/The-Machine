"""tm-auto-editor
Standalone wrapper for Auto-Editor so The-Machine can quickly remove long silences
and other unwanted parts from an already–anonymised audio file.

Usage (example):
    python tools/auto_editor_cli.py input.wav \
        --margin 0.2sec --silent-speed 99999 --video-speed 1

This script lives outside the core pipeline so it can be called as an
independent tool. It assumes that a **local** editable install of Auto-Editor
exists in `external_apps/auto_editor` (see README). If Auto-Editor is missing
or FFmpeg is not available, the script exits with a clear message.

All logging/output is privacy-safe: reports only include anonymised filenames
and parameters.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Third-party dependency; we only import if available to check durations.
# soundfile is already in project requirements.
import soundfile as sf  # type: ignore

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _find_run_folder(path: Path) -> Path:
    """Find nearest ancestor whose name starts with 'run-' (case-sensitive).

    If none is found, return the input's parent directory.
    """
    for parent in [path.parent] + list(path.parents):
        if parent.name.startswith("run-"):
            return parent
    return path.parent


def _measure_duration(path: Path) -> float:
    """Return audio duration in seconds using soundfile.info."""
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return -1.0  # Unknown / failed


def _build_auto_editor_cmd(
    input_path: Path,
    output_path: Path,
    params: Dict[str, Any],
) -> List[str]:
    """Return a list representing the auto-editor CLI command."""

    cmd: List[str] = [
        sys.executable,
        "-m",
        "auto_editor",
        str(input_path),
        "--margin",
        params["margin"],
        "--silent-speed",
        str(params["silent_speed"]),
        "--video-speed",
        str(params["video_speed"]),
        "--edit",
        f"audio:threshold={params['threshold']}",
        "--audio-codec",
        params["audio_codec"],
        "--output",
        str(output_path),
    ]

    return cmd


# ---------------------------------------------------------------------------
# Main wrapper logic
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tm-auto-editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Stand-alone wrapper around Auto-Editor for The-Machine.
            Example:
                tm-auto-editor input.wav --margin 0.2sec --silent-speed 99999
            """
        ),
    )

    parser.add_argument("input", type=Path, help="Input WAV/MP3 file to edit.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for edited file. Default: <stem>_ae.<ext> in same folder.",
    )
    parser.add_argument("--margin", default="0.2sec", help="Auto-Editor --margin value.")
    parser.add_argument(
        "--silent-speed",
        default="99999",
        help="Speed for silent sections (effectively cut).",
    )
    parser.add_argument(
        "--video-speed", default="1", help="Speed for kept sections (audio-only mode)."
    )
    parser.add_argument(
        "--threshold",
        default="0.04",
        help="Audio loudness threshold for detecting silence (Auto-Editor edit arg).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path. Default: run-folder/<stem>_ae.json",
    )
    parser.add_argument(
        "--audio-codec",
        default=None,
        help=(
            "Audio codec to pass through to Auto-Editor (maps to --audio-codec). "
            "Defaults to 'pcm_s16le' when exporting WAV, or 'libmp3lame' when "
            "exporting MP3."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose wrapper logging (does not pass to auto-editor).")

    return parser.parse_args(argv)



def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    input_path: Path = args.input.resolve()
    if not input_path.exists():
        sys.exit(f"[tm-auto-editor] Input file not found: {input_path}")

    # Determine default extension based on input if not explicitly given.
    default_ext = ".wav" if input_path.suffix.lower() in {".wav", ".wave"} else ".mp3"

    # Determine output path
    output_path: Path
    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = input_path.with_name(input_path.stem + "_ae" + default_ext)

    # Decide audio codec (can be overridden by --audio-codec)
    inferred_codec = "pcm_s16le" if output_path.suffix.lower() == ".wav" else "libmp3lame"
    audio_codec = args.audio_codec or inferred_codec

    # Determine report path
    run_folder = _find_run_folder(input_path)
    if args.report:
        report_path = args.report.resolve()
    else:
        report_path = run_folder / (output_path.stem + ".json")

    params: Dict[str, Any] = {
        "margin": args.margin,
        "silent_speed": args.silent_speed,
        "video_speed": args.video_speed,
        "threshold": args.threshold,
        "audio_codec": audio_codec,
        "verbose": args.verbose,
    }

    # Build and run command
    cmd = _build_auto_editor_cmd(input_path, output_path, params)
    if args.verbose:
        print("[tm-auto-editor] Running:", " ".join(map(str, cmd)), file=sys.stderr)

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            "Auto-Editor is not installed. Please clone external_apps/auto_editor "
            "and run 'pip install -e external_apps/auto_editor'."
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Auto-Editor failed with exit code {exc.returncode}.")

    # Build JSON report
    report: Dict[str, Any] = {
        "input": input_path.name,
        "output": output_path.name,
        "tool": "auto-editor",
        "params": params,
        "duration_before": _measure_duration(input_path),
        "duration_after": _measure_duration(output_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    if args.verbose:
        print(f"[tm-auto-editor] Wrote report: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main() 