#!/usr/bin/env python
"""cut_and_process.py

Usage
-----
python cut_and_process.py <audio_file> \
    [--run-name myrun] [--confidence 0.45] [--chunk-length 3] [--overlap 2] \
    [--backend whisper] [--nms-gap 1] [--asr_engine parakeet] [--call-tones]

This convenience wrapper performs two steps:
1. Runs extensions/clap_segmentation_experiment.py to cut the *audio_file* into
   individual call WAVs using CLAP. The cuts are written below
       <run_folder>/clap_experiments/segmented_calls/
2. Feeds that folder to pipeline_orchestrator.py in batch-processing mode so
   every cut call is processed through the full Machine pipeline.

The script simply shells out to the existing entry-points so it stays
maintenance-free.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR


def run(cmd: list[str]):
    """Run *cmd* synchronously, forwarding stdout/stderr. Abort on failure."""
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    p = argparse.ArgumentParser(description="Cut long recording into calls with CLAP, then process each call.")
    p.add_argument("audio_file", help="Path to the long recording to cut")
    p.add_argument("--run-name", help="Optional run folder name (default run-YYYYmmdd-HHMMSS)")
    # CLAP cutter knobs
    p.add_argument("--confidence", type=float, default=None, help="Cosine similarity threshold")
    p.add_argument("--chunk-length", type=int, default=None, help="Chunk length seconds for detection")
    p.add_argument("--overlap", type=int, default=None, help="Overlap seconds between chunks")
    p.add_argument("--backend", choices=["utils", "whisper"], default="whisper", help="Detector backend to use")
    p.add_argument("--nms-gap", type=float, default=None, help="Temporal NMS seconds")
    # Orchestrator knobs
    p.add_argument("--asr_engine", choices=["parakeet", "whisper"], default="parakeet", help="ASR engine (default: parakeet)")
    p.add_argument("--call-tones", action="store_true", help="Insert tones between calls in final show")
    args = p.parse_args()

    audio_path = Path(args.audio_file).expanduser().resolve()
    if not audio_path.exists():
        sys.exit(f"Audio file not found: {audio_path}")

    run_ts = args.run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_folder = Path("outputs") / run_ts
    run_folder.mkdir(parents=True, exist_ok=True)

    # ------------------ Step 1: CLAP cutter ------------------
    segments_dir = run_folder / "segments"
    cutter_cmd = [sys.executable, str(PROJECT_ROOT / "clap_segment.py"),
                  str(audio_path),
                  "--outdir", str(segments_dir)]
    if args.confidence is not None:
        cutter_cmd += ["--threshold", str(args.confidence)]
    if args.backend == "utils":
        # new cutter only supports utils backend; whisper kept for backward compatibility
        pass  # placeholder to maintain arg structure

    run(cutter_cmd)

    seg_dir = segments_dir
    if not seg_dir.exists() or not any(seg_dir.glob("*.wav")):
        sys.exit("No call segments found – aborting pipeline run.")

    # ------------------ Step 2: Pipeline orchestrator --------
    orch_cmd = [sys.executable, str(PROJECT_ROOT / "pipeline_orchestrator.py"),
                str(seg_dir),
                "--output-folder", str(run_folder),
                "--mode", "calls",
                "--asr_engine", args.asr_engine]
    if args.call_tones:
        orch_cmd.append("--call-tones")

    run(orch_cmd)

    print(f"✅ Completed. Outputs in {run_folder}")


if __name__ == "__main__":
    main() 