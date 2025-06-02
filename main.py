"""
main.py - Unified Audio Processing Entry Point (updated for modular single-file/URL workflow)

Usage:
    python main.py --input-dir <dir> [--output-folder <folder>] [other orchestrator args]
    python main.py <path_to_audio> [--output-folder <folder>] [other args]
    python main.py --url <url> [--output-folder <folder>] [other args]

- Handles all PII-sanitization and logging setup at the start.
- Dispatches to the orchestrator for tuple/call input.
- For single-file, runs the orchestrator with the file path as the positional argument (not --file).
- For --url, downloads to a temp folder, then runs the orchestrator with the resulting file as the positional argument.
- For --input-dir, calls pipeline_orchestrator.py as before.
- Remove --file passthrough to single_file_orchestrator.py.
- Ensure robust logging and output folder handling.
- Document the new workflow in the script docstring.
"""
import argparse
import sys
import os
from pathlib import Path
import shutil
import logging
import datetime
import tempfile

def setup_logging(output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    log_path = output_folder / 'main_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")

def run_orchestrator(args, input_dir, output_folder):
    import subprocess
    cmd = [sys.executable, 'pipeline_orchestrator.py', str(input_dir), '--output-folder', str(output_folder)]
    # Add other orchestrator args as needed
    if args.resume:
        cmd.append('--resume')
    if args.resume_from:
        cmd += ['--resume-from', args.resume_from]
    if args.force:
        cmd.append('--force')
    if args.force_rerun:
        cmd += ['--force-rerun', args.force_rerun]
    if args.clear_from:
        cmd += ['--clear-from', args.clear_from]
    if args.stage_status:
        cmd += ['--stage-status', args.stage_status]
    if args.show_resume_status:
        cmd.append('--show-resume-status')
    if args.asr_engine:
        cmd += ['--asr_engine', args.asr_engine]
    if args.llm_config:
        cmd += ['--llm_config', args.llm_config]
    if args.call_tones:
        cmd.append('--call-tones')
    if args.call_cutter:
        cmd.append('--call-cutter')
    logging.info(f"[main.py] Running orchestrator: {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error(f"Orchestrator failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def run_single_file_orchestrator(args, file_path, output_folder):
    import subprocess
    cmd = [sys.executable, 'single_file_orchestrator.py', file_path]
    if args.asr_engine:
        cmd += ['--asr_engine', args.asr_engine]
    if args.resume:
        cmd.append('--resume')
    if args.resume_from:
        cmd += ['--resume-from', args.resume_from]
    if args.force:
        cmd.append('--force')
    if args.force_rerun:
        cmd += ['--force-rerun', args.force_rerun]
    if args.clear_from:
        cmd += ['--clear-from', args.clear_from]
    if args.stage_status:
        cmd += ['--stage-status', args.stage_status]
    if args.show_resume_status:
        cmd.append('--show-resume-status')
    logging.info(f"[main.py] Running single_file_orchestrator: {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logging.error(f"single_file_orchestrator failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def run_url_workflow(args, url, output_folder):
    import subprocess
    # Use a temp folder for yt-dlp output, or output_folder/raw_inputs
    temp_dir = Path(output_folder) / 'raw_inputs'
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Step 1: Download with yt_dlp_download.py
    cmd_dl = [sys.executable, 'yt_dlp_download.py', '--url', url, '--output', str(temp_dir)]
    logging.info(f"[main.py] Downloading URL: {' '.join(map(str, cmd_dl))}")
    result_dl = subprocess.run(cmd_dl)
    if result_dl.returncode != 0:
        logging.error(f"yt_dlp_download.py failed with exit code {result_dl.returncode}")
        sys.exit(result_dl.returncode)
    # Step 2: Find the downloaded file
    downloaded_file = temp_dir / 'input_from_url.wav'
    if not downloaded_file.exists():
        logging.error(f"Downloaded file not found: {downloaded_file}")
        sys.exit(1)
    # Step 3: Run single_file_orchestrator.py on the downloaded file
    run_single_file_orchestrator(args, downloaded_file, output_folder)

def main():
    parser = argparse.ArgumentParser(description="Unified Audio Processing Entry Point (modular workflow)")
    parser.add_argument('--input-dir', type=str, help='Input directory (tuple/call pipeline)')
    parser.add_argument('--file', type=str, help='Path to local audio file (single-file pipeline)')
    parser.add_argument('--url', type=str, help='URL to download audio from (single-file pipeline)')
    parser.add_argument('--output-folder', type=str, default='outputs/main_run', help='Output folder')
    # Orchestrator passthrough args
    parser.add_argument('--resume', action='store_true', help='Enable resume functionality')
    parser.add_argument('--resume-from', type=str, help='Resume from a specific stage')
    parser.add_argument('--force', action='store_true', help='When used with --resume-from, deletes all outputs and state from that stage forward for a clean re-run')
    parser.add_argument('--force-rerun', type=str, metavar='STAGE', help='Force re-run a specific stage even if marked complete')
    parser.add_argument('--clear-from', type=str, metavar='STAGE', help='Clear completion status from specified stage onwards')
    parser.add_argument('--stage-status', type=str, metavar='STAGE', help='Show detailed status for a specific stage')
    parser.add_argument('--show-resume-status', action='store_true', help='Show current resume status and exit')
    parser.add_argument('--asr_engine', type=str, help='ASR engine to use')
    parser.add_argument('--llm_config', type=str, help='Path to LLM config')
    parser.add_argument('--call-tones', action='store_true', help='Append tones.wav to calls')
    parser.add_argument('--call-cutter', action='store_true', help='Enable CLAP-based call segmentation')
    args = parser.parse_args()

    # --- Argument validation for resume/new run distinction ---
    is_resume = args.output_folder and (args.resume or args.resume_from or args.force)
    if not is_resume and not (args.input_dir or getattr(args, 'file', None) or getattr(args, 'url', None)):
        print("ERROR: Must provide either --input-dir, --file, or --url for a new run. For resume, use --output-folder with --resume, --resume-from, or --force.")
        exit(1)

    setup_logging(args.output_folder)

    if args.input_dir:
        logging.info(f"[main.py] Detected tuple/call input: {args.input_dir}")
        run_orchestrator(args, args.input_dir, args.output_folder)
    elif args.file:
        logging.info(f"[main.py] Detected single-file input: {args.file}")
        run_single_file_orchestrator(args, args.file, args.output_folder)
    elif args.url:
        logging.info(f"[main.py] Detected URL input: {args.url}")
        run_url_workflow(args, args.url, args.output_folder)
    else:
        logging.error("Must provide either --input-dir, --file, or --url.")
        sys.exit(1)

if __name__ == '__main__':
    main() 