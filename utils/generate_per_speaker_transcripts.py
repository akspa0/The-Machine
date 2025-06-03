#!/usr/bin/env python3
"""
Utility to backfill per-speaker transcripts in outputs/run-*/speakers/.
For each <call_id>/<channel>/<speaker>/, concatenates all *.txt files in timestamp order into speaker_transcript.txt.
Usage:
    python utils/generate_per_speaker_transcripts.py <run_folder> [--force]
Example:
    python utils/generate_per_speaker_transcripts.py outputs/run-20250602-171604 --force
"""
import sys
import os
from pathlib import Path
import argparse

def extract_ts(f):
    parts = f.stem.split('-')
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return 0

def generate_transcripts(speakers_dir, force=False):
    speakers_dir = Path(speakers_dir)
    count = 0
    for call_id_dir in speakers_dir.iterdir():
        if not call_id_dir.is_dir():
            continue
        for channel_dir in call_id_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            for speaker_dir in channel_dir.iterdir():
                if not speaker_dir.is_dir():
                    continue
                out_path = speaker_dir / 'speaker_transcript.txt'
                if out_path.exists() and not force:
                    print(f"[SKIP] {out_path} already exists. Use --force to overwrite.")
                    continue
                txt_files = sorted(speaker_dir.glob('*.txt'))
                if not txt_files:
                    continue
                txt_files = sorted(txt_files, key=extract_ts)
                lines = []
                for txt_file in txt_files:
                    text = txt_file.read_text(encoding='utf-8').strip()
                    if text:
                        lines.append(text)
                if lines:
                    out_path.write_text('\n'.join(lines), encoding='utf-8')
                    print(f"[WRITE] {out_path} ({len(lines)} utterances)")
                    count += 1
    print(f"[DONE] Wrote {count} speaker_transcript.txt files.")

def main():
    parser = argparse.ArgumentParser(description="Backfill per-speaker transcripts in speakers/ folders.")
    parser.add_argument('run_folder', type=str, help='Path to outputs/run-*/speakers/')
    parser.add_argument('--force', action='store_true', help='Overwrite existing speaker_transcript.txt files')
    args = parser.parse_args()
    speakers_dir = Path(args.run_folder) / 'speakers'
    if not speakers_dir.exists():
        print(f"[ERROR] speakers/ directory not found in {args.run_folder}")
        sys.exit(1)
    generate_transcripts(speakers_dir, force=args.force)

if __name__ == '__main__':
    main() 