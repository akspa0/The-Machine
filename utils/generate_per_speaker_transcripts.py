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
import logging

def extract_ts(f):
    parts = f.stem.split('-')
    if len(parts) > 1 and parts[1].isdigit():
        return int(parts[1])
    return 0

def generate_per_speaker_transcripts(speakers_root, force=False):
    speakers_root = Path(speakers_root)
    if not speakers_root.exists():
        print(f"Speakers root {speakers_root} does not exist.")
        return
    for call_folder in speakers_root.iterdir():
        if not call_folder.is_dir():
            continue
        call_id = call_folder.name
        for channel_folder in call_folder.iterdir():
            if not channel_folder.is_dir():
                continue
            channel = channel_folder.name
            for speaker_folder in channel_folder.iterdir():
                if not speaker_folder.is_dir() or not speaker_folder.name.startswith('S'):
                    continue
                speaker_id = speaker_folder.name
                transcript_path = speaker_folder / 'speaker_transcript.txt'
                if transcript_path.exists():
                    if not force:
                        print(f"[SKIP] {transcript_path} already exists.")
                        continue
                    else:
                        print(f"[FORCE] Overwriting {transcript_path}.")
                # Gather all .txt files for this speaker only (never mix channels or calls)
                txt_files = sorted(speaker_folder.glob('*.txt'))
                lines = []
                for txt_file in txt_files:
                    content = txt_file.read_text(encoding='utf-8').strip()
                    if content:
                        lines.append(content)
                if len(lines) < 5:
                    print(f"[SKIP] {speaker_folder} has only {len(lines)} non-empty lines (<5), skipping transcript generation.")
                    continue
                transcript = '\n'.join(lines)
                transcript_path.write_text(transcript, encoding='utf-8')
                print(f"[OK] Wrote {len(lines)} lines to {transcript_path}")

def main():
    parser = argparse.ArgumentParser(description="Backfill per-speaker transcripts in speakers/ folders.")
    parser.add_argument('run_folder', type=str, help='Path to outputs/run-*/speakers/')
    parser.add_argument('--force', action='store_true', help='Overwrite existing speaker_transcript.txt files')
    args = parser.parse_args()
    speakers_dir = Path(args.run_folder) / 'speakers'
    if not speakers_dir.exists():
        print(f"[ERROR] speakers/ directory not found in {args.run_folder}")
        sys.exit(1)
    generate_per_speaker_transcripts(speakers_dir, force=args.force)

if __name__ == '__main__':
    main() 