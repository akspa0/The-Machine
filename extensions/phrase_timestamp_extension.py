from __future__ import annotations

"""Phrase timestamp builder.

Copies each per-speaker segment WAV (already cut during speaker segmentation) into
`phrase_ts/<call>/<channel>/<speaker>/` and writes a JSON manifest with start/end
relative to original call and transcript text. Provides a consistent library of
whole phrases for later collage work.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import shutil
import sys as _sys

# ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from extensions.extension_base import ExtensionBase

class PhraseTimestampExtension(ExtensionBase):
    """Collect speaker phrases (segments) into a uniform library."""

    name = "phrase_timestamp"
    stage = "phrase_timestamp"
    description = "Copies speaker segment phrases into phrase_ts directory with manifest."

    def __init__(self, output_root: str | Path, *, min_dur: float = 0.4, max_dur: float = 10.0):
        super().__init__(output_root)
        self.output_root = Path(output_root)
        self.min_dur = min_dur
        self.max_dur = max_dur
        self.phrase_root = self.output_root / "phrase_ts"
        self.phrase_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    def run(self):
        speakers_dir = self.output_root / "speakers"
        if not speakers_dir.exists():
            self.log("speakers/ folder not found; run segmentation first.")
            return

        count = 0
        for call_dir in sorted(speakers_dir.iterdir()):
            if not call_dir.is_dir():
                continue
            call_id = call_dir.name
            for channel_dir in call_dir.iterdir():
                if not channel_dir.is_dir():
                    continue
                for speaker_dir in channel_dir.iterdir():
                    if not speaker_dir.is_dir():
                        continue
                    phrases: List[Dict[str, Any]] = []
                    phrase_out_dir = self.phrase_root / call_id / channel_dir.name / speaker_dir.name
                    phrase_out_dir.mkdir(parents=True, exist_ok=True)
                    for wav_file in sorted(speaker_dir.glob("*.wav")):
                        if wav_file.name.endswith("_16k.wav"):
                            continue  # skip 16k copies
                        parts = wav_file.stem.split("-")
                        if len(parts) < 3:
                            continue
                        index, start_h, end_h = parts[0], parts[1], parts[2]
                        start = int(start_h) / 100.0
                        end = int(end_h) / 100.0
                        dur = end - start
                        if dur < self.min_dur or dur > self.max_dur:
                            continue
                        # transcript txt if exists
                        txt_path = wav_file.with_suffix(".txt")
                        text = txt_path.read_text(encoding="utf-8").strip() if txt_path.exists() else ""
                        # copy wav
                        dst_wav = phrase_out_dir / wav_file.name
                        if not dst_wav.exists():
                            shutil.copy2(wav_file, dst_wav)
                        phrases.append({
                            "file": dst_wav.name,
                            "start": start,
                            "end": end,
                            "duration": dur,
                            "text": text,
                        })
                        count += 1
                    if phrases:
                        man_path = phrase_out_dir / "phrases.json"
                        with open(man_path, "w", encoding="utf-8") as f:
                            json.dump(phrases, f, indent=2, ensure_ascii=False)
        self.log(f"Collected {count} phrase WAVs into {self.phrase_root}.")

# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Build phrase_ts library from speakers folder.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder")
    p.add_argument("--min-dur", type=float, default=0.4, help="Minimum phrase duration seconds (default 0.4)")
    p.add_argument("--max-dur", type=float, default=10.0, help="Maximum phrase duration seconds (default 10)")
    p.add_argument("--device", help=argparse.SUPPRESS)
    p.add_argument("--model-size", help=argparse.SUPPRESS)
    p.add_argument("--sentences", help=argparse.SUPPRESS)
    args = p.parse_args()

    PhraseTimestampExtension(
        args.run_folder,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    ).run()

if __name__ == "__main__":  # pragma: no cover
    main() 