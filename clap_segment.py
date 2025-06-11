#!/usr/bin/env python
"""clap_segment.py

Minimal one-command call segmentation:
1. (Optional) Vocal separation into stems using existing pipeline util.
2. Run two CLAP passes – one on vocal stem, one on instrumental stem.
3. Segment original input based on alternating detections between instrumental and vocal events.
4. Write WAV segments and a PII-free manifest.

The script keeps <250 LOC to respect workspace rules.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

import torchaudio

from extensions.clap_utils import detect_clap_events, pair_alternating_sets, find_calls_cross_track_ring_speech_ring, merge_contiguous
from audio_separation import separate_audio_file

# ---------------------------------------------------------------------------
#  Prompt sets
# ---------------------------------------------------------------------------

VOCAL_PROMPTS: List[str] = [
    "human speech",
    "speech",
    "conversation",
]

INSTR_PROMPTS: List[str] = [
    "music",
    "telephone ring",
    "telephone ringing",
    "telephone hang up",
    "telephone hang-up tones",
]


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _audio_duration_sec(path: Path) -> float:
    info = torchaudio.info(str(path))
    return info.num_frames / info.sample_rate


def _extract_segments(original: Path, seg_pairs: List[Tuple[float, float]], out_dir: Path) -> List[Dict]:
    """Cut *original* into segments given by *seg_pairs* and write WAVs.

    Returns list of dicts with index, start, end, and output path (string).
    """
    waveform, sr = torchaudio.load(str(original))
    meta = []
    for idx, (s, e) in enumerate(seg_pairs):
        start_frame = int(s * sr)
        end_frame = int(e * sr)
        if end_frame <= start_frame:
            continue
        segment_wave = waveform[:, start_frame:end_frame]
        if segment_wave.shape[1] == 0:
            continue
        seg_name = f"{idx:04d}_segment.wav"
        seg_path = out_dir / seg_name
        torchaudio.save(str(seg_path), segment_wave, sr)
        meta.append({
            "index": idx,
            "start": round(s, 3),
            "end": round(e, 3),
            "output_path": str(seg_path),
        })
    return meta


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAP-based call segmentation (one-shot).")
    parser.add_argument("input_file", type=str, help="Path to input audio file (wav/mp3/etc.)")
    parser.add_argument("--outdir", type=str, default=None, help="Directory for output (default: <input>_segments)")
    parser.add_argument("--no-separation", action="store_true", help="Skip vocal/instrumental separation step")
    parser.add_argument("--threshold", type=float, default=0.54, help="Confidence threshold (default 0.54)")
    parser.add_argument("--instrumental-threshold", type=float, default=0.54, help="Threshold for instrumental prompts")
    parser.add_argument("--chunk-length", type=int, default=5, help="Detection chunk length seconds (default: 5)")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap seconds between chunks")

    args = parser.parse_args()

    inp_path = Path(args.input_file).expanduser().resolve()
    if not inp_path.is_file():
        raise FileNotFoundError(inp_path)

    if args.outdir:
        out_root = Path(args.outdir)
    else:
        base_dir = Path("CLAP_jobs")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hint = inp_path.stem[:24]
        out_root = base_dir / f"{timestamp}_{hint}"
    out_root = out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Vocal separation (optional)
    # ---------------------------------------------------------------------
    stems_dir = out_root / "stems"
    stems_dir.mkdir(exist_ok=True)

    vocals_path: Path | None = None
    instr_path: Path | None = None
    separation_meta: Dict = {}

    if not args.no_separation:
        sep_res = separate_audio_file(inp_path, stems_dir, "mel_band_roformer_vocals_fv4_gabox.ckpt")
        separation_meta = sep_res
        for stem in sep_res.get("output_stems", []):
            if stem["stem_type"] == "vocals":
                vocals_path = Path(stem["output_path"])
            elif stem["stem_type"] == "instrumental":
                instr_path = Path(stem["output_path"])

    # Fallbacks if separation skipped or failed
    if not vocals_path:
        vocals_path = inp_path
    if not instr_path:
        instr_path = inp_path

    # ---------------------------------------------------------------------
    # 2. CLAP detection passes
    # ---------------------------------------------------------------------
    events_v = detect_clap_events(
        vocals_path,
        VOCAL_PROMPTS,
        confidence_threshold=args.threshold,
        chunk_length_sec=args.chunk_length,
        overlap_sec=args.overlap,
        nms_gap_sec=0.0,
    )
    for ev in events_v:
        ev["track"] = "vocal"

    # Instrumental detection for contextual boundaries
    events_i = detect_clap_events(
        instr_path,
        INSTR_PROMPTS,
        confidence_threshold=args.instrumental_threshold,
        chunk_length_sec=args.chunk_length,
        overlap_sec=args.overlap,
        nms_gap_sec=2.0,
    )
    for ev in events_i:
        ev["track"] = "instrumental"

    # Save raw event logs for traceability
    with open(out_root / "events_speech.json", "w", encoding="utf-8") as f:
        json.dump(events_v, f, indent=2)
    with open(out_root / "events_instrumental.json", "w", encoding="utf-8") as f:
        json.dump(events_i, f, indent=2)

    # ---------------------------------------------------------------------
    # 3. Segment derivation (speech activity)
    # ---------------------------------------------------------------------
    raw_pairs = merge_contiguous(events_v, max_gap_sec=args.chunk_length)
    seg_pairs = [pair for pair in raw_pairs if pair[1] - pair[0] >= args.chunk_length]

    if not seg_pairs:
        dur = _audio_duration_sec(inp_path)
        seg_pairs = [(0.0, dur)]

    # ---------------------------------------------------------------------
    # 4. WAV segment extraction & manifest
    # ---------------------------------------------------------------------
    seg_meta = _extract_segments(inp_path, seg_pairs, out_root)

    manifest = {
        "audio_uuid": uuid.uuid4().hex,
        "input_basename": inp_path.stem,
        "clap_threshold": args.threshold,
        "instrumental_threshold": args.instrumental_threshold,
        "separation_status": separation_meta.get("separation_status", "skipped" if args.no_separation else "failed"),
        "num_events_vocal": len(events_v),
        "num_events_instrumental": len(events_i),
        "events_speech_log": "events_speech.json",
        "events_instrumental_log": "events_instrumental.json",
        "segments": seg_meta,
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] {len(seg_meta)} segments written to {out_root}")


if __name__ == "__main__":
    main() 