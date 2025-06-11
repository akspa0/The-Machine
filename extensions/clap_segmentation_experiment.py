import os
import sys
import json
from pathlib import Path
# Ensure project root is in sys.path for extension_base import
project_root = Path(__file__).resolve().parent.parent
print(f"[DEBUG] sys.path before extension_base import: {sys.path}")
print(f"[DEBUG] project_root: {project_root}")
ext_base_path = project_root / 'extension_base.py'
print(f"[DEBUG] extension_base.py exists: {ext_base_path.exists()}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    from extensions.extension_base import ExtensionBase
except ModuleNotFoundError as e:
    print(f"[WARN] Normal import failed: {e}. Attempting importlib fallback.")
    import importlib.util
    spec = importlib.util.spec_from_file_location('extension_base', str(ext_base_path))
    extmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extmod)
    ExtensionBase = extmod.ExtensionBase
from typing import List, Optional
import torchaudio
# NOTE: CLAP model interactions are now handled via extensions.clap_utils
import numpy as np
import tempfile
import shutil
import re
import datetime
# If LLM chunking/tokenization is needed, import from llm_utils
# from llm_utils import split_into_chunks_advanced

# Third-party
import numpy as np
import tempfile
import shutil
import re
import datetime

# Project imports
import tempfile, shutil, re, datetime, json
import torchaudio
from extensions.clap_utils import detect_clap_events
from audio_separation import separate_audio_file
from extensions.clap_whisper_detector import detect_clap_events_wb

class ClapSegmentationExperiment(ExtensionBase):
    """
    CLAP Segmentation Extension
    -----------------------------------
    Runs CLAP-based segmentation on either a batch of files (outputs/run-*) or a single audio file.
    Outputs results to clap_experiments/ and segmented_calls/.
    Creates or updates clap_segments annotation files.
    Usage:
        python clap_segmentation_experiment.py <output_root> [--audio-file path/to/audio.wav] [--confidence 0.6]
    """
    def __init__(
        self,
        output_root,
        audio_file: Optional[str] = None,
        confidence: float = 0.15,
        chunk_length_sec: int = 10,
        overlap_sec: int = 0,
        config_path: Optional[str] = None,
        no_separation: bool = False,
        auto_calibrate: Optional[int] = None,
        similarity_metric: str = "sigmoid",
        nms_gap: float = 0.0,
        backend: str = "utils",
        pairing_override: Optional[str] = None,
        min_segment_length: float | None = None,
    ):
        """Create segmentation experiment extension.

        Args:
            output_root: root directory of run (outputs/run-*)
            audio_file: optional single audio file to process
            confidence: default fallback threshold (overridden by config)
            chunk_length_sec: detection chunk length (overridden by config)
            overlap_sec: detection overlap length (overridden by config)
            config_path: JSON workflow file (defaults to workflows/clap_segmentation.json)
            no_separation: skip vocal/instrumental separation and run CLAP directly on input
            auto_calibrate: seconds for auto-threshold calibration (uses mean+3σ)
            similarity_metric: similarity metric to threshold on (sigmoid probability or raw cosine)
            nms_gap: temporal NMS gap in seconds (0 disables)
            backend: backend to use for CLAP detection
            pairing_override: pairing mode override (e.g., raw_events)
            min_segment_length: minimum segment length in seconds (override config)
        """
        super().__init__(output_root)
        self.audio_file = audio_file
        # Load segmentation workflow config if present
        cfg_path = (
            Path(config_path)
            if config_path
            else Path(__file__).resolve().parent.parent / "workflows" / "clap_segmentation.json"
        )
        if cfg_path.exists():
            import json

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_json = json.load(f)
            # Accept root-level or nested key
            self.seg_config = cfg_json.get("clap_segmentation", cfg_json)
        else:
            self.seg_config = {}

        # Allow pairing override
        if pairing_override is not None:
            self.seg_config["pairing"] = pairing_override

        # Min-segment-length override or default for raw_events
        if min_segment_length is not None:
            self.seg_config["min_segment_length_sec"] = float(min_segment_length)
        elif self.seg_config.get("pairing") == "raw_events" and "min_segment_length_sec" not in self.seg_config:
            self.seg_config["min_segment_length_sec"] = 2.0

        # Apply config overrides
        self.confidence = self.seg_config.get("confidence_threshold", confidence)
        self.chunk_length_sec = self.seg_config.get("chunk_length_sec", chunk_length_sec)
        self.overlap_sec = self.seg_config.get("overlap_sec", overlap_sec)
        self.prompts = self.seg_config.get(
            "prompts",
            [
                "telephone ring tones",
                "hang-up tones",
            ],
        )

        self.model_id = "laion/clap-htsat-fused"

        self.no_separation = no_separation

        self.auto_calibrate = auto_calibrate
        self.similarity_metric = self.seg_config.get("similarity_metric", similarity_metric)
        self.nms_gap = self.seg_config.get("nms_gap_sec", nms_gap)
        self.backend = self.seg_config.get("backend", backend)

        # Output dirs
        self.experiments_dir = self.output_root / "clap_experiments"
        self.segments_dir = self.experiments_dir / "segmented_calls"
        self.experiments_dir.mkdir(exist_ok=True, parents=True)
        self.segments_dir.mkdir(exist_ok=True, parents=True)

    def run(self):
        if self.audio_file:
            self.log(f"Running CLAP segmentation on single file: {self.audio_file}")
            self.process_file(Path(self.audio_file))
        else:
            self.log(f"Running CLAP segmentation in batch mode for: {self.output_root}")
            audio_dir = self.output_root / 'finalized' / 'calls'
            if not audio_dir.exists():
                self.log(f"No finalized/calls directory found in {self.output_root}")
                return
            for audio_file in audio_dir.glob('*.wav'):
                self.process_file(audio_file)

    def process_file(self, audio_path: Path):
        # Multi-track CLAP segmentation: process both vocal and instrumental tracks if available
        seg_file = self.experiments_dir / f"{audio_path.stem}_clap_segments.json"
        events = []
        # Prepare temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            stems = {}
            if not self.no_separation:
                # --- Attempt separation first ---
                pattern = r'\d{4}-(left|right|out)-[\d-]+\.wav'
                if not re.match(pattern, audio_path.name):
                    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                    temp_input = temp_dir_path / f'0000-out-{timestamp}.wav'
                    # Convert to wav if needed
                    if audio_path.suffix.lower() != '.wav':
                        import soundfile as sf
                        data, sr = sf.read(str(audio_path))
                        sf.write(str(temp_input), data, sr)
                    else:
                        shutil.copy2(audio_path, temp_input)
                    input_for_sep = temp_input
                else:
                    input_for_sep = audio_path

                model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
                sep_result = separate_audio_file(input_for_sep, temp_dir_path, model_path)
                if sep_result.get('separation_status') == 'success':
                    stems = {s['stem_type']: Path(s['output_path']) for s in sep_result.get('output_stems', [])}
                else:
                    self.log(f"[SEPARATION ERROR] {sep_result.get('stderr', '')}. Falling back to original audio for CLAP.")

            # --- Fallback: if no stems or self.no_separation, create stems dict with 'fallback' key ---
            if not stems:
                stems = {'fallback': audio_path}

            # Run CLAP on vocals
            if 'vocals' in stems and stems['vocals'].exists():
                self.log(f"[SEPARATION] Running CLAP on vocals: {stems['vocals']}")
                if self.backend == "utils":
                    vocal_events = detect_clap_events(
                        stems['vocals'],
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        overlap_sec=self.overlap_sec,
                        confidence_threshold=self.confidence,
                        auto_calibrate_seconds=self.auto_calibrate,
                        similarity_metric=self.similarity_metric,
                        nms_gap_sec=self.nms_gap,
                    )
                else:
                    vocal_events = detect_clap_events_wb(
                        stems['vocals'],
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        threshold=self.confidence,
                        nms_gap_sec=self.nms_gap,
                    )
                for e in vocal_events:
                    e['track'] = 'vocal'
                events.extend(vocal_events)
            else:
                self.log(f"[SEPARATION] No vocals track found for {audio_path.name}")
            # Run CLAP on instrumental
            if 'instrumental' in stems and stems['instrumental'].exists():
                self.log(f"[SEPARATION] Running CLAP on instrumental: {stems['instrumental']}")
                if self.backend == "utils":
                    instr_events = detect_clap_events(
                        stems['instrumental'],
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        overlap_sec=self.overlap_sec,
                        confidence_threshold=self.confidence,
                        auto_calibrate_seconds=self.auto_calibrate,
                        similarity_metric=self.similarity_metric,
                        nms_gap_sec=self.nms_gap,
                    )
                else:
                    instr_events = detect_clap_events_wb(
                        stems['instrumental'],
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        threshold=self.confidence,
                        nms_gap_sec=self.nms_gap,
                    )
                for e in instr_events:
                    e['track'] = 'instrumental'
                events.extend(instr_events)
            else:
                self.log(f"[SEPARATION] No instrumental track found for {audio_path.name}")

            # If fallback key present, run directly on original / temp wav
            if 'fallback' in stems and Path(stems['fallback']).exists():
                fb_path = Path(stems['fallback'])
                # Ensure WAV format for torchaudio reliability
                if fb_path.suffix.lower() != '.wav':
                    conv_path = temp_dir_path / (fb_path.stem + '_conv.wav')
                    self.log(f"[CONVERT] ffmpeg {fb_path.name} -> {conv_path.name}")
                    import subprocess, shlex
                    cmd = [
                        'ffmpeg', '-y', '-i', str(fb_path),
                        '-ac', '1', '-ar', '48000', str(conv_path)
                    ]
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if res.returncode != 0:
                        self.log(f"[FFMPEG ERROR] {res.stderr.decode()[:200]}")
                    else:
                        fb_path = conv_path
                self.log(f"[CLAP] Running fallback CLAP on {fb_path}")
                if self.backend == "utils":
                    fb_events = detect_clap_events(
                        fb_path,
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        overlap_sec=self.overlap_sec,
                        confidence_threshold=self.confidence,
                        auto_calibrate_seconds=self.auto_calibrate,
                        similarity_metric=self.similarity_metric,
                        nms_gap_sec=self.nms_gap,
                    )
                else:
                    fb_events = detect_clap_events_wb(
                        fb_path,
                        self.prompts,
                        model_id=self.model_id,
                        chunk_length_sec=self.chunk_length_sec,
                        threshold=self.confidence,
                        nms_gap_sec=self.nms_gap,
                    )
                for e in fb_events:
                    e['track'] = 'fallback'
                events.extend(fb_events)

        # Save merged events; segment construction is handled by decode_clap_segments
        with open(seg_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2)
        self.log(f"Wrote {len(events)} CLAP events (multi-track, separated) to {seg_file}")

    def decode_clap_segments(self):
        """
        Robust event pairing for new event format, now using multi-track events:
        - For each event, check if any detected prompt matches start or end prompt lists.
        - Use event['start_time_s'] and event['end_time_s'] for segment boundaries.
        - Allow for multiple prompts per event (an event can be both start and end).
        - Use all events from both vocal and instrumental tracks.
        - Log when a segment is started/ended, and if segments are skipped (e.g., too short).
        - Diagnostics: print all unique prompt strings found in the events JSON.
        """
        min_segment_length = self.seg_config.get("min_segment_length_sec", 10.0)
        padding = self.seg_config.get("segment_padding_sec", 0.5)

        pairing_mode = self.seg_config.get("pairing", "contiguous")
        thr_dict = self.seg_config.get("thresholds", {
            "ring": 0.45,
            "speech": 0.35,
            "music": 0.4,
        })

        seg_files = list(self.experiments_dir.glob("*_clap_segments.json"))
        for seg_file in seg_files:
            with open(seg_file, 'r', encoding='utf-8') as f:
                events = json.load(f)
            if not events:
                self.log(f"No events in {seg_file.name}")
                continue
            # DIAGNOSTIC: Print all unique prompt strings and tracks found in the events JSON
            unique_prompts = set()
            unique_tracks = set()
            for event in events:
                unique_prompts.update(event.get('prompts', {}).keys())
                unique_tracks.add(event.get('track', 'unknown'))
            self.log(f"[DIAG] Unique prompts in events: {sorted(unique_prompts)}")
            self.log(f"[DIAG] Tracks in events: {sorted(unique_tracks)}")
            audio_stem = seg_file.stem.replace('_clap_segments', '')
            audio_candidates = list((self.output_root / 'finalized' / 'calls').glob(f'{audio_stem}.*'))
            if not audio_candidates:
                # Fallback: use original audio_file if provided and matches stem
                if self.audio_file and Path(self.audio_file).stem == audio_stem:
                    audio_candidates = [Path(self.audio_file)]
                else:
                    # Try temp converted wav in same folder as seg_file stem
                    alt_wav = Path(self.output_root) / f"{audio_stem}_conv.wav"
                    if alt_wav.exists():
                        audio_candidates = [alt_wav]
            if not audio_candidates:
                self.log(f"No audio file found for {audio_stem}")
                continue
            audio_path = audio_candidates[0]
            waveform, sr = torchaudio.load(str(audio_path))
            audio_len = waveform.shape[1] / sr
            # Pairing & segment extraction---------------------------------
            from extensions.clap_utils import (
                find_calls_cross_track_ring_speech_ring,
                merge_contiguous,
            )

            if pairing_mode == "cross_track_ring_speech_ring":
                pairs = find_calls_cross_track_ring_speech_ring(events, thr_dict)
            elif pairing_mode == "raw_events":
                from extensions.clap_utils import events_to_segments
                pairs = events_to_segments(
                    events,
                    padding,
                    self.seg_config.get("min_segment_length_sec", 2.0),
                    audio_len,
                )
            elif pairing_mode == "tone_alternating":
                from extensions.clap_utils import pair_alternating_prompt
                prompt = self.seg_config.get("tone_prompt", self.seg_config.get("start_prompt", "telephone hang-up tones"))
                pairs = pair_alternating_prompt(events, prompt)
                # padding/min_len handled below
            else:
                # fallback: contiguous
                pairs = merge_contiguous(events)

            segments: list = []
            for idx_pair, (seg_start, seg_end) in enumerate(pairs):
                seg_start_p = max(0.0, seg_start - padding)
                seg_end_p = min(audio_len, seg_end + padding)
                seg_len = seg_end_p - seg_start_p
                if seg_len < min_segment_length:
                    continue
                seg_wave = waveform[:, int(seg_start_p * sr): int(seg_end_p * sr)]
                seg_name = f"{audio_stem}-call_{idx_pair:04d}.wav"
                seg_path = self.segments_dir / seg_name
                torchaudio.save(str(seg_path), seg_wave, sr)
                segments.append({
                    'segment_index': idx_pair,
                    'start': seg_start_p,
                    'end': seg_end_p,
                    'output_path': str(seg_path),
                    'source_file': str(audio_path),
                    'start_event': None,
                    'end_event': None,
                })
            manifest_path = self.segments_dir / f"{audio_stem}_clap_segment_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            txt_path = self.segments_dir / f"{audio_stem}_clap_segments.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(f"Segment {seg['segment_index']}: {seg['start']:.2f}s - {seg['end']:.2f}s\n")
                    f.write(f"  Start event: {seg.get('start_event')}
")
                    f.write(f"  End event: {seg.get('end_event')}

")
            self.log(f"Decoded {len(segments)} segments for {audio_stem}, manifest: {manifest_path.name}, txt: {txt_path.name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLAP Segmentation Experiment Extension")
    parser.add_argument('output_root', type=str, help='Root output directory (outputs/run-*)')
    parser.add_argument('--audio-file', type=str, default=None, help='Optional: path to a single audio file to process')
    parser.add_argument('--confidence', type=float, default=0.15, help='Confidence threshold for CLAP events')
    parser.add_argument('--chunk-length', type=int, default=10, help='Chunk length in seconds (default: 10)')
    parser.add_argument('--overlap', type=float, default=0.0, help='Overlap seconds between chunks.')
    parser.add_argument('--config', type=str, default=None, help='Path to clap_segmentation workflow JSON')
    parser.add_argument('--decode-clap-segments', action='store_true', help='Decode clap_segments into paired segments and output manifest/txt')
    parser.add_argument('--no-separation', action='store_true', help='Skip vocal/instrumental separation and run CLAP directly on input')
    parser.add_argument('--auto-calibrate', type=int, default=None, help='Seconds for auto-threshold calibration (uses mean+3σ)')
    parser.add_argument('--similarity', type=str, choices=['sigmoid', 'cosine'], default='sigmoid', help='Similarity metric to threshold on (sigmoid probability or raw cosine).')
    parser.add_argument('--nms-gap', type=float, default=0.0, help='Temporal NMS gap in seconds (0 disables).')
    parser.add_argument('--backend', type=str, choices=['utils', 'whisper'], default='utils', help='Backend to use for CLAP detection')
    parser.add_argument('--pairing', type=str, default=None, help='Pairing mode override (e.g., raw_events)')
    parser.add_argument('--min-segment-length', type=float, default=None, help='Minimum segment length seconds (override config)')
    args = parser.parse_args()
    ext = ClapSegmentationExperiment(
        args.output_root,
        audio_file=args.audio_file,
        confidence=args.confidence,
        chunk_length_sec=args.chunk_length,
        overlap_sec=args.overlap,
        config_path=args.config,
        no_separation=args.no_separation,
        auto_calibrate=args.auto_calibrate,
        similarity_metric=args.similarity,
        nms_gap=args.nms_gap,
        backend=args.backend,
        pairing_override=args.pairing,
        min_segment_length=args.min_segment_length,
    )
    if args.decode_clap_segments:
        ext.decode_clap_segments()
    else:
        ext.run()

if __name__ == "__main__":
    main() 