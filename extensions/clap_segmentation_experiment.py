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
from transformers import ClapProcessor, ClapModel
import torch
import numpy as np
import tempfile
from audio_separation import separate_audio_file
import shutil
import re
import datetime

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
    def __init__(self, output_root, audio_file: Optional[str] = None, confidence: float = 0.6, chunk_length_sec: int = 10):
        super().__init__(output_root)
        self.audio_file = audio_file
        self.confidence = confidence
        self.chunk_length_sec = chunk_length_sec  # Default 10s, tunable via CLI
        self.experiments_dir = self.output_root / 'clap_experiments'
        self.segments_dir = self.experiments_dir / 'segmented_calls'
        self.experiments_dir.mkdir(exist_ok=True, parents=True)
        self.segments_dir.mkdir(exist_ok=True, parents=True)
        self.model_id = "laion/clap-htsat-fused"
        self.prompts = [
            "telephone ring tones",
            "hang-up tones",
            # Add more prompts as needed
        ]

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
        # Prepare temp dir for separation outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            # Ensure input file matches expected pattern for separation
            pattern = r'\d{4}-(left|right|out)-[\d-]+\.wav'
            if not re.match(pattern, audio_path.name):
                # Create temp copy with compatible name: 0000-out-YYYYMMDD-HHMMSS.wav
                timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                temp_input = temp_dir_path / f'0000-out-{timestamp}.wav'
                # Convert to wav if needed
                if audio_path.suffix.lower() != '.wav':
                    import soundfile as sf
                    import numpy as np
                    data, sr = sf.read(str(audio_path))
                    sf.write(str(temp_input), data, sr)
                    self.log(f"[DIAG] Converted {audio_path} to {temp_input} (wav)")
                else:
                    shutil.copy2(audio_path, temp_input)
                input_for_sep = temp_input
            else:
                input_for_sep = audio_path
            # Use only the model filename, let audio_separation.py handle model resolution
            model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
            # Run separation using local audio_separation.py wrapper (calls audio-separator CLI)
            sep_result = separate_audio_file(input_for_sep, temp_dir_path, model_path)
            if sep_result.get('separation_status') != 'success':
                self.log(f"[SEPARATION ERROR] Return code: {sep_result.get('returncode')}, stderr: {sep_result.get('stderr')}")
            stems = {s['stem_type']: Path(s['output_path']) for s in sep_result.get('output_stems', [])}
            # Run CLAP on vocals
            if 'vocals' in stems and stems['vocals'].exists():
                self.log(f"[SEPARATION] Running CLAP on vocals: {stems['vocals']}")
                vocal_events = self.segment_audio_with_clap(stems['vocals'])
                for e in vocal_events:
                    e['track'] = 'vocal'
                events.extend(vocal_events)
            else:
                self.log(f"[SEPARATION] No vocals track found for {audio_path.name}")
            # Run CLAP on instrumental
            if 'instrumental' in stems and stems['instrumental'].exists():
                self.log(f"[SEPARATION] Running CLAP on instrumental: {stems['instrumental']}")
                instr_events = self.segment_audio_with_clap(stems['instrumental'])
                for e in instr_events:
                    e['track'] = 'instrumental'
                events.extend(instr_events)
            else:
                self.log(f"[SEPARATION] No instrumental track found for {audio_path.name}")
        # Save merged events; segment construction is handled by decode_clap_segments
        with open(seg_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2)
        self.log(f"Wrote {len(events)} CLAP events (multi-track, separated) to {seg_file}")

    def segment_audio_with_clap(self, audio_path: Path) -> List[dict]:
        """
        CLAP event detection with chunking. 10s chunks and 0.6 confidence work well for most call audio.
        """
        chunk_length_sec = self.chunk_length_sec  # Use instance variable
        overlap_sec = 0  # No overlap for now
        processor = ClapProcessor.from_pretrained(self.model_id)
        model = ClapModel.from_pretrained(self.model_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 48000:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sr, 48000)
            sr = 48000
        total_samples = waveform.shape[1]
        duration = total_samples / sr
        chunk_samples = int(chunk_length_sec * sr)
        n_chunks = int(np.ceil(total_samples / chunk_samples))
        # Precompute text features
        text_inputs = processor(text=self.prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        events = []
        for i in range(n_chunks):
            start = i * chunk_length_sec
            end = min((i + 1) * chunk_length_sec, duration)
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            chunk_audio = waveform[:, s_idx:e_idx]
            # Ensure mono and 1D
            if chunk_audio.ndim == 2 and chunk_audio.shape[0] > 1:
                chunk_audio = chunk_audio.mean(dim=0)
            chunk_audio = chunk_audio.squeeze().cpu().numpy()
            if chunk_audio.size == 0:
                continue
            audio_inputs = processor(audios=chunk_audio, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                audio_features = model.get_audio_features(**audio_inputs)
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                logits_per_audio = audio_features @ text_features.T
                probs = torch.sigmoid(logits_per_audio).cpu().detach().numpy()
            # Fix shape: should be (num_prompts,) or (1, num_prompts)
            if probs.ndim == 2 and probs.shape[0] == 1:
                probs = probs[0]
            elif probs.ndim == 0:
                probs = np.array([probs])
            elif probs.ndim != 1:
                self.log(f"[WARNING] Unexpected probs shape: {probs.shape} at chunk {i}")
                probs = probs.flatten()
            chunk_probabilities = {prompt: float(prob) for prompt, prob in zip(self.prompts, probs)}
            filtered_prompts = {p: prob for p, prob in chunk_probabilities.items() if prob >= self.confidence}
            if filtered_prompts:
                events.append({
                    "start_time_s": round(start, 3),
                    "end_time_s": round(end, 3),
                    "prompts": filtered_prompts
                })
        # Save events for downstream pairing
        return events

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
        min_segment_length = 2.0  # DEBUG: Lowered for diagnosis; restore to 12.0 after
        padding = 0.5
        start_prompts = [
            'telephone ringing', 'music', 'speech', 'laughter', 'telephone greeting', 'telephone noises', 'telephone ring tones'
        ]
        end_prompts = [
            'telephone hang-up tones', 'laughter', 'music', 'telephone noises', 'telephone hang-up noises', 'hang-up tone'
        ]
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
                self.log(f"No audio file found for {audio_stem}")
                continue
            audio_path = audio_candidates[0]
            waveform, sr = torchaudio.load(str(audio_path))
            audio_len = waveform.shape[1] / sr
            # Sort events by start_time_s for robust pairing
            events = sorted(events, key=lambda e: e['start_time_s'])
            segments = []
            in_segment = False
            seg_start = None
            seg_start_event = None
            for idx, event in enumerate(events):
                event_prompts = set(event.get('prompts', {}).keys())
                is_start = bool(event_prompts & set(start_prompts))
                is_end = bool(event_prompts & set(end_prompts))
                if is_start:
                    self.log(f"[DEBUG] Start event at idx {idx}, {event['start_time_s']:.2f}s, prompts: {event_prompts}, track: {event.get('track','?')}")
                if is_end:
                    self.log(f"[DEBUG] End event at idx {idx}, {event['end_time_s']:.2f}s, prompts: {event_prompts}, track: {event.get('track','?')}")
                if not in_segment and is_start:
                    seg_start = event['start_time_s']
                    seg_start_event = event
                    in_segment = True
                    self.log(f"[SEG] Start at idx {idx}, {seg_start:.2f}s, prompts: {event_prompts}, track: {event.get('track','?')}")
                if in_segment and is_end:
                    seg_end = event['end_time_s']
                    seg_end_event = event
                    seg_start_p = max(0.0, seg_start - padding)
                    seg_end_p = min(audio_len, seg_end + padding)
                    seg_len = seg_end_p - seg_start_p
                    self.log(f"[SEG] Attempt: start idx {idx}, {seg_start_p:.2f}s, end idx {idx}, {seg_end_p:.2f}s, len {seg_len:.2f}s")
                    if seg_len >= min_segment_length:
                        seg_wave = waveform[:, int(seg_start_p * sr):int(seg_end_p * sr)]
                        seg_name = f"{audio_stem}-call_{len(segments):04d}.wav"
                        seg_path = self.segments_dir / seg_name
                        torchaudio.save(str(seg_path), seg_wave, sr)
                        segments.append({
                            'segment_index': len(segments),
                            'start': seg_start_p,
                            'end': seg_end_p,
                            'output_path': str(seg_path),
                            'source_file': str(audio_path),
                            'start_event': seg_start_event,
                            'end_event': seg_end_event
                        })
                        self.log(f"[SEG] End at idx {idx}, {seg_end:.2f}s, prompts: {event_prompts}, track: {event.get('track','?')} -> Segment {len(segments)-1} saved. Length: {seg_len:.2f}s")
                    else:
                        self.log(f"[SEG] Skipped short segment: {seg_start_p:.2f}s - {seg_end_p:.2f}s (len {seg_len:.2f}s)")
                    in_segment = False
                    seg_start = None
                    seg_start_event = None
            if in_segment and seg_start is not None:
                seg_end = audio_len
                seg_start_p = max(0.0, seg_start - padding)
                seg_end_p = audio_len
                seg_len = seg_end_p - seg_start_p
                self.log(f"[SEG] Attempt: start idx END, {seg_start_p:.2f}s, end idx END, {seg_end_p:.2f}s, len {seg_len:.2f}s")
                if seg_len >= min_segment_length:
                    seg_wave = waveform[:, int(seg_start_p * sr):int(seg_end_p * sr)]
                    seg_name = f"{audio_stem}-call_{len(segments):04d}.wav"
                    seg_path = self.segments_dir / seg_name
                    torchaudio.save(str(seg_path), seg_wave, sr)
                    segments.append({
                        'segment_index': len(segments),
                        'start': seg_start_p,
                        'end': seg_end_p,
                        'output_path': str(seg_path),
                        'source_file': str(audio_path),
                        'start_event': seg_start_event,
                        'end_event': None
                    })
                    self.log(f"[SEG] Closed at end of audio: {seg_start_p:.2f}s - {seg_end_p:.2f}s -> Segment {len(segments)-1} saved. Length: {seg_len:.2f}s")
                else:
                    self.log(f"[SEG] Skipped short segment at end: {seg_start_p:.2f}s - {seg_end_p:.2f}s (len {seg_len:.2f}s)")
            manifest_path = self.segments_dir / f"{audio_stem}_clap_segment_manifest.json"
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(segments, f, indent=2)
            txt_path = self.segments_dir / f"{audio_stem}_clap_segments.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                for seg in segments:
                    f.write(f"Segment {seg['segment_index']}: {seg['start']:.2f}s - {seg['end']:.2f}s\n")
                    f.write(f"  Start event: {seg['start_event']}\n")
                    f.write(f"  End event: {seg['end_event']}\n\n")
            self.log(f"Decoded {len(segments)} segments for {audio_stem}, manifest: {manifest_path.name}, txt: {txt_path.name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CLAP Segmentation Experiment Extension")
    parser.add_argument('output_root', type=str, help='Root output directory (outputs/run-*)')
    parser.add_argument('--audio-file', type=str, default=None, help='Optional: path to a single audio file to process')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold for CLAP events')
    parser.add_argument('--chunk-length', type=int, default=10, help='Chunk length in seconds (default: 10)')
    parser.add_argument('--decode-clap-segments', action='store_true', help='Decode clap_segments into paired segments and output manifest/txt')
    args = parser.parse_args()
    ext = ClapSegmentationExperiment(args.output_root, audio_file=args.audio_file, confidence=args.confidence, chunk_length_sec=args.chunk_length)
    if args.decode_clap_segments:
        ext.decode_clap_segments()
    else:
        ext.run()

if __name__ == "__main__":
    main() 