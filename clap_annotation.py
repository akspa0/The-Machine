import os
import json
from pathlib import Path
from typing import List, Dict
from transformers import ClapProcessor, ClapModel
import torch
import torchaudio
import re
import numpy as np
import soundfile as sf
from shutil import copyfile
import logging
from extensions.clap_utils import detect_clap_events
# If LLM chunking/tokenization is needed, import from llm_utils
# from llm_utils import split_into_chunks_advanced

def parse_anonymized_filename(filename):
    # Example: 0000-out-20250511-221253.wav
    m = re.match(r'(\d{4})-(out)-([\d-]+)\.wav', filename)
    if not m:
        return None, None, None
    call_id, channel, timestamp = m.groups()
    return call_id, channel, timestamp

def chunk_audio(waveform, sr, chunk_length_sec=5, overlap_sec=2):
    chunk_size = int(sr * chunk_length_sec)
    overlap_size = int(sr * overlap_sec)
    total_samples = waveform.shape[1]
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] > 0:
            chunks.append((chunk, start / sr, end / sr))  # (waveform, start_time, end_time)
        if end == total_samples:
            break
        start += chunk_size - overlap_size
    return chunks

def run_clap_annotation_on_chunk(chunk_waveform, sr, prompts, model, processor, confidence_threshold=0.5):
    # Convert to mono if not already
    if chunk_waveform.shape[0] > 1:
        chunk_waveform = chunk_waveform.mean(dim=0)
    else:
        chunk_waveform = chunk_waveform.squeeze(0)
    chunk_np = chunk_waveform.cpu().numpy()
    # Pass as a list of 1D numpy arrays
    inputs = processor(text=prompts, audios=[chunk_np], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_audio = outputs.logits_per_audio[0].softmax(dim=0)
    results = []
    for i, prompt in enumerate(prompts):
        score = logits_per_audio[i].item()
        if score >= confidence_threshold:
            results.append({
                'prompt': prompt,
                'confidence': score
            })
    return results

def run_clap_annotation(audio_path: Path, prompts: List[str], model_id: str = "laion/clap-htsat-unfused", chunk_length_sec=5, overlap_sec=2, confidence_threshold=0.5) -> List[Dict]:
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info(f"[CLAP] Model loaded: {model_id} on device: {device}")
    model.eval()
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.numel() == 0:
            raise ValueError("Audio file is empty (0 elements)")
        if sr != 48000:
            waveform = torchaudio.functional.resample(waveform, sr, 48000)
            sr = 48000
    except Exception as e:
        # Raise a controlled error to be caught by the caller
        raise RuntimeError(f"Failed to load or process audio: {audio_path.name} ({str(e)})")
    chunks = chunk_audio(waveform, sr, chunk_length_sec, overlap_sec)
    all_results = []
    for idx, (chunk, start_time, end_time) in enumerate(chunks):
        chunk_results = run_clap_annotation_on_chunk(chunk, sr, prompts, model, processor, confidence_threshold)
        for event in chunk_results:
            all_results.append({
                'chunk_index': idx,
                'start_time': start_time,
                'end_time': end_time,
                'prompt': event['prompt'],
                'confidence': event['confidence']
            })
    return all_results

def annotate_clap_for_out_files(
    input_dir: Path,
    output_dir: Path,
    prompts: list = None,
    model_id: str = "laion/clap-htsat-unfused",
    chunk_length_sec=5,
    overlap_sec=2,
    confidence_threshold=0.6
) -> list:
    """High-level directory annotator powered by extensions.clap_utils.detect_clap_events.

    For each audio file that looks like an anonymised `*-out-*.wav` (or mp3/flac) we
    run CLAP detection, store the accepted events in a companion JSON, and return a
    manifest-friendly list summarising the work.  All logging is anonymised: we never
    output the original path, only the call-ID extracted from the filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for audio_file in input_dir.glob('*'):
        if not audio_file.is_file() or audio_file.suffix.lower() not in {'.wav', '.mp3', '.flac'}:
            continue

        call_id, channel, timestamp = parse_anonymized_filename(audio_file.name)
        if call_id is None:
            # Skip anything that isn't already anonymised
            continue

        logging.info(f"[CLAP] Processing call-id {call_id}, channel {channel}")

        events = detect_clap_events(
            audio_file,
            prompts or [],
            model_id=model_id,
            chunk_length_sec=chunk_length_sec,
            overlap_sec=overlap_sec,
            confidence_threshold=confidence_threshold,
        )

        # Flatten events → per-prompt list for manifest compatibility
        accepted_annotations = []
        for ev in events:
            for prompt, conf in ev['prompts'].items():
                accepted_annotations.append({
                    'call_id': call_id,
                    'prompt': prompt,
                    'confidence': conf,
                    'start_time': ev['start_time_s'],
                    'end_time': ev['end_time_s'],
                })

        out_json = output_dir / f"{audio_file.stem}_clap_annotations.json"
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(accepted_annotations, f, indent=2)

        logging.info(f"[CLAP] {audio_file.name}: {len(accepted_annotations)} accepted annotations written to {out_json.name}")

        results.append({
            'input_name': audio_file.name,
            'annotation_path': str(out_json),
            'accepted_annotations': accepted_annotations,
            'call_id': call_id,
        })

    return results

def pair_clap_events(events, config, total_duration):
    """
    Pair start and end events using config parameters for robust segmentation.
    Returns a list of (start, end) tuples.
    """
    start_prompt = config.get("start_prompt", "telephone ring tones")
    end_prompt = config.get("end_prompt", "hang-up tones")
    min_gap = config.get("min_gap_sec", 2.0)
    min_call = config.get("min_call_duration_sec", 10.0)
    noise_gap = config.get("noise_gap_sec", 1.5)
    pairs = []
    i = 0
    n = len(events)
    while i < n:
        e = events[i]
        if e["prompt"] == start_prompt:
            start_time = e["start_time"]
            # Find the next end_prompt after this start
            j = i + 1
            while j < n:
                e2 = events[j]
                if e2["prompt"] == end_prompt:
                    end_time = e2["start_time"]
                    # Ignore if too close (noise)
                    if end_time - start_time < noise_gap:
                        j += 1
                        continue
                    # Only accept if duration is long enough
                    if end_time - start_time >= min_call:
                        pairs.append((start_time, end_time))
                        i = j  # Move to after this end
                        break
                j += 1
            else:
                # No end found, segment goes to file end
                if total_duration - start_time >= min_call:
                    pairs.append((start_time, total_duration))
                i = n  # Done
                break
        i += 1
    return pairs

def segment_audio_with_clap(
    audio_path: Path,
    segmentation_config: dict,
    output_dir: Path,
    model_id: str = "laion/clap-htsat-unfused",
    chunk_length_sec=5,
    overlap_sec=2
) -> list:
    """
    Segment a long audio file into calls using CLAP-based event detection and intelligent pairing.
    Returns a list of segment metadata dicts.
    """
    prompts = segmentation_config.get("prompts", ["telephone ring tones", "hang-up tones"])
    confidence_threshold = segmentation_config.get("confidence_threshold", 0.6)
    min_segment_length = segmentation_config.get("min_segment_length_sec", 10)
    padding = segmentation_config.get("segment_padding_sec", 0.5)

    # Run CLAP event detection
    events = run_clap_annotation(
        audio_path,
        prompts,
        model_id=model_id,
        chunk_length_sec=chunk_length_sec,
        overlap_sec=overlap_sec,
        confidence_threshold=confidence_threshold
    )
    # Only keep events with relevant prompts
    start_prompt = segmentation_config.get("start_prompt", "telephone ring tones")
    end_prompt = segmentation_config.get("end_prompt", "hang-up tones")
    filtered_events = [e for e in events if e["prompt"] in (start_prompt, end_prompt)]
    # Load audio
    waveform, sr = torchaudio.load(str(audio_path))
    total_duration = waveform.shape[1] / sr
    # Pair events intelligently
    pairs = pair_clap_events(filtered_events, segmentation_config, total_duration)
    # Build segments
    segments = []
    for i, (seg_start, seg_end) in enumerate(pairs):
        seg_start = max(0.0, seg_start - padding)
        seg_end = min(total_duration, seg_end + padding)
        if seg_end - seg_start < min_segment_length:
            continue
        seg_wave = waveform[:, int(seg_start * sr):int(seg_end * sr)]
        seg_name = f"{audio_path.stem}-seg{i:04d}.wav"
        seg_path = output_dir / seg_name
        torchaudio.save(str(seg_path), seg_wave, sr)
        segments.append({
            "segment_index": i,
            "start": seg_start,
            "end": seg_end,
            "output_path": str(seg_path),
            "source_file": str(audio_path),
            "events": [e for e in filtered_events if seg_start <= e["start_time"] < seg_end]
        })
    return segments

# ---------------------------------------------------------------------------
# Convenience wrapper using workflow JSON

def annotate_clap_from_workflow(
    input_dir: Path,
    output_dir: Path,
    workflow_path: Path = Path(__file__).resolve().parent / 'workflows' / 'clap_annotation.json',
) -> list:
    """Load *workflow_path* JSON and run annotate_clap_for_out_files.

    The JSON is expected to look like::

        {
            "prompts": [...],
            "confidence_threshold": 0.4,
            "chunk_length_sec": 8,
            "overlap_sec": 3
        }
    """
    import json

    if not workflow_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {workflow_path}")

    with open(workflow_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    prompts = cfg.get('prompts', [])
    confidence = cfg.get('confidence_threshold', 0.5)
    chunk_len = cfg.get('chunk_length_sec', 10)
    overlap = cfg.get('overlap_sec', 0)

    return annotate_clap_for_out_files(
        input_dir,
        output_dir,
        prompts=prompts,
        model_id='laion/clap-htsat-unfused',
        chunk_length_sec=chunk_len,
        overlap_sec=overlap,
        confidence_threshold=confidence,
    ) 