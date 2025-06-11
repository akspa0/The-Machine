"""extensions/clap_whisper_detector.py
Minimal port of the WhisperBite proof-of-concept CLAP detector so we can
switch back-and-forth between the "classic" logic and the newer
clap_utils implementation.

This file stays <250 LOC and depends only on torch/transformers and the
helpers already in clap_utils.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
import soundfile as sf

from extensions.clap_utils import get_clap_model, _load_audio_mono_48k, apply_temporal_nms

__all__ = ["detect_clap_events_wb"]


def _prep_text_features(processor, model, device, prompts: List[str]):
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_feats = model.get_text_features(**text_inputs)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    return text_feats


def detect_clap_events_wb(
    audio_path: Path | str,
    prompts: List[str],
    model_id: str = "laion/clap-htsat-fused",
    chunk_length_sec: float = 5.0,
    threshold: float = 0.15,
    nms_gap_sec: float = 1.0,
) -> List[Dict]:
    """Re-implementation of WhisperBite's run_clap_event_detection().

    Returns events in the unified format expected by downstream pairing:
        {
            "start_time_s": float,
            "end_time_s": float,
            "prompts": {prompt: score, ...},
            "track": "fallback",
        }
    """

    audio_path = Path(audio_path)
    processor, model, device = get_clap_model(model_id)
    text_feats = _prep_text_features(processor, model, device, prompts)

    # Load audio to mono 48k tensor
    waveform, sr = _load_audio_mono_48k(audio_path)
    waveform = waveform.squeeze(0)  # 1-D
    audio_np = waveform.cpu().numpy()
    total_samples = len(audio_np)

    chunk_samples = int(chunk_length_sec * 48_000)  # after resample always 48k
    num_chunks = int(np.ceil(total_samples / chunk_samples))

    all_events: List[Dict] = []

    for i in range(num_chunks):
        s_idx = i * chunk_samples
        e_idx = min((i + 1) * chunk_samples, total_samples)
        chunk = audio_np[s_idx:e_idx].astype("float32")
        if len(chunk) < 4800:  # <0.1 s guard
            continue
        start_t = s_idx / 48_000.0
        end_t = e_idx / 48_000.0
        # CLAP forward
        with torch.no_grad():
            inputs = processor(audios=[chunk], sampling_rate=48_000, return_tensors="pt", padding=True).to(device)
            a_feats = model.get_audio_features(**inputs)
            a_feats = a_feats / a_feats.norm(dim=-1, keepdim=True)
            sims = torch.nn.functional.cosine_similarity(a_feats[:, None], text_feats[None, :], dim=-1)
            sims = sims.squeeze(0).cpu().numpy()
        # record events above threshold
        for prompt, score in zip(prompts, sims):
            if score >= threshold:
                all_events.append({
                    "start_time_s": round(start_t, 3),
                    "end_time_s": round(end_t, 3),
                    "prompts": {prompt: float(score)},
                    "track": "fallback",
                })

    # Temporal NMS (per-prompt handled jointly)
    if nms_gap_sec > 0.0:
        all_events = apply_temporal_nms(all_events, nms_gap_sec)
    return all_events 