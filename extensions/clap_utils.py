from __future__ import annotations

"""extensions/clap_utils.py
Reusable utilities for CLAP-based audio event detection and segmentation.
All functions here are import-safe for both root-level scripts and
extensions.* modules.

NOTE:  This file purposely remains <250 LOC to respect workspace rules.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torchaudio
from transformers import ClapProcessor, ClapModel
import numpy as np
import soundfile as sf

__all__ = [
    "get_clap_model",
    "detect_clap_events",
    "pair_events",
    "pair_first_start_first_end",
    "merge_contiguous",
    "apply_temporal_nms",
    "find_calls_ring_speech_ring",
    "find_calls_cross_track_ring_speech_ring",
    "segment_audio",
    "events_to_segments",
    "pair_alternating_prompt",
    "pair_alternating_sets",
]

# Default prompts that proved reliable in PoC
DEFAULT_PROMPT_GENERIC = "telephone noises"
DEFAULT_START_PROMPT = "telephone ringing sounds"
DEFAULT_END_PROMPT = "telephone hang-up tones"
DEFAULT_CONF = 0.15

# Prompt groups for advanced call detection
RING_PROMPTS = ["telephone noises", "telephone ringing sounds", "telephone hang-up tones"]
SPEECH_PROMPTS = ["speech", "conversation"]
MUSIC_PROMPTS = ["music", "sound effects"]

def get_clap_model(model_id: str = "laion/clap-htsat-fused") -> Tuple[ClapProcessor, ClapModel, torch.device]:
    """Load a CLAP model/processor on the best available device (GPU preferred)."""
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return processor, model, device


def _load_audio_mono_48k(path: Path) -> Tuple[torch.Tensor, int]:
    """Load *path* via torchaudio → mono 48 kHz tensor of shape (1, n)."""
    waveform, sr = torchaudio.load(str(path))
    # To mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != 48_000:
        waveform = torchaudio.functional.resample(waveform, sr, 48_000)
        sr = 48_000
    return waveform, sr


def detect_clap_events(
    audio_path: Path,
    prompts: List[str],
    model_id: str = "laion/clap-htsat-fused",
    chunk_length_sec: int = 10,
    overlap_sec: int = 0,
    confidence_threshold: float = 0.15,
    similarity_metric: str = "sigmoid",  # "sigmoid" (prob) or "cosine"
    nms_gap_sec: float = 0.0,
    auto_calibrate_seconds: int | None = None,
    k_sigma: float = 3.0,
) -> List[Dict]:
    """Run CLAP over *audio_path* and return detected events.

    Returned event format →
        {
            "start_time_s": float,
            "end_time_s": float,
            "prompts": {prompt: confidence, ...},
        }
    An event entry exists only if at least one prompt ≥ *confidence_threshold*.
    """
    processor, model, device = get_clap_model(model_id)

    # Pre-compute text features once for efficiency.
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    events: List[Dict] = []

    # Optional calibration phase (quick first pass) ------------------------
    if auto_calibrate_seconds:
        calib_scores = []  # flat list of all probs in calib window
        with sf.SoundFile(str(audio_path)) as f_cal:
            sr_c = f_cal.samplerate
            frames_limit = min(int(auto_calibrate_seconds * sr_c), len(f_cal))
            audio_cal = f_cal.read(frames_limit, dtype="float32").T
        if audio_cal.size:
            wave_c = torch.from_numpy(audio_cal)
            if wave_c.shape[0] > 1:
                wave_c = wave_c.mean(dim=0, keepdim=True)
            if sr_c != 48_000:
                wave_c = torchaudio.functional.resample(wave_c, sr_c, 48_000)
                sr_c = 48_000
            audio_np_c = wave_c.cpu().numpy().astype('float32').flatten()
            with torch.no_grad():
                aud_in = processor(audios=[audio_np_c], sampling_rate=sr_c, return_tensors="pt", padding=True).to(device)
                feat_c = model.get_audio_features(**aud_in)
                feat_c = feat_c / feat_c.norm(dim=-1, keepdim=True)
                logits_c = (feat_c @ text_features.T).squeeze(0)
                if similarity_metric == "sigmoid":
                    scores_c = torch.sigmoid(logits_c).cpu().numpy()
                else:  # cosine
                    scores_c = logits_c.cpu().numpy()
            calib_scores.extend(scores_c.tolist())
        if calib_scores:
            mu, sigma = float(np.mean(calib_scores)), float(np.std(calib_scores))
            dynamic_th = max(mu + k_sigma * sigma, 0.15)
            confidence_threshold = dynamic_th
            # Informative print (can be removed in prod)
            print(f"[CLAP-AUTO] calib mu={mu:.3f}, sigma={sigma:.3f} → threshold={confidence_threshold:.3f}")

    with sf.SoundFile(str(audio_path)) as f:
        sr_orig = f.samplerate
        total_frames = len(f)
        chunk_frames = int(chunk_length_sec * sr_orig)
        overlap_frames = int(overlap_sec * sr_orig)
        step = chunk_frames - overlap_frames if chunk_frames > overlap_frames else chunk_frames
        start_frame = 0
        while start_frame < total_frames:
            f.seek(start_frame)
            frames_to_read = min(chunk_frames, total_frames - start_frame)
            audio_block = f.read(frames_to_read, dtype='float32').T  # (channels, n)
            if audio_block.size == 0:
                break
            wave = torch.from_numpy(audio_block)
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            # Resample per-block if needed
            if sr_orig != 48_000:
                wave = torchaudio.functional.resample(wave, sr_orig, 48_000)
                sr_proc = 48_000
            else:
                sr_proc = sr_orig

            start_t = start_frame / sr_orig
            end_t = (start_frame + frames_to_read) / sr_orig

            audio_np = wave.squeeze(0).cpu().numpy().astype('float32').flatten()
            with torch.no_grad():
                audio_inputs = processor(audios=[audio_np], sampling_rate=sr_proc, return_tensors="pt", padding=True).to(device)
                audio_features = model.get_audio_features(**audio_inputs)
                audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
                logits = (audio_features @ text_features.T).squeeze(0)
                if similarity_metric == "sigmoid":
                    scores = torch.sigmoid(logits).cpu().numpy()
                else:
                    scores = logits.cpu().numpy()
            prompt_scores = {p: float(score) for p, score in zip(prompts, scores) if score >= confidence_threshold}
            if prompt_scores:
                events.append({
                    "start_time_s": round(start_t, 3),
                    "end_time_s": round(end_t, 3),
                    "prompts": prompt_scores,
                })

            if start_frame + chunk_frames >= total_frames:
                break
            start_frame += step

    # Optional temporal NMS
    if nms_gap_sec and nms_gap_sec > 0.0:
        events = apply_temporal_nms(events, nms_gap_sec)

    return events


# ---- Heuristics for pairing & segmentation ---------------------------------

def pair_events(events: List[Dict], config: Dict, total_duration: float) -> List[Tuple[float, float]]:
    """Heuristic start/end pairing similar to legacy implementation."""
    start_prompt = config.get("start_prompt", "telephone ring tones")
    end_prompt = config.get("end_prompt", "hang-up tones")
    min_gap = config.get("min_gap_sec", 2.0)
    min_call = config.get("min_call_duration_sec", 10.0)
    noise_gap = config.get("noise_gap_sec", 1.5)

    def _is_prompt(event: Dict, target: str) -> bool:
        return target in event.get("prompts", {})

    pairs = []
    i = 0
    n = len(events)
    while i < n:
        if _is_prompt(events[i], start_prompt):
            start_time = events[i]["start_time_s"]
            j = i + 1
            while j < n and not _is_prompt(events[j], end_prompt):
                j += 1
            if j == n:
                # No explicit end, use EOF
                if total_duration - start_time >= min_call:
                    pairs.append((start_time, total_duration))
                break
            end_time = events[j]["start_time_s"]
            if end_time - start_time < noise_gap:
                i = j + 1
                continue
            if end_time - start_time >= min_call:
                pairs.append((start_time, end_time))
            i = j + 1
        else:
            i += 1
    return pairs


def pair_first_start_first_end(
    events: List[Dict],
    start_prompt: str = DEFAULT_START_PROMPT,
    end_prompt: str = DEFAULT_END_PROMPT,
) -> List[Tuple[float, float]]:
    """Return list of (start,end) where each pair is first START then first END that follows.
    Discards any additional STARTs before the matching END; continues scanning after END."""

    def _is_prompt(ev: Dict, target: str) -> bool:
        return target in ev.get("prompts", {})

    pairs: List[Tuple[float, float]] = []
    i = 0
    n = len(events)
    while i < n:
        # find first START
        while i < n and not _is_prompt(events[i], start_prompt):
            i += 1
        if i >= n:
            break
        start_t = events[i]["start_time_s"]
        i += 1
        # find first END after this START
        while i < n and not _is_prompt(events[i], end_prompt):
            i += 1
        if i >= n:
            break  # no end found – drop last incomplete pair
        end_t = events[i]["end_time_s"]
        if end_t > start_t:
            pairs.append((start_t, end_t))
        i += 1
    return pairs


def segment_audio(
    audio_path: Path,
    segmentation_config: Dict,
    output_dir: Path,
    model_id: str = "laion/clap-htsat-fused",
    chunk_length_sec: int = 10,
    overlap_sec: int = 0,
) -> List[Dict]:
    """High-level helper: detect events → pair → write *wav* segments → return metadata."""
    prompts = segmentation_config.get("prompts", [DEFAULT_PROMPT_GENERIC])
    conf = segmentation_config.get("confidence_threshold", DEFAULT_CONF)
    padding = segmentation_config.get("segment_padding_sec", 0.5)
    min_len = segmentation_config.get("min_segment_length_sec", 10.0)
    pairing_mode = segmentation_config.get("pairing", None)

    # Derive default pairing mode
    if pairing_mode is None:
        pairing_mode = "contiguous" if len(prompts) == 1 else "first_start_first_end"

    sim_metric = segmentation_config.get("similarity_metric", "sigmoid")
    nms_gap_sec = segmentation_config.get("nms_gap_sec", 0.0)

    events = detect_clap_events(
        audio_path,
        prompts,
        model_id=model_id,
        chunk_length_sec=chunk_length_sec,
        overlap_sec=overlap_sec,
        confidence_threshold=conf,
        similarity_metric=sim_metric,
        nms_gap_sec=nms_gap_sec,
    )

    waveform, sr = _load_audio_mono_48k(audio_path)
    total_duration = waveform.shape[1] / sr
    if pairing_mode == "ring_speech_ring":
        thr_dict = segmentation_config.get("thresholds", {
            "ring": 0.51,
            "speech": 0.4,
            "music": 0.4,
        })
        pairs = find_calls_ring_speech_ring(events, thr_dict)
    elif pairing_mode == "cross_track_ring_speech_ring":
        thr_dict = segmentation_config.get("thresholds", {
            "ring": 0.45,
            "speech": 0.35,
            "music": 0.4,
        })
        pairs = find_calls_cross_track_ring_speech_ring(events, thr_dict)
    elif pairing_mode == "raw_events":
        pairs = events_to_segments(events, padding, min_len, total_duration)
    elif pairing_mode == "contiguous":
        pairs = merge_contiguous(events)
    elif pairing_mode == "first_start_first_end":
        pairs = pair_first_start_first_end(events)
    elif pairing_mode == "tone_alternating":
        pairs = pair_alternating_prompt(events, prompts[0])
    else:
        pairs = pair_events(events, segmentation_config, total_duration)

    output_dir.mkdir(parents=True, exist_ok=True)
    segments: List[Dict] = []
    for idx, (s, e) in enumerate(pairs):
        s_p = max(0.0, s - padding)
        e_p = min(total_duration, e + padding)
        if e_p - s_p < min_len:
            continue
        wav_chunk = waveform[:, int(s_p * sr): int(e_p * sr)]
        seg_name = f"{audio_path.stem}-seg{idx:04d}.wav"
        seg_path = output_dir / seg_name
        torchaudio.save(str(seg_path), wav_chunk, sr)
        segments.append({
            "segment_index": idx,
            "start": s_p,
            "end": e_p,
            "output_path": str(seg_path),
            "source_file": str(audio_path),
        })
    return segments


# ---------------------------------------------------------------------------
#  Merge contiguous/overlapping events for single-prompt use-case
# ---------------------------------------------------------------------------


def merge_contiguous(
    events: List[Dict],
    max_gap_sec: float = 1.0,
) -> List[Tuple[float, float]]:
    """Given a list of detection *events* (any prompt), merge those whose gaps
    are ≤ *max_gap_sec* into one (start,end) interval. Returns merged list."""
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e["start_time_s"])
    merged: List[Tuple[float, float]] = []
    cur_start = events_sorted[0]["start_time_s"]
    cur_end = events_sorted[0]["end_time_s"]
    for ev in events_sorted[1:]:
        if ev["start_time_s"] - cur_end <= max_gap_sec:
            cur_end = max(cur_end, ev["end_time_s"])
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = ev["start_time_s"], ev["end_time_s"]
    merged.append((cur_start, cur_end))
    return merged


# ---------------------------------------------------------------------------
#  Ring–Speech–Ring state-machine call detector
# ---------------------------------------------------------------------------


def _classify_window(prompts_dict: Dict[str, float], thr: Dict[str, float]) -> str:
    """Return class label for window given prompt->prob mapping."""
    if any(prompts_dict.get(p, 0) >= thr.get("ring", 0.51) for p in RING_PROMPTS):
        return "ring"
    if any(prompts_dict.get(p, 0) >= thr.get("speech", 0.4) for p in SPEECH_PROMPTS):
        return "speech"
    if any(prompts_dict.get(p, 0) >= thr.get("music", 0.4) for p in MUSIC_PROMPTS):
        return "music"
    return "other"


def find_calls_ring_speech_ring(
    events: List[Dict],
    thr: Dict[str, float],
    min_speech_windows: int = 3,
    max_non_speech_gap: int = 2,
) -> List[Tuple[float, float]]:
    """Find (start,end) by pattern: ring -> speech block -> ring."""
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e["start_time_s"])
    call_pairs: List[Tuple[float, float]] = []
    state = "idle"
    speech_count = 0
    start_t = None
    gap_since_speech = 0

    for ev in events_sorted:
        cls = _classify_window(ev["prompts"], thr)
        if state == "idle":
            if cls == "ring":
                state = "candidate"
                start_t = ev["start_time_s"]
                speech_count = 0
                gap_since_speech = 0
        elif state == "candidate":
            if cls == "speech":
                speech_count += 1
                gap_since_speech = 0
            else:
                gap_since_speech += 1
            if cls == "ring" and speech_count >= min_speech_windows:
                end_t = ev["end_time_s"]
                if end_t > start_t:
                    call_pairs.append((start_t, end_t))
                state = "idle"
            elif gap_since_speech > max_non_speech_gap and speech_count == 0:
                # give up, false start
                state = "idle"
    return call_pairs


# ---------------------------------------------------------------------------
#  Cross-track Ring–Speech–Ring detector (instrumental vs vocal)
# ---------------------------------------------------------------------------


def find_calls_cross_track_ring_speech_ring(
    events: List[Dict],
    thr: Dict[str, float],
    ring_tracks: set[str] | None = None,
    speech_tracks: set[str] | None = None,
    min_speech_windows: int = 3,
    max_non_speech_gap: int = 2,
) -> List[Tuple[float, float]]:
    """Like find_calls_ring_speech_ring but enforces that ring events come from
    *ring_tracks* (default {'instrumental'}) and speech events come from
    *speech_tracks* (default {'vocal'})."""

    if not events:
        return []

    ring_tracks = ring_tracks or {"instrumental"}
    speech_tracks = speech_tracks or {"vocal"}

    events_sorted = sorted(events, key=lambda e: e["start_time_s"])

    call_pairs: List[Tuple[float, float]] = []
    state = "idle"
    speech_count = 0
    start_t: float | None = None
    gap_since_speech = 0

    for ev in events_sorted:
        track = ev.get("track", "unknown")
        cls = _classify_window(ev["prompts"], thr)

        if state == "idle":
            if cls == "ring" and track in ring_tracks:
                state = "candidate"
                start_t = ev["start_time_s"]
                speech_count = 0
                gap_since_speech = 0
        elif state == "candidate":
            if cls == "speech" and track in speech_tracks:
                speech_count += 1
                gap_since_speech = 0
            else:
                gap_since_speech += 1

            if cls == "ring" and track in ring_tracks and speech_count >= min_speech_windows:
                end_t = ev["end_time_s"]
                if end_t > (start_t or 0):
                    call_pairs.append((start_t, end_t))
                state = "idle"
            elif gap_since_speech > max_non_speech_gap and speech_count == 0:
                state = "idle"  # false start

    return call_pairs


# ---------------------------------------------------------------------------
# Temporal Non-Maximum Suppression (per-prompt) – ported from WhisperBite PoC
# ---------------------------------------------------------------------------


def _event_confidence(ev: Dict) -> float:
    """Return max confidence across prompts for sorting."""
    return max(ev.get("prompts", {}).values(), default=0.0)


def apply_temporal_nms(events: List[Dict], min_gap_seconds: float = 1.0) -> List[Dict]:
    """Suppress events that occur within *min_gap_seconds* of a higher-confidence one.

    This is a simplified version of the WhisperBite PoC logic and works across
    all prompts jointly. The time reference is the *start_time_s* of each
    event; an accepted event blocks a ±min_gap window around its start.
    """
    if not events:
        return []

    # Sort by confidence descending
    sorted_events = sorted(events, key=_event_confidence, reverse=True)
    accepted: List[Dict] = []

    for ev in sorted_events:
        s = ev["start_time_s"]
        if all(abs(s - acc["start_time_s"]) > min_gap_seconds for acc in accepted):
            accepted.append(ev)

    # Return chronologically sorted list
    return sorted(accepted, key=lambda e: e["start_time_s"])


# ---------------------------------------------------------------------------
#  Simple per-event segmenter (no pairing)                                   
# ---------------------------------------------------------------------------


def events_to_segments(
    events: List[Dict],
    padding: float,
    min_len: float,
    total_duration: float,
) -> List[Tuple[float, float]]:
    """Return (start,end) for EACH event independently with padding applied."""
    segs: List[Tuple[float, float]] = []
    for ev in events:
        s = max(0.0, ev["start_time_s"] - padding)
        e = min(total_duration, ev["end_time_s"] + padding)
        if e - s >= min_len:
            segs.append((s, e))
    return segs


# ---------------------------------------------------------------------------
#  Alternating-start/stop prompt pairer
# ---------------------------------------------------------------------------


def pair_alternating_prompt(
    events: List[Dict],
    prompt: str,
) -> List[Tuple[float, float]]:
    """Treat every occurrence of *prompt* as alternating START/END markers.

    Example: times [t0, t1, t2, t3] → pairs (t0,t1), (t2,t3).
    If there is an unmatched final START it is discarded.
    Uses event['start_time_s'] as the boundary.
    """
    candidates = [ev for ev in sorted(events, key=lambda e: e["start_time_s"]) if prompt in ev.get("prompts", {})]
    pairs: List[Tuple[float, float]] = []
    for idx in range(0, len(candidates) - 1, 2):
        s = candidates[idx]["start_time_s"]
        e = candidates[idx + 1]["start_time_s"]
        if e > s:
            pairs.append((s, e))
    return pairs


# ---------------------------------------------------------------------------
#  Alternating event-set pairer (V vs I)
# ---------------------------------------------------------------------------


def pair_alternating_sets(
    events_vocal: List[Dict],
    events_instr: List[Dict],
    min_len: float = 2.0,
) -> List[Tuple[float, float]]:
    """Return segments whenever detection toggles between instrumental and vocal sets.

    Parameters
    ----------
    events_vocal : list
        Events detected on the vocal track.
    events_instr : list
        Events detected on the instrumental track.
    min_len : float, default 2.0
        Minimum segment duration (seconds) to accept.

    Returns
    -------
    list[tuple[float, float]]
        Chronologically ordered (start,end) pairs.
    """
    combined: List[Tuple[float, str]] = []
    for ev in events_vocal:
        combined.append((ev["start_time_s"], "V"))
    for ev in events_instr:
        combined.append((ev["start_time_s"], "I"))

    if len(combined) < 2:
        return []

    # Collapse consecutive events from the same set to avoid tiny repeated toggles
    combined.sort(key=lambda x: x[0])
    collapsed: List[Tuple[float, str]] = []
    for time_t, set_id in combined:
        if not collapsed or set_id != collapsed[-1][1]:
            collapsed.append((time_t, set_id))

    if len(collapsed) < 2:
        return []

    segments: List[Tuple[float, float]] = []
    for (start_t, set_a), (end_t, set_b) in zip(collapsed, collapsed[1:]):
        if set_a != set_b and (end_t - start_t) >= min_len:
            segments.append((start_t, end_t))
    return segments 