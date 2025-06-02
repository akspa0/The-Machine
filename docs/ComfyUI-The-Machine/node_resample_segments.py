from .utils import privacy
from .utils import manifest as manifest_utils
import os

class ResampleSegmentsNode:
    """
    Resample Segments Node
    Resamples each segment audio file to 16kHz mono using torchaudio. Updates manifest with resampled file paths.
    """
    @classmethod
    def input_types(cls):
        return {"segment_files": "list[str]", "manifest": "dict"}

    @classmethod
    def output_types(cls):
        return {"resampled_segments": "list[dict]", "manifest": "dict"}

    def process(self, segment_files, manifest):
        import torchaudio
        import torch
        resampled_segments = []
        errors = []
        for idx, seg_path in enumerate(segment_files):
            try:
                sanitized_path = privacy.scrub_filename(seg_path)
                waveform, sr = torchaudio.load(sanitized_path)
                if waveform.numel() == 0 or (waveform.ndim == 2 and waveform.shape[1] == 0):
                    raise ValueError("Segment file is empty")
                # Convert to mono if needed
                if waveform.ndim == 2 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Resample if needed
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                    sr = 16000
                if waveform.ndim != 2 or waveform.shape[0] != 1:
                    raise ValueError(f"Shape after mono/resample is {waveform.shape}, expected [1, time]")
                out_dir = os.path.join("resampled_segments", f"{idx:04d}")
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, os.path.basename(sanitized_path).replace('.wav', '_16k.wav'))
                torchaudio.save(out_file, waveform, sr)
                entry = {
                    "tuple_index": f"{idx:04d}",
                    "input_file": sanitized_path,
                    "output_resampled": out_file,
                    "resample_timestamp": "2025-06-01T01:05:00Z",
                    "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "speaker_segmentation", "resample_segments"]
                }
                resampled_segments.append(entry)
            except Exception as e:
                errors.append({"file": seg_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, resampled_segments)
        if errors:
            manifest["resample_segments_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"resampled_segments": resampled_segments, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for resample settings, progress
        pass 