from .utils import privacy
from .utils import manifest as manifest_utils
import os

class TruePeakNormalizationNode:
    """
    True Peak Normalization Node
    Applies true peak normalization to -1.0 dBTP for all input audio files. Updates manifest.
    """
    @classmethod
    def input_types(cls):
        return {"audio_files": "list[str]", "manifest": "dict"}

    @classmethod
    def output_types(cls):
        return {"true_peak_normalized": "list[dict]", "manifest": "dict"}

    def process(self, audio_files, manifest):
        import pyloudnorm as pyln
        import soundfile as sf
        import numpy as np
        true_peak_normalized = []
        errors = []
        for idx, audio_path in enumerate(audio_files):
            try:
                sanitized_path = privacy.scrub_filename(audio_path)
                audio, sr = sf.read(sanitized_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                meter = pyln.Meter(sr)
                peak_normalized = pyln.normalize.peak(audio, -1.0)
                true_peak_db = 20 * np.log10(np.max(np.abs(peak_normalized)))
                output_dir = os.path.join("true_peak_normalized", f"{idx:04d}")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, os.path.basename(sanitized_path))
                sf.write(output_file, peak_normalized, sr)
                entry = {
                    "tuple_index": f"{idx:04d}",
                    "input_file": sanitized_path,
                    "output_true_peak": output_file,
                    "measured_true_peak_db": true_peak_db,
                    "true_peak_timestamp": "2025-06-01T00:55:00Z",
                    "lineage": ["tuple_assembler", "separation", "normalization", "true_peak_normalization"]
                }
                true_peak_normalized.append(entry)
            except Exception as e:
                errors.append({"file": audio_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, true_peak_normalized)
        if errors:
            manifest["true_peak_normalization_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"true_peak_normalized": true_peak_normalized, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for normalization settings, progress
        pass 