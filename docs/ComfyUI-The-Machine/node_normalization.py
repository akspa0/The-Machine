from .utils import privacy
from .utils import manifest as manifest_utils
import os
from pathlib import Path

class NormalizationNode:
    """
    Normalization Node
    Applies true peak/loudness normalization (ITU-R BS.1770, pyloudnorm) to separated files, updates manifest, and ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"separated": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"normalized": "list[dict]", "manifest": "dict"}

    def process(self, separated, manifest, config=None):
        from tqdm import tqdm
        import pyloudnorm as pyln
        import soundfile as sf
        import numpy as np
        normalized = []
        errors = []
        output_dir = Path('normalized')
        output_dir.mkdir(exist_ok=True)
        meter = pyln.Meter(44100)  # Default sample rate, will update per file
        for entry in tqdm(separated, desc="Normalizing audio"):
            tuple_index = entry["tuple_index"]
            for key in ("vocals_path", "instrumental_path"):
                in_path = entry.get(key)
                if not in_path or not Path(in_path).exists():
                    continue
                try:
                    data, rate = sf.read(in_path)
                    meter = pyln.Meter(rate)
                    loudness = meter.integrated_loudness(data)
                    target_loudness = config.get("target_lufs", -16.0) if config else -16.0
                    loudnorm = pyln.normalize.loudness(data, loudness, target_loudness)
                    out_name = Path(in_path).stem + "-norm.wav"
                    out_path = output_dir / privacy.scrub_filename(f"{tuple_index}-{out_name}")
                    sf.write(out_path, loudnorm, rate)
                    norm_entry = {
                        "tuple_index": tuple_index,
                        "input_type": entry.get("input_type"),
                        "source": key,
                        "normalized_path": str(out_path),
                        "lineage": [in_path],
                    }
                    normalized.append(norm_entry)
                except Exception as e:
                    errors.append({"tuple_index": tuple_index, "input_type": entry.get("input_type"), "source": key, "error": str(e)})
        manifest_utils.update_manifest(manifest, normalized, key="normalized")
        if errors:
            manifest["normalization_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"normalized": normalized, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for normalization.
        - separated: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "separated": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Normalization Config (JSON)", "default": {}}
        } 