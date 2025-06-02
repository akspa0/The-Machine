from .utils import privacy
from .utils import manifest as manifest_utils
import os
from pathlib import Path

class SeparationNode:
    """
    Separation Node
    Runs source separation (e.g., Demucs/Spleeter) on input files, outputs separated vocals/instrumentals, updates manifest, and ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"tuples": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"separated": "list[dict]", "manifest": "dict"}

    def process(self, tuples, manifest, config=None):
        from tqdm import tqdm
        import subprocess
        separated = []
        errors = []
        output_dir = Path('separated')
        output_dir.mkdir(exist_ok=True)
        # For each tuple, run separation on left/right/out if present
        for tup in tqdm(tuples, desc="Source separation"):
            tuple_index = tup["tuple_index"]
            for key in ("left", "right", "out"):
                input_path = tup.get(key)
                if not input_path:
                    continue
                try:
                    # Output file names
                    base = Path(input_path).stem
                    vocal_path = output_dir / privacy.scrub_filename(f"{tuple_index}-{key}-vocals.wav")
                    inst_path = output_dir / privacy.scrub_filename(f"{tuple_index}-{key}-instrumental.wav")
                    # Run Demucs (or Spleeter) via subprocess (assumes demucs CLI installed)
                    # Example: demucs -n htdemucs --two-stems=vocals -o separated input_path
                    subprocess.run([
                        'demucs', '-n', 'htdemucs', '--two-stems', 'vocals', '-o', str(output_dir), str(input_path)
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Demucs output is in separated/htdemucs/base/vocals.wav and instrumental.wav
                    demucs_dir = output_dir / 'htdemucs' / base
                    demucs_vocals = demucs_dir / 'vocals.wav'
                    demucs_inst = demucs_dir / 'no_vocals.wav'
                    # Move and scrub
                    if demucs_vocals.exists():
                        os.rename(demucs_vocals, vocal_path)
                    if demucs_inst.exists():
                        os.rename(demucs_inst, inst_path)
                    sep_entry = {
                        "tuple_index": tuple_index,
                        "input_type": key,
                        "vocals_path": str(vocal_path),
                        "instrumental_path": str(inst_path),
                        "lineage": [str(input_path)],
                    }
                    separated.append(sep_entry)
                except Exception as e:
                    errors.append({"tuple_index": tuple_index, "input_type": key, "error": str(e)})
        manifest_utils.update_manifest(manifest, separated, key="separated")
        if errors:
            manifest["separation_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"separated": separated, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for separation.
        - tuples: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "tuples": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Separation Config (JSON)", "default": {}}
        } 