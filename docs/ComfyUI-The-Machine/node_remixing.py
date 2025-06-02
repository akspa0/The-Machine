from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

class RemixingNode:
    """
    Remixing Node
    Mixes vocals/instrumentals for trans_out/recv_out pairs, applies stereo panning, volume, and optional tones, updates manifest, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"soundbites": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"remixed": "list[dict]", "manifest": "dict"}

    def process(self, soundbites, manifest, config=None):
        from tqdm import tqdm
        from pydub import AudioSegment
        remixed = []
        errors = []
        output_dir = Path('remixed')
        output_dir.mkdir(exist_ok=True)
        instrumental_volume = config.get("instrumental_volume", 0.5) if config else 0.5
        pan_amount = config.get("pan_amount", 0.2) if config else 0.2
        for entry in tqdm(soundbites, desc="Remixing"):
            sb_path = entry["soundbite_path"]
            tuple_index = entry["tuple_index"]
            speaker = entry["speaker"]
            try:
                vocals = AudioSegment.from_wav(sb_path)
                # For demonstration, use vocals as both channels
                left = vocals.pan(-pan_amount)
                right = vocals.pan(pan_amount)
                # Optionally add instrumental (not available in soundbite, so skip here)
                # Optionally add tones (not implemented here)
                stereo = AudioSegment.from_mono_audiosegments(left, right)
                out_name = f"{tuple_index}-{speaker}-remixed.wav"
                out_path = output_dir / privacy.scrub_filename(out_name)
                stereo.export(out_path, format="wav")
                remix_entry = {
                    "tuple_index": tuple_index,
                    "speaker": speaker,
                    "remixed_path": str(out_path),
                    "lineage": [sb_path],
                }
                remixed.append(remix_entry)
            except Exception as e:
                errors.append({"tuple_index": tuple_index, "soundbite_path": sb_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, remixed, key="remixed")
        if errors:
            manifest["remixing_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"remixed": remixed, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for remixing.
        - soundbites: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "soundbites": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Remixing Config (JSON)", "default": {}}
        } 