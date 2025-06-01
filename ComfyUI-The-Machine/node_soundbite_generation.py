import os
from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class SoundbiteGenerationNode:
    """
    Soundbite Generation Node
    Generates soundbites from transcribed segments, validates transcript/duration, renames, updates manifest, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"transcribed": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"soundbites": "list[dict]", "manifest": "dict"}

    def process(self, transcribed, manifest, config=None):
        from tqdm import tqdm
        soundbites = []
        errors = []
        output_dir = Path('soundbites')
        output_dir.mkdir(exist_ok=True)
        min_duration = config.get("min_duration", 1.0) if config else 1.0
        max_duration = config.get("max_duration", 30.0) if config else 30.0
        for entry in tqdm(transcribed, desc="Soundbite generation"):
            seg_path = entry["segment_path"]
            transcript = entry["transcript"]
            tuple_index = entry["tuple_index"]
            speaker = entry["speaker"]
            try:
                audio = AudioSegment.from_wav(seg_path)
                duration = len(audio) / 1000.0
                if duration < min_duration or duration > max_duration:
                    continue  # Skip invalid soundbites
                # Rename segment file to include index and short transcript
                short_trans = transcript[:48].replace(" ", "_")
                out_name = f"{tuple_index}-{speaker}-{short_trans}.wav"
                out_path = output_dir / privacy.scrub_filename(out_name)
                audio.export(out_path, format="wav")
                sb_entry = {
                    "tuple_index": tuple_index,
                    "speaker": speaker,
                    "soundbite_path": str(out_path),
                    "transcript": transcript,
                    "duration": duration,
                    "lineage": [seg_path],
                }
                soundbites.append(sb_entry)
            except Exception as e:
                errors.append({"tuple_index": tuple_index, "segment_path": seg_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, soundbites, key="soundbites")
        if errors:
            manifest["soundbite_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"soundbites": soundbites, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for soundbite generation.
        - transcribed: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "transcribed": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Soundbite Config (JSON)", "default": {}}
        } 