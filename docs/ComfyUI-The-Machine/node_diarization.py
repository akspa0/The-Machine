import os
from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

class DiarizationNode:
    """
    Diarization Node
    Runs speaker diarization (e.g., pyannote) on normalized files, segments vocals per speaker, stores in speakers/SXX/, updates manifest, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"normalized": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"diarized": "list[dict]", "manifest": "dict"}

    def process(self, normalized, manifest, config=None):
        from tqdm import tqdm
        # from pyannote.audio import Pipeline
        diarized = []
        errors = []
        output_dir = Path('diarized')
        output_dir.mkdir(exist_ok=True)
        # pipeline = Pipeline.from_pretrained(config.get("pyannote_model", "pyannote/speaker-diarization"))
        for entry in tqdm(normalized, desc="Diarization"):
            audio_path = entry["normalized_path"]
            tuple_index = entry["tuple_index"]
            try:
                # result = pipeline(audio_path)
                # For now, simulate result:
                result = [
                    {"speaker": "S01", "start": 0.0, "end": 2.0},
                    {"speaker": "S02", "start": 2.0, "end": 4.0}
                ]
                for seg in result:
                    speaker_dir = output_dir / f"{tuple_index}-speakers" / seg["speaker"]
                    speaker_dir.mkdir(parents=True, exist_ok=True)
                    seg_name = f"{tuple_index}-{seg['speaker']}-{seg['start']:.2f}-{seg['end']:.2f}.wav"
                    seg_path = speaker_dir / privacy.scrub_filename(seg_name)
                    # Real: extract segment audio from audio_path using start/end
                    # For now, just touch the file
                    seg_path.touch()
                    diar_entry = {
                        "tuple_index": tuple_index,
                        "speaker": seg["speaker"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "segment_path": str(seg_path),
                        "lineage": [audio_path],
                    }
                    diarized.append(diar_entry)
            except Exception as e:
                errors.append({"tuple_index": tuple_index, "audio_path": audio_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, diarized, key="diarized")
        if errors:
            manifest["diarization_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"diarized": diarized, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for diarization.
        - normalized: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "normalized": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Diarization Config (JSON)", "default": {}}
        } 