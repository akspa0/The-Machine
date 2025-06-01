import os
from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

class TranscriptionNode:
    """
    Transcription Node
    Runs transcription (e.g., Parakeet) on diarized segments, saves .txt, updates manifest, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"diarized": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"transcribed": "list[dict]", "manifest": "dict"}

    def process(self, diarized, manifest, config=None):
        from tqdm import tqdm
        # from parakeet import Transcriber
        transcribed = []
        errors = []
        for entry in tqdm(diarized, desc="Transcription"):
            seg_path = entry["segment_path"]
            tuple_index = entry["tuple_index"]
            speaker = entry["speaker"]
            try:
                # Real: run Parakeet or other ASR
                # transcriber = Transcriber.load_from_config(config)
                # transcript = transcriber.transcribe(seg_path)
                # For now, simulate transcript:
                transcript = f"Simulated transcript for {Path(seg_path).name}"
                txt_path = Path(seg_path).with_suffix(".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                trans_entry = {
                    "tuple_index": tuple_index,
                    "speaker": speaker,
                    "segment_path": seg_path,
                    "transcript_path": str(txt_path),
                    "transcript": transcript,
                    "lineage": [seg_path],
                }
                transcribed.append(trans_entry)
            except Exception as e:
                errors.append({"tuple_index": tuple_index, "segment_path": seg_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, transcribed, key="transcribed")
        if errors:
            manifest["transcription_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"transcribed": transcribed, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for transcription.
        - diarized: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "diarized": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Transcription Config (JSON)", "default": {}}
        } 