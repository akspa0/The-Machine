from .utils import privacy
from .utils import manifest as manifest_utils
import os

class SoundbiteValidationNode:
    """
    Soundbite Validation/Renaming Node
    Validates each soundbite for transcript presence and duration >= 1s. Renames valid soundbites to <index>-<short_transcription>.wav. Updates manifest.
    """
    @classmethod
    def input_types(cls):
        return {"soundbite_files": "list[str]", "transcripts": "list[str]", "manifest": "dict"}

    @classmethod
    def output_types(cls):
        return {"validated_soundbites": "list[dict]", "manifest": "dict"}

    def process(self, soundbite_files, transcripts, manifest):
        import soundfile as sf
        import shutil
        validated = []
        errors = []
        for idx, (wav_path, transcript) in enumerate(zip(soundbite_files, transcripts)):
            try:
                sanitized_path = privacy.scrub_filename(wav_path)
                if not transcript or not transcript.strip():
                    continue
                info = sf.info(sanitized_path)
                duration = info.duration
                if duration < 1.0:
                    continue
                # Rename to <index>-<short_transcription>.wav (≤48 chars, sanitized)
                short_trans = transcript[:48].replace(" ", "_")
                out_dir = os.path.dirname(sanitized_path)
                out_name = f"{idx:04d}-{short_trans}.wav"
                out_path = os.path.join(out_dir, out_name)
                if sanitized_path != out_path:
                    shutil.copy2(sanitized_path, out_path)
                entry = {
                    "tuple_index": f"{idx:04d}",
                    "input_file": sanitized_path,
                    "output_validated": out_path,
                    "transcript": transcript,
                    "duration": duration,
                    "soundbite_validation_timestamp": "2025-06-01T01:10:00Z",
                    "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "speaker_segmentation", "resample_segments", "transcription", "soundbite", "soundbite_validation"]
                }
                validated.append(entry)
            except Exception as e:
                errors.append({"file": wav_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, validated)
        if errors:
            manifest["soundbite_validation_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"validated_soundbites": validated, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for validation settings, preview
        pass 