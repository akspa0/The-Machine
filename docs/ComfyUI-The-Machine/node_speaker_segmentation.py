from .utils import privacy
from .utils import manifest as manifest_utils
import os

class SpeakerSegmentationNode:
    """
    Speaker Segmentation Node
    Segments each diarized file into per-speaker audio files, saving in speakers/<tuple_index>/SXX/. Updates manifest with segment metadata.
    """
    @classmethod
    def input_types(cls):
        return {"diarized_jsons": "list[str]", "audio_files": "list[str]", "manifest": "dict"}

    @classmethod
    def output_types(cls):
        return {"segments": "list[dict]", "manifest": "dict"}

    def process(self, diarized_jsons, audio_files, manifest):
        import json
        import soundfile as sf
        segments_out = []
        errors = []
        for idx, (json_path, audio_path) in enumerate(zip(diarized_jsons, audio_files)):
            try:
                sanitized_audio = privacy.scrub_filename(audio_path)
                with open(json_path, 'r', encoding='utf-8') as f:
                    diarization = json.load(f)
                audio, sr = sf.read(sanitized_audio)
                tuple_index = f"{idx:04d}"
                for seg in diarization.get('segments', []):
                    speaker_id = seg.get('speaker', 'S00')
                    start = float(seg.get('start', 0))
                    end = float(seg.get('end', 0))
                    seg_idx = seg.get('index', 0)
                    out_dir = os.path.join('speakers', tuple_index, speaker_id)
                    os.makedirs(out_dir, exist_ok=True)
                    out_name = f"{seg_idx:04d}-{int(start*100):07d}-{int(end*100):07d}.wav"
                    out_path = os.path.join(out_dir, out_name)
                    start_sample = int(start * sr)
                    end_sample = int(end * sr)
                    sf.write(out_path, audio[start_sample:end_sample], sr)
                    entry = {
                        "tuple_index": tuple_index,
                        "segment_index": f"{seg_idx:04d}",
                        "speaker_id": speaker_id,
                        "start_time": start,
                        "end_time": end,
                        "output_segment": out_path,
                        "speaker_segmentation_timestamp": "2025-06-01T01:00:00Z",
                        "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "speaker_segmentation"]
                    }
                    segments_out.append(entry)
            except Exception as e:
                errors.append({"file": audio_path, "json": json_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, segments_out)
        if errors:
            manifest["speaker_segmentation_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"segments": segments_out, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for segment display, progress
        pass 