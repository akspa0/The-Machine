from .utils import privacy
from .utils import manifest as manifest_utils
import os

class RawInputIngestionNode:
    """
    Raw Input Ingestion Node
    Ingests raw audio files (single or batch), removes PII, assigns indices, and creates initial manifest entries. Handles video/audio extraction, tuple grouping, and robust error handling.
    """
    @classmethod
    def input_types(cls):
        return {"input_dir": "str", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"ingested": "list[dict]", "manifest": "dict"}

    def process(self, input_dir, manifest, config=None):
        import shutil
        import subprocess
        from pathlib import Path
        from datetime import datetime
        from tqdm import tqdm
        SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov', '.webm'}
        TUPLE_SUBID = {'left': 'a', 'right': 'b', 'out': 'c'}
        TYPE_MAP = {'left': 'left', 'right': 'right', 'out': 'out'}
        ingested = []
        errors = []
        input_path = Path(input_dir)
        all_files = []
        # --- Discover files ---
        if input_path.is_file():
            ext = input_path.suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                all_files.append((input_path, input_path.name))
            elif ext in VIDEO_EXTENSIONS:
                audio_name = input_path.stem + '.wav'
                audio_path = input_path.parent / audio_name
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', str(input_path),
                        '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)
                    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    all_files.append((audio_path, audio_name))
                except Exception as e:
                    errors.append({"file": str(input_path), "error": str(e)})
        elif input_path.is_dir():
            for root, _, files in os.walk(input_path):
                for file in files:
                    ext = Path(file).suffix.lower()
                    orig_path = Path(root) / file
                    if ext in SUPPORTED_EXTENSIONS:
                        all_files.append((orig_path, file))
                    elif ext in VIDEO_EXTENSIONS:
                        audio_name = Path(file).stem + '.wav'
                        audio_path = Path(root) / audio_name
                        try:
                            subprocess.run([
                                'ffmpeg', '-y', '-i', str(orig_path),
                                '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', str(audio_path)
                            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            all_files.append((audio_path, audio_name))
                        except Exception as e:
                            errors.append({"file": str(orig_path), "error": str(e)})
        else:
            errors.append({"error": f"Input path does not exist or is invalid: {input_dir}"})
        if not all_files:
            errors.append({"error": "No supported audio files found in input."})
        # --- Group files into tuples by timestamp ---
        tuple_groups = {}
        for orig_path, base_name in all_files:
            ts = os.path.getmtime(orig_path)
            ts_str = datetime.fromtimestamp(ts).strftime('%Y%m%d-%H%M%S')
            ext = Path(base_name).suffix.lower()
            file_type = 'out'
            for key in TYPE_MAP:
                if key in base_name:
                    file_type = TYPE_MAP[key]
            tuple_key = ts_str
            if tuple_key not in tuple_groups:
                tuple_groups[tuple_key] = []
            tuple_groups[tuple_key].append((orig_path, base_name, file_type, ts_str, ext))
        # --- Process tuples and create manifest entries ---
        for idx, (ts_str, files) in enumerate(sorted(tuple_groups.items())):
            files_sorted = sorted(files, key=lambda x: ['left', 'right', 'out'].index(x[2]) if x[2] in ['left', 'right', 'out'] else 99)
            for subidx, (orig_path, base_name, file_type, ts_str, ext) in enumerate(files_sorted):
                subid = TUPLE_SUBID.get(file_type, chr(100 + subidx))
                index_str = f"{idx:04d}"
                new_name = f"{index_str}-{subid}-{file_type}-{ts_str}{ext}"
                output_dir = Path('renamed')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / privacy.scrub_filename(new_name)
                shutil.copy2(orig_path, output_path)
                manifest_entry = {
                    "output_name": str(output_path.name),
                    "output_path": str(output_path),
                    "tuple_index": index_str,
                    "type": file_type,
                    "timestamp": ts_str,
                    "original_ext": ext,
                    "original_duration": None,
                    "sample_rate": None,
                    "channels": None,
                    "subid": subid,
                    "lineage": [],
                }
                # Read audio metadata
                try:
                    import soundfile as sf
                    info = sf.info(str(output_path))
                    manifest_entry["original_duration"] = info.duration
                    manifest_entry["sample_rate"] = info.samplerate
                    manifest_entry["channels"] = info.channels
                except Exception:
                    pass
                ingested.append(manifest_entry)
        manifest_utils.update_manifest(manifest, ingested)
        if errors:
            manifest["input_ingestion_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"ingested": ingested, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for input ingestion.
        - input_dir: file/dir picker
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "input_dir": {"type": "file", "label": "Input File or Directory", "mode": "open"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Advanced Config (JSON)", "default": {}}
        } 