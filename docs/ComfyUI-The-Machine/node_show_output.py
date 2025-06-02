from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

class ShowOutputNode:
    """
    Show Output Node
    Concatenates valid calls into a single WAV, inserts tones, generates call order text file, updates manifest, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"remixed": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"show_output": "dict", "manifest": "dict"}

    def process(self, remixed, manifest, config=None):
        from tqdm import tqdm
        from pydub import AudioSegment
        show_dir = Path('show_output')
        show_dir.mkdir(exist_ok=True)
        tones_path = config.get("tones_path", "tones.wav") if config else "tones.wav"
        min_duration = config.get("min_duration", 10.0) if config else 10.0
        show_wav = show_dir / "show.wav"
        call_order_txt = show_dir / "call_order.txt"
        errors = []
        # Load tones if available
        try:
            tones = AudioSegment.from_wav(tones_path)
        except Exception:
            tones = None
        # Concatenate valid calls
        calls = []
        call_order = []
        for entry in tqdm(remixed, desc="Show output"):
            try:
                audio = AudioSegment.from_wav(entry["remixed_path"])
                duration = len(audio) / 1000.0
                if duration < min_duration:
                    continue
                calls.append(audio)
                call_order.append(f"{entry['tuple_index']}\t{entry['speaker']}\t{duration:.2f}s")
            except Exception as e:
                errors.append({"remixed_path": entry["remixed_path"], "error": str(e)})
        # Insert tones between calls
        if calls:
            output = calls[0]
            for call in calls[1:]:
                if tones:
                    output += tones
                output += call
            output.export(show_wav, format="wav")
        # Write call order file
        with open(call_order_txt, "w", encoding="utf-8") as f:
            for line in call_order:
                f.write(line + "\n")
        show_output = {
            "show_wav": str(show_wav),
            "call_order_txt": str(call_order_txt),
            "call_count": len(calls),
            "lineage": [e["remixed_path"] for e in remixed],
        }
        manifest_utils.update_manifest(manifest, [show_output], key="show_output")
        if errors:
            manifest["show_output_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"show_output": show_output, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for show output.
        - remixed: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "remixed": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "Show Output Config (JSON)", "default": {}}
        } 