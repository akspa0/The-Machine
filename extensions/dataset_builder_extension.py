from __future__ import annotations
"""DatasetBuilderExtension – aggregates phrases, words, and audio into a clean dataset folder.

Example CLI:
    python extensions/dataset_builder_extension.py outputs/ --dataset-dir dataset/ --all-runs --speaker-map speaker_map.yaml
"""

from pathlib import Path
import sys as _sys
import shutil
import json
from typing import List, Dict, Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from extensions.extension_base import ExtensionBase
from extensions.vocal_separation_extension import VocalSeparationExtension
from extensions.phrase_timestamp_extension import PhraseTimestampExtension
from extensions.word_timestamp_extension import WordTimestampExtension


class DatasetBuilderExtension(ExtensionBase):
    name = "dataset_builder"
    stage = "dataset_builder"
    description = "Aggregates data across runs into a dataset folder with manifests."

    def __init__(
        self,
        source_root: str | Path,
        *,
        dataset_dir: str | Path = "dataset",
        all_runs: bool = False,
        speaker_map_path: str | Path | None = None,
    ) -> None:
        super().__init__(source_root)
        self.src_root = Path(source_root)
        self.dataset_dir = Path(dataset_dir)
        self.all_runs = all_runs
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.dataset_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        self.labels_dir = self.dataset_dir / "labels"
        self.labels_dir.mkdir(exist_ok=True)

        self.speaker_map: Dict[str, str] = {}
        if speaker_map_path and Path(speaker_map_path).exists():
            with open(speaker_map_path, "r", encoding="utf-8") as f:
                if str(speaker_map_path).endswith(('.yml', '.yaml')):
                    self.speaker_map = yaml.safe_load(f) or {}
                else:
                    self.speaker_map = json.load(f)

    # ------------------------------------------------------------------
    def run(self):
        runs = self._discover_runs()
        if not runs:
            self.log("No run-* folders found – aborting dataset build.")
            return

        phrases_out: List[Dict[str, Any]] = []
        words_out: List[Dict[str, Any]] = []

        for run in runs:
            self.log(f"Processing run {run.name} …")
            self._ensure_prereqs(run)
            phrases_root = run / "phrase_ts"
            words_root = run / "word_ts_whisper"
            # Aggregate phrases
            for phrases_json in phrases_root.glob("*/**/phrases.json"):
                call_id, channel, speaker = phrases_json.relative_to(phrases_root).parts[:3]
                spk_key = f"{call_id}/{channel}/{speaker}"
                spk_name = self.speaker_map.get(spk_key, speaker)
                phrases_data = json.loads(phrases_json.read_text(encoding="utf-8"))
                # Ensure audio copies
                dest_dir = self.audio_dir / call_id / speaker
                dest_dir.mkdir(parents=True, exist_ok=True)
                for p in phrases_data:
                    src_wav = phrases_json.parent / p["file"]
                    dst_wav = dest_dir / f"{p['file']}"
                    if not dst_wav.exists():
                        shutil.copy2(src_wav, dst_wav)
                    phrases_out.append({
                        **p,
                        "call_id": call_id,
                        "channel": channel,
                        "speaker": speaker,
                        "speaker_name": spk_name,
                        "wav": str(dst_wav.relative_to(self.dataset_dir)),
                    })
            # Aggregate words
            for words_json in words_root.glob("*/**/words.json"):
                call_id, channel, speaker = words_json.relative_to(words_root).parts[:3]
                spk_key = f"{call_id}/{channel}/{speaker}"
                spk_name = self.speaker_map.get(spk_key, speaker)
                words_data = json.loads(words_json.read_text(encoding="utf-8"))
                words_out.append({
                    "call_id": call_id,
                    "channel": channel,
                    "speaker": speaker,
                    "speaker_name": spk_name,
                    "words": words_data,
                })

        # Write manifests
        (self.labels_dir / "phrases.json").write_text(json.dumps(phrases_out, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.labels_dir / "words.json").write_text(json.dumps(words_out, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.speaker_map:
            (self.labels_dir / "speaker_map.json").write_text(json.dumps(self.speaker_map, indent=2, ensure_ascii=False), encoding="utf-8")
        self.log(f"Dataset build complete. Total phrases: {len(phrases_out)}, speakers: {len(set(p['speaker'] for p in phrases_out))}.")

    # ------------------------------------------------------------------
    def _discover_runs(self) -> List[Path]:
        if self.all_runs:
            if self.src_root.name.startswith("run-"):
                return [p for p in self.src_root.parent.glob("run-*") if p.is_dir()]
            else:
                return [p for p in self.src_root.glob("run-*") if p.is_dir()]
        else:
            return [self.src_root]

    # ------------------------------------------------------------------
    def _ensure_prereqs(self, run: Path):
        # Separation
        VocalSeparationExtension(run).run()
        # Phrase timestamps
        PhraseTimestampExtension(run).run()
        # Word timestamps (whisper)
        WordTimestampExtension(run, engine="whisper").run()


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Aggregate dataset across run folders.")
    p.add_argument("source_root", type=str, help="outputs/ folder or single run-* folder")
    p.add_argument("--dataset-dir", type=str, default="dataset", help="Destination dataset directory")
    p.add_argument("--all-runs", action="store_true", help="Process all run-* folders found under source_root")
    p.add_argument("--speaker-map", type=str, help="YAML or JSON mapping speaker keys to names")
    args = p.parse_args()

    DatasetBuilderExtension(
        args.source_root,
        dataset_dir=args.dataset_dir,
        all_runs=args.all_runs,
        speaker_map_path=args.speaker_map,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main() 