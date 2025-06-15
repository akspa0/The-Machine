from __future__ import annotations

"""VocalSeparationExtension – runs audio_separation to create stems.

Usage (stand-alone):
    python extensions/vocal_separation_extension.py <run-folder> [--model ckpt] [--all-runs]

When imported by other extensions, instantiate and .run(); it will skip work if
`separated/` already contains *-vocals.wav files.
"""

from pathlib import Path
import sys as _sys
import json
from typing import List, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from extensions.extension_base import ExtensionBase
from audio_separation import separate_audio_file

DEFAULT_MODEL = "mel_band_roformer_vocals_fv4_gabox.ckpt"


class VocalSeparationExtension(ExtensionBase):
    name = "vocal_separation"
    stage = "vocal_separation"
    description = "Runs audio_separation to produce vocals/instrumental stems for each input wav."

    def __init__(
        self,
        output_root: str | Path,
        *,
        model_path: str = DEFAULT_MODEL,
    ) -> None:
        super().__init__(output_root)
        self.output_root = Path(output_root)
        self.model_path = model_path
        self.renamed_dir = self.output_root / "renamed"
        self.separated_dir = self.output_root / "separated"
        self.separated_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self):
        # Detect if work already done
        existing = list(self.separated_dir.glob("*/*-vocals.wav"))
        if existing:
            self.log("Separated stems already exist – skipping separation stage.")
            return

        if not self.renamed_dir.exists():
            self.log("renamed/ folder not found – nothing to separate.")
            return

        inputs = [p for p in self.renamed_dir.iterdir() if p.suffix.lower() == ".wav"]
        if not inputs:
            self.log("No WAV files found in renamed/ – nothing to separate.")
            return

        manifest: List[Dict] = []
        for wav in inputs:
            try:
                out_subdir = self.separated_dir / wav.stem
                out_subdir.mkdir(exist_ok=True)
                result = separate_audio_file(wav, out_subdir, self.model_path)
                manifest.append(result)
            except Exception as e:
                self.log(f"[ERROR] Separation failed for {wav.name}: {e}")

        manifest_path = self.separated_dir / "separation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.log(f"Audio separation complete. Manifest written to {manifest_path.relative_to(self.output_root)}.")


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Run vocal separation on a single run-* folder or all sibling runs.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder OR outputs/ containing many runs")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="audio-separator model checkpoint path")
    p.add_argument("--all-runs", action="store_true", help="Process every run-* folder inside given path (if path is outputs/)")
    args = p.parse_args()

    root_path = Path(args.run_folder)
    runs: List[Path]
    if args.all_runs:
        runs = [p for p in root_path.glob("run-*") if p.is_dir()]
        if not runs:
            print("[WARN] No run-* folders found under", root_path)
            return
    else:
        runs = [root_path]

    for run in runs:
        print(f"[INFO] Separating audio in {run.name} …")
        VocalSeparationExtension(run, model_path=args.model).run()


if __name__ == "__main__":  # pragma: no cover
    main() 