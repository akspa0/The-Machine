from __future__ import annotations

"""Word-level timestamp extension for The-Machine.

Runs on a completed outputs/run-* folder.  For every call inside the
`separated/` sub-folder it chooses the vocal track(s) and produces
Whisper-based word-level timestamp JSON + plain-text transcripts.

Designed as an ExtensionBase subclass so it can later be hooked by the
PipelineOrchestrator, yet it is fully runnable as a standalone CLI tool.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import argparse
import sys
import subprocess
import tempfile

# Add project root to sys.path when running directly ------------------------
from pathlib import Path
import sys as _sys
_CURR = Path(__file__).resolve()
_ROOT = _CURR.parent.parent  # project root (one level above extensions/)
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError as exc:
    print("❌ faster-whisper is required:  pip install faster-whisper", file=sys.stderr)
    raise

from extensions.extension_base import ExtensionBase
from extensions.llm_utils import LLMTaskManager  # Optional – only used if --llm-config supplied

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def transcribe_with_whisper(audio_path: Path, model: WhisperModel) -> List[Dict[str, Any]]:
    """Run Whisper with word_timestamps enabled and flatten into a list.

    Returns list of dicts: {word, start, end, conf}
    """
    # segments is a generator; we must iterate.
    results = []
    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )
    for seg in segments:
        for w in seg.words:
            results.append(
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": getattr(w, "probability", None),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Extension class
# ---------------------------------------------------------------------------


class WordTimestampExtension(ExtensionBase):
    """Generate word-level timestamps for each vocal track."""

    name = "word_timestamp"
    stage = "word_timestamp"
    description = "Whisper-based word-level transcripts for separated vocals."

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(
        self,
        output_root: str | Path,
        *,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str = "float16",
        engine: str = "whisper",
        llm_config: Optional[Path] = None,
        **_: Any,
    ) -> None:
        super().__init__(output_root)
        self.model_size = model_size
        self.device = device or ("cuda" if device is None else device)
        self.compute_type = compute_type
        self.llm_config_path = Path(llm_config) if llm_config else None
        self.engine = engine.lower()

        # Lazy-initialise Whisper
        self.model: Optional[WhisperModel] = None  # whisper model (loaded on demand)

        # Prepare out directory
        self.out_root = Path(self.output_root) / f"word_ts_{self.engine}"
        self.out_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def run(self):
        if self.engine == "whisper":
            self._load_whisper()

        speakers_root = Path(self.output_root) / "speakers"
        if not speakers_root.exists():
            self.log(f"No speakers/ folder found at {speakers_root}; nothing to do.")
            return

        processed = 0
        for call_dir in sorted(speakers_root.iterdir()):
            if not call_dir.is_dir():
                continue
            call_id = call_dir.name

            for channel_dir in sorted(call_dir.iterdir()):
                if not channel_dir.is_dir():
                    continue

                for speaker_dir in sorted(channel_dir.iterdir()):
                    if not speaker_dir.is_dir():
                        continue

                    speaker_id = speaker_dir.name

                    # Gather segment WAVs (exclude _16k versions to use original quality)
                    seg_files = sorted([p for p in speaker_dir.glob('*.wav') if not p.name.endswith('_16k.wav')])
                    if not seg_files:
                        continue

                    # Concatenate audio
                    import soundfile as sf
                    import numpy as np
                    import librosa

                    audio_chunks: List[np.ndarray] = []
                    target_sr = None
                    for wav in seg_files:
                        data, sr = sf.read(str(wav))
                        if data.ndim > 1:
                            data = data.mean(axis=1)  # mono simplify
                        if target_sr is None:
                            target_sr = sr
                        if sr != target_sr:
                            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                        audio_chunks.append(data)

                    if not audio_chunks:
                        continue

                    concat_audio = np.concatenate(audio_chunks)
                    # Write temp concatenated file inside out_root
                    out_dir = self.out_root / call_id / channel_dir.name / speaker_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    concat_path = out_dir / "speaker_audio.wav"
                    sf.write(str(concat_path), concat_audio, target_sr)

                    self.log(f"Transcribing {concat_path.relative_to(self.out_root)}")
                    try:
                        if self.engine == "whisper":
                            words = transcribe_with_whisper(concat_path, self.model)
                        else:
                            words = self._transcribe_with_parakeet(concat_path)
                            if not words:  # fallback
                                self.log("[WARN] Parakeet failed; falling back to Whisper for %s" % concat_path.name)
                                self._load_whisper()
                                words = transcribe_with_whisper(concat_path, self.model)
                    except Exception as e:
                        self.log(f"[ERROR] {self.engine} failed on {concat_path.name}: {e}")
                        continue

                    json_path = out_dir / "words.json"
                    txt_path = out_dir / "transcript.txt"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(words, f, indent=2, ensure_ascii=False)
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(" ".join(w["word"] for w in words))

                    meta = {"words": len(words), "model": self.model_size, "segments": len(seg_files)}
                    self._add_manifest_entry(
                        call_id=call_id,
                        channel=f"{channel_dir.name}/{speaker_id}",
                        input_file=concat_path,
                        output_files=[json_path, txt_path],
                        metadata=meta,
                    )
                    processed += 1

        self.log(f"Completed word-timestamp generation for {processed} speaker file(s).")
        # Optional LLM post-processing --------------------------------------
        if self.llm_config_path and self.llm_config_path.exists():
            self._run_llm_tasks()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_manifest_entry(
        self,
        *,
        call_id: str,
        channel: str,
        input_file: Path,
        output_files: List[Path],
        metadata: Dict[str, Any],
    ) -> None:
        entry = {
            "stage": "word_timestamped",
            "call_id": call_id,
            "channel": channel,
            "input_files": [str(input_file)],
            "output_files": [str(p) for p in output_files],
            "metadata": metadata,
            "result": "success",
        }
        # Update manifest in memory and on disk
        if self.manifest is None:
            self.manifest = []
        self.manifest.append(entry)
        manifest_path = Path(self.output_root) / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    # ------------------------------------------------------------------

    def _run_llm_tasks(self):
        self.log("Running LLM post-processing as requested …")
        with open(self.llm_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        llm_cfg = cfg.get("llm_config", {})
        tasks = cfg.get("llm_tasks", [])
        # Build prompts (assume using master transcript)
        mgr = LLMTaskManager(llm_cfg)
        for call_dir in (self.out_root).iterdir():
            if not call_dir.is_dir():
                continue
            call_id = call_dir.name
            for channel_dir in call_dir.iterdir():
                transcript_path = channel_dir / "transcript.txt"
                if not transcript_path.exists():
                    continue
                text = transcript_path.read_text(encoding="utf-8")
                for task in tasks:
                    prompt_template = task.get("prompt_template", "{transcript}")
                    out_file = task.get("output_file", "summary.txt")
                    mgr.add(
                        prompt_template.format(transcript=text),
                        output_path=channel_dir / out_file,
                        **{k: v for k, v in task.items() if k not in {"prompt_template", "output_file"}},
                    )
        mgr.run_all()
        self.log("LLM tasks completed.")

    def _transcribe_with_parakeet(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Use paddlespeech ASR cli to get word timestamps. Fallback: split evenly."""
        try:
            tmp_json = Path(tempfile.gettempdir()) / (audio_path.stem + "_asr.json")
            cmd = [
                "paddlespeech", "asr",
                "--input", str(audio_path),
                "--output", str(tmp_json),
                "--word_time_stamp", "True",
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            if tmp_json.exists():
                data = json.loads(tmp_json.read_text(encoding="utf-8"))
                words = []
                for utt in data.get("result", []):
                    words.extend(utt.get("words", []))
                if words:
                    # format to our schema
                    return [
                        {"word": w[0], "start": w[1], "end": w[2]} for w in words
                    ]
        except Exception as e:
            self.log(f"[WARN] Parakeet ASR failed: {e}")
        # fallback: no data
        return []

    def _load_whisper(self):
        """Load WhisperModel lazily once."""
        if self.model is None:
            self.log("Loading Whisper model …")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    p = argparse.ArgumentParser(description="Generate word-level Whisper transcripts for a run folder.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder")
    p.add_argument("--model-size", default="base", help="Whisper model size or path (default: base)")
    p.add_argument("--device", default="auto", help="cuda, cpu, or auto (default)")
    p.add_argument("--compute-type", default="float16", help="See faster-whisper compute_type (default: float16)")
    p.add_argument("--engine", choices=["whisper", "parakeet"], default="whisper", help="ASR engine for word timestamps")
    p.add_argument("--llm-config", type=str, help="Optional JSON with llm_config + tasks for post-processing")
    args = p.parse_args()

    ext = WordTimestampExtension(
        args.run_folder,
        model_size=args.model_size,
        device=None if args.device == "auto" else args.device,
        compute_type=args.compute_type,
        engine=args.engine,
        llm_config=args.llm_config,
    )
    ext.run()


if __name__ == "__main__":  # pragma: no cover
    main() 