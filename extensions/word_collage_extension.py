from __future__ import annotations

"""Word-collage LLM extension

Reads `word_ts/` speaker word-timestamp libraries and asks the LLM to
construct surreal, comedic sentences using ONLY the words that actually
have audio.  Results are written to `collage/` inside the same run
folder, one text file per speaker.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import sys as _sys

# Ensure project root on sys.path when run directly -----------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from extensions.extension_base import ExtensionBase
from extensions.llm_utils import LLMTaskManager
from extensions.word_timestamp_extension import WordTimestampExtension


def _clean_words(raw: List[str]) -> List[str]:
    """Filter to alphabetic words, strip punctuation, make unique preserving order."""
    seen = set()
    cleaned: List[str] = []
    for w in raw:
        word = re.sub(r"[^A-Za-z']", "", w)  # keep apostrophes
        if not word:
            continue
        lw = word.lower()
        if lw not in seen:
            seen.add(lw)
            cleaned.append(word)
    return cleaned


class WordCollageExtension(ExtensionBase):
    """Generate surreal audio-collage scripts from word libraries."""

    name = "word_collage"
    stage = "word_collage"
    description = "LLM builds surreal sentences from per-speaker word libraries."

    def __init__(
        self,
        output_root: str | Path,
        *,
        llm_config: Optional[Path] = None,
        sentences: int = 12,
        # Forward params for timestamping if needed
        model_size: str = "base",
        device: str | None = None,
        compute_type: str = "float16",
        **_: Any,
    ) -> None:
        super().__init__(output_root)
        self.sentences = sentences
        self.llm_config_path = Path(llm_config) if llm_config else Path("workflows/llm_tasks.json")
        if not self.llm_config_path.exists():
            raise FileNotFoundError(f"LLM config JSON not found: {self.llm_config_path}")
        self.out_root = Path(self.output_root) / "collage"
        self.out_root.mkdir(parents=True, exist_ok=True)

        # Store Whisper params for potential auto-run of timestamp extension
        self._ts_params = dict(model_size=model_size, device=device, compute_type=compute_type)

    # ------------------------------------------------------------------
    def run(self):
        word_root = Path(self.output_root) / "word_ts"
        if not word_root.exists():
            self.log("word_ts/ not found – running WordTimestampExtension automatically …")
            WordTimestampExtension(
                self.output_root,
                **self._ts_params,
            ).run()
            if not word_root.exists():
                self.log("WordTimestampExtension did not create word_ts; aborting collage.")
                return

        # Load base LLM config
        llm_raw = json.loads(self.llm_config_path.read_text(encoding="utf-8"))
        llm_cfg = llm_raw.get("llm_config", {})

        mgr = LLMTaskManager(llm_cfg)
        task_count = 0

        for speaker_json in word_root.glob("*/**/words.json"):
            # Example path: word_ts/0000/left-vocals/S00/words.json
            rel = speaker_json.relative_to(word_root)
            parts = rel.parts  # (call_id, channel, speaker, words.json)
            if len(parts) < 3:
                continue
            call_id, channel, speaker = parts[0], parts[1], parts[2]
            words_data = json.loads(speaker_json.read_text(encoding="utf-8"))
            words_list = _clean_words([w["word"] for w in words_data if isinstance(w.get("word"), str)])
            if len(words_list) < 10:
                continue  # not enough material

            prompt = (
                "SYSTEM:\n"  # will be turned into system_prompt param
                "You are a Dadaist playwright constructing bizarre, surreal yet *pronounceable* sentences. "
                "RULES: Use ONLY the supplied words, do NOT invent new tokens, you may repeat words, but result must be "
                f"{self.sentences} distinct sentences separated by newline. Keep it weird but grammatically valid."
            )

            user_prompt = "Word library:\n" + ", ".join(words_list)
            out_dir = self.out_root / call_id / channel / speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt = out_dir / "collage.txt"

            mgr.add(
                user_prompt,
                output_path=out_txt,
                system_prompt=prompt,
                single_output=True,
            )
            task_count += 1

            # Manifest entry prepared after run
            meta = {"words": len(words_list), "sentences": self.sentences}
            self._add_manifest(call_id, f"{channel}/{speaker}", str(speaker_json), str(out_txt), meta)

        results = mgr.run_all()
        self.log(f"LLM collage generation finished – {task_count} speaker prompts, {len(results)} responses.")

        # Build WAV collages -------------------------------------------------
        self._build_audio_collages(word_root)

    # ------------------------------------------------------------------
    def _add_manifest(self, call_id: str, channel: str, input_file: str, output_file: str, metadata: Dict[str, Any]):
        entry = {
            "stage": "word_collage",
            "call_id": call_id,
            "channel": channel,
            "input_files": [input_file],
            "output_files": [output_file],
            "metadata": metadata,
            "result": "success",
        }
        if self.manifest is None:
            self.manifest = []
        self.manifest.append(entry)
        (Path(self.output_root) / "manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def _build_audio_collages(self, word_root: Path):
        """Cut word audio and concatenate per speaker to match collage.txt."""
        import soundfile as sf
        import numpy as np

        for collage_txt in self.out_root.glob("*/**/collage.txt"):
            lines = collage_txt.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                continue

            speaker_dir = collage_txt.parent
            words_json_path = speaker_dir / "words.json"
            audio_path = speaker_dir / "speaker_audio.wav"
            if not (words_json_path.exists() and audio_path.exists()):
                continue

            word_entries = json.loads(words_json_path.read_text(encoding="utf-8"))
            # build mapping lower->list[(start,end)]
            word_map: Dict[str, List[tuple[float, float]]] = {}
            for w in word_entries:
                key = re.sub(r"[^A-Za-z']", "", w["word"]).lower()
                if not key:
                    continue
                word_map.setdefault(key, []).append((w["start"], w["end"]))

            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            sentence_audios: List[np.ndarray] = []
            silence = np.zeros(int(sr * 0.15), dtype=audio.dtype)

            for sentence in lines:
                sentence_audio_parts: List[np.ndarray] = []
                for raw in sentence.split():
                    key = re.sub(r"[^A-Za-z']", "", raw).lower()
                    entries = word_map.get(key)
                    if not entries:
                        continue  # skip missing
                    start, end = entries[0]  # first occurrence
                    start_idx = int(start * sr)
                    end_idx = int(end * sr)
                    if start_idx >= end_idx or end_idx > len(audio):
                        continue
                    segment = audio[start_idx:end_idx]
                    sentence_audio_parts.extend([segment, silence])
                if sentence_audio_parts:
                    sentence_audio = np.concatenate(sentence_audio_parts)
                    # a bit longer silence after sentence
                    sentence_audios.extend([sentence_audio, np.zeros(int(sr*0.4), dtype=audio.dtype)])

            if not sentence_audios:
                continue
            final_audio = np.concatenate(sentence_audios)
            out_wav = speaker_dir / "collage.wav"
            sf.write(str(out_wav), final_audio, sr)
            self.log(f"Wrote {out_wav.relative_to(self.out_root)}")
            # update manifest
            self._add_manifest(
                call_id=speaker_dir.parents[3].name,  # word_ts/<call>/<channel>/<speaker>
                channel="/".join(speaker_dir.parts[-3:]),
                input_file=str(audio_path),
                output_file=str(out_wav),
                metadata={"duration_sec": len(final_audio)/sr},
            )

# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Generate surreal LLM collage sentences from word_ts speaker libraries.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder")
    p.add_argument("--llm-config", type=str, help="Optional JSON with llm_config + tasks", default=None)
    p.add_argument("--sentences", type=int, default=12, help="Number of sentences per speaker (default 12)")
    # Ignored but accepted for parity with user call
    p.add_argument("--device", default="auto", help=argparse.SUPPRESS)
    p.add_argument("--model-size", default="base", help="Whisper model size for auto timestamping")
    p.add_argument("--compute-type", default="float16", help="Compute type for Whisper")
    args = p.parse_args()

    ext = WordCollageExtension(
        args.run_folder,
        llm_config=args.llm_config,
        sentences=args.sentences,
        model_size=args.model_size,
        device=None if args.device == "auto" else args.device,
        compute_type=args.compute_type,
    )
    ext.run()


if __name__ == "__main__":  # pragma: no cover
    main() 