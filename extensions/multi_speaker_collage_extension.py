from __future__ import annotations

"""Multi-speaker phrase collage extension.

Builds a surreal audio collage by remixing phrases from *all* speakers (optionally
across every run-* folder) into mixed-speaker sentences.
"""

from pathlib import Path
import sys as _sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from typing import List, Dict, Any
import json
import random
import re
import numpy as np
import soundfile as sf

from extensions.extension_base import ExtensionBase
from extensions.llm_utils import run_llm_task


class MultiSpeakerCollageExtension(ExtensionBase):
    name = "multi_speaker_collage"
    stage = "multi_speaker_collage"
    description = "Creates surreal collage sentences that interleave phrases from many speakers."

    def __init__(
        self,
        output_root: str | Path,
        *,
        all_runs: bool = False,
        distinct_speakers: bool = False,
        phrases_per_sentence: int = 3,
        sentences: int = 10,
        gap: float = 0.3,
        preview_len: int = 120,
        retries: int = 3,
        temperature: float = 0.3,
        llm_config: Path | None = None,
    ) -> None:
        super().__init__(output_root)
        self.output_root = Path(output_root)
        self.all_runs = all_runs
        self.distinct_speakers = distinct_speakers
        self.phrases_per_sentence = phrases_per_sentence
        self.sentences = sentences
        self.gap = gap
        self.preview_len = preview_len
        self.retries = retries
        self.temperature = temperature

        self.out_root = self.output_root / "multi_speaker_collage"
        self.out_root.mkdir(parents=True, exist_ok=True)

        cfg_path = llm_config or Path("workflows/llm_tasks.json")
        self.llm_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8")) if Path(cfg_path).exists() else {}

        # Will be populated later
        self.global_phrases: List[Dict[str, Any]] = []  # each dict holds file, text, speaker_key

    # ------------------------------------------------------------------
    def run(self):
        self._collect_global_phrases()
        if len(self.global_phrases) < self.phrases_per_sentence:
            self.log("Not enough phrases collected; aborting.")
            return

        plan_path = self.out_root / "montage_plan.txt"
        llm_plan = self._create_plan()
        if not llm_plan:
            self.log("[WARN] Falling back to random plan generation …")
            llm_plan = self._create_random_plan()
        plan_path.write_text(llm_plan, encoding="utf-8")
        self.log(f"Wrote {plan_path.relative_to(self.out_root)}")

        self._assemble_audio(llm_plan, plan_path.with_suffix(".wav"))
        self._write_montage_text(llm_plan, plan_path.with_suffix(".txt"))

    # ------------------------------------------------------------------
    def _collect_global_phrases(self):
        runs: List[Path]
        if self.all_runs:
            # If given path is itself a run-* folder, use siblings; otherwise treat it as container of runs
            if self.output_root.name.startswith("run-"):
                runs = [p for p in self.output_root.parent.glob("run-*") if p.is_dir()]
            else:
                runs = [p for p in self.output_root.glob("run-*") if p.is_dir()]
        else:
            # Single run mode – if path is container, need to choose latest? default to path
            runs = [self.output_root]
        idx = 0
        for run_dir in runs:
            phrase_root = run_dir / "phrase_ts"
            if not phrase_root.exists():
                continue
            for phrases_json in phrase_root.glob("*/**/phrases.json"):
                call_id, channel, speaker = phrases_json.relative_to(phrase_root).parts[:3]
                speaker_key = f"{call_id}/{channel}/{speaker}"
                phrases_data = json.loads(phrases_json.read_text(encoding="utf-8"))
                for p in phrases_data:
                    text = p.get("text", "").strip()
                    if not text:
                        continue
                    wav_path = phrases_json.parent / p["file"]
                    if not wav_path.exists():
                        continue
                    self.global_phrases.append({
                        "idx": idx,
                        "text": text,
                        "wav_path": wav_path,
                        "speaker": speaker_key,
                    })
                    idx += 1
        random.shuffle(self.global_phrases)
        self.log(f"Collected {len(self.global_phrases)} phrases from {'all runs' if self.all_runs else 'current run'}.")

    # ------------------------------------------------------------------
    def _create_plan(self) -> str | None:
        # Build numbered list with previews limited for prompt
        previews = [f"#{p['idx']}: {p['text'][: self.preview_len]}" for p in self.global_phrases[:120]]
        user_prompt = "Phrase list:\n" + "\n".join(previews) + "\n\nNow output the montage lines (one per line)."

        system_prompt = (
            f"You will receive a numbered list of phrase previews from many speakers.\n"
            f"Return EXACTLY {self.sentences} lines.\n"
            f"Each line must contain {self.phrases_per_sentence} IDs in the format '#N' separated by spaces.\n"
            "Use only IDs in the list. Output NOTHING else."
        )
        if self.distinct_speakers:
            system_prompt += " Each line should use phrases from *different* speakers if possible."

        llm_base = self.llm_cfg.get("llm_config", {})
        llm_cfg = {**llm_base, "lm_studio_temperature": self.temperature, "lm_studio_max_tokens": 128}

        return self._generate_plan_with_retry(user_prompt, system_prompt, llm_cfg)

    # ------------------------------------------------------------------
    def _generate_plan_with_retry(self, prompt: str, system_prompt: str, llm_cfg: dict) -> str | None:
        original_prompt = prompt
        best_lines: List[str] = []
        for attempt in range(self.retries):
            reply = run_llm_task(prompt, llm_cfg, chunking=False, single_output=True, system_prompt=system_prompt)
            raw_lines = [l.strip() for l in reply.splitlines() if l.strip()]
            valid_lines: List[str] = []
            for l in raw_lines:
                idx_tokens = re.findall(r"#?(\d+)", l)
                if len(idx_tokens) < self.phrases_per_sentence:
                    continue
                # take first N tokens
                idx_tokens = idx_tokens[: self.phrases_per_sentence]
                if self.distinct_speakers:
                    speakers = {self._speaker_of(int(t)) for t in idx_tokens}
                    if len(speakers) < len(idx_tokens):
                        continue
                sanitized_line = " ".join(f"#{t}" for t in idx_tokens)
                valid_lines.append(sanitized_line)
                if len(valid_lines) == self.sentences:
                    break
            if len(valid_lines) >= len(best_lines):
                best_lines = valid_lines
            if len(valid_lines) == self.sentences:
                return "\n".join(valid_lines)
            # Retry reminder
            prompt = original_prompt + "\n\nReminder: Follow the rules exactly and output only the required lines."
        return "\n".join(best_lines) if best_lines else None

    # ------------------------------------------------------------------
    def _speaker_of(self, idx: int) -> str:
        return self.global_phrases[idx]["speaker"] if 0 <= idx < len(self.global_phrases) else ""

    # ------------------------------------------------------------------
    def _assemble_audio(self, plan: str, out_wav: Path):
        lines = plan.splitlines()
        audio_parts: List[np.ndarray] = []
        sr = None
        gap_sil = None
        for line in lines:
            idx_tokens = re.findall(r"#?(\d+)", line)
            for idx_token in idx_tokens[: self.phrases_per_sentence]:
                idx = int(idx_token)
                if idx >= len(self.global_phrases):
                    continue
                wav_path: Path = self.global_phrases[idx]["wav_path"]
                if not wav_path.exists():
                    continue
                audio, sr_ = sf.read(str(wav_path))
                if sr is None:
                    sr = sr_
                    gap_sil = np.zeros(int(sr * self.gap), dtype=audio.dtype)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio_parts.extend([audio, gap_sil])
        if not audio_parts or sr is None:
            self.log("[WARN] No audio assembled – empty parts.")
            return
        final_audio = np.concatenate(audio_parts)
        sf.write(str(out_wav), final_audio, sr)
        self.log(f"Wrote {out_wav.relative_to(self.out_root)}")

    # ------------------------------------------------------------------
    def _write_montage_text(self, plan: str, out_txt: Path):
        lines = plan.splitlines()
        resolved: List[str] = []
        for line in lines:
            idx_tokens = re.findall(r"#?(\d+)", line)
            words: List[str] = []
            for idx_token in idx_tokens[: self.phrases_per_sentence]:
                idx = int(idx_token)
                if idx >= len(self.global_phrases):
                    continue
                text = self.global_phrases[idx]["text"]
                if text:
                    words.append(text)
            if words:
                resolved.append(" ".join(words))
        if resolved:
            out_txt.write_text("\n".join(resolved), encoding="utf-8")
            self.log(f"Wrote {out_txt.relative_to(self.out_root)}")

    # ------------------------------------------------------------------
    def _create_random_plan(self) -> str:
        """Create a random plan locally if LLM fails."""
        selected_lines: List[str] = []
        attempt = 0
        while len(selected_lines) < self.sentences and attempt < 1000:
            attempt += 1
            tokens = []
            used_speakers = set()
            while len(tokens) < self.phrases_per_sentence and len(tokens) < len(self.global_phrases):
                p = random.choice(self.global_phrases)
                if self.distinct_speakers and p["speaker"] in used_speakers:
                    continue
                if f"#{p['idx']}" in tokens:
                    continue
                tokens.append(f"#{p['idx']}")
                used_speakers.add(p["speaker"])
            if len(tokens) == self.phrases_per_sentence:
                selected_lines.append(" ".join(tokens))
        return "\n".join(selected_lines)


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Create multi-speaker phrase collage from phrase_ts libraries.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder (used as current run)")
    p.add_argument("--all-runs", action="store_true", help="Include phrase_ts from ALL sibling run-* folders")
    p.add_argument("--distinct-speakers", action="store_true", help="Force each sentence to use phrases from different speakers")
    p.add_argument("--sentences", type=int, default=10, help="Number of lines in montage (default 10)")
    p.add_argument("--phrases-per-sentence", type=int, default=3, help="Phrases per sentence (default 3)")
    p.add_argument("--gap", type=float, default=0.3, help="Silence gap seconds between phrases")
    p.add_argument("--preview-len", type=int, default=120, help="Chars shown per phrase preview in prompt")
    p.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    p.add_argument("--retries", type=int, default=3, help="LLM retry attempts if invalid output")
    args = p.parse_args()

    MultiSpeakerCollageExtension(
        args.run_folder,
        all_runs=args.all_runs,
        distinct_speakers=args.distinct_speakers,
        phrases_per_sentence=args.phrases_per_sentence,
        sentences=args.sentences,
        gap=args.gap,
        preview_len=args.preview_len,
        temperature=args.temperature,
        retries=args.retries,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main() 