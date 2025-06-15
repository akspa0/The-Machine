from __future__ import annotations

"""Phrase collage extension.

Uses phrase_ts library to build surreal sentence sequences by concatenating
whole phrase WAVs.  Uses the LLM to choose phrase order.
"""

from pathlib import Path
import sys as _sys

# ensure project root path before extension imports
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from typing import Any, List, Dict
import json
import random
import re
import numpy as np
import soundfile as sf

from extensions.extension_base import ExtensionBase
from extensions.llm_utils import LLMTaskManager, run_llm_task
from extensions.phrase_timestamp_extension import PhraseTimestampExtension

class PhraseCollageExtension(ExtensionBase):
    name = "phrase_collage"
    stage = "phrase_collage"
    description = "Creates surreal audio collage using whole phrases."

    def __init__(
        self,
        output_root: str | Path,
        *,
        llm_config: Path | None = None,
        phrases_per_sentence: int = 3,
        sentences: int = 10,
        gap: float = 0.3,
        preview_len: int = 120,
        retries: int = 3,
        temperature: float = 0.3,
    ) -> None:
        super().__init__(output_root)
        self.output_root = Path(output_root)
        self.phrase_root = self.output_root / "phrase_ts"
        self.out_root = self.output_root / "phrase_collage"
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.phrases_per_sentence = phrases_per_sentence
        self.sentences = sentences
        self.gap = gap
        self.preview_len = preview_len
        self.retries = retries
        self.temperature = temperature
        cfg_path = llm_config or Path("workflows/llm_tasks.json")
        self.llm_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8")) if Path(cfg_path).exists() else {}

    # ------------------------------------------------------------------
    def run(self):
        if not self.phrase_root.exists():
            self.log("phrase_ts/ not found – running PhraseTimestampExtension automatically …")
            PhraseTimestampExtension(self.output_root).run()
            if not self.phrase_root.exists():
                self.log("PhraseTimestampExtension did not create phrase_ts; aborting collage.")
                return

        base_llm_cfg = self.llm_cfg.get("llm_config", {})
        task_count = 0

        for phrases_json in self.phrase_root.glob("*/**/phrases.json"):
            phrases_data = json.loads(phrases_json.read_text(encoding="utf-8"))
            if len(phrases_data) < self.phrases_per_sentence:
                continue
            call_id, channel, speaker = phrases_json.relative_to(self.phrase_root).parts[:3]

            # Build prompt list (shortened text)
            phrase_texts = [p.get("text", "")[: self.preview_len] for p in phrases_data]
            random.shuffle(phrase_texts)

            system_prompt = (
                f"You will receive a numbered list of phrase previews.\n"
                f"Return EXACTLY {self.sentences} lines.\n"
                f"Each line must contain {self.phrases_per_sentence} IDs in the format '#N' separated by spaces (e.g., '#3 #7 #1').\n"
                "Use only IDs that appear in the list. Output NOTHING except the lines of IDs."
            )

            # limit list size for prompt brevity
            prompt_limit = 40
            if len(phrase_texts) > prompt_limit:
                phrase_texts = phrase_texts[:prompt_limit]
            numbered_list = "\n".join(f"#{i}: {t}" for i, t in enumerate(phrase_texts))
            user_prompt = "Phrase list:\n" + numbered_list + "\n\nNow output the montage lines (one per line)."
            out_dir = self.out_root / call_id / channel / speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            plan_txt = out_dir / "montage_plan.txt"

            llm_cfg_local = {**base_llm_cfg, "lm_studio_temperature": self.temperature, "lm_studio_max_tokens": 128}
            reply = self._generate_plan_with_retry(user_prompt, system_prompt, llm_cfg_local) or ""

            if reply:
                plan_txt.write_text(reply, encoding="utf-8")
                task_count += 1
            else:
                self.log(f"[WARN] Could not obtain valid plan for {call_id}/{channel}/{speaker}")

        self.log(f"Generated {task_count} montage plans; assembling audio …")
        self._assemble_audio()

        # After assembling audio, also create human-readable montage text files
        self._write_montage_text()

    # ------------------------------------------------------------------
    def _assemble_audio(self):
        for plan_file in self.out_root.glob("*/**/montage_plan.txt"):
            lines = [l.strip() for l in plan_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            rel = plan_file.relative_to(self.out_root)
            call_id, channel, speaker = rel.parts[:3]
            phrase_dir = self.phrase_root / call_id / channel / speaker
            phrases_data = json.loads((phrase_dir / "phrases.json").read_text(encoding="utf-8"))
            sr = None
            audio_parts: List[np.ndarray] = []
            gap_sil = None
            for line in lines:
                import re
                idx_tokens = re.findall(r"#(\d+)", line)
                if len(idx_tokens) < self.phrases_per_sentence:
                    continue
                for idx_token in idx_tokens[: self.phrases_per_sentence]:
                    if not idx_token.isdigit():
                        continue
                    idx = int(idx_token)
                    if idx >= len(phrases_data):
                        continue
                    wav_name = phrases_data[idx]["file"]
                    wav_path = phrase_dir / wav_name
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
                continue
            final_audio = np.concatenate(audio_parts)
            out_wav = plan_file.parent / "phrase_collage.wav"
            sf.write(str(out_wav), final_audio, sr)
            self.log(f"Wrote {out_wav.relative_to(self.out_root)}")

    # ------------------------------------------------------------------
    def _generate_plan_with_retry(self, prompt: str, system_prompt: str, llm_cfg: dict) -> str | None:
        original_prompt = prompt  # keep full phrase list for retries
        for attempt in range(self.retries):
            reply = run_llm_task(prompt, {**llm_cfg, "lm_studio_temperature": self.temperature}, chunking=False, single_output=True, system_prompt=system_prompt)
            raw_lines = [l.strip() for l in reply.splitlines() if l.strip()]
            valid_lines = []
            for l in raw_lines:
                if re.fullmatch(r"(#[0-9]+\s+){%d}#[0-9]+" % (self.phrases_per_sentence-1), l):
                    valid_lines.append(l)
                if len(valid_lines) == self.sentences:
                    break
            if len(valid_lines) == self.sentences:
                return "\n".join(valid_lines)
            # Otherwise retry with the same phrase list but an additional reminder
            prompt = original_prompt + "\n\nReminder: Follow the rules exactly and output only the required lines."
        return None

    # ------------------------------------------------------------------
    def _write_montage_text(self) -> None:
        """Translate each montage_plan into plain text sentences and write montage_text.txt."""
        for plan_file in self.out_root.glob("*/**/montage_plan.txt"):
            lines = [l.strip() for l in plan_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                continue
            rel = plan_file.relative_to(self.out_root)
            call_id, channel, speaker = rel.parts[:3]
            phrase_dir = self.phrase_root / call_id / channel / speaker
            phrases_path = phrase_dir / "phrases.json"
            if not phrases_path.exists():
                continue
            phrases_data = json.loads(phrases_path.read_text(encoding="utf-8"))

            resolved_sentences: List[str] = []
            for line in lines:
                idx_tokens = re.findall(r"#(\d+)", line)
                if len(idx_tokens) < self.phrases_per_sentence:
                    continue
                words: List[str] = []
                for idx_token in idx_tokens[: self.phrases_per_sentence]:
                    idx_int = int(idx_token)
                    if idx_int >= len(phrases_data):
                        continue
                    text = phrases_data[idx_int].get("text", "").strip()
                    if text:
                        words.append(text)
                if words:
                    resolved_sentences.append(" ".join(words))

            if not resolved_sentences:
                continue

            text_path = plan_file.parent / "montage_text.txt"
            text_path.write_text("\n".join(resolved_sentences), encoding="utf-8")
            self.log(f"Wrote {text_path.relative_to(self.out_root)}")

# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser(description="Create phrase collage from phrase_ts library.")
    p.add_argument("run_folder", type=str, help="Path to outputs/run-* folder")
    p.add_argument("--sentences", type=int, default=10, help="Number of lines in montage (default 10)")
    p.add_argument("--phrases-per-sentence", type=int, default=3, help="Phrases per line (default 3)")
    p.add_argument("--gap", type=float, default=0.3, help="Silence gap seconds between phrases (default 0.3)")
    p.add_argument("--preview-len", type=int, default=120, help="Max chars shown per phrase preview")
    p.add_argument("--temperature", type=float, default=0.3, help="LLM sampling temperature (default 0.3)")
    p.add_argument("--retries", type=int, default=3, help="Max retries if LLM output invalid")
    args = p.parse_args()

    PhraseCollageExtension(
        args.run_folder,
        sentences=args.sentences,
        phrases_per_sentence=args.phrases_per_sentence,
        gap=args.gap,
        preview_len=args.preview_len,
        temperature=args.temperature,
        retries=args.retries,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main() 