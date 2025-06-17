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
        temperature: float = 0.6,
        include_speakers: List[str] | None = None,
        exclude_speakers: List[str] | None = None,
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
        self.include_speakers = include_speakers or []
        self.exclude_speakers = exclude_speakers or []

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

        # Apply speaker include/exclude filters -----------------------------------
        if self.include_speakers:
            self.global_phrases = [p for p in self.global_phrases if any(tok in p["speaker"] for tok in self.include_speakers)]
        if self.exclude_speakers:
            self.global_phrases = [p for p in self.global_phrases if not any(tok in p["speaker"] for tok in self.exclude_speakers)]

        random.shuffle(self.global_phrases)
        self.log(
            f"Collected {len(self.global_phrases)} phrases after filtering from {'all runs' if self.all_runs else 'current run'} "
        )

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
        """Collect phrase candidates.

        Preferred source: *soundbites* directories produced by the stable
        pipeline.  If those are missing we gracefully fall back to the older
        phrase_ts/ library (to remain backward-compatible).
        """
        runs: List[Path]

        def _discover_runs(base: Path) -> List[Path]:
            if base.name.startswith("run-"):
                return [base] if not self.all_runs else [p for p in base.parent.glob("run-*") if p.is_dir()]
            run_candidates = sorted([p for p in base.glob("run-*") if p.is_dir()], key=lambda p: p.name, reverse=True)
            if not run_candidates:
                return [base]
            return run_candidates if self.all_runs else [run_candidates[0]]

        runs = _discover_runs(self.output_root)

        idx = 0
        for run_dir in runs:
            # 1️⃣ Preferred: soundbites/ --------------------------------------------------
            sb_root = run_dir / "soundbites"
            if sb_root.exists():
                for wav in sb_root.glob("**/*.wav"):
                    txt = wav.with_suffix(".txt")
                    if not txt.exists():
                        continue
                    text = txt.read_text(encoding="utf-8").strip()
                    if not text or len(text.split()) < 3:  # skip ultra-shorts
                        continue
                    # speaker key approximation: call/channel/speaker derived from path parts if possible
                    parts = wav.relative_to(sb_root).parts
                    speaker_key = "/".join(parts[:3]) if len(parts) >= 3 else "unknown"
                    self.global_phrases.append({
                        "idx": idx,
                        "text": text,
                        "wav_path": wav,
                        "speaker": speaker_key,
                    })
                    idx += 1

            # 2️⃣ Fallback: phrase_ts/ (legacy) -----------------------------------------
            phrase_root = run_dir / "phrase_ts"
            if phrase_root.exists():
                for phrases_json in phrase_root.glob("*/**/phrases.json"):
                    try:
                        call_id, channel, speaker = phrases_json.relative_to(phrase_root).parts[:3]
                    except ValueError:
                        call_id = channel = speaker = "legacy"
                    speaker_key = f"{call_id}/{channel}/{speaker}"
                    try:
                        phrases_data = json.loads(phrases_json.read_text(encoding="utf-8"))
                    except Exception:  # pragma: no cover – corrupt json
                        continue
                    for p in phrases_data:
                        text = p.get("text", "").strip()
                        if not text:
                            continue
                        wav_path = phrases_json.parent / p.get("file", "")
                        if not wav_path.exists():
                            continue
                        self.global_phrases.append({
                            "idx": idx,
                            "text": text,
                            "wav_path": wav_path,
                            "speaker": speaker_key,
                        })
                        idx += 1

    # ------------------------------------------------------------------
    def _create_plan(self) -> str | None:
        # Build numbered list with previews limited for prompt
        previews = [f"#{p['idx']}: {p['text'][: self.preview_len]}" for p in self.global_phrases[:120]]
        user_prompt = "Phrase list:\n" + "\n".join(previews) + "\n\nNow output the montage lines (one per line)."

        system_prompt = (
            "You are an expert dialogue editor. Your task is to combine short spoken phrases "
            "into surprisingly coherent yet quirky sentences.\n"
            "You will receive a numbered list of phrase previews (not full text).\n"
            f"Return EXACTLY {self.sentences} lines. Each line must contain {self.phrases_per_sentence} IDs "
            "in the format '#N' separated by spaces, representing the order in which the audio clips "
            "should be concatenated.\n"
            "Rules:\n"
            "1. Use ONLY IDs present in the list.\n"
            "2. The resulting sentence should be logically plausible (grammar can be playful).\n"
            "3. Avoid obvious nonsense; tiny surrealism is OK, but maintain subject-verb agreement.\n"
            "4. Do NOT output anything except the lines of ID sequences."
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
                if best_lines and len(best_lines) < self.sentences:
                    # Top-up with random lines so caller always gets the requested count
                    missing = self.sentences - len(best_lines)
                    backup = self._create_random_plan().splitlines()
                    best_lines.extend(backup[:missing])
                return "\n".join(best_lines)
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

        # --- Loudness + True Peak normalization ------------------------------
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(final_audio)
            audio_lufs = pyln.normalize.loudness(final_audio, loudness, -14.0)
            final_audio = pyln.normalize.peak(audio_lufs, -3.0)
            self.log("Applied -14 LUFS + -3 dBTP normalization.")
        except Exception as exc:  # pragma: no cover – pyloudnorm optional
            self.log(f"[WARN] Loudness normalization failed ({exc}); saving raw mix.")

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
    p.add_argument("--include-speakers", type=str, nargs="*", help="Only keep speakers containing these substrings (ANY match)")
    p.add_argument("--exclude-speakers", type=str, nargs="*", help="Skip speakers containing these substrings (ANY match)")
    p.add_argument("--sentences", type=int, default=10, help="Number of lines in montage (default 10)")
    p.add_argument("--phrases-per-sentence", type=int, default=3, help="Phrases per sentence (default 3)")
    p.add_argument("--gap", type=float, default=0.3, help="Silence gap seconds between phrases")
    p.add_argument("--preview-len", type=int, default=120, help="Chars shown per phrase preview in prompt")
    p.add_argument("--temperature", type=float, default=0.6, help="LLM temperature (default 0.6)")
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
        include_speakers=args.include_speakers,
        exclude_speakers=args.exclude_speakers,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main() 