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

# Additional audio formats accepted for --extra-audio-dir ingestion
AUDIO_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

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
        sentences: int = 2,
        gap: float = 0.2,
        preview_len: int = 120,
        retries: int = 3,
        temperature: float = 0.6,
        include_speakers: List[str] | None = None,
        exclude_speakers: List[str] | None = None,
        extra_audio_dirs: List[Path] | None = None,
        out_dir: Path | None = None,
        llm_config: Path | None = None,
        no_diarize: bool = False,
        variations: int = 6,
        split_long_phrases: bool = True,
        max_words: int = 20,
        whisper_model: str = "large-v3",
        verbose: bool = False,
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
        self.extra_audio_dirs = extra_audio_dirs or []
        self.no_diarize = no_diarize
        self.variations = max(1, variations)
        self.split_long_phrases = split_long_phrases
        self.max_words = max_words
        self.whisper_model = whisper_model
        self.verbose = verbose

        # Decide GPU/CPU for Whisper
        try:
            import torch  # type: ignore
            self._whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            self._whisper_device = "cpu"

        self._whisper_model_obj = None  # lazy init

        # Resolve output root
        self.out_root = Path(out_dir) if out_dir else self.output_root / "multi_speaker_collage"
        self.out_root.mkdir(parents=True, exist_ok=True)

        cfg_path = llm_config or Path("workflows/llm_tasks.json")
        self.llm_cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8")) if Path(cfg_path).exists() else {}

        # Will be populated later
        self.global_phrases: List[Dict[str, Any]] = []  # each dict holds file, text, speaker_key
        # Where temp files from extra dirs will be written
        self._temp_dir = self.out_root / "_tmp_external"
        self._temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _slugify(name: str) -> str:
        """Return a filesystem-safe slug limited to ASCII, without trailing spaces/dots."""
        import re, unicodedata

        slug = unicodedata.normalize("NFKD", name)
        slug = slug.encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^A-Za-z0-9\-_.]+", "_", slug)  # replace illegal chars
        slug = slug.strip(" .-_")  # trim problematic trailing chars
        return slug[:64] or "unnamed"

    def run(self):
        self._collect_global_phrases()

        # 1️⃣ Optionally ingest extra audio *before* we validate phrase availability
        if self.extra_audio_dirs:
            added = self._collect_phrases_from_external_dirs()
            self.log(f"Added {added} phrases from extra audio dirs.")

        # 2️⃣ Speaker include/exclude filters ----------------------------------------
        if self.include_speakers:
            self.global_phrases = [
                p for p in self.global_phrases if any(tok in p["speaker"] for tok in self.include_speakers)
            ]
        if self.exclude_speakers:
            self.global_phrases = [
                p for p in self.global_phrases if not any(tok in p["speaker"] for tok in self.exclude_speakers)
            ]

        # Optionally split long phrases before validation ---------------------------
        self._split_long_phrases()

        # 3️⃣ Validate after we tried every source -----------------------------------
        if len(self.global_phrases) < self.phrases_per_sentence:
            self.log("Not enough phrases collected; aborting.")
            return

        # 4️⃣ Shuffle phrases for variety and keep idx mapping consistent ------------
        random.shuffle(self.global_phrases)
        for new_idx, phrase in enumerate(self.global_phrases):
            phrase["idx"] = new_idx

        self.log(
            f"Collected {len(self.global_phrases)} phrases after filtering from {'all runs' if self.all_runs else 'current run'}"
        )

        # ------------------------------------------------------------------
        # Generate one or more collage variants
        # ------------------------------------------------------------------
        existing_plans: set[str] = set()
        for variant_idx in range(1, self.variations + 1):
            # Re-shuffle & re-index for each variation to enhance randomness
            random.shuffle(self.global_phrases)
            for new_idx, phrase in enumerate(self.global_phrases):
                phrase["idx"] = new_idx

            llm_plan = self._create_plan()
            if not llm_plan:
                self.log("[WARN] Falling back to random plan generation …")
                llm_plan = self._create_random_plan()

            if llm_plan in existing_plans:
                # Duplicate plan; try again once with random plan
                llm_plan = self._create_random_plan()
                if llm_plan in existing_plans:
                    self.log("[INFO] Duplicate plan detected; skipping variant.")
                    continue

            existing_plans.add(llm_plan)

            vtag = f"v{variant_idx:02d}"
            plan_path = self.out_root / f"montage_{vtag}.plan"
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
            "You are a dialogue writer crafting a tiny two-line micro-story from a collage of prerecorded "
            "phrases. Each line must read naturally—no obvious jump-cuts or word salad—and together they "
            "should flow like the beginning and end of a short anecdote.\n\n"
            "You will receive a numbered list of phrase previews (only partial text).\n"
            f"Return EXACTLY {self.sentences} lines. Each line must contain {self.phrases_per_sentence} IDs "
            "in the format '#N' separated by spaces, indicating the order the audio clips should be played.\n"
            "Strict rules:\n"
            "1. Use ONLY IDs from the list—no invented numbers.\n"
            "2. Maintain coherent grammar and narrative continuity; it should NOT sound like a patchwork quilt.\n"
            "3. Subject-verb agreement and pronoun consistency are mandatory.\n"
            "4. Output ONLY the ID sequences (no extra words, no punctuation beyond spaces)."
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

    # ------------------------------------------------------------------
    # External audio helpers
    # ------------------------------------------------------------------

    def _collect_phrases_from_external_dirs(self) -> int:
        """Diarize and transcribe loose audio files in *extra_audio_dirs*.

        Returns number of phrases added.
        """
        import whisper  # type: ignore
        diar_available = True
        if not self.no_diarize:
            try:
                from speaker_diarization import batch_diarize, segment_speakers_from_diarization
            except Exception as exc:
                self.log(f"[WARN] Diarization deps missing: {exc} – running in --no-diarize mode.")
                diar_available = False
        else:
            diar_available = False

        count = 0
        for audio_dir in self.extra_audio_dirs:
            for src in Path(audio_dir).rglob('*'):
                if not src.is_file() or src.suffix.lower() not in AUDIO_EXT:
                    continue

                # ------------------------------------------------------------------
                # Caching: reuse prior processing results if available
                # ------------------------------------------------------------------
                safe_stem = self._slugify(src.stem)
                tmp_run = self._temp_dir / safe_stem
                sep_dir = tmp_run / "separated"
                diar_dir = tmp_run / "diarized"
                speak_dir = tmp_run / "speakers"

                if speak_dir.exists() and any(speak_dir.rglob('*.wav')):
                    # Cached diarization / segmentation already present – harvest phrases
                    cached = self._add_phrases_from_cached_segments(speak_dir, src, safe_stem)
                    count += cached
                    if self.verbose and cached:
                        self.log(f"[CACHE] reused {cached} phrases from {src.name}")
                    continue

                # Ensure sub-directories exist for fresh processing path
                for d in (sep_dir, diar_dir, speak_dir):
                    d.mkdir(parents=True, exist_ok=True)

                # Convert non-wav to wav in temp folder
                if src.suffix.lower() != '.wav':
                    conv_wav = (self._temp_dir / safe_stem).with_suffix('.wav')
                    conv_wav.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        'ffmpeg', '-y', '-i', str(src),
                        '-ar', '44100', '-ac', '1', str(conv_wav)
                    ]
                    try:
                        import subprocess
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        wav_path = conv_wav
                    except Exception as exc:
                        if self.verbose:
                            self.log(f"⤫ ffmpeg failed for {src.name}: {exc}")
                        continue
                else:
                    wav_path = src

                # Copy or link original to temp conversation path
                conv_path = sep_dir / f"{wav_path.stem}-conversation.wav"
                import shutil
                try:
                    shutil.copy2(wav_path, conv_path)
                except Exception:
                    continue

                if diar_available:
                    try:
                        from speaker_diarization import batch_diarize, segment_speakers_from_diarization
                        batch_diarize(str(sep_dir), str(diar_dir), min_speakers=2, max_speakers=2, progress=False)
                        segment_speakers_from_diarization(str(diar_dir), str(sep_dir), str(speak_dir), progress=False)
                    except Exception as exc:
                        if self.verbose:
                            self.log(f"⤫ diarization failed for {wav_path.name}: {exc}")
                        diar_available = False  # fallback below

                # Load Whisper once
                wmodel = self._get_whisper_model()

                segment_files = list(speak_dir.rglob("*_16k.wav")) if diar_available else []

                # Fallback: treat whole file as single segment
                if not segment_files:
                    segment_files = [conv_path]

                for seg_wav in segment_files:
                    if seg_wav.is_relative_to(speak_dir):
                        rel = seg_wav.relative_to(speak_dir)
                        speaker_key = str(rel.parent)
                    else:
                        speaker_key = f"external/{wav_path.stem}"

                    txt_path = seg_wav.with_suffix(".txt")
                    if wmodel is not None:
                        try:
                            text = self._transcribe(seg_wav)
                        except Exception as e:
                            if self.verbose:
                                self.log(f"⤫ whisper failed on {seg_wav.name}: {e}")
                            text = ""
                    else:
                        text = ""
                    if not text:
                        import re
                        text = re.sub(r'[\-_]+', ' ', seg_wav.stem).strip() or '(untitled)'
                    try:
                        txt_path.write_text(text, encoding='utf-8')
                    except Exception as e:
                        if self.verbose:
                            self.log(f"⤫ could not write txt for {seg_wav.name}: {e}")
                        pass
                    self.global_phrases.append({
                        "idx": len(self.global_phrases)+1,
                        "text": text,
                        "wav_path": seg_wav,
                        "speaker": speaker_key,
                    })
                    count += 1
                    if self.verbose:
                        self.log(f"✓ added phrase '{text[:40]}' from {seg_wav.name}")
        if self.verbose:
            self.log(f"[VERBOSE] Added {count} phrases from external dirs (total inspected).")
        return count

    # ------------------------------------------------------------------
    def _add_phrases_from_cached_segments(self, speak_dir: Path, src_file: Path, safe_stem: str) -> int:
        """Harvest speaker-segment WAVs + texts from an existing cached directory."""
        added = 0
        segment_files = list(speak_dir.rglob('*.wav'))
        for seg_wav in segment_files:
            rel = seg_wav.relative_to(speak_dir)
            speaker_key = str(rel.parent) if rel.parent != Path('.') else f"external/{safe_stem}"
            txt_path = seg_wav.with_suffix('.txt')
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8').strip()
            else:
                import re
                text = re.sub(r'[\-_]+', ' ', seg_wav.stem).strip() or '(untitled)'
            self.global_phrases.append({
                "idx": -1,
                "text": text,
                "wav_path": seg_wav,
                "speaker": speaker_key,
            })
            added += 1
        return added

    # ------------------------------------------------------------------
    def _split_long_phrases(self):
        """Split entries in self.global_phrases whose text exceeds *max_words* into
        smaller chunks and slice corresponding audio proportionally.
        """
        if not self.split_long_phrases or self.max_words <= 0:
            return

        new_entries: List[Dict[str, Any]] = []
        for phrase in list(self.global_phrases):
            words = phrase["text"].split()
            if len(words) <= self.max_words:
                continue  # keep as-is

            # remove original long item
            self.global_phrases.remove(phrase)

            # Load audio – we'll attempt precise word-timestamp splitting first
            try:
                import soundfile as sf
                audio, sr = sf.read(str(phrase["wav_path"]))
            except Exception as exc:
                self.log(f"[WARN] could not load {phrase['wav_path'].name}: {exc}")
                continue

            # Whisper word-timestamp path ------------------------------------------------
            precise_chunks: List[tuple[float, float, List[str]]] = []
            try:
                wmodel = self._get_whisper_model()
                if wmodel is not None and hasattr(wmodel, "transcribe"):
                    wres = wmodel.transcribe(str(phrase["wav_path"]), word_timestamps=True, verbose=False)

                words_meta = []
                for seg in wres.get("segments", []):
                    words_meta.extend(seg.get("words", []))

                chunk_words: List[str] = []
                start_time: float | None = None
                for w_meta in words_meta:
                    w = w_meta["word"].strip()
                    if start_time is None:
                        start_time = w_meta["start"]
                    chunk_words.append(w)
                    if len(chunk_words) >= self.max_words or w.endswith(('.', '!', '?', ';', ':')):
                        end_time = w_meta["end"]
                        precise_chunks.append((start_time, end_time, chunk_words.copy()))
                        chunk_words.clear()
                        start_time = None
                if chunk_words and start_time is not None:
                    end_time = words_meta[-1]["end"] if words_meta else (len(audio) / sr)
                    precise_chunks.append((start_time, end_time, chunk_words.copy()))
            except Exception:
                precise_chunks = []

            if precise_chunks:
                chunks_src = precise_chunks
            else:
                # Fallback proportional splitting ----------------------------------
                total_samples = len(audio)
                chunks_src = []
                for i in range(0, len(words), self.max_words):
                    part_words = words[i : i + self.max_words]
                    ratio_start = i / len(words)
                    ratio_end = (i + len(part_words)) / len(words)
                    start_time = ratio_start * (total_samples / sr)
                    end_time = ratio_end * (total_samples / sr)
                    chunks_src.append((start_time, end_time, part_words))

            for idx_chunk, (start_t, end_t, chunk_words) in enumerate(chunks_src, start=1):
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                chunk_audio = audio[start_sample:end_sample]

                sub_name = f"{phrase['wav_path'].stem}_sub{idx_chunk:02d}"
                sub_wav = self._temp_dir / f"{self._slugify(sub_name)}.wav"
                try:
                    sf.write(sub_wav, chunk_audio, sr)
                    # also save caption for cache reuse
                    sub_txt = sub_wav.with_suffix('.txt')
                    sub_txt.write_text(" ".join(chunk_words), encoding='utf-8')
                except Exception as exc:
                    self.log(f"[WARN] could not write split wav {sub_name}: {exc}")
                    continue

                new_entries.append({
                    "idx": -1,  # placeholder; will be reindexed later
                    "text": " ".join(chunk_words),
                    "wav_path": sub_wav,
                    "speaker": phrase["speaker"],
                })

        # extend global phrases
        self.global_phrases.extend(new_entries)

    # ------------------------------------------------------------------
    # Whisper helper
    # ------------------------------------------------------------------
    def _get_whisper_model(self):
        if self._whisper_model_obj is not None:
            return self._whisper_model_obj
        try:
            import whisper  # type: ignore
            # First try requested model
            try:
                self._whisper_model_obj = whisper.load_model(self.whisper_model, device=self._whisper_device)
                if self.verbose:
                    self.log(f"[WHISPER] Loaded model '{self.whisper_model}' on {self._whisper_device}")
            except Exception as exc_inner:
                # Fallback to 'base' if requested model unavailable
                fallback = "base"
                self.log(f"[WARN] Could not load Whisper model '{self.whisper_model}' ({exc_inner}). Falling back to '{fallback}'.")
                try:
                    self._whisper_model_obj = whisper.load_model(fallback, device=self._whisper_device)
                    if self.verbose:
                        self.log(f"[WHISPER] Loaded fallback model '{fallback}' on {self._whisper_device}")
                except Exception as exc_fallback:
                    self.log(f"[ERROR] Failed to load fallback Whisper model '{fallback}' ({exc_fallback}). ASR disabled.")
                    self._whisper_model_obj = None
            return self._whisper_model_obj
        except Exception as exc:
            self.log(f"[WARN] Whisper library unavailable ({exc}); proceeding without ASR.")
            self._whisper_model_obj = None
            return None

    # ------------------------------------------------------------------
    def _transcribe(self, wav_path: Path, *, word_ts: bool = False) -> str:
        """Utility wrapper for whisper.transcribe with sane defaults.

        Returns empty string on failure.
        """
        wmodel = self._get_whisper_model()
        if wmodel is None:
            return ""
        try:
            result = wmodel.transcribe(
                str(wav_path),
                word_timestamps=word_ts,
                verbose=False,
                fp16=False,  # using fp32 avoids silent OOM failures
                temperature=0.0,
            )
            txt = result.get("text", "").strip()
            if txt:
                return txt
            # Retry with a bit of temperature and beam search if empty ----------------
            result = wmodel.transcribe(
                str(wav_path),
                word_timestamps=word_ts,
                verbose=False,
                fp16=False,
                temperature=0.2,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
            )
            return result.get("text", "").strip()
        except Exception as exc:
            if self.verbose:
                self.log(f"[WARN] whisper failed on {wav_path.name}: {exc}")
            return ""


# ---------------------------------------------------------------------------
# Stand-alone CLI
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="Create multi-speaker phrase collage from run folder and/or extra audio snippets.")
    p.add_argument("run_folder", type=str, nargs="?", default=".", help="Path to outputs/run-* folder (default: current directory). Use '.' when only --extra-audio-dir is supplied.")
    p.add_argument("--all-runs", action="store_true", help="Include phrase_ts from ALL sibling run-* folders")
    p.add_argument("--distinct-speakers", action="store_true", help="Force each sentence to use phrases from different speakers")
    p.add_argument("--include-speakers", type=str, nargs="*", help="Only keep speakers containing these substrings (ANY match)")
    p.add_argument("--exclude-speakers", type=str, nargs="*", help="Skip speakers containing these substrings (ANY match)")
    p.add_argument("--sentences", type=int, default=2, help="Number of lines in montage (default 2)")
    p.add_argument("--phrases-per-sentence", type=int, default=3, help="Phrases per sentence (default 3)")
    p.add_argument("--gap", type=float, default=0.2, help="Silence gap seconds between phrases (default 0.2)")
    p.add_argument("--preview-len", type=int, default=120, help="Chars shown per phrase preview in prompt")
    p.add_argument("--temperature", type=float, default=0.6, help="LLM temperature (default 0.6)")
    p.add_argument("--retries", type=int, default=3, help="LLM retry attempts if invalid output")
    p.add_argument("--extra-audio-dir", type=str, action="append", help="Add loose audio files from DIR (recursive). Supports multiple.")
    p.add_argument("--out-dir", type=str, help="Custom directory for collage outputs (will be created if absent)")
    p.add_argument("--no-diarize", action="store_true", help="Skip speaker diarization for extra audio clips; treat each file as single phrase.")
    p.add_argument("--variations", type=int, default=6, help="Number of different collage variants to generate (default 6)")
    p.add_argument("--split-long-phrases", action="store_true", help="Split collected phrases longer than --max-words into smaller chunks")
    p.add_argument("--max-words", type=int, default=20, help="Maximum words per phrase when splitting (default 20)")
    p.add_argument("--whisper-model", type=str, default="large-v3", help="Whisper model name (default 'large-v3'). If CUDA available, runs on GPU.")
    p.add_argument("--verbose", action="store_true", help="Verbose diagnostic output")
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
        extra_audio_dirs=[Path(p) for p in (args.extra_audio_dir or [])],
        out_dir=Path(args.out_dir) if args.out_dir else None,
        no_diarize=args.no_diarize,
        variations=args.variations,
        split_long_phrases=args.split_long_phrases,
        max_words=args.max_words,
        whisper_model=args.whisper_model,
        verbose=args.verbose,
    ).run()


if __name__ == "__main__":  # pragma: no cover
    main() 