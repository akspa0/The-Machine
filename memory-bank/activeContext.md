# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions for The-Machine.

## Current Focus (2025-06-14)

• 🔧 Hot-fix the *separation bug*: every non-tuple file (i.e. lacking `-left-`/`-right-`) is now sent through the vocal separation stage **unless** the filename truly starts with `out-` (legacy whole-conversation input).  This guarantees a proper `*-vocals.wav` track for downstream diarisation.
• 🆕 Added `word_timestamp_extension.py` – Whisper word-level timestamps written to `word_ts/`.
• 📦 Dependencies updated (`faster-whisper`).
• ✅ Memory bank trimmed: only live features & roadmap retained.

## Updates 2025-06-16

• 🛠️ **Dependency hot-fix:** Added `numpy<=2.1.*` pin in `requirements.txt` because NeMo → Numba chain is incompatible with NumPy 2.2 +.  Also added early guard in `transcription.py` that aborts with a clear message when a higher NumPy version is detected.
• 🔄 **ASR fallback path:** `transcription.py` now auto-falls back to Whisper when Parakeet import fails, preventing pipeline aborts.
• 🐢 **Slow-tempo audio caveat:** A recent run used 0.25× speed audio; ASR produced zero transcripts. Decision pending: either pre-restore tempo (sox tempo 4.0) or accept transcript-less flow.
• 🎨 **Collage auto-build:** `MultiSpeakerCollageExtension` now auto-runs `PhraseTimestampExtension` if `phrase_ts/` is absent and aggregates phrases across runs correctly.

## Updates 2025-06-17

• 🐞 **Pipeline blockage for single-file (out) inputs:** Discovered that `add_separation_jobs()` drops stems directly in `separated/` when handling conversation `-out-` files. Downstream stages (normalization → true-peak → diarization → ASR) expect the canonical layout `separated/<call_id>/…`, so they found no inputs and skipped work → zero transcripts and soundbites.
• 🔍 **Root cause:** Directory contract mismatch introduced during tuple-prefix refactor.
• 💡 **Decision point:** (a) Hot-fix: always create `<call_id>` subfolder even for single-file mode (ensures uniform layout). (b) Mid-term: carve tuple-specific paths into an extension or alternate orchestrator to simplify the base pipeline.
• 🚧 **Next Action:** Implement quick subfolder fix, then run regression. Consider splitting tuple logic later.

## Updates 2025-06-20

• 📝 **Show-notes refinement:** `tools/assemble_show_v2.py` now writes track-lists with the canonical format:
  - Header: `# Show Title: …` and `# Part NN of MM   |  Duration: HH:MM:SS`.
  - Call lines: `NN  HH:MM:SS  Call Title` (index, start-time, human title).
  - Tone lines: `[TONE] HH:MM:SS – HH:MM:SS` (en-dash separator, no stray punctuation).
  - Last tone omitted when `--no-tail-tone` is supplied.
• 🎨 Helpers `fmt_call`, `fmt_tone`, `part_header` ensure consistency; extra stray parenthesis removed.
• 📏 File still <750 LOC; audio concat logic untouched.

## Recent Changes

- Extension registry & task manager stable.
- Word-timestamp extension merged; extensions can safely assume granular transcripts.
- Separation logic patched (see code 2025-06-14).

## Recent Changes (addendum)

- Dependency pin + runtime guard implemented (see above).

## Next Steps

1. **Modular pipeline split (Phase-2)** – extract each stage into its own module (e.g. `separation_stage.py`).
2. Introduce YAML pipeline config & stage registry.
3. Continue database prototype after modular refactor lands.

## Next Steps (amendment)

4. Implement optional **tempo restore stage** or CLI flag (`--fix-tempo 4.0`) before separation to handle slowed recordings.

## Next Steps (addendum 2025-06-20)
5. Add automated validation in CI to assert track-list files comply with the new pattern.
6. Wire the show-notes generator (title + synopsis) into finalisation v2 once modularised.

## Active Decisions & Considerations

- Keep privacy/PII rules as hard guardrails.
- Maintain ≤750 LOC per file rule; move helpers to `utils/` as necessary.
- Extensions should never compensate for missing core-pipeline features—fix the pipeline first.

### Extension-First v2 Roadmap (draft)
We're prototyping a new architecture where *finalisation itself* is split into small, pluggable sub-stages.  Each sub-stage can be fulfilled by any registered ExtensionBase subclass.

Planned sub-stage IDs (subject to change):
• F0  pre-filters  – Auto-Editor silence trim, profanity Bleeper, etc.
• F1  loudness     – LUFS and true-peak normalisation.
• F2  remix        – instrumentals/vocals mix & panning.
• F3  tagging      – metadata / manifests / ID3.
• F4  export       – MP3, show concat, archiving.

CLI concepts under consideration (comma-delimited lists):
  --finalise-pre    <ext1,ext2>   inject before built-ins
  --finalise-post   <ext>         run after built-ins
  --skip-calls      0002,0005     (tuple indexes)
  --only-calls      0000-0003     ranges

Nothing is wired yet— we're accumulating extensions first.  The registry & `LLMTaskManager` are already merged and will power this v2 when the orchestrator changes land. 