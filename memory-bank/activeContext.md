# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions for The-Machine.

## Current Focus (2025-06-14)

• 🔧 Hot-fix the *separation bug*: every non-tuple file (i.e. lacking `-left-`/`-right-`) is now sent through the vocal separation stage **unless** the filename truly starts with `out-` (legacy whole-conversation input).  This guarantees a proper `*-vocals.wav` track for downstream diarisation.
• 🆕 Added `word_timestamp_extension.py` – Whisper word-level timestamps written to `word_ts/`.
• 📦 Dependencies updated (`faster-whisper`).
• ✅ Memory bank trimmed: only live features & roadmap retained.

## Recent Changes

- Extension registry & task manager stable.
- Word-timestamp extension merged; extensions can safely assume granular transcripts.
- Separation logic patched (see code 2025-06-14).

## Next Steps

1. **Modular pipeline split (Phase-2)** – extract each stage into its own module (e.g. `separation_stage.py`).
2. Introduce YAML pipeline config & stage registry.
3. Continue database prototype after modular refactor lands.

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