# progress.md

**Purpose:**
Tracks what works, what's next, and current status for The-Machine.

## What Works (as of 2025-06-14)

- PipelineOrchestrator reliably processes tuple inputs through all stages.
- Hot-fix applied: non-tuple files now undergo vocal separation automatically, giving proper `*-vocals.wav` stems for diarisation.
- Word-level timestamp extension operational (`word_ts/`).
- Extension registry & task manager driving new plugins (FlashSR, Bleeper, etc.).
- Privacy/PII guardrails enforced; manifest tracking intact.

## What's Next

- Phase-2: modularise each pipeline stage into standalone `*_stage.py` modules; orchestrator becomes a dispatcher.
- YAML pipeline config for easier re-ordering / skipping.
- Prototype SQLite registry once modularisation complete.

## Current Status

System healthy after separation patch; ready for modular refactor. 