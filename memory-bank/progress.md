# progress.md

**Purpose:**
Tracks what works, what's next, and current status for the extension-driven, librarian-orchestrator version of The-Machine.

## What Works

- The main program (librarian) orchestrates all jobs and data flow, invoking extensions as needed.
- All core features are implemented as modular extensions (stacks).
- API-based job/data transfer is the standard for all external integrations (e.g., ComfyUI).
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- All jobs, files, and data are uniquely identified and tracked for traceability.
- Privacy and PII removal are enforced at every stage.
- Canonical output copying: all outputs are copied from external tools into the project structure with standardized naming and manifest tracking.
- The system is ready for future database integration (e.g., SQLite) for job/data/metadata management.
- **Updated:** Speech-only CLAP segmentation via `clap_segment.py` reliably produces variable-length call WAVs. Outputs are stored in timestamped `CLAP_jobs/` folders with anonymised manifests and raw event logs. Cutter integrates optional separation, dual-pass CLAP, and contiguous-speech merging.
- LLM chunking, tokenization, and summarization logic is now unified in `extensions/llm_utils.py`.
- All scripts/extensions import from this single utility; redundant scripts have been removed.
- CLI entry points for chunking/summarization are now via `llm_utils.py`.
- All extensions remain independently runnable and functional.
- Persona builder and transcript utility now robustly enforce per-speaker, per-channel, per-call boundaries.
- FlashSR extension successfully enhances low-quality audio; supports GPU batching, chunk parameters, and custom output directory.
- Bleeper extension automatically censors profanity in first 3 minutes, generates bleeped WAV/MP3.
- Only process transcripts ≥300 bytes and ≥15s audio for persona; skip and log otherwise.
- Utility script only generates transcripts if ≥5 lines, never mixes channels/calls.
- LLM call synopsis appended to persona input if available.
- All logging and debug output is clear and detailed.
- Persona builder audio samples are now lossless, using numpy+soundfile to concatenate original .wav files (not _16k.wav), with no resampling or pydub.
- System prompt for persona generation now instructs LLM to be concise, allow for absurdity, and keep responses below 300 tokens.
- All LLM chunking/continuation logic is removed; only direct responses are used.
- Logging and debug output is robust and clear.

## What's Next

- Develop and refine new extensions for audio, AI, and data processing.
- Design and prototype a database-backed registry for jobs, files, and metadata.
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.
- **Next:** Enforce duration-based validation (≥10s) for all audio files before separation, log file validity after renaming, and ensure metadata transfer from single-file inputs to output soundbites.

## Current Status

- The system is fully extension-driven, robust, and ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions.
- Next iteration: use instrumental detections as hard boundaries and move cutter logic into PipelineOrchestrator. 