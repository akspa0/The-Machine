# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations for the extension-driven, librarian-orchestrator version of The-Machine.

## Current Focus

- The main program (librarian) orchestrates all jobs and data flow, invoking extensions as needed.
- All new features and improvements are implemented as extensions (stacks).
- API-based job/data transfer is the standard for all external integrations (e.g., ComfyUI).
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- All jobs, files, and data are uniquely identified and tracked for traceability.
- Preparing for future database integration to manage jobs, data, and reference lookups.
- **New:** Audio file validation is now based on duration (≥10s) rather than file size, to ensure only valid files are processed for separation. Pre-processing after renaming logs all files, their durations, and validity status. Single-file input metadata is now transferred to output soundbites during finalization.
- All LLM chunking, tokenization, and summarization logic is now unified in `extensions/llm_utils.py`.
- All scripts and extensions import from this single utility; redundant scripts have been removed.
- CLI entry points for chunking/summarization are now via `llm_utils.py`.
- All extensions remain independently runnable and functional.
- Persona builder now strictly uses per-speaker, per-channel, per-call transcripts (≥300 bytes, ≥15s audio).
- Skips and logs insufficient data cases.
- LLM call synopsis is appended if available.
- Utility script for per-speaker transcripts never mixes channels/calls, only processes if ≥5 lines.
- All logging and output structure is robust and clear.
- Debug output added for every decision point.
- Persona builder audio samples are now lossless, using numpy+soundfile to concatenate original .wav files (not _16k.wav), with no resampling or pydub.
- System prompt for persona generation now instructs LLM to be concise, allow for absurdity, and keep responses below 300 tokens.
- All LLM chunking/continuation logic is removed; only direct responses are used.
- Logging and debug output is robust and clear.
- **NEW (2025-06-10):** Introduced `clap_segment.py` one-shot cutter which:
  - Optionally separates a long recording into vocal / instrumental stems.
  - Runs two CLAP passes (speech prompts on vocal, ring/music prompts on instrumental).
  - Merges contiguous speech windows into variable-length segments (≥chunk_length).
  - Saves raw event logs and a manifest in a timestamped folder under `CLAP_jobs/`.
  - Respects privacy rule: no PII in filenames, all logs anonymised.
  - Default confidence raised to 0.54; chunk length default 5 s.
  - Script ≤250 LOC and now the canonical way to cut long recordings before feeding PipelineOrchestrator.
- `cut_and_process.py` updated to call the new cutter and pipe the segments directly into the pipeline.
- **NEW (2025-06-12):** Added FlashSR super-resolution extension (`flashsr_extension.py`) with automatic weight download, GPU batching, and chunked processing. Integrated optional `--flashsr` flag and output folder support. Added profanity Bleeper extension (`bleeper_extension.py`) auto-invoked in finalization to censor first 180 s.

## Recent Changes

- Archived legacy memory bank and regenerated all documentation based on the current codebase and architecture.
- Automated persona manifest generation and prompt caching for downstream workflows.
- Switched all file transfer to API-based methods (no direct file system access for ComfyUI input).
- Improved error handling, logging, and output tracking throughout the pipeline.
- Implemented CLAP speech-only segmentation (speech prompts only) after discovering ring prompts are not always present. Instrumental pass is logged for diagnostics.
- Added `merge_contiguous` logic with dynamic gap based on `chunk_length`, removing prior 10-second uniform segment artefact.
- Default thresholds harmonised across CLAP utilities (0.15 experimental, 0.54 production).

## Next Steps

- Focus on developing and refining extensions for new audio, AI, and data processing capabilities.
- Design and prototype a database-backed registry for jobs, files, and metadata (e.g., SQLite).
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.
- **Next:** Implement duration-based validation (≥10s) for all audio files before separation, log file validity after renaming, and ensure metadata transfer from single-file inputs to output soundbites.
- Leverage instrumental detections as optional hard boundaries to split merged speech segments when ring / hang-up tones exist.
- Consider folding `clap_segment.py` logic into PipelineOrchestrator as a pre-segmentation stage.

## Active Decisions & Considerations

- All new work is focused on extensions and the librarian orchestrator.
- The system is ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions. 