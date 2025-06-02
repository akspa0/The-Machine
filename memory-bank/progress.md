# progress.md

**Purpose:**
Tracks what works, what's next, and current status for the extension-driven, librarian-orchestrator version of The-Machine.

## [2024-06-01] Most Recent Update
- Segment/soundbite bug is fixed: all diarized/segmented utterances are now transcribed and output as soundbites in both single-file and batch workflows.
- main.py is now the canonical entry point for all workflows.
- Output structure and manifest logic are unified and robust.

## What Works

- The main program (librarian) orchestrates all jobs and data flow, invoking extensions as needed.
- All core features are implemented as modular extensions (stacks).
- API-based job/data transfer is the standard for all external integrations (e.g., ComfyUI).
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- All jobs, files, and data are uniquely identified and tracked for traceability.
- Privacy and PII removal are enforced at every stage.
- Canonical output copying: all outputs are copied from external tools into the project structure with standardized naming and manifest tracking.
- The system is ready for future database integration (e.g., SQLite) for job/data/metadata management.
- All pipeline stages (ingestion, separation, diarization, segmentation, transcription, soundbite extraction, LLM, remix, show, finalization) are robust for both single-file and batch/call workflows.
- Output directory structure is unified and compatible with all downstream extensions.
- Resume, force, and extension system are stable and tested.

## What's Next

- Develop and refine new extensions for audio, AI, and data processing.
- Design and prototype a database-backed registry for jobs, files, and metadata.
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.

## What's Left to Build

- More extension documentation and usage examples.
- Further edge case testing (multi-speaker, long audio, CLAP-driven segmentation).
- Ongoing improvements to manifest, logging, and traceability.
- Additional extension patterns for downstream tasks.

## Current Status

- The system is fully extension-driven, robust, and ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions.
- ComfyUI node suite: all nodes implemented, requirements and README updated, but project is paused.
- Next steps: debug and refactor `clap_segmentation_experiment.py`, explore Gradio UI for interactive use.
- Known issues: ComfyUI integration not fully functional; segmentation experiment script not working.
- Documentation and requirements are current for all completed work.
- The pipeline is stable and ready for both production and extension development.
- All major bugs affecting segment/soundbite handling are resolved.

## Recent Changes (2024-06-01)

- Single-file/URL pipeline logic is now separated into its own script (`single_file_pipeline.py`) for robust, isolated debugging and processing.
- The orchestrator and single-file logic are now decoupled to prevent accidental breakage and allow focused iteration.
- The next step is to create `main.py` as a unified entry point that handles PII-sanitization/logging and dispatches to the correct pipeline based on input type.
- The current `single_file_pipeline.py` only performs separation; it needs to be extended to run the full pipeline (diarization, transcription, soundbites, LLM, etc.) on the vocals stem.
- The goal is a single script for all workloads, with robust, PII-safe logging and output throughout.

## Known Issues

- Continue to test edge cases and extension compatibility.
- Monitor for any new issues as extensions and workflows evolve. 