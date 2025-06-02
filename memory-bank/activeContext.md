# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations for the extension-driven, librarian-orchestrator version of The-Machine.

## Current Focus

- Ensuring robust, unified audio processing for both single-file and batch/call workflows.
- Supporting advanced LLM chunking, tokenization, and summarization for both per-speaker and entire-file outputs.
- Improving user feedback and progress reporting, especially for GPU-accelerated stages like separation.
- Maintaining strict privacy, traceability, and manifest requirements.
- Supporting modular extension-driven architecture.
- The main program (librarian) orchestrates all jobs and data flow, invoking extensions as needed.
- All new features and improvements are implemented as extensions (stacks).
- API-based job/data transfer is the standard for all external integrations (e.g., ComfyUI).
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- All jobs, files, and data are uniquely identified and tracked for traceability.
- Preparing for future database integration to manage jobs, data, and reference lookups.
- The ComfyUI node suite is feature-complete and fully documented, but further work is paused.
- Current focus: debugging and refining `clap_segmentation_experiment.py` for robust CLAP-based segmentation, and exploring a Gradio interface for interactive experimentation.
- All requirements, setup, and documentation for the node suite are current and ready for future resumption.
- Experimentation and refinement of new tools and interfaces are encouraged.

## Recent Changes

- Archived legacy memory bank and regenerated all documentation based on the current codebase and architecture.
- Automated persona manifest generation and prompt caching for downstream workflows.
- Switched all file transfer to API-based methods (no direct file system access for ComfyUI input).
- Improved error handling, logging, and output tracking throughout the pipeline.
- Fixed a critical bug in soundbite and transcript handling: all diarized/segmented utterances are now correctly transcribed, output, and included in the master transcript for both single-file and batch workflows.
- Updated the pipeline to always use the correct call/job index for output directory structure, preventing segment lookup errors.
- The canonical entry point for all workflows is now `main.py` (do not use `pipeline_orchestrator.py` directly).
- README and user instructions updated to reflect this change.
- Resume, force, and extension system remain robust and tested.
- LLM handling logic is being enhanced to use chunking/tokenization/summarization (see extensions/llm_tokenize.py and llm_summarize.py).
- Multiple project resets/rebuilds have led to a stable, extensible pipeline.

## Next Steps

- Continue developing and documenting extension patterns for downstream tasks (e.g., persona generation, image generation).
- Further test edge cases (multi-speaker, long audio, CLAP-driven segmentation).
- Refine manifest and logging for even greater traceability.
- Gather user feedback on new workflow and extension system.
- Focus on developing and refining extensions for new audio, AI, and data processing capabilities.
- Design and prototype a database-backed registry for jobs, files, and metadata (e.g., SQLite).
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.
- Integrate advanced LLM chunking/caching for large transcripts.
- Improve progress reporting for all sub-tasks, especially GPU-accelerated ones.
- Continue refining user experience and extension compatibility.

## Active Decisions & Considerations

- All new work is focused on extensions and the librarian orchestrator.
- The system is ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions.
- All user-facing documentation and examples now reference `main.py` as the entry point.
- All outputs, logs, and manifests are strictly PII-free and fully auditable.
- Extensions must be robust to both batch and single-file workflows.

## Architectural Update (2024-06-01)

- The project is moving to a unified entry point (`main.py`) for all audio processing workflows.
- All PII-sanitization and logging setup will be handled at the very start of `main.py`.
- `main.py` will parse CLI arguments and dispatch to the appropriate pipeline:
  - Tuple/call pipeline (the orchestrator)
  - Single-file/URL pipeline (new, robust, and isolated logic)
- The single-file/URL pipeline is now separated into its own script (`single_file_pipeline.py`) for robust, isolated debugging and processing.
- The orchestrator and single-file logic are now decoupled to prevent accidental breakage and to allow focused debugging.
- The next step is to create `main.py` as the single entry point for all workloads, ensuring all outputs and logs are PII-safe and that the user only needs to run one script for any workflow.
- single_file_orchestrator.py now always creates a run-<timestamp> folder in outputs/ for every run, mirroring the structure of pipeline_orchestrator.py.
- All subfolders (raw_inputs, separated, etc.) are created inside this run folder, and all file movement, naming, and manifest logic is handled internally.
- However, downstream stages (diarization, segmentation, transcription) are not finding or processing the separated vocals stem, resulting in empty results.
- The next step is to add robust debug logging before each stage and ensure all downstream stages are pointed at the correct input file(s) and structure for single-file runs.
- Batch functions (e.g., batch_diarize) may need to be adapted or replaced with single-file logic for robust single-file workflows.
- main.py now only dispatches to the orchestrator; it does not dictate file/folder structure. 