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
- **New:** Audio file validation now uses duration (≥10s) instead of file size, with pre-processing and logging of file validity. Single-file input metadata is transferred to output soundbites during finalization.

## What's Next

- Develop and refine new extensions for audio, AI, and data processing.
- Design and prototype a database-backed registry for jobs, files, and metadata.
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.
- **Next:** Enforce duration-based validation (≥10s) for all audio files before separation, log file validity after renaming, and ensure metadata transfer from single-file inputs to output soundbites.

## Current Status

- The system is fully extension-driven, robust, and ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions. 