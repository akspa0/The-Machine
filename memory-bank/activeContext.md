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

## Recent Changes

- Archived legacy memory bank and regenerated all documentation based on the current codebase and architecture.
- Automated persona manifest generation and prompt caching for downstream workflows.
- Switched all file transfer to API-based methods (no direct file system access for ComfyUI input).
- Improved error handling, logging, and output tracking throughout the pipeline.

## Next Steps

- Focus on developing and refining extensions for new audio, AI, and data processing capabilities.
- Design and prototype a database-backed registry for jobs, files, and metadata (e.g., SQLite).
- Continue to improve traceability, privacy, and automation in all extensions.
- Update documentation and memory bank as the extension ecosystem evolves.

## Active Decisions & Considerations

- All new work is focused on extensions and the librarian orchestrator.
- The system is ready for database integration and further extension-driven development.
- Legacy/monolithic pipeline logic is deprecated in favor of modular, API-driven extensions. 