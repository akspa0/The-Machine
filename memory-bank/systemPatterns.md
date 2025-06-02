# systemPatterns.md

**Purpose:**
Documents the system architecture, key technical decisions, design patterns, and component relationships for the extension-driven, librarian-orchestrator version of The-Machine.

## Architecture Overview

- The main program acts as a "librarian" orchestrator, managing all job flow, data, and extension invocation.
- All core logic is implemented as modular extensions ("stacks") that can be independently developed, tested, and swapped.
- All jobs, files, and data are uniquely identified and tracked for full traceability.
- Privacy and PII removal are enforced at every stage.
- All file and job transfers to external tools (e.g., ComfyUI) use robust API-based methods, never direct file system access.
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- Canonical output copying: all outputs are copied from external tools into the project structure with standardized naming and manifest tracking.
- The system is designed for future database integration (e.g., SQLite) to manage jobs, data, and reference lookups.

## Key Technical Decisions

- **Extension-Driven Design:** All new features are implemented as extensions, not as monolithic pipeline logic.
- **API-First Orchestration:** All external integrations (e.g., ComfyUI) use API-based file transfer and job submission.
- **Canonical Output Handling:** All outputs are named and stored in a standardized, traceable way, and tracked in the manifest.
- **Prompt Caching:** Prompts and metadata are cached and reused for downstream workflows, ensuring consistency and traceability.
- **Database-Ready:** The system is architected to support a future database for job/data management and reference lookups.

## Design Patterns

- **Librarian/Stacks Model:** The main program is the librarian, extensions are the stacks. All orchestration, job management, and data flow are handled by the librarian.
- **Plug-and-Play Extensions:** Extensions can be added, removed, or updated without modifying the core system.
- **Traceability:** Every job, file, and data artifact is uniquely identified and tracked from ingestion to output.
- **Privacy-First:** No PII is ever logged or output; privacy is enforced at every stage.
- **API-Driven:** All file and job transfers to external tools are done via robust APIs, never direct file system access.
- **Database Integration (Planned):** All jobs, files, and metadata will be referenceable and queryable via a database registry.
- **New:** Audio file validation is performed by duration (≥10s) before separation, with pre-processing and logging of file validity. Metadata from single-file inputs is transferred to output soundbites during finalization.

## Component Relationships

- Librarian orchestrator <-> Extensions (stacks): all job/data flow is managed by the librarian, with extensions providing specialized processing.
- Extensions <-> External tools (e.g., ComfyUI): all communication is via API, with outputs copied back into the project structure.
- All jobs, files, and outputs are tracked in the manifest and (future) database. 