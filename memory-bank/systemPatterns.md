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
- **Dependency Pins for Transitive Constraints:** When a downstream lib (e.g. NeMo → Numba) imposes a strict version cap, we explicitly pin the upstream library in `requirements.txt` (e.g. `numpy<=2.1.*`) and add an early runtime guard to fail fast if the constraint is violated.

## Design Patterns

- **Librarian/Stacks Model:** The main program is the librarian, extensions are the stacks. All orchestration, job management, and data flow are handled by the librarian.
- **Plug-and-Play Extensions:** Extensions can be added, removed, or updated without modifying the core system.
- **Traceability:** Every job, file, and data artifact is uniquely identified and tracked from ingestion to output.
- **Privacy-First:** No PII is ever logged or output; privacy is enforced at every stage.
- **API-Driven:** All file and job transfers to external tools are done via robust APIs, never direct file system access.
- **Database Integration (Planned):** All jobs, files, and metadata will be referenceable and queryable via a database registry.
- **Lazy Prerequisite Invocation:** Extensions may detect missing prerequisite output (e.g., `phrase_ts/`) and invoke the responsible helper extension automatically, logging any failure but continuing when possible.
- **Canonical Output Tree Contract (2025-06-17):** All downstream stages assume audio stems live under `separated/<call_id>/`. Separation stage MUST honor this regardless of input modality (tuple, conversation, single-file). Any future mode changes must preserve or formally version this contract before merge.
- **New:** Audio file validation is performed by duration (≥10s) before separation, with pre-processing and logging of file validity. Metadata from single-file inputs is transferred to output soundbites during finalization.
- **All LLM chunking, tokenization, and summarization logic is unified in `extensions/llm_utils.py`.**
- **All scripts/extensions import from this single utility; redundant scripts have been removed.**
- **CLI entry points for chunking/summarization are now via `llm_utils.py`.**
- **All extensions remain independently runnable and functional.**
- **ShowAssemblerExtension Pattern (2025-06-21):** Final show creation (balancing calls, tone insertion, compression, per-part titling) is implemented in a dedicated extension (`stage = "finalise.F4"`).  It consumes workflow config and centralises all LLM calls through `LLMTaskManager`.

## Component Relationships

- Librarian orchestrator <-> Extensions (stacks): all job/data flow is managed by the librarian, with extensions providing specialized processing.
- Extensions <-> External tools (e.g., ComfyUI): all communication is via API, with outputs copied back into the project structure.
- All jobs, files, and outputs are tracked in the manifest and (future) database. 