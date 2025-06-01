# cursor-rules.md

## Project Intelligence & Patterns

- **Extension-Driven Development:**
  - All new features and improvements must be implemented as modular extensions (stacks), not as monolithic pipeline logic.
  - Extensions should be plug-and-play, independently testable, and not require changes to the core librarian.

- **Librarian Orchestrator:**
  - The main program (librarian) is responsible for all job orchestration, extension invocation, and data flow.
  - All jobs, files, and data must be uniquely identified and tracked for full traceability.

- **API-First Orchestration:**
  - All external integrations (e.g., ComfyUI) must use robust API-based file transfer and job submission.
  - No direct file system access for external tool input/output.

- **Canonical Output Handling:**
  - All outputs must be copied from external tools into the project structure with standardized naming and manifest tracking.
  - Prompts, metadata, and outputs must be cached and reused for downstream tasks.

- **Privacy and Traceability:**
  - Privacy and PII removal must be enforced at every stage.
  - No PII in filenames, logs, or outputs.
  - All jobs, files, and data must be referenceable and queryable (future: via database).

- **Database-Ready Design:**
  - The system must be architected for future integration with a database (e.g., SQLite) for job/data/metadata management and reference lookups.

- **Documentation and Memory Bank:**
  - The memory bank must always reflect the current codebase, extension ecosystem, and system architecture.
  - Outdated or legacy content should be archived and not referenced in active documentation. 