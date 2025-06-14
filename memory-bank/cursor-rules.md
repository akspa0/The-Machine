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

- **External Tools Pattern:**
  - Third-party projects are cloned into `external_apps/<tool>/` and installed locally via editable pip installs (`pip install -e external_apps/<tool>`). Wrapper scripts (≤250 LOC) live in `tools/` and invoke these local installs. This ensures latest upstream versions while keeping project-relative paths.

- **Extension Registry Pattern:**
  - All extension scripts subclass `ExtensionBase`. During import, subclasses auto-register into a global dictionary keyed by `name` (or class name).
  - Orchestrator can call `extension_base.run_extensions_for_stage(output_root, stage)` to execute all extensions targeting that stage.
  - Each extension declares `stage` metadata so ordering/selection is transparent and configurable.

## Draft rules for finalisation sub-stage extensions (not enforced yet)
• Naming convention: `stage = "finalise.F#"` where `F#` is F0–F4.
• Extension can declare `call_scope = "per_tuple" | "per_call" | "global"`.
• CLI flags (comma-delimited) will map to lists of extension *names*.
• `--skip-calls` / `--only-calls` accept indexes or ranges (e.g., 0001,0003-0005).
• Pipeline must respect privacy rule: skip-list evaluation must happen before any logging. 