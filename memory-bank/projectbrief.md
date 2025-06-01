# projectbrief.md

**Purpose:**
Defines the core requirements, goals, and architecture for the current, extension-driven version of The-Machine.

## Project Name

The-Machine: Audio Context Librarian

## Overview

The-Machine is a modular, privacy-focused audio processing system. The main program acts as a "librarian" orchestrator, managing a growing set of specialized extensions (the "stacks") for all audio, metadata, and AI-driven tasks. The system is designed for extensibility, traceability, and robust automation.

## Core Requirements

- The main program (librarian) manages all job orchestration, extension invocation, and data flow.
- All core logic is implemented as extensions (stacks) that can be independently developed, tested, and swapped.
- All jobs, files, and data are uniquely identified and tracked for full traceability.
- Privacy and PII removal are enforced at every stage.
- All file and job transfers to external tools (e.g., ComfyUI) use robust API-based methods, never direct file system access.
- Prompts, metadata, and outputs are cached and reused for downstream tasks.
- The system is ready for a future database (e.g., SQLite) to manage jobs, data, and reference lookups like a real library.

## Goals

- Enable rapid development and integration of new extensions for audio, AI, and data processing.
- Ensure all data and jobs are traceable, auditable, and privacy-preserving.
- Provide a robust, automated, and extensible foundation for future research and production use.
- Support a database-backed registry for all jobs, files, and metadata.

## Scope

- In scope: Extension-driven architecture, librarian orchestrator, API-based job/data transfer, privacy enforcement, traceability, database-ready design.
- Out of scope: UI, deployment, legacy pipeline logic, non-extension-based workflows.

## 2025-XX-XX: Project Milestone
- The system is fully extension-driven, robust, and ready for database integration and further extension work. 