# Product Context

The product provides robust, auditable, and privacy-compliant audio processing for research and production environments. It is designed for modularity, extensibility, and ease of experimentation, enabling users to build, test, and refine audio workflows with strict privacy guarantees and manifest-based traceability.

## Why This Project Exists

- Manual audio processing is slow, error-prone, and privacy-sensitive.
- Existing tools are monolithic, hard to extend, and lack robust traceability.
- There is a need for a modular, extensible, and privacy-first system that can evolve with new AI and audio processing capabilities.

## Product Philosophy

- The main program is a "librarian" that manages all jobs, data, and extensions ("stacks").
- All core features are implemented as extensions, making the system easy to extend, test, and maintain.
- Every job, file, and data artifact is uniquely identified and tracked for full traceability.
- Privacy and PII removal are enforced at every stage.
- All external integrations (e.g., ComfyUI) use robust API-based methods.
- The system is designed to be database-ready, supporting future reference lookups and job/data management.

## User Experience Goals

- Simple, automated processing of large batches of audio and metadata.
- Plug-and-play extension model: users can add, remove, or update extensions without touching the core system.
- Clear, auditable traceability for every job, file, and output.
- Privacy-first: no PII in filenames, logs, or outputs.
- All jobs and data are referenceable and queryable (future: via database).
- CLI and config-driven orchestration for power users and researchers.

## 2025-XX-XX: User Experience Milestone
- The system is modular, robust, and ready for extension-driven research and production use.
- The user experience is focused on automation, traceability, and extensibility, with a clear path to database-backed job/data management.

- The user experience now includes robust LLM chunking/tokenization/summarization for both per-speaker and entire-file outputs.
- Progress reporting for separation and other GPU-accelerated stages is a priority for user feedback and transparency.
- Project stability and extensibility are the result of multiple resets and rebuilds, with robust resume/debug/logging.
- Ongoing focus on user feedback and continuous refinement of the pipeline and extensions.
- Documentation and examples have been updated to reflect these improvements.

- The user experience is now unified for both single-file and batch/call workflows.
- Users launch all workflows via `main.py`, which handles all orchestration, job management, and extension invocation.
- All outputs are consistent, privacy-preserving, and compatible with downstream extensions.
- Recent fixes ensure that all diarized/segmented utterances are processed, transcribed, and included in the outputs and master transcript. 