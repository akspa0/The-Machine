# Privacy & Manifest Logic Specification

## Purpose
Defines the privacy, PII removal, anti-logging, and manifest logic that must be enforced across all ComfyUI nodes in The-Machine.

## Privacy & PII Removal
- No logging or output of original filenames, paths, or PII at any stage.
- All logging and manifest writing only use anonymized fields:
  - output_name
  - output_path
  - tuple_index
  - subid
  - type
  - timestamp
- Privacy checks are enforced in the earliest node(s) and validated downstream.
- PII removal includes:
  - Renaming files on ingestion
  - Scrubbing metadata (ID3 tags, etc.)
  - Ensuring no PII leaks in logs, manifests, or outputs

## Anti-Logging Logic
- No console or file logging of PII or original filenames/paths.
- All logs must be privacy-compliant and reference only anonymized fields.
- Logging and manifest writing are only permitted after files are fully anonymized and originals are deleted.

## Manifest Schema
- Manifest is a JSON/dict object updated at every stage.
- Tracks:
  - Anonymized filenames
  - Indices (zero-padded, chronological)
  - Processing lineage (input → output mapping)
  - Timestamps
  - Speaker IDs
  - Transcriptions
  - CLAP events
  - LLM outputs
  - All relevant metadata for traceability
- No PII or original filenames/paths are ever stored or logged.

## Manifest Update & Disk Write Rules
- Each node receives and updates the manifest object.
- Manifest is written to disk at every stage (after node processing).
- Batch and single-file support:
  - Batch: Manifest tracks all files/jobs in the batch.
  - Single-file: Manifest tracks the current file/job.
- All nodes must validate manifest privacy and completeness before processing.

## Reusability
- Privacy and manifest logic should be implemented as reusable utilities or base classes for all nodes.
- All nodes must import and use these utilities to ensure consistency.

## Validation
- Unit and integration tests must validate privacy, manifest updates, and correct processing for both batch and single-file workflows. 