# Node Specification: Export/Database Node

## Purpose
This node exports the manifest and metadata to a database (e.g., SQLite), supports querying and reference lookups, and ensures all exported data is privacy-compliant and traceable.

## Inputs
- Final manifest and metadata from the Show Output Node
- (Optional) Additional metadata or user-supplied reference data

## Outputs
- Populated database file (e.g., manifest.db in /database/)
- (Optional) Exported metadata files (CSV, JSON) for external use
- Updated manifest with export/database details and lineage

## Supported Database Types
- SQLite (default, file-based)
- Optional: PostgreSQL, MySQL, or other DBs (configurable for advanced users)

## Core Responsibilities
- Create or update database schema to store all manifest fields and metadata
- Insert all manifest entries, lineage, and metadata into the database
- Support efficient querying and reference lookups (e.g., by tuple_index, speaker_id, timestamp)
- Export additional metadata files as needed (CSV, JSON)
- Update manifest with:
    - database_file (filename/path)
    - export_timestamp
    - lineage (add export_database step)
- Ensure all exported data and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in database or exported files
- Validate that all exported data is anonymized before writing to database
- All manifest and database updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Export all manifest entries in the batch
- Single-file: Export single manifest entry
- Database schema must support both modes (tables for batch, single row for single-file)

## Error Handling
- Flag entries that fail export (e.g., DB error, schema mismatch) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include an 'export_errors' section for any failures

## UI/UX Notes
- Allow user to select database type and export options (config panel)
- Display export status, database file info, and query interface
- Teal-themed UI elements for export settings, progress bars, and query results
- Show manifest summary (number of entries exported, database file location)

## Example Manifest Export Entry
```json
{
  "database_file": "database/manifest.db",
  "export_timestamp": "2025-06-01T00:50:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task", "remixing", "show_output", "export_database"]
}
```

## Validation
- Unit tests for database export, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for export settings, query interface, and summary 