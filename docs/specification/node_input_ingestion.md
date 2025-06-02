# Node Specification: Raw Input Ingestion Node

## Purpose
This node ingests raw audio files (single or batch), removes all PII from filenames and metadata, assigns unique indices, and creates the initial manifest entries for downstream processing.

## Inputs
- One or more raw audio files (WAV, MP3, or supported formats)
- Optional: metadata files (JSON, CSV, etc.)

## Outputs
- Anonymized audio files (renamed, PII-free, stored in a temp/output folder)
- Initial manifest (JSON/dict) with one entry per ingested file

## Core Responsibilities
- Remove all PII from filenames and audio metadata (ID3 tags, etc.)
- Assign a unique, zero-padded chronological index to each file (e.g., 0000, 0001, ...)
- Rename files to <index>-<type>.<ext> (e.g., 0000-out.wav, 0001-trans_out.wav)
- Store anonymized files in a dedicated output folder (e.g., /renamed/)
- Create a manifest entry for each file with:
  - output_name (anonymized filename)
  - output_path (relative path to anonymized file)
  - tuple_index (chronological index)
  - type (out, trans_out, recv_out, etc.)
  - timestamp (ingestion time, ISO8601)
  - original_ext (original file extension)
  - original_duration (seconds, from audio metadata)
  - sample_rate (from audio metadata)
  - channels (from audio metadata)
  - subid (optional, for multi-part files)
  - lineage (empty or initial, to be updated downstream)
- Write the manifest to disk as /renamed/manifest.json
- Ensure no original filenames, paths, or PII are present in the manifest or output files

## Privacy & PII Logic
- All renaming and metadata scrubbing must occur before any logging or manifest writing
- Use a robust PII detection/removal utility for filenames and audio metadata
- Validate that all output files and manifest entries are PII-free before passing to downstream nodes

## Batch & Single-File Support
- Batch: Accept and process multiple files at once, assigning indices in chronological order (by file timestamp or user-specified order)
- Single-file: Accept and process one file, assigning the next available index
- Manifest must support both modes (list of entries for batch, single entry for single-file)

## Error Handling
- Skip or flag files that cannot be anonymized or have corrupt metadata
- Log only anonymized, non-PII information about errors
- Manifest should include an 'errors' section for any files not processed

## UI/UX Notes
- File picker supports multi-select (batch) and single-file modes
- Display anonymized filenames and indices after processing
- Teal-themed UI elements for buttons, progress bars, and file lists
- Show manifest summary (number of files, types, total duration)

## Example Manifest Entry
```json
{
  "output_name": "0000-out.wav",
  "output_path": "renamed/0000-out.wav",
  "tuple_index": "0000",
  "type": "out",
  "timestamp": "2025-05-31T23:59:00Z",
  "original_ext": ".wav",
  "original_duration": 123.45,
  "sample_rate": 44100,
  "channels": 2,
  "subid": null,
  "lineage": []
}
```

## Validation
- Unit tests for PII removal, renaming, and manifest creation
- Integration tests for batch and single-file workflows
- UI tests for file picker and manifest summary display 