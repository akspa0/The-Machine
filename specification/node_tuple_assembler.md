# Node Specification: Phone Call Tuple Assembler Node

## Purpose
This node groups anonymized audio files into logical phone call tuples (e.g., out, trans_out, recv_out), updates the manifest with tuple structure and lineage, and prepares data for downstream processing.

## Inputs
- Anonymized audio files and manifest from the Raw Input Ingestion Node
- Each file must have a unique tuple_index and type (out, trans_out, recv_out, etc.)

## Outputs
- Updated manifest with tuple groupings and lineage
- (Optionally) reorganized files into tuple-based folders (e.g., /tuples/0000/)

## Core Responsibilities
- Group files by tuple_index, forming tuples (e.g., 0000-out.wav, 0000-trans_out.wav, 0000-recv_out.wav)
- Validate that each tuple contains the required file types (configurable: out, trans_out, recv_out, etc.)
- Update manifest to include a 'tuples' section, e.g.:
  - tuple_index
  - files (list of anonymized filenames/paths)
  - types present
  - tuple_start_time (earliest timestamp in tuple)
  - tuple_end_time (latest timestamp in tuple)
  - lineage (list of processing steps/files)
- Optionally move/copy grouped files into /tuples/<tuple_index>/
- Ensure all manifest updates are PII-free and reference only anonymized fields

## Privacy & PII Logic
- No original filenames, paths, or PII in tuple groupings or manifest
- Validate that all files in a tuple are anonymized before grouping
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Group all files in the batch into tuples by index
- Single-file: Pass through as a single tuple (if only one file)
- Manifest must support both modes (list of tuples for batch, single tuple for single-file)

## Error Handling
- Flag incomplete tuples (missing required types) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include an 'incomplete_tuples' section for any tuples missing files

## UI/UX Notes
- Display tuple groupings and completeness (e.g., checkmarks for complete tuples)
- Teal-themed UI elements for tuple lists and status indicators
- Show manifest summary (number of tuples, completeness, types present)

## Example Manifest Tuple Entry
```json
{
  "tuple_index": "0000",
  "files": [
    "tuples/0000/0000-out.wav",
    "tuples/0000/0000-trans_out.wav",
    "tuples/0000/0000-recv_out.wav"
  ],
  "types_present": ["out", "trans_out", "recv_out"],
  "tuple_start_time": "2025-05-31T23:59:00Z",
  "tuple_end_time": "2025-05-31T24:01:00Z",
  "lineage": []
}
```

## Validation
- Unit tests for tuple grouping, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for tuple completeness and manifest summary display 