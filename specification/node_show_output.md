# Node Specification: Show Output Node

## Purpose
This node concatenates all valid calls into a single WAV file, inserts tones between calls, generates a text file listing call order, names, timestamps, and metadata, and updates the manifest with show output details and lineage.

## Inputs
- Manifest and remixed call files from the Remixing Node (e.g., /remixed/<tuple_index>/)
- Tones file (tones.wav) for insertion between calls
- All valid calls (duration > 10s, error-free)

## Outputs
- Concatenated show WAV file, stored in /show/
- Call order text file, stored in /show/
- Updated manifest with show output details and lineage

## Core Responsibilities
- Select all valid calls (duration > 10s, error-free) for inclusion
- Concatenate calls in chronological order into a single WAV file
- Insert tones (tones.wav) between calls (optional, configurable)
- Generate a text file listing call order, call start timestamps, and metadata
- Output show files: show.wav, call_order.txt in /show/
- Update manifest with:
    - show_output_file (filename/path)
    - call_order_file (filename/path)
    - calls_included (list of tuple_indices)
    - show_generation_timestamp
    - lineage (add show_output step)
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all valid calls in the batch
- Single-file: Process one call at a time (if only one valid call)
- Manifest must support both modes (list of show outputs for batch, single result for single-file)

## Error Handling
- Flag calls that fail concatenation or are invalid (e.g., too short, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'show_output_errors' section for any failures

## UI/UX Notes
- Allow user to configure tone insertion (on/off, tone file selection)
- Display show file info, call order, and audio preview
- Teal-themed UI elements for show settings, progress bars, and output list
- Show manifest summary (number of calls included, total duration)

## Example Manifest Show Output Entry
```json
{
  "show_output_file": "show/show.wav",
  "call_order_file": "show/call_order.txt",
  "calls_included": ["0000", "0001", "0002"],
  "show_generation_timestamp": "2025-06-01T00:45:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task", "remixing", "show_output"]
}
```

## Validation
- Unit tests for show output generation, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for show settings, output list, and summary 