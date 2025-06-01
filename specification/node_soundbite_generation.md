# Node Specification: Soundbite Generation Node

## Purpose
This node generates soundbites from the highest quality, normalized split vocal audio, based on diarized segments and/or user-defined criteria. It updates the manifest with soundbite lineage and context.

## Inputs
- Speaker-segmented, normalized audio files and manifest from the Transcription Node
- Each tuple contains segments in /speakers/<tuple_index>/<channel>/SXX/
- Optional: user-defined soundbite criteria (e.g., duration, keywords, timestamps)

## Outputs
- Soundbite audio files, stored in /soundbites/<tuple_index>/<channel>/SXX/
- Updated manifest with soundbite results and lineage

## Core Responsibilities
- For each eligible segment (or user-defined region):
  - Cut soundbite from the highest quality, normalized split vocal audio
  - Output soundbite as WAV file: <segment_index>-<short_name>.wav in /soundbites/<tuple_index>/<channel>/SXX/
  - Update manifest with:
    - soundbite_index (chronological, zero-padded)
    - speaker_id (SXX)
    - start_time, end_time (seconds)
    - output_soundbite (filename/path)
    - source_segment (reference to original segment)
    - soundbite_criteria (e.g., keyword, duration)
    - soundbite_timestamp
    - lineage (add soundbite step)
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all eligible segments/files in the batch
- Single-file: Process one segment/file at a time
- Manifest must support both modes (list of soundbite results for batch, single result for single-file)

## Error Handling
- Flag segments that fail soundbite generation (e.g., corrupt audio, invalid criteria) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'soundbite_errors' section for any failures

## UI/UX Notes
- Allow user to define soundbite criteria (duration, keywords, timestamps) via config panel
- Display soundbite list, segment info, and audio preview for each soundbite
- Teal-themed UI elements for soundbite settings, progress bars, and soundbite list
- Show manifest summary (number of soundbites, total duration, speakers)

## Example Manifest Soundbite Entry
```json
{
  "tuple_index": "0000",
  "soundbite_index": "0000",
  "speaker_id": "S01",
  "start_time": 0.0,
  "end_time": 2.5,
  "output_soundbite": "soundbites/0000/0000-out/S01/0000-Hello_world.wav",
  "source_segment": "speakers/0000/0000-out/S01/0000-Hello_world.wav",
  "soundbite_criteria": {"keyword": "hello", "duration": 2.5},
  "soundbite_timestamp": "2025-06-01T00:30:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite"]
}
```

## Validation
- Unit tests for soundbite generation, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for soundbite settings, soundbite list, and summary 