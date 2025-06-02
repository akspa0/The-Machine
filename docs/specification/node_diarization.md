# Node Specification: Diarization Node

## Purpose
This node performs speaker diarization on normalized vocal audio using a custom pyannote-based implementation, segments audio per speaker, and updates the manifest and master transcript with speaker information and segment metadata.

## Inputs
- Normalized vocal audio files and manifest from the CLAP Annotation Node
- Each tuple contains normalized vocal tracks (e.g., <index>-<type>-vocals-normalized.wav)

## Outputs
- Speaker-segmented audio files for each speaker, stored in /speakers/<tuple_index>/<channel or conversation>/SXX/
- Updated manifest with diarization results, segment metadata, and lineage
- Updated master transcript with diarization events (utterances per speaker)

## Diarization Implementation
- Custom pyannote-based diarization (not using off-the-shelf wrappers)
- Configurable parameters: model checkpoint, min/max segment length, overlap handling, VAD settings
- Segments normalized vocals per speaker, stores in speakers/SXX/ folders
- Each segment is named with a chronological index and a short, meaningful name (≤ 48 characters)

## Core Responsibilities
- For each normalized vocal file:
  - Run diarization to detect speaker segments
  - Output segments as WAV files: <segment_index>-<start>-<end>.wav in /speakers/<tuple_index>/<channel>/SXX/
  - Name each segment with index and short, meaningful name (≤ 48 chars)
  - For each segment, generate a manifest entry with:
    - segment_index (chronological, zero-padded)
    - speaker_id (SXX)
    - start_time, end_time (seconds)
    - output_segment (filename/path)
    - channel or conversation
    - diarization_model (name, version, parameters)
    - diarization_timestamp
    - lineage (add diarization step)
  - Update master transcript with diarization events (format: [CHANNEL][SPEAKER][START-END])
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files, manifest, or transcript
- Validate that all outputs are anonymized before updating manifest/transcript
- All manifest and transcript updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all tuples/files in the batch
- Single-file: Process one tuple/file at a time
- Manifest and transcript must support both modes (list of diarization results for batch, single result for single-file)

## Error Handling
- Flag files that fail diarization (e.g., model error, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'diarization_errors' section for any failures

## UI/UX Notes
- Allow user to configure diarization parameters (model, segment length, VAD, etc.)
- Display detected speakers, segment boundaries, and audio previews for each file
- Teal-themed UI elements for diarization settings, progress bars, and segment lists
- Show manifest and transcript summary (number of speakers, segments, total duration)

## Example Manifest Diarization Entry
```json
{
  "tuple_index": "0000",
  "input_file": "normalized/0000/0000-out-vocals-normalized.wav",
  "diarization_model": {
    "name": "pyannote-custom",
    "version": "2025.06.01",
    "parameters": {"min_segment": 0.5, "max_segment": 30.0}
  },
  "segments": [
    {
      "segment_index": "0000",
      "speaker_id": "S01",
      "start_time": 0.0,
      "end_time": 2.5,
      "output_segment": "speakers/0000/0000-out/S01/0000-0000000-0000250.wav"
    }
  ],
  "diarization_timestamp": "2025-06-01T00:20:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization"]
}
```

## Example Transcript Diarization Event
```
[OUT][S01][0.00-2.50]
```

## Validation
- Unit tests for diarization, manifest/transcript updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for diarization settings, segment display, and summary 