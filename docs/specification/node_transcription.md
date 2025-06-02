# Node Specification: Transcription Node

## Purpose
This node transcribes each diarized speaker segment using a supported ASR model (e.g., Parakeet), saves .txt files in segment folders, renames segments to include transcription, and updates the manifest and master transcript with transcription data.

## Inputs
- Speaker-segmented audio files and manifest from the Diarization Node
- Each tuple contains segments in /speakers/<tuple_index>/<channel>/SXX/

## Outputs
- Transcription .txt files for each segment, saved in the same folder as the segment audio
- Renamed segment files to include index and a short version of the transcription (≤ 48 characters)
- Updated manifest with transcription results and lineage
- Updated master transcript with transcription events (utterances per speaker)

## Supported Models
- Parakeet (default, HuggingFace implementation)
- Optional: Whisper, custom ASR models (configurable)
- Model selection and parameters are configurable per node instance

## Core Responsibilities
- For each segment audio file:
  - Run ASR model to generate transcription
  - Save transcription as .txt file in the same folder
  - Rename segment audio file to include index and short version of transcription (≤ 48 chars, sanitized)
  - Update manifest with:
    - segment_index
    - speaker_id
    - start_time, end_time
    - output_segment (renamed filename/path)
    - transcription (full text)
    - asr_model (name, version, parameters)
    - transcription_timestamp
    - lineage (add transcription step)
  - Update master transcript with transcription event (format: [SPEAKER][START-END]: text)
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files, manifest, or transcript
- Validate that all outputs are anonymized before updating manifest/transcript
- All manifest and transcript updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all segments/files in the batch
- Single-file: Process one segment/file at a time
- Manifest and transcript must support both modes (list of transcription results for batch, single result for single-file)

## Error Handling
- Flag segments that fail transcription (e.g., model error, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'transcription_errors' section for any failures

## UI/UX Notes
- Allow user to select ASR model and parameters (config panel)
- Display transcription text, segment info, and audio preview for each segment
- Teal-themed UI elements for ASR settings, progress bars, and transcript display
- Show manifest and transcript summary (number of segments, total words, speakers)

## Example Manifest Transcription Entry
```json
{
  "tuple_index": "0000",
  "segment_index": "0000",
  "speaker_id": "S01",
  "start_time": 0.0,
  "end_time": 2.5,
  "output_segment": "speakers/0000/0000-out/S01/0000-Hello_world.wav",
  "transcription": "Hello world.",
  "asr_model": {
    "name": "parakeet",
    "version": "1.0.0",
    "parameters": {}
  },
  "transcription_timestamp": "2025-06-01T00:25:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription"]
}
```

## Example Transcript Transcription Event
```
[S01][0.00-2.50]: Hello world.
```

## Validation
- Unit tests for transcription, manifest/transcript updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for ASR settings, transcript display, and summary 