# Node Specification: Remixing Node

## Purpose
This node mixes vocals and instrumentals for trans_out and recv_out pairs into left and right channels, applies stereo panning and volume adjustments, and updates the manifest with remixing details and lineage.

## Inputs
- Manifest and separated/normalized audio files from previous nodes (e.g., /normalized/<tuple_index>/)
- Each tuple contains relevant tracks (e.g., trans_out-vocals, trans_out-instrumental, recv_out-vocals, recv_out-instrumental)

## Outputs
- Remixed call files (stereo WAV), stored in /remixed/<tuple_index>/
- Updated manifest with remixing results and lineage

## Core Responsibilities
- For each tuple:
  - Mix vocals and instrumentals for trans_out and recv_out pairs
  - Instrumentals at 50% volume
  - Stereo channels panned 20% from center for a 40% separation effect
  - Output remixed file: <tuple_index>-remixed.wav in /remixed/<tuple_index>/
  - Update manifest with:
    - remixing_parameters (volume, panning, etc.)
    - output_remixed (filename/path)
    - remixing_timestamp
    - lineage (add remixing step)
- Ensure all outputs and manifest updates are PII-free

## Stereo Panning & Volume Logic
- Instrumentals at 50% volume
- Stereo channels panned 20% from center (e.g., left: 60% vocals, 40% instrumentals; right: 40% vocals, 60% instrumentals)
- All vocals used in remix/show must be normalized to -14.0 LUFS
- Configurable parameters for advanced users

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all tuples in the batch
- Single-file: Process one tuple at a time
- Manifest must support both modes (list of remixing results for batch, single result for single-file)

## Error Handling
- Flag tuples that fail remixing (e.g., missing tracks, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'remixing_errors' section for any failures

## UI/UX Notes
- Allow user to adjust remixing parameters (volume, panning) via config panel
- Display remixed file info and audio preview for each tuple
- Teal-themed UI elements for remixing settings, progress bars, and output list
- Show manifest summary (number of remixed files, parameters used)

## Example Manifest Remixing Entry
```json
{
  "tuple_index": "0000",
  "remixing_parameters": {
    "instrumental_volume": 0.5,
    "stereo_pan": 0.2,
    "lufs": -14.0
  },
  "output_remixed": "remixed/0000/0000-remixed.wav",
  "remixing_timestamp": "2025-06-01T00:40:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task", "remixing"]
}
```

## Validation
- Unit tests for remixing, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for remixing settings, output list, and summary 