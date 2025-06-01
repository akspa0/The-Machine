# Node Specification: Normalization Node

## Purpose
This node normalizes audio files for volume (LUFS), sample rate, and channel configuration. It ensures all model inference uses 16kHz mono, but final outputs use the highest available quality. Updates the manifest with normalization details and lineage.

## Inputs
- Separated audio files and manifest from the Separation Node
- Each tuple contains separated tracks (e.g., vocals, instrumental)

## Outputs
- Normalized audio files (e.g., <index>-<type>-vocals-normalized.wav) for each input, stored in /normalized/<tuple_index>/
- Updated manifest with normalization results and lineage

## Supported Normalization Types/Algorithms
- Volume normalization to -14.0 LUFS (default, configurable)
- Sample rate conversion (default: 16kHz mono for model inference, retain original for final outputs)
- Channel configuration (mono/stereo as required)
- Algorithms: pyloudnorm, ffmpeg, librosa, or custom (configurable)

## Core Responsibilities
- For each separated file:
  - Normalize volume to target LUFS
  - Convert sample rate as required (16kHz mono for inference, original for outputs)
  - Convert to mono or stereo as required
  - Output normalized files: <index>-<type>-vocals-normalized.wav, etc.
  - Store outputs in /normalized/<tuple_index>/
  - Update manifest with:
    - normalization_target (LUFS, sample rate, channels)
    - normalization_algorithm (name, version, parameters)
    - output_normalized (filename/path)
    - normalization_timestamp
    - lineage (add normalization step)
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all tuples/files in the batch
- Single-file: Process one tuple/file at a time
- Manifest must support both modes (list of normalization results for batch, single result for single-file)

## Error Handling
- Flag files that fail normalization (e.g., corrupt audio, algorithm error) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'normalization_errors' section for any failures

## UI/UX Notes
- Allow user to set normalization targets (LUFS, sample rate, channels) via config panel
- Display progress and output file list for each tuple
- Teal-themed UI elements for normalization settings, progress bars, and output lists
- Show manifest summary (number of files processed, normalization targets used)

## Example Manifest Normalization Entry
```json
{
  "tuple_index": "0000",
  "input_file": "separated/0000/0000-out-vocals.wav",
  "normalization_target": {
    "lufs": -14.0,
    "sample_rate": 16000,
    "channels": 1
  },
  "normalization_algorithm": {
    "name": "pyloudnorm",
    "version": "0.1.0",
    "parameters": {"lufs": -14.0}
  },
  "output_normalized": "normalized/0000/0000-out-vocals-normalized.wav",
  "normalization_timestamp": "2025-06-01T00:10:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization"]
}
```

## Validation
- Unit tests for normalization, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for normalization settings, progress, and output display 