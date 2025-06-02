# Node Specification: Separation Node

## Purpose
This node performs vocal/instrumental separation on each audio file in a tuple, outputs separated tracks, and updates the manifest with lineage and processing details.

## Inputs
- Tuple-based anonymized audio files and manifest from the Tuple Assembler Node
- Each tuple contains one or more audio files (e.g., out, trans_out, recv_out)

## Outputs
- Separated audio files (e.g., vocals.wav, instrumental.wav) for each input file, stored in a /separated/<tuple_index>/ folder
- Updated manifest with separation results and lineage

## Supported Models/Algorithms
- Default: Spleeter (2-stem or 4-stem)
- Optional: Demucs, custom separation models (configurable)
- Model selection and parameters are configurable per node instance

## Core Responsibilities
- For each file in a tuple:
  - Run the selected separation model
  - Output at least two files: <index>-<type>-vocals.wav, <index>-<type>-instrumental.wav
  - Store outputs in /separated/<tuple_index>/
  - Update manifest with:
    - separation_model (name, version, parameters)
    - output_vocals (filename/path)
    - output_instrumental (filename/path)
    - separation_timestamp
    - lineage (add separation step)
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all tuples/files in the batch
- Single-file: Process one tuple/file at a time
- Manifest must support both modes (list of separation results for batch, single result for single-file)

## Error Handling
- Flag files that fail separation (e.g., model error, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'separation_errors' section for any failures

## UI/UX Notes
- Allow user to select separation model and parameters (dropdown/config panel)
- Display progress and output file list for each tuple
- Teal-themed UI elements for model selection, progress bars, and output lists
- Show manifest summary (number of files processed, separation model used)

## Example Manifest Separation Entry
```json
{
  "tuple_index": "0000",
  "input_file": "tuples/0000/0000-out.wav",
  "separation_model": {
    "name": "spleeter",
    "version": "2.3.0",
    "parameters": {"stems": 2}
  },
  "output_vocals": "separated/0000/0000-out-vocals.wav",
  "output_instrumental": "separated/0000/0000-out-instrumental.wav",
  "separation_timestamp": "2025-06-01T00:05:00Z",
  "lineage": ["tuple_assembler", "separation"]
}
```

## Validation
- Unit tests for separation, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for model selection, progress, and output display 