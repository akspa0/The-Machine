# Node Specification: CLAP Annotation Node

## Purpose
This node runs the CLAP model on normalized audio to detect context events (e.g., ringing, DTMF, yelling), filters results by confidence, and updates the manifest and master transcript with accepted annotations.

## Inputs
- Normalized audio files and manifest from the Normalization Node
- Configurable list of CLAP prompts (e.g., 'telephone ring tones', 'DTMF', 'dogs barking')
- Confidence threshold (default: 0.6, configurable)

## Outputs
- CLAP annotation events for each audio file, stored in /clap/<tuple_index>/
- Updated manifest with CLAP results and lineage
- Updated master transcript with CLAP events

## Supported Models
- CLAP (Contrastive Language-Audio Pretraining, e.g., HuggingFace implementation)
- Model version and parameters are configurable per node instance

## Core Responsibilities
- For each normalized file:
  - Run CLAP model with all configured prompts
  - Accept only annotations with confidence >= threshold
  - Output annotation events as JSON (start, end, label, confidence)
  - Store outputs in /clap/<tuple_index>/
  - Update manifest with:
    - clap_model (name, version, parameters)
    - clap_events (list of accepted events)
    - clap_prompts (list of prompts used)
    - clap_confidence_threshold
    - clap_annotation_timestamp
    - lineage (add CLAP step)
  - Update master transcript with CLAP events (format: [CLAP][START-END][annotation])
- Ensure all outputs and manifest updates are PII-free

## Privacy & PII Logic
- No original filenames, paths, or PII in output files, manifest, or transcript
- Validate that all outputs are anonymized before updating manifest/transcript
- All manifest and transcript updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all tuples/files in the batch
- Single-file: Process one tuple/file at a time
- Manifest and transcript must support both modes (list of CLAP results for batch, single result for single-file)

## Error Handling
- Flag files that fail CLAP annotation (e.g., model error, corrupt audio) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'clap_errors' section for any failures

## UI/UX Notes
- Allow user to configure CLAP prompts and confidence threshold (config panel)
- Display detected events, confidence scores, and prompt matches for each file
- Teal-themed UI elements for prompt config, progress bars, and event lists
- Show manifest and transcript summary (number of events, types detected)

## Example Manifest CLAP Entry
```json
{
  "tuple_index": "0000",
  "input_file": "normalized/0000/0000-out-vocals-normalized.wav",
  "clap_model": {
    "name": "clap-hf",
    "version": "1.0.0",
    "parameters": {"prompts": ["ringing", "DTMF"], "threshold": 0.6}
  },
  "clap_events": [
    {"start": 0.0, "end": 2.5, "label": "ringing", "confidence": 0.82},
    {"start": 10.0, "end": 10.5, "label": "DTMF", "confidence": 0.91}
  ],
  "clap_prompts": ["ringing", "DTMF"],
  "clap_confidence_threshold": 0.6,
  "clap_annotation_timestamp": "2025-06-01T00:15:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap"]
}
```

## Example Transcript CLAP Event
```
[CLAP][0.00-2.50][ringing]
[CLAP][10.00-10.50][DTMF]
```

## Validation
- Unit tests for CLAP annotation, manifest/transcript updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for prompt config, event display, and summary 