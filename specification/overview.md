# System Overview: The-Machine as ComfyUI Nodes

## Purpose
This document provides a high-level architecture and guiding principles for rebuilding The-Machine as a modular, privacy-first, extension-driven audio processing system using ComfyUI custom nodes.

## Key Principles
- All processing is node-based (no CLI orchestration).
- Privacy and PII removal are enforced at the earliest possible stage.
- Every job, file, and data artifact is uniquely identified and tracked (manifest/lineage).
- All nodes are plug-and-play, reusable, and composable.
- Batch and single-file processing are supported throughout.
- The system is ready for future database integration.

## Node Graph (Pipeline Example)
1. Raw Input Ingestion Node
2. Phone Call Tuple Assembler Node
3. Separation Node
4. Normalization Node
5. CLAP Annotation Node
6. Diarization Node
7. Transcription Node
8. Soundbite Generation Node
9. LLM Task Node
10. Remixing Node
11. Show Output Node
12. Export/Database Node (future)

Each node receives a manifest, updates it, and passes it to the next node.

## Data Flow
- Nodes pass file paths and manifest objects (dict/JSON) between each other.
- All nodes validate input manifest for privacy and completeness before processing.
- Manifest is updated and written to disk at every stage.

## Batch and Single-File Support
- All nodes must support both batch and single-file workflows.
- Batch: Process multiple calls/files in parallel or sequence.
- Single-file: Process one call/file at a time for granular control.

## Extensibility
- Users can add, remove, or swap nodes in the ComfyUI graph.
- Each node is self-contained and only depends on manifest + file inputs.
- Workflow presets (JSON/YAML) define LLM tasks, CLAP prompts, segmentation rules, etc.

## UI/UX
- Node design emphasizes simplicity, clarity, and a clean (teal-themed, if possible) UI.
- Example node layouts and user flows will be provided in a separate UI/UX guidelines document. 