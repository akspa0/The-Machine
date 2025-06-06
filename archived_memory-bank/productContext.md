# productContext.md

**Purpose:**
Why this project exists, the problems it solves, how it should work, and user experience goals.

## Problem Statement

Editing and contextualizing raw phone call audio is tedious, error-prone, and privacy-sensitive. Manual workflows are slow, do not scale, and risk privacy breaches due to PII exposure and lack of traceability.

## Solution Overview

Automate the ingestion, PII removal, file tracking, and AI-based processing of phone call audio. Use state-of-the-art models for separation, annotation, normalization, diarization, and transcription, with robust file management, manifesting, and privacy enforcement. The pipeline is fully automated, privacy-first, API-driven, and supports robust, auditable, and extensible processing for large batches of audio.

- Persona manifest is auto-generated if missing.
- Prompts are cached and reused for video workflows.
- All ComfyUI jobs use API-based file upload and robust polling.
- All outputs are copied and tracked in the project structure.

## User Experience Goals

- Simple, automated processing of large batches of audio
- Privacy-first: no PII in filenames, logs, or outputs
- Clear traceability and auditability of files and processing lineage
- Output ready for further analysis, dataset use, or downstream LLM tasks
- User-configurable via CLI and workflow JSONs
- **Ready for production testing and a fresh git commit.**

## 2024-06-XX: User Experience Achieved
- Privacy, traceability, resumability, extensibility, and robust error handling are all achieved.
- Pipeline is stable, user-friendly, and ready for production use. 
- **Full automation, API-driven workflow, and robust error handling are implemented.** 