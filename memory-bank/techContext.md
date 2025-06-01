# techContext.md

**Purpose:**
Describes the technologies, APIs, and technical context for the extension-driven, librarian-orchestrator version of The-Machine.

## Core Technologies

- **Python 3.x**: Main language for the librarian orchestrator and all extensions.
- **API-First Integrations**: All external tools (e.g., ComfyUI) are integrated via robust HTTP APIs (e.g., /prompt, /upload/image).
- **Modular Extensions**: All core logic is implemented as Python modules/extensions that can be independently developed, tested, and swapped.
- **CLI/Config-Driven**: The system is orchestrated via CLI and config files for maximum flexibility and reproducibility.
- **Database-Ready**: The architecture is designed for future integration with a database (e.g., SQLite) for job/data/metadata management and reference lookups.

## Development Setup

- Install dependencies via pip (requirements.txt provided).
- Extensions are developed as standalone Python modules in the extensions/ directory.
- All job/data transfer to external tools is done via API, not direct file system access.
- All outputs are copied into the project structure and tracked in the manifest.

## Technical Constraints

- All jobs, files, and data must be uniquely identified and tracked for traceability.
- Privacy and PII removal must be enforced at every stage.
- Extensions must be plug-and-play and not require changes to the core librarian.
- The system must be ready for database integration for job/data/metadata management.

## Dependencies

- Python 3.x
- requests (for API calls)
- tqdm, rich (for CLI UX)
- pydub, numpy, soundfile, torchaudio, librosa (for audio processing in extensions)
- ComfyUI (external, API-driven)
- SQLite (planned, for future database registry)

## Tech Context

- Python 3.9+
- PyTorch, torchaudio
- Demucs, Spleeter (audio separation)
- pyloudnorm (loudness normalization)
- pyannote.audio (speaker diarization)
- transformers (CLAP, ASR, etc.)
- ComfyUI (node-based UI, currently on hold)
- ffmpeg (system dependency)
- Gradio (optional, for web UI)
- Environment management: conda or pip
- CUDA (optional, for GPU acceleration)

- The system is now fully extension-driven, API-first, and ready for database-backed job/data management. 