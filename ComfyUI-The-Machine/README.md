# ComfyUI-The-Machine

A modular, privacy-first extension suite for ComfyUI, implementing The-Machine's audio processing pipeline as custom nodes.

## Node List
- Raw Input Ingestion Node
- Phone Call Tuple Assembler Node
- Separation Node
- Normalization Node
- CLAP Annotation Node
- Diarization Node
- Transcription Node
- Soundbite Generation Node
- LLM Task Node
- Remixing Node
- Show Output Node
- Export/Database Node

## Setup
1. Place this folder in your ComfyUI custom nodes directory.
2. Install required dependencies (see requirements.txt).
3. Restart ComfyUI to load the nodes.

## Extension Philosophy
- Each node is modular, plug-and-play, and privacy-first.
- All processing is tracked via a manifest, with strict PII removal at every stage.
- Batch and single-file workflows are supported throughout.
- Shared utilities enforce privacy, manifest logic, and error handling.

## Development
- Each node is implemented as a Python class, following ComfyUI's custom node API.
- Shared utilities are in `utils/`.
- Tests are in `tests/`.

See the specification folder for detailed node requirements and responsibilities.

# ComfyUI Node Suite: Requirements & Setup

## Python Dependencies
See `requirements.txt` for the full list. Key packages:
- torch, torchaudio (install with CUDA support if available)
- tqdm, numpy, soundfile, mutagen, pydub, scipy
- transformers, datasets, huggingface_hub
- demucs, spleeter (audio separation)
- pyloudnorm (loudness normalization)
- pyannote.audio (speaker diarization)
- comfyui (node-based UI)
- ffmpeg-python (optional, but system ffmpeg required)
- pytest, pytest-cov (testing)

## System Requirements
- **ffmpeg** (must be installed and in your PATH)
- **CUDA** (for GPU acceleration with torch/torchaudio, optional but recommended)
- **Python 3.9–3.11** (recommended)

## ComfyUI Installation
If not available on PyPI, install from source:
```
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -e .
```
Or see the [ComfyUI repo](https://github.com/comfyanonymous/ComfyUI) for latest instructions.

## Node Usage & Pipeline Chaining
- All nodes use explicit, pipeable inputs/outputs (`list[dict]` or `dict`), with `manifest` always passed and updated.
- All config is via JSON, and all intermediate data is hidden for seamless chaining.
- Batch and single-file operation are both supported.
- No PII is ever exposed in outputs, logs, or UI.
- Manifest is always updated and validated at each step.

### Example Minimal Pipeline (ComfyUI Graph)
```
RawInputIngestionNode → TupleAssemblerNode → SeparationNode → NormalizationNode → CLAPAnnotationNode → DiarizationNode → TranscriptionNode → SoundbiteGenerationNode → RemixingNode → ShowOutputNode
```

## Troubleshooting
- **Missing ffmpeg:** Install from https://ffmpeg.org/download.html and ensure it is in your PATH.
- **CUDA errors:** Ensure you have the correct torch/torchaudio version for your GPU and CUDA version.
- **Demucs/Spleeter errors:** Check their documentation for model downloads and system requirements.
- **pyannote.audio:** May require additional HuggingFace authentication for some models.
- **ComfyUI not found:** Install from source as above.

For more details, see the comments in `requirements.txt` and the docstrings in each node file. 