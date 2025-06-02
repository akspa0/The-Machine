# The-Machine

---

Dedicated to Carlito Cross - RIP - [MadhouseLive](https://www.madhouselive.com/)

---

A privacy-focused, modular pipeline for processing phone call audio and other recordings. Automates ingestion, PII removal, file tracking, audio separation, CLAP annotation, loudness normalization, speaker diarization, transcription, soundbite extraction, LLM integration, remixing, and show creation. All steps are orchestrated for strict privacy, traceability, and manifest/logging requirements.

---

**GitHub Repository:** [https://github.com/akspa0/The-Machine](https://github.com/akspa0/The-Machine)

---

## 🚀 Modern Architecture: Extension-Driven, API-First, Librarian-Orchestrator

**The-Machine** is now built around a modular, extension-driven, API-first architecture:
- The main program acts as a **librarian orchestrator**: it manages jobs, invokes extensions, and handles all data flow, but contains no monolithic pipeline logic.
- All core logic is implemented as plug-and-play **extensions** ("stacks"), which are independently testable and reusable.
- All integrations (e.g., ComfyUI) use robust API-based file transfer and job submission—never direct file system access.
- Every job, file, and data object is uniquely identified, tracked, and privacy-preserving, with robust error handling and traceability.
- Prompts and metadata are cached for reuse in downstream workflows, ensuring consistency and manifest traceability.
- The system is designed for future database integration (e.g., SQLite) to manage jobs, data, and reference lookups.

---

## ⚠️ Environment & Installation (Conda Recommended)

> **The-Machine is a complex, GPU-accelerated pipeline with many dependencies. We strongly recommend using [Anaconda/conda](https://docs.conda.io/en/latest/) to manage your Python environment, especially for PyTorch and GPU support.**
>
> - Conda ensures correct versions of PyTorch, torchaudio, and CUDA for your hardware.
> - Pip-only installs are possible but not recommended for most users.

### 1. Create and Activate a Conda Environment
```sh
conda create -n themachine python=3.10
conda activate themachine
```

### 2. Install PyTorch (Choose the right CUDA version for your system)
See [PyTorch.org](https://pytorch.org/get-started/locally/) for the latest command.
- **CPU only:**
  ```sh
  conda install pytorch torchaudio cpuonly -c pytorch
  ```
- **CUDA 11.8 (NVIDIA GPU):**
  ```sh
  conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  ```

### 3. Install the rest of the dependencies
```sh
pip install -r requirements.txt
```

---

## 🧠 Local LLM Inference with LM Studio

The-Machine supports local LLM inference using [LM Studio](https://lmstudio.ai/), allowing you to run powerful language models on your own hardware for privacy and speed.

### 1. Install LM Studio
- Download LM Studio from [https://lmstudio.ai/](https://lmstudio.ai/).
- Available for Windows, Mac, and Linux.
- Follow the installation instructions for your platform.

### 2. Download a Recommended Model
- We recommend using a high-quality, chat-optimized model in GGUF format. For example:
  - **llama-3.1-8b-supernova-etherealhermes (Q4_K_M GGUF):** [Model Card & Download](https://huggingface.co/etherealhermes/llama-3.1-8b-supernova-etherealhermes-GGUF)
- Download the desired quantization (e.g., Q4_K_M for a good balance of speed and quality).

### 3. Load the Model in LM Studio
- Open LM Studio and use the 'Download Model' feature, or manually place the GGUF file in your models directory.
- In LM Studio, select the model and click 'Launch'.
- Ensure the local API server is enabled (default: `http://localhost:1234/v1`).
  - You can check this in the LM Studio settings under 'API Server'.

### 4. Configure The-Machine to Use LM Studio
- In your LLM config (e.g., `workflows/llm_tasks.json`), set the API endpoint to `http://localhost:1234/v1`.
- Or, use the CLI flag to specify the config:
  ```sh
  python main.py <input_dir> --llm_config workflows/llm_tasks.json
  ```
- Make sure your LLM config matches the model's expected prompt format (e.g., chat template, system prompt).

### 5. Example LLM Config Snippet
```json
{
  "api_base": "http://localhost:1234/v1",
  "model": "llama-3.1-8b-supernova-etherealhermes",
  "temperature": 0.7,
  "max_tokens": 2048,
  "system_prompt": "You are a helpful assistant."
}
```

### 6. Troubleshooting
- **LM Studio not running:** Make sure LM Studio is open and the API server is enabled.
- **Port conflicts:** Ensure nothing else is using port 1234, or change the port in LM Studio and your config.
- **Model loading errors:** Verify you downloaded the correct GGUF file and quantization for your hardware.
- **API errors:** Test the endpoint in your browser: [http://localhost:1234/v1](http://localhost:1234/v1) should return a JSON response if running.

---

## Major Features & Recent Changes
- **Soundbites folders** are now named after sanitized call titles, not just call IDs.
- **SFW (safe-for-work) call titles** and show summaries are generated and included in outputs.
- **Show notes** are generated by the LLM and appended to show summaries.
- **Extensions system:** Users can add scripts (like `character_persona_builder.py`) to an `extensions/` folder, which run after the main pipeline and can use all outputs.
- **Advanced Character.AI persona generation:** The extension system supports per-channel and per-speaker persona creation, robust to folder naming and batch/single-file workflows.
- **CLAP segmentation and annotation** are now handled exclusively by the extension (`clap_segmentation_experiment.py`) in the `extensions/` folder. The main pipeline is fully decoupled from CLAP.
- **All outputs, logs, and manifests** are strictly PII-free and fully auditable.
- **Timestamps** in show summaries are now in HH:MM:SS format, with emoji for calls/tones.
- **Resume and force:** The pipeline is robust to resume, with `--resume` and `--resume-from`, and `--force` now archives outputs instead of deleting them.

---

## Features
- **Privacy-first:** No PII in filenames, logs, or outputs. All logging/manifesting is anonymized.
- **Modular pipeline:** Ingestion, separation, CLAP, diarization, normalization, transcription, soundbite, remix, show, LLM, and more.
- **CLAP-based segmentation:** (Extension only) Detects call boundaries in long audio using CLAP via the `clap_segmentation_experiment.py` extension (configurable prompts, thresholds).
- **Speaker diarization:** Segments audio by speaker, with per-speaker transcripts and outputs.
- **LLM integration:** Workflow-driven LLM tasks (titles, synopses, categories, image prompts, songs, etc.)
- **Batch processing:** Handles large folders of audio, tuples, or single files (including YouTube/URL inputs).
- **Extensible:** All workflows and prompts are JSON-configurable in the `workflows/` folder.
- **Traceability:** Full manifest and metadata lineage for every file and output.

---

## Directory Structure
```
The-Machine/
  memory-bank/           # Project docs, context, and rules
  outputs/               # All run outputs (timestamped folders)
  workflows/             # All pipeline, CLAP, and LLM configs (JSON)
  extensions/            # Custom extension scripts (see below)
  ...                    # Pipeline scripts and modules
```

---

## Core Pipeline & Librarian Orchestrator
- The main program (librarian) orchestrates all jobs, invokes extensions, and manages data flow.
- All core logic is implemented as modular extensions (see below).
- All external integrations (e.g., ComfyUI) use API-based file transfer and job submission—never direct file system access.
- All outputs are copied into the canonical project structure with standardized naming and manifest tracking.
- Privacy and PII removal are enforced at every stage; no PII in filenames, logs, or outputs.
- The system is architected for future database integration for job/data/metadata management and reference lookups.

---

## Extensions System

Extensions are the heart of The-Machine. All new features and improvements are implemented as modular, plug-and-play scripts in the `extensions/` folder.

### How to Use Extensions
- Place your extension scripts in the `extensions/` directory.
- Each extension should inherit from `ExtensionBase` (see `extension_base.py`).
- Extensions are run manually or can be invoked automatically after pipeline completion.
- Extensions receive the root output directory as their argument and should only access finalized, anonymized outputs.

#### Example: Running the Character Persona Builder Extension
```sh
python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
```

#### Example: Running the Avatar SDXL Generator Extension
```sh
python extensions/avatar/sdxl_avatar_generator.py \
  --persona-manifest outputs/run-YYYYMMDD-HHMMSS/characters/persona_manifest.json \
  --output-root outputs/run-YYYYMMDD-HHMMSS \
  --initial-prompt "a drawing of"
```

#### Example: Running the ComfyUI Image Generator Extension
```sh
python extensions/comfyui_image_generator.py \
  --run-folder outputs/run-YYYYMMDD-HHMMSS \
  --workflow extensions/ComfyUI/theMachine_SDXL_Basic.json \
  --image --window-seconds 90 --max-tokens 4096
```

See `extensions/README.md` and `extensions/avatar/README.md` for more details and authoring tips.

### Best Practices for Extension Authors
- Be robust to folder naming (handle run-specific prefixes, normalize for output).
- Log only anonymized, PII-free information.
- Support both batch and single-file workflows.
- Document your extension's purpose and usage.
- Follow privacy, traceability, and idempotence rules.
- Use the manifest and output folder structure for all data lineage.
- Handle missing or partial data gracefully.

---

## Configuration
- **CLAP segmentation/annotation:** Now handled exclusively by the extension (`clap_segmentation_experiment.py`). You may still configure prompts and thresholds in `workflows/clap_segmentation.json` and `workflows/clap_annotation.json` for use by the extension.
- **LLM tasks:** `
