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

> **Note:** For LLM utilities, you may need to install `tiktoken`:
> ```sh
> pip install tiktoken
> ```

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

## LLM Utilities: Tokenization & Summarization

The-Machine includes two powerful utilities for LLM-driven workflows:

### 1. `llm_tokenize.py`: Token-Aware Chunking
Splits large text files into token-limited chunks for LLM processing, using `tiktoken`.

**Usage Example:**
```sh
python extensions/llm_tokenize.py --input-file path/to/large.txt --max-tokens 4096 --output-dir path/to/chunks/
```
- `--input-file`: Path to the input text file.
- `--max-tokens`: Max tokens per chunk (default: 4096).
- `--output-dir`: Directory to write chunk files (prints to stdout if not set).
- `--model`: (Optional) Model name for tiktoken encoding (default: gpt-3.5-turbo).

### 2. `llm_summarize.py`: LLM-Driven Summarization
Summarizes multiple text chunks into a single, creative SDXL prompt or summary using the LLM.

**Usage Example:**
```sh
python extensions/llm_summarize.py --input-files path/to/chunks/*.txt --output path/to/summary.txt --llm-config workflows/llm_tasks.json
```
- `--input-files`: List of chunk text files to summarize.
- `--output`: Output file for the summary.
- `--llm-config`: Path to LLM config JSON (default: workflows/llm_tasks.json).

**When to Use:**
- Prepping large persona files for SDXL prompt generation.
- Summarizing transcripts, personas, or other large text artifacts for downstream LLM or image workflows.

---

## Usage Examples

### Basic Pipeline Run
```sh
python pipeline_orchestrator.py <input_dir>
```

### With CLAP-based Call Segmentation (Extension Only)
```sh
python extensions/clap_segmentation_experiment.py outputs/run-YYYYMMDD-HHMMSS --config workflows/clap_segmentation.json
```

### Processing a YouTube/URL Input
```sh
python pipeline_orchestrator.py --url "https://www.youtube.com/watch?v=..."
```

### Resume/Debug
```sh
python pipeline_orchestrator.py --output-folder outputs/run-YYYYMMDD-HHMMSS --resume
```

### Other CLI Options
- `--asr_engine parakeet|whisper` (choose ASR model)
- `--llm_config workflows/llm_tasks.json` (custom LLM task config)
- `--call-tones` (insert tones between calls in show output)
- `--resume-from <stage>` (resume from a specific stage)

---

## Customization & Configuration
- **CLAP/LLM prompts, thresholds, and logic** are fully tweakable in the `workflows/` JSON files.
- Change prompts, add/remove tasks, adjust thresholds, and rerun the pipeline—no code changes needed.

---

## Troubleshooting & FAQ
- **PyTorch install issues:** Use conda and follow the [official instructions](https://pytorch.org/get-started/locally/).
- **No segments detected:** Lower the CLAP confidence threshold or add more prompts in `clap_segmentation.json`.
- **Too many/false segments:** Raise the threshold or adjust pairing/gap settings.
- **LLM token limit errors:** Use `llm_tokenize.py` to chunk large files before LLM processing.
- **Missing tiktoken:** Install with `pip install tiktoken`.
- **Manifest/logs:** Check the output run folder for detailed logs and manifest.json.
- **If prompts or scene prompt JSONs do not update after changing window or token settings, use the `--force` option to force regeneration. Otherwise, the script will reuse existing files.**

---

## Contributing & Support
- PRs and issues welcome!
- For questions, open an issue or contact the maintainer.
- Extension contributions are encouraged—see `extensions/README.md` for best practices.
- The memory bank must always reflect the current codebase, extension ecosystem, and system architecture. Outdated or legacy content should be archived and not referenced in active documentation.

---

**Happy hacking with The-Machine!** 