# 🧩 The-Machine Extensions System

---

## Overview

The-Machine is built to be **extension-driven**: every new feature, analysis, or creative output is implemented as a modular, plug-and-play extension. Extensions can:
- Run after the main pipeline or independently
- Use all context, transcripts, and outputs from any run folder
- Add new analysis, creative outputs, or integrations
- Be shared, swapped, and improved without touching the core pipeline

**Philosophy:**
> Extensions are the heart of The-Machine. They let you build on top of audio context and transcription to create new tools, datasets, and experiences.

---

## 📦 Extension Catalog

Below are the main extensions included in this project. Each can be run independently or as part of a workflow.

### 🎭 `character_persona_builder.py`
- **Purpose:** Generates creative personas for each speaker/channel using LLMs and context.
- **Usage:**
  ```sh
  python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
  ```

### 🖼️ `avatar/sdxl_avatar_generator.py`
- **Purpose:** Generates SDXL/ComfyUI avatars and backdrops for each persona, using context and persona manifests.
- **Usage:**
  ```sh
  python extensions/avatar/sdxl_avatar_generator.py \
    --persona-manifest outputs/run-YYYYMMDD-HHMMSS/characters/persona_manifest.json \
    --output-root outputs/run-YYYYMMDD-HHMMSS
  ```

### 🎬 `avatar_animation_orchestrator.py`
- **Purpose:** Orchestrates persona, avatar, and animation generation for a run folder.
- **Usage:**
  ```sh
  python extensions/avatar_animation_orchestrator.py --run-folder outputs/run-YYYYMMDD-HHMMSS
  ```

### 🖌️ `comfyui_image_generator.py`
- **Purpose:** Generates images (and optionally videos) using ComfyUI workflows and LLM-generated prompts.
- **Usage:**
  ```sh
  python extensions/comfyui_image_generator.py --run-folder outputs/run-YYYYMMDD-HHMMSS --image
  ```

### 🔊 `clap_segmentation_experiment.py`
- **Purpose:** Segments long audio files into calls using CLAP event detection and context.
- **Usage:**
  ```sh
  python extensions/clap_segmentation_experiment.py outputs/run-YYYYMMDD-HHMMSS
  ```

### 🤹 `comedic_show_story_builder.py`
- **Purpose:** Generates creative, story-like show summaries using LLMs and show context.
- **Usage:**
  ```sh
  python extensions/comedic_show_story_builder.py outputs/run-YYYYMMDD-HHMMSS
  ```

### 🚀 `flashsr_extension.py`
- **Purpose:** Upsamples / enhances low-quality audio using FlashSR super-resolution (auto-downloads weights from Hugging Face).
- **Usage:**
  ```sh
  python extensions/flashsr_extension.py --input outputs/run-YYYYMMDD-HHMMSS/call/0003_vocals_only --device cuda
  ```
  Runs automatically when the pipeline is invoked with `--flashsr`.

### 🔔 `bleeper_extension.py`
- **Purpose:** Detects profanity in the first 180 seconds of the compiled show and overlays a configurable beep (or mute) tone; produces `show_bleeped.wav`.
- **Usage (stand-alone):**
  ```sh
  python -m extensions.bleeper_extension --input outputs/run-YYYYMMDD-HHMMSS/finalized/show
  ```
  Automatically executed during `finalization_stage.py`; both clean and bleeped versions are kept.

### 🧹 `transcript_and_soundbite_cleanup.py`
- **Purpose:** Aggregates transcripts and cleans up obsolete soundbites after processing.
- **Usage:**
  ```sh
  python extensions/transcript_and_soundbite_cleanup.py outputs/run-YYYYMMDD-HHMMSS
  ```

### 🗣️ `talking_head_pipeline.py`
- **Purpose:** Orchestrates persona, avatar, animation, and lipsync workflows for talking head video generation.
- **Usage:**
  ```sh
  python extensions/talking_head_pipeline.py --run-folder outputs/run-YYYYMMDD-HHMMSS --sdxl-workflow ... --framepack-workflow ... --latentsync-workflow ...
  ```

### 🧠 `llm_utils.py`
- **Purpose:** Provides unified LLM chunking, tokenization, and summarization utilities for all extensions and the pipeline.
- **Usage:**
  ```sh
  python extensions/llm_utils.py --help
  ```

## 🎛️ New Audio Collage & Separation Extensions

### 🔀 `vocal_separation_extension.py`
* **Purpose:** Runs the open-source `audio-separator` model to create `*-vocals.wav` / `*-instrumental.wav` stems for every WAV in `renamed/`.
* **Auto-Invoke:** Extensions that need stems (phrase/word timestamps, collages) should call this automatically if no `separated/` folder exists.
* **CLI:**
  ```sh
  # Single run
  python extensions/vocal_separation_extension.py outputs/run-20250611-000850

  # Process ALL run-* folders inside outputs/
  python extensions/vocal_separation_extension.py outputs/ --all-runs
  ```

### 🗣️ `word_timestamp_extension.py`
* Generates word-level timestamps for each **speaker** (concatenated vocal segments) using Whisper or Parakeet.
* Outputs to `word_ts_whisper/` (or `word_ts_parakeet/`).

### 🗨️ `phrase_timestamp_extension.py`
* Copies speaker segments into `phrase_ts/<call>/<channel>/<speaker>/` and writes a JSON manifest (start/end, transcript).
* Provides the phrase library used by the collage extensions.

### 🎵 `phrase_collage_extension.py`
* Builds surreal sentences **per speaker** by asking a local LLM to pick phrase IDs.
* Concatenates chosen phrases → `phrase_collage/…/phrase_collage.wav`.
* Writes `montage_plan.txt` (ID lines) and `montage_text.txt` (human-readable).
  ```sh
  python extensions/phrase_collage_extension.py outputs/run-20250611-000850 \
      --sentences 5 --phrases-per-sentence 3
  ```

### 🌐 `multi_speaker_collage_extension.py`
* NEW! Remixes phrases **across all speakers** (and optionally across every `run-*` folder) to form multi-voice sentences.
* Flags:
  * `--all-runs` Include every run inside `outputs/` or siblings of the given run folder.
  * `--distinct-speakers` Force each sentence to use phrases from different speakers.
* Example:
  ```sh
  python extensions/multi_speaker_collage_extension.py outputs/ --all-runs \
      --sentences 4 --phrases-per-sentence 4 --distinct-speakers
  ```
* Falls back to a random plan if the LLM cannot comply.

---

## 🏗️ How Extensions Use Context

- Extensions can access all context, transcripts, and outputs from any run folder.
- They can build new features, analyses, or creative outputs using this context (e.g., personas, image prompts, show summaries).
- Many extensions use LLMs to generate new context or creative outputs from transcripts.
- You can chain extensions together for complex workflows.

---

## 🚦 Quickstart: Running & Authoring Extensions

Some extensions (e.g., **bleeper** during finalization or **flashsr** when `--flashsr` flag is set) run automatically in the core pipeline. Others can be invoked manually as shown.

1. **Run any extension** with the output folder as its argument (see above examples).
2. **Author your own:**
   - Inherit from `ExtensionBase`.
   - Use context, transcripts, and outputs as needed.
   - Log only anonymized, PII-free information.
   - Document your extension and add it to this catalog!

---

## 🛡️ Best Practices for Extension Authors

- Be robust to folder naming and output structure.
- Log only anonymized, PII-free information.
- Support both batch and single-file workflows.
- Document your extension's purpose, usage, and options.
- Follow privacy, traceability, and idempotence rules.
- Use the manifest and output folder structure for all data lineage.
- Handle missing or partial data gracefully.

---

## 📂 Subfolders & Special Extensions

- `avatar/` — Persona/avatar/backdrop generation workflows
- `ComfyUI/` — Workflow JSONs for image/video generation
- See each subfolder for more details and workflow templates.

---

## 🤝 Contributing Extensions

- New extensions are welcome! Please follow the best practices above and add your extension to this catalog.
- Open a PR or issue with your extension and documentation.

---

**Extensions are the engine of context. Build, share, and remix!** 