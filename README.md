# The-Machine 🧠🔊

Dedicated to the memory of Carlito Cross [Madhouse Live](https://madhouselive.com)
---

**A context-driven, privacy-first, modular pipeline for understanding, transforming, and building on top of audio recordings.**

---

## 🚀 What is The-Machine?

The-Machine is a powerful, extensible toolkit for:
- Adding rich context to audio recordings (calls, music, podcasts, etc.)
- Preparing audio and metadata for dataset use, research, and creative projects
- Building new tools and workflows on top of audio context and transcriptions
- Enabling privacy-first, traceable, and reproducible audio processing

**Why?**
> Audio is more than just sound — it is context, story, and data. The-Machine helps you unlock, organize, and use that context for anything from dataset curation to creative AI workflows.

---

## ✨ Features

- 🎙️ **Audio Ingestion & PII Removal**: Ingests audio, removes PII from filenames, and anonymizes all logs/outputs.
- 🗂️ **Context-Driven Processing**: Every file is tracked, indexed, and processed with full lineage and manifesting.
- 🧩 **Extension System**: Modular, plug-and-play extensions for everything—transcription, CLAP annotation, LLM tasks, remixing, show creation, and more.
- 🦾 **LLM Integration**: Local LLM support (LM Studio, etc.) for titles, summaries, image prompts, and more—fully privacy-safe.
- 🗣️ **Speaker Diarization & Transcription**: Segments audio by speaker, transcribes with Parakeet/Whisper, and aligns with context.
- 🥁 **CLAP Annotation & Segmentation**: Detects events (e.g., ringing, hang-up) and segments calls using CLAP.
- 🎚️ **Normalization & Remixing**: Loudness normalization, true peak, and creative remixing for dataset or show use.
- 🖼️ **Image/Video Generation**: Extensions for SDXL/ComfyUI image and video generation from transcripts and personas.
- 📜 **Manifest & Traceability**: Every output is tracked in a manifest—no lost context, ever.
- 🔒 **Privacy-First**: No PII in logs, outputs, or manifests. All processing is anonymized by design.
- 🧠 **Memory Bank**: Project context, progress, and system patterns are tracked for robust, extension-driven development.
- 🛠️ **Workflow-Driven**: All logic and configuration is defined in JSON workflows—easy to extend, modify, and share.
- 🏗️ **Ready for Dataset Prep**: Designed to help you build, clean, and annotate audio datasets for ML/AI.
- 🔄 **Resume & Robustness**: Pipeline can resume from any stage, with full error recovery and validation.
- 🧬 **Designed for Extensibility**: Build your own extensions to add new context, analysis, or creative outputs.
- Persona builder audio samples are now lossless, using numpy+soundfile to concatenate original .wav files (not _16k.wav), with no resampling or pydub, guaranteeing high fidelity for all persona samples.
- System prompt for persona generation now instructs the LLM to be concise, allow for absurdity, and keep responses below 300 tokens.
- All LLM chunking/continuation logic is removed; only direct responses are used for persona and all LLM tasks.
- Logging and debug output is robust and clear for all pipeline and extension stages.

---

## 🧩 Extension System

All new features are implemented as modular **extensions** in the `extensions/` folder. Extensions can:
- Run after the main pipeline or independently
- Use all context, transcripts, and outputs
- Add new analysis, creative outputs, or integrations

**See [`extensions/README.md`](./extensions/README.md) for a full catalog and authoring guide.**

---

## 🛠️ Example Usage

### Ingest and Process Audio
```sh
python pipeline_orchestrator.py input_audio/
```

### Run an Extension (e.g., Persona Builder)
```sh
python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
```

### Generate Avatars/Images
```sh
python extensions/avatar/sdxl_avatar_generator.py \
  --persona-manifest outputs/run-YYYYMMDD-HHMMSS/characters/persona_manifest.json \
  --output-root outputs/run-YYYYMMSS
```

### Use the LLM Utilities (chunking, summarization, etc.)
```sh
python extensions/llm_utils.py --help
```

---

## 📚 Project Structure

- `extensions/` — All modular extensions (see README inside)
- `workflows/` — JSON configs for pipeline, CLAP, LLM, etc.
- `memory-bank/` — Project context, progress, and system patterns
- `outputs/` — All run outputs (timestamped folders)
- `specification/` — System and node documentation

---

## 🧠 How to Build Your Own Extensions

1. Copy `extension_base.py` and inherit from `ExtensionBase`.
2. Use context, transcripts, and outputs from any run folder.
3. Add your logic—analysis, creative output, new integrations, etc.
4. Log only anonymized, PII-free information.
5. Document your extension and add it to the catalog!

See [`extensions/README.md`](./extensions/README.md) for more.

---

## 🌟 Vision & Future

- **Context Everywhere:** Audio is just the start — The-Machine is designed to add, use, and build on context for any data.
- **Multimodal Workflows:** Future extensions will support image→text→audio pipelines, creative AI, and dataset generation in all directions.
- **Reverse Pipelines:** Imagine describing an image with a local LLM, then generating audio or music from that description—The-Machine will make it possible.
- **Open, Extensible, and Privacy-First:** Built for researchers, creators, and anyone who wants to understand and use audio context.

---

## 📝 Documentation & Resources

- [Extension Catalog & Guide](./extensions/README.md)
- [Workflow Configs](./workflows/README.md)
- [Memory Bank & Project Context](./memory-bank/README.md)
- [System Specifications](./specification/README.md)

---

## 🤝 Contributing

- Contributions, new extensions, and feedback are welcome!
- Please see the extension authoring guide and open an issue or PR.

---

## Auto-Editor Wrapper (`tm-auto-editor`)

The-Machine ships a small stand-alone CLI wrapper (≤250 LOC) located at `tools/auto_editor_cli.py`.
It leverages the open-source [Auto-Editor](https://github.com/WyattBlue/auto-editor) project to
remove long silences and awkward pauses from your already-anonymised call WAV/MP3 files.

### 1. Clone & install Auto-Editor once
```bash
# Inside project root
git clone --depth 1 https://github.com/WyattBlue/auto-editor external_apps/auto_editor
pip install -e external_apps/auto_editor  # inside your virtualenv
```

### 2. Run the wrapper manually
```bash
python tools/auto_editor_cli.py path/to/0123_call.wav \
    --margin 0.2sec --silent-speed 99999 --video-speed 1 --verbose
```
This creates `0123_call_ae.wav` and a JSON report (placed in the nearest `run-*` folder if present).

### 3. Integrate into PipelineOrchestrator
Call the wrapper as a post-finalisation step:
```python
subprocess.run([
    sys.executable,
    "tools/auto_editor_cli.py",
    call_wav,
    "--report", manifest_tmp_json,
])
```
The JSON report is designed to be merged into the main manifest.

---

**Built for context, privacy, and creativity.**
