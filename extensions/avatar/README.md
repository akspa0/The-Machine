# 🖼️ Avatar & Persona Image Generation Extensions

---

## Overview

The avatar extension system lets you generate creative, privacy-safe persona and backdrop images for each call, speaker, or channel—using only anonymized context and transcripts. These images can be used for:
- Dataset enrichment
- Creative projects
- Visualizing call context
- Downstream AI/ML workflows

**Philosophy:**
> Every voice has a story—and a face. Avatars and backdrops help bring context to life, while preserving privacy and traceability.

---

## 🧬 Workflow

1. **Persona Manifest**: Generated by the persona builder extension, this manifest contains anonymized persona descriptions for each call/speaker.
2. **SDXL/ComfyUI Workflows**: Use persona descriptions and transcripts to generate images via SDXL/ComfyUI workflows.
3. **Backdrops**: Optionally, generate a backdrop image for each call using transcript context.
4. **Integration**: All outputs are saved in the run folder, fully anonymized and tracked in the manifest.

---

## 🚀 Main Script: `sdxl_avatar_generator.py`

### Purpose
- Generates persona (avatar) and backdrop images for each call/persona using SDXL/ComfyUI workflows and context.

### Example Usage
```sh
python extensions/avatar/sdxl_avatar_generator.py \
  --persona-manifest outputs/run-YYYYMMDD-HHMMSS/characters/persona_manifest.json \
  --output-root outputs/run-YYYYMMDD-HHMMSS
```

### CLI Options
- `--persona-manifest` (required): Path to persona_manifest.json
- `--output-root` (required): Path to run-* output folder
- `--avatar-workflow`: Path to avatar SDXL workflow JSON (default provided)
- `--backdrop-workflow`: Path to backdrop SDXL workflow JSON (default provided)
- `--initial-prompt`: Text to prepend to the positive prompt (e.g., "a drawing of")
- `--api-url`: ComfyUI API URL (default: http://127.0.0.1:8188)
- `--batch-size`: Number of ComfyUI jobs to submit per batch (default: 10)

---

## 🧠 How Context & Transcription Are Used

- Persona descriptions are generated from transcripts and context—never from PII.
- Prompts for SDXL/ComfyUI are crafted to be creative, privacy-safe, and visually rich.
- Backdrops are generated from call transcripts, visualizing the environment or mood.
- All outputs are anonymized and tracked in the manifest for full traceability.

---

## 🛡️ Best Practices for Avatar/Image Extensions

- Never use or log PII—only use anonymized context and transcripts.
- Use creative, privacy-safe prompt engineering for SDXL/ComfyUI.
- Track all outputs in the manifest for traceability.
- Document your extension's purpose, usage, and options.
- Support both batch and single-file workflows.

---

## 📚 Related Extensions & Docs

- See [`../README.md`](../README.md) for the full extension system overview.
- See [`../comfyui_image_generator.py`](../comfyui_image_generator.py) for advanced image/video generation.
- See [`../../workflows/`](../../workflows/) for workflow JSONs.

---

**Bring context to life—safely, creatively, and traceably.** 