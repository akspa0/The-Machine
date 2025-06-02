# 🖌️ ComfyUI Workflow Templates

---

## Overview

This folder contains **workflow JSONs** for use with [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and The-Machine's image/video generation extensions. These workflows define how images, avatars, backdrops, and videos are generated from context, transcripts, and persona descriptions.

---

## 📂 Main Workflow Files

- `theMachine_SDXL_Basic.json` — General-purpose SDXL image generation workflow.
- `avatar/avatar_sdxl_workflow.json` — Workflow for generating persona/avatar images.
- `avatar/backdrop_sdxl_workflow.json` — Workflow for generating backdrop/environment images.
- `avatar/img2wan-avatar.json` — Workflow for image-to-video avatar animation.
- `avatar/image_to_video_wan.json` — Workflow for converting images to video using WAN.
- `avatar/latentsync1.5_comfyui_basic.json` — Workflow for lipsyncing avatars to audio.

---

## 🚀 How to Use

- These workflows are used by extensions such as:
  - [`comfyui_image_generator.py`](../comfyui_image_generator.py)
  - [`avatar/sdxl_avatar_generator.py`](../avatar/sdxl_avatar_generator.py)
  - [`avatar_animation_orchestrator.py`](../avatar_animation_orchestrator.py)
- You can specify the workflow file via CLI options, or use the defaults provided by each extension.
- To create your own workflow, copy and modify any of these JSONs to suit your needs.

---

## 🛡️ Best Practices for Workflow Design

- Ensure all prompts and outputs are privacy-safe and anonymized.
- Use context and transcripts to drive creative, meaningful outputs.
- Keep workflows modular and easy to modify for new tasks.
- Document any changes or custom workflows for reproducibility.

---

## 📚 Related Extensions & Docs

- See [`../README.md`](../README.md) for the full extension system overview.
- See [`../../workflows/`](../../workflows/) for workflow config examples.

---

**ComfyUI workflows are the creative engine for image and video generation in The-Machine.** 