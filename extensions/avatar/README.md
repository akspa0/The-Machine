# Avatar SDXL Generator Extension for The-Machine

This extension generates persona (avatar) and backdrop images for each call/persona using SDXL workflows, with per-task workflow customization. It is designed for use with The-Machine pipeline and supports compositing, chroma-keying, and advanced avatar video workflows.

---

## Purpose
- **Persona/Avatar Images:** Generate 1024x1024 green-screened images for each speaker/persona, suitable for lipsyncing and compositing.
- **Backdrop Images:** Generate wide (e.g., 1368x768) environmental images for each call, to serve as the set or background for composited avatars.
- **Workflow Customization:** Use separate, tweakable SDXL workflow JSONs for avatars and backdrops, allowing for per-task optimization (LoRA, prompt, negative prompt, etc.).

---

## Output Structure
- `comfyui_images/avatar/backdrops/` — Backdrop images for each call.
- `comfyui_images/avatar/{call_id}/{speaker}/` — Persona images for each speaker in each call.
- `comfyui_images/avatar/image_manifest.json` — Manifest mapping call_id to backdrop and (call_id, speaker) to persona image.

---

## How to Use

### CLI Example
```sh
python extensions/avatar/sdxl_avatar_generator.py \
  --persona-manifest outputs/run-YYYYMMDD-HHMMSS/characters/persona_manifest.json \
  --output-root outputs/run-YYYYMMDD-HHMMSS \
  --avatar-workflow extensions/ComfyUI/avatar/avatar_sdxl_workflow.json \
  --backdrop-workflow extensions/ComfyUI/avatar/backdrop_sdxl_workflow.json \
  --initial-prompt "a drawing of"
```

### CLI Options
- `--persona-manifest` (required): Path to persona_manifest.json
- `--output-root` (required): Path to run-* output folder
- `--avatar-workflow`: Path to avatar SDXL workflow JSON (default: extensions/ComfyUI/avatar/avatar_sdxl_workflow.json)
- `--backdrop-workflow`: Path to backdrop SDXL workflow JSON (default: extensions/ComfyUI/avatar/backdrop_sdxl_workflow.json)
- `--initial-prompt`: Initial text to prepend to the positive prompt (e.g., "a drawing of", "a photograph of")

---

## Workflow Customization
- **Avatar Workflow:**
  - Node 10 (EmptyLatentImage) is set to 1024x1024 for lipsync compatibility.
  - Designed for single-person, green-screened images.
- **Backdrop Workflow:**
  - Node 10 is set to a wide size (e.g., 1368x768) for environmental scenes.
  - Designed for backgrounds/sets for compositing avatars.
- You can further tweak LoRA, negative prompt, sampler, etc., in each workflow JSON independently.

---

## Integration with The-Machine
- Run after persona generation (`character_persona_builder.py`).
- Uses the persona manifest to generate images for each persona and call.
- Outputs are ready for downstream animation, lipsync, and compositing steps.

---

## Troubleshooting & Best Practices
- If a workflow file is missing or broken, the script will warn and skip that task.
- Use the `--initial-prompt` option to control style (drawing, photo, etc.).
- Tweak workflow JSONs for best results per task.
- All outputs and logs are PII-free and traceable.

---

## Example Usage
```sh
python extensions/avatar/sdxl_avatar_generator.py \
  --persona-manifest outputs/run-20250530-002206/characters/persona_manifest.json \
  --output-root outputs/run-20250530-002206 \
  --initial-prompt "a photograph of"
```

---

For more, see the main project README and the extensions/README.md. 