# ComfyUI Image/Video Generator Extension for The-Machine

## Purpose

This script (`comfyui_image_generator.py`) is an extension for The-Machine pipeline that generates images or video scenes using ComfyUI workflows, based on LLM-generated prompts from audio transcripts. It supports both single-image and scene-based video workflows, with robust privacy and traceability.

---

## Features
- **Scene-based prompt generation**: Segments transcripts by time or utterance, generates a prompt for each segment using an LLM.
- **Flexible workflow**: Supports both image and video generation via ComfyUI API.
- **Modular output**: Outputs are named by call name or call id for easy traceability.
- **CLI-driven**: All options are configurable via command line.
- **Privacy-focused**: No PII is logged or output; all prompts are anonymized.

---

## Usage

### Basic CLI
```sh
python extensions/comfyui_image_generator.py --run-folder <run-folder> [OPTIONS]
```

### Key CLI Options
- `--run-folder <path>`: Path to a run-YYYYMMDD-HHMMSS output folder (required)
- `--workflow <path>`: Path to the ComfyUI workflow JSON (default: SDXL image workflow)
- `--output-dir <path>`: Output directory for images/videos (default: auto-named)
- `--video` / `--image`: Select video or image workflow (mutually exclusive, default: image)
- `--master-transcript <path>`: Use a specific transcript file for prompt generation (overrides auto-detection)
- `--segmentation-mode <mode>`: `time` (default, by N seconds) or `utterance` (per transcript line)
- `--window-seconds <N>`: Time window size for scene segmentation (default: 30)
- `--update-manifest`: Update manifest.json with output metadata

### Example: Image Generation
```sh
python extensions/comfyui_image_generator.py \
  --run-folder outputs/run-20250530-002206 \
  --workflow extensions/ComfyUI/theMachine_SDXL_Basic.json \
  --image
```

### Example: Video Generation with Custom Transcript
```sh
python extensions/comfyui_image_generator.py \
  --run-folder outputs/run-20250530-002206 \
  --workflow extensions/ComfyUI/text_to_video_wan.json \
  --master-transcript "outputs/run-20250530-002206/finalized/soundbites/Turbo's Guam Gloryhole Camp/0000_master_transcript.txt" \
  --video --segmentation-mode time --window-seconds 10
```

---

## How It Works

1. **Prompt Generation**
   - Segments the transcript (by time or utterance).
   - For each segment, generates a concise, visually descriptive prompt using the LLM.
   - Saves all prompts (with time ranges) to a `.scene_prompts.json` file for each call.

2. **ComfyUI Workflow Execution**
   - For images: Uses the summary prompt to generate one or more images per call.
   - For video: Iterates over scene prompts, updating the workflow's positive prompt for each segment, and calls the ComfyUI API to generate video segments.
   - Outputs are saved in `comfyui_images/<call_name>/` or `comfyui_videos/<call_name>/`.

3. **Output Naming**
   - All outputs are named using the call name or call id, and (for video) the scene index or time range.

---

## Integration with ComfyUI
- The script calls the ComfyUI API with the selected workflow JSON.
- For video, each scene prompt is sent as a separate job; you may need to concatenate video segments downstream if your workflow does not do this natively.
- You can swap in new ComfyUI workflows (e.g., for video) without changing the prompt generation logic.

---

## Troubleshooting
- **No scene prompts JSON found**: The script will now always generate `.scene_prompts.json` if missing, as long as a valid transcript is available.
- **No transcript found**: Ensure the correct transcript file is present or use `--master-transcript`.
- **ComfyUI API errors**: Check that the ComfyUI server is running and accessible at the specified `--api-url`.
- **Output not found**: Check the output directory and naming; outputs are organized by call name/id.

---

## Advanced Options
- Use `--segmentation-mode utterance` to generate a prompt for every transcript line (useful for highly dynamic video).
- Use `--window-seconds N` to control the length of each scene in time-based segmentation.
- Use `--update-manifest` to track all outputs in the pipeline manifest.

---

## Example Output Structure
```
outputs/run-20250530-002206/
  comfyui_images/
    0000/
      ...
  comfyui_videos/
    Turbo's Guam Gloryhole Camp/
      ...
```

---

## Contact
For questions or issues, see the main The-Machine README or contact the project maintainer. 