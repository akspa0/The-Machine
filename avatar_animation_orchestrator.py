import os
import json
import argparse
from pathlib import Path

# --- CONFIGURABLE ---
DEFAULT_SDXL_DIR = 'outputs/sdxl_images'  # Example default, update as needed
DEFAULT_PERSONA_DIR = 'outputs/personas'  # Example default, update as needed
DEFAULT_OUTPUT_DIR = 'outputs/animated_webp'
DEFAULT_WORKFLOW_PATH = 'extensions/ComfyUI/avatar/image_to_video_wan.json'
DEFAULT_MANIFEST_PATH = 'outputs/animated_webp/manifest.json'

# --- UTILS ---
def find_sdxl_images(sdxl_dir):
    """Scan SDXL output dir and build mapping: (call_id, speaker, timestamps) -> image path."""
    mapping = {}
    for root, _, files in os.walk(sdxl_dir):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Example: callid_SPEAKER00_0000-0123_closeup.png
                parts = fname.split('_')
                if len(parts) < 3:
                    continue
                call_id = parts[0]
                speaker = parts[1]
                timestamps = parts[2].split('.')[0]  # Remove extension
                mapping[(call_id, speaker, timestamps)] = os.path.join(root, fname)
    return mapping

def load_persona(persona_dir, speaker):
    """Load persona text for a given speaker (expects persona.md or .txt)."""
    for ext in ('.md', '.txt'):
        path = os.path.join(persona_dir, f'{speaker}{ext}')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return f.read().strip()
    return ''

def generate_prompt(persona, action=None, shot=None):
    """Generate a positive prompt for animation."""
    prompt = persona
    if action:
        prompt += f" {action}"
    if shot:
        prompt += f" ({shot})"
    return prompt.strip()

def update_workflow_json(base_json, positive_prompt, image_path, filename_prefix):
    """Return a modified workflow dict with updated node 6, 52, 28."""
    wf = json.loads(json.dumps(base_json))  # Deep copy
    # Node 6: positive prompt
    wf['6']['inputs']['text'] = positive_prompt
    # Node 52: image path
    wf['52']['inputs']['image'] = image_path
    # Node 28: filename_prefix
    wf['28']['inputs']['filename_prefix'] = filename_prefix
    return wf

def run_comfyui_workflow(workflow_json, output_dir):
    """Placeholder for running the workflow via ComfyUI API/CLI. Save the workflow JSON for now."""
    # TODO: Integrate with ComfyUI API/CLI
    wf_path = os.path.join(output_dir, f"workflow_{workflow_json['28']['inputs']['filename_prefix']}.json")
    with open(wf_path, 'w', encoding='utf-8') as f:
        json.dump(workflow_json, f, indent=2)
    # Simulate output path
    webp_path = os.path.join(output_dir, f"{workflow_json['28']['inputs']['filename_prefix']}.webp")
    return webp_path

def main():
    parser = argparse.ArgumentParser(description='Orchestrate avatar animation using image_to_video_wan.json workflow.')
    parser.add_argument('--sdxl-dir', type=str, default=DEFAULT_SDXL_DIR, help='Directory with SDXL images.')
    parser.add_argument('--persona-dir', type=str, default=DEFAULT_PERSONA_DIR, help='Directory with persona files.')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Directory for animated webp outputs.')
    parser.add_argument('--workflow', type=str, default=DEFAULT_WORKFLOW_PATH, help='Path to base workflow JSON.')
    parser.add_argument('--manifest', type=str, default=DEFAULT_MANIFEST_PATH, help='Path to output manifest JSON.')
    parser.add_argument('--segments-json', type=str, required=True, help='JSON file with segment metadata (call_id, speaker, timestamps, action, shot, etc.).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SDXL mapping
    sdxl_map = find_sdxl_images(args.sdxl_dir)

    # 2. Load base workflow
    with open(args.workflow, 'r', encoding='utf-8') as f:
        base_workflow = json.load(f)

    # 3. Load segments metadata
    with open(args.segments_json, 'r', encoding='utf-8') as f:
        segments = json.load(f)

    manifest = []
    for seg in segments:
        call_id = seg['call_id']
        speaker = seg['speaker']
        timestamps = seg['timestamps']
        action = seg.get('action')
        shot = seg.get('shot')
        key = (call_id, speaker, timestamps)
        sdxl_image = sdxl_map.get(key)
        if not sdxl_image:
            print(f"[WARN] No SDXL image for {key}, skipping.")
            continue
        persona = load_persona(args.persona_dir, speaker)
        positive_prompt = generate_prompt(persona, action, shot)
        filename_prefix = f"{call_id}_{speaker}_{timestamps}"
        wf_json = update_workflow_json(base_workflow, positive_prompt, sdxl_image, filename_prefix)
        webp_path = run_comfyui_workflow(wf_json, args.output_dir)
        manifest.append({
            'call_id': call_id,
            'speaker': speaker,
            'timestamps': timestamps,
            'shot': shot,
            'action': action,
            'persona': persona,
            'sdxl_image': sdxl_image,
            'animated_webp': webp_path,
            'positive_prompt': positive_prompt
        })

    # 4. Write manifest
    with open(args.manifest, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Manifest written to {args.manifest}")
    print("[TODO] Integrate lipsync step using generated animated webps.")

if __name__ == '__main__':
    main() 