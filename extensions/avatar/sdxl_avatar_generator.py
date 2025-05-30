import os
import json
import argparse
from pathlib import Path

# --- UTILS ---
def aggregate_transcript(speakers_root, call_id, channel_preference):
    """Aggregate transcript text from preferred channel (right-vocals, fallback to left-vocals)."""
    call_folder = speakers_root / call_id
    if not call_folder.exists():
        return None
    # Find channel folders
    channel_folders = {d.name: d for d in call_folder.iterdir() if d.is_dir()}
    for channel in channel_preference:
        for folder_name, folder_path in channel_folders.items():
            if channel in folder_name:
                # Aggregate all .txt files from all speakers
                lines = []
                for speaker_folder in folder_path.iterdir():
                    if not speaker_folder.is_dir():
                        continue
                    for txt_file in speaker_folder.glob('*.txt'):
                        try:
                            text = txt_file.read_text(encoding='utf-8').strip()
                            if text:
                                lines.append(text)
                        except Exception:
                            continue
                if lines:
                    return '\n'.join(lines)
    return None

def read_persona_md(persona_path):
    try:
        return Path(persona_path).read_text(encoding='utf-8').strip()
    except Exception:
        return ''

def update_workflow_json(base_json, positive_prompt, width, height):
    wf = json.loads(json.dumps(base_json))  # Deep copy
    # Find the node for latent/image size (by class_type or known node id)
    for node_id, node in wf.items():
        if node.get('class_type', '').lower().startswith('emptylatent') or node.get('class_type', '').lower().startswith('latent'):
            node['inputs']['width'] = width
            node['inputs']['height'] = height
        # Set positive prompt if this is the prompt node
        if node.get('class_type', '').lower().startswith('cliptextencode') and 'positive' in node.get('_meta', {}).get('title', '').lower():
            node['inputs']['text'] = positive_prompt
    return wf

def run_comfyui_workflow(workflow_json, output_path):
    # TODO: Integrate with ComfyUI API/CLI. For now, just save the workflow JSON.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(workflow_json, f, indent=2)
    # Simulate output image path
    return output_path.with_suffix('.png')

def main():
    parser = argparse.ArgumentParser(description='Generate SDXL persona and backdrop images for each call/persona.')
    parser.add_argument('--persona-manifest', type=str, required=True, help='Path to persona_manifest.json')
    parser.add_argument('--output-root', type=str, required=True, help='Path to run-* output folder')
    script_dir = Path(__file__).parent.resolve()
    default_avatar_workflow = script_dir.parent.parent / 'ComfyUI' / 'avatar' / 'avatar_sdxl_workflow.json'
    default_backdrop_workflow = script_dir.parent.parent / 'ComfyUI' / 'avatar' / 'backdrop_sdxl_workflow.json'
    parser.add_argument('--avatar-workflow', type=str, default=str(default_avatar_workflow), help='Path to avatar SDXL workflow JSON')
    parser.add_argument('--backdrop-workflow', type=str, default=str(default_backdrop_workflow), help='Path to backdrop SDXL workflow JSON')
    parser.add_argument('--initial-prompt', type=str, default='', help='Initial text to prepend to the positive prompt (e.g., "a drawing of", "a photograph of")')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    speakers_root = output_root / 'speakers'
    comfyui_images = output_root / 'comfyui_images' / 'avatar'
    backdrops_dir = comfyui_images / 'backdrops'
    backdrops_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load persona manifest
    with open(args.persona_manifest, 'r', encoding='utf-8') as f:
        persona_manifest = json.load(f)

    # 2. Load workflows
    try:
        with open(args.avatar_workflow, 'r', encoding='utf-8') as f:
            avatar_workflow = json.load(f)
    except Exception:
        print(f"[WARN] Avatar workflow not found at {args.avatar_workflow}, skipping persona image generation.")
        avatar_workflow = None
    try:
        with open(args.backdrop_workflow, 'r', encoding='utf-8') as f:
            backdrop_workflow = json.load(f)
    except Exception:
        print(f"[WARN] Backdrop workflow not found at {args.backdrop_workflow}, skipping backdrop generation.")
        backdrop_workflow = None

    # 3. Backdrop generation (one per call_id)
    call_ids = sorted(set(p['call_id'] for p in persona_manifest))
    backdrop_map = {}
    if backdrop_workflow:
        for call_id in call_ids:
            transcript = aggregate_transcript(speakers_root, call_id, channel_preference=['right-vocals', 'left-vocals'])
            if not transcript:
                print(f"[WARN] No transcript found for {call_id}, skipping backdrop.")
                continue
            prompt = f"{args.initial_prompt} {transcript}\nGenerate an image of the setting or environment described, suitable as a backdrop for this call.".strip()
            wf_json = update_workflow_json(backdrop_workflow, prompt, backdrop_workflow['10']['inputs']['width'], backdrop_workflow['10']['inputs']['height'])
            output_path = backdrops_dir / f"{call_id}_backdrop_workflow.json"
            image_path = run_comfyui_workflow(wf_json, output_path)
            backdrop_map[call_id] = str(image_path)
    else:
        print("[WARN] No backdrop workflow loaded, skipping all backdrops.")

    # 4. Persona image generation
    persona_map = {}
    if avatar_workflow:
        for entry in persona_manifest:
            call_id = entry['call_id']
            speaker = entry['speaker']
            persona_md = read_persona_md(entry['persona_path'])
            prompt = f"{args.initial_prompt} {persona_md}\nGenerate an image of the person as described, sitting in the center of the frame, in front of a solid green backdrop.".strip()
            persona_dir = comfyui_images / call_id / speaker
            persona_dir.mkdir(parents=True, exist_ok=True)
            wf_json = update_workflow_json(avatar_workflow, prompt, avatar_workflow['10']['inputs']['width'], avatar_workflow['10']['inputs']['height'])
            output_path = persona_dir / 'persona_workflow.json'
            image_path = run_comfyui_workflow(wf_json, output_path)
            persona_map[f"{call_id}_{speaker}"] = str(image_path)
    else:
        print("[WARN] No avatar workflow loaded, skipping all persona images.")

    # 5. Write manifest
    manifest = {
        'backdrops': backdrop_map,
        'personas': persona_map
    }
    manifest_path = comfyui_images / 'image_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Done. Image manifest written to {manifest_path}")

if __name__ == '__main__':
    main() 