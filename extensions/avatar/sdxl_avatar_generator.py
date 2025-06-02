import os
import json
import argparse
from pathlib import Path
import requests
import random
import sys
import time
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from llm_utils import run_llm_task, split_into_chunks_advanced

# --- UTILS ---
def aggregate_transcript(speakers_root, call_id, channel_preference):
    """Aggregate transcript text from preferred channel (right-vocals, fallback to left-vocals)."""
    call_folder = speakers_root / call_id
    if not call_folder.exists():
        return None
    # Look for channel_segments.txt in each channel folder
    for channel in channel_preference:
        channel_folder = call_folder / channel
        if channel_folder.exists():
            seg_file = channel_folder / f"{channel}_segments.txt"
            if seg_file.exists():
                try:
                    text = seg_file.read_text(encoding='utf-8').strip()
                    if text:
                        return text
                except Exception:
                    continue
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

# --- API-COMPLIANT COMFYUI UTILS ---
def submit_comfyui_workflow(workflow_json, api_url):
    payload = {"prompt": workflow_json}
    resp = requests.post(f"{api_url.rstrip('/')}/prompt", json=payload)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR] ComfyUI API error: {resp.status_code} {resp.text}")
        raise
    data = resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Failed to submit workflow: {data}")
    return prompt_id

def wait_for_comfyui_completion(prompt_id, api_url, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{api_url.rstrip('/')}/history/{prompt_id}")
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "completed":
            return data
        time.sleep(2)
    raise TimeoutError("ComfyUI job did not complete in time.")

def get_output_files_from_result(result, file_exts=(".png", ".webp")):
    files = []
    for out in result.get('outputs', []):
        fname = out.get('filename', '')
        if fname.lower().endswith(file_exts):
            files.append(fname)
    return files

def score_segments_with_llm(segments, llm_config):
    prompt_template = (
        "Rate the following transcript segment for how visually interesting or funny it would be as an image prompt, on a scale from 1 (boring) to 10 (very interesting/funny). Only output the number.\n\nSegment:\n{segment}\n\nScore:"
    )
    scores = []
    for segment in segments:
        prompt = prompt_template.format(segment=segment)
        response = run_llm_task(prompt, llm_config, single_output=True, chunking=False)
        try:
            score = float(response.strip().split()[0])
        except Exception:
            score = 5.0  # fallback if LLM output is not a number
        scores.append(score)
    return scores

def check_llm_api_available(llm_config, timeout=5):
    base_url = llm_config.get('lm_studio_base_url', 'http://localhost:1234/v1')
    try:
        response = requests.get(base_url, timeout=timeout)
        return response.status_code == 200 or response.status_code == 404  # 404 is OK for /v1
    except Exception:
        return False

def generate_llm_prompt_for_persona(persona_md, llm_config):
    """
    Generate a visually rich, detailed SDXL image prompt for the given persona description.
    If the persona_md is long (>2048 tokens), chunk it and aggregate LLM responses.
    """
    prompt_template = (
        "Generate a visually rich, detailed SDXL image prompt for the following persona description. Focus on unique visual details, mood, and style.\n\nPersona:\n{persona}\n\nPrompt:"
    )
    # Token-aware chunking for long persona descriptions
    chunks = split_into_chunks_advanced(persona_md, max_tokens=2048, model=llm_config.get('lm_studio_model_identifier', 'gpt-3.5-turbo'))
    llm_available = check_llm_api_available(llm_config)
    responses = []
    if llm_available:
        try:
            for chunk in chunks:
                prompt = prompt_template.format(persona=chunk)
                llm_output = run_llm_task(prompt, llm_config, single_output=True, chunking=False)
                responses.append(llm_output.strip())
            # Aggregate responses if chunked
            if len(responses) > 1:
                return '\n'.join(responses)
            else:
                return responses[0]
        except Exception as e:
            print(f"[WARN] LLM API call failed: {e}. Using fallback prompt.")
            return prompt_template.format(persona=persona_md)
    else:
        print("[WARN] LLM API not available. Using fallback prompt.")
        return prompt_template.format(persona=persona_md)

def set_random_seed_in_workflow(wf_json):
    for node in wf_json.values():
        if node.get('class_type') == 'KSamplerAdvanced':
            node['inputs']['noise_seed'] = random.randint(0, 2**32 - 1)
    return wf_json

# Define the system prompt for persona processing
SYSTEM_PROMPT = (
    "You are an expert at writing SDXL image prompts for creative, visually rich character portraits. "
    "Always treat the input as a character or person, even if it is a business, object, or odd name—personify it as needed. "
    "Favor creativity, absurd humor, and surreal situations in your prompt generation. "
    "Your output will be used to generate a unique, memorable, and visually striking image."
)

def run_comfyui_workflow(workflow_json, api_url):
    prompt_id = submit_comfyui_workflow(workflow_json, api_url)
    print(f"[INFO] Submitted workflow to ComfyUI, prompt_id: {prompt_id}")
    result = wait_for_comfyui_completion(prompt_id, api_url)
    print(f"[INFO] ComfyUI job completed for prompt_id: {prompt_id}")
    output_files = get_output_files_from_result(result)
    if not output_files:
        print(f"[WARN] No output files found for prompt_id: {prompt_id}")
    return output_files

def main():
    parser = argparse.ArgumentParser(description='Generate SDXL persona and backdrop images for each call/persona.')
    parser.add_argument('--persona-manifest', type=str, required=True, help='Path to persona_manifest.json')
    parser.add_argument('--output-root', type=str, required=True, help='Path to run-* output folder')
    script_dir = Path(__file__).parent.resolve()
    default_avatar_workflow = script_dir.parent / 'ComfyUI' / 'avatar' / 'avatar_sdxl_workflow.json'
    default_backdrop_workflow = script_dir.parent / 'ComfyUI' / 'avatar' / 'backdrop_sdxl_workflow.json'
    parser.add_argument('--avatar-workflow', type=str, default=str(default_avatar_workflow), help='Path to avatar SDXL workflow JSON')
    parser.add_argument('--backdrop-workflow', type=str, default=str(default_backdrop_workflow), help='Path to backdrop SDXL workflow JSON')
    parser.add_argument('--initial-prompt', type=str, default='', help='Initial text to prepend to the positive prompt (e.g., "a drawing of", "a photograph of")')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:8188', help='ComfyUI API URL (default: http://127.0.0.1:8188)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of ComfyUI jobs to submit per batch before pausing (default: 10)')
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

    # Load LLM config for scoring
    llm_config_path = Path('workflows/llm_tasks.json')
    if llm_config_path.exists():
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
    else:
        llm_config = {}

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
            # API-compliant: submit and wait for output
            output_files = run_comfyui_workflow(wf_json, api_url=args.api_url)
            if output_files:
                backdrop_map[call_id] = output_files[0]  # Use first output file
            else:
                print(f"[WARN] No backdrop image generated for {call_id}.")
    else:
        print("[WARN] No backdrop workflow loaded, skipping all backdrops.")

    # 4. Persona image generation
    persona_map = {}
    if avatar_workflow:
        # Group persona_manifest entries by (call_id, speaker)
        from collections import defaultdict
        speaker_entries = defaultdict(list)
        for entry in persona_manifest:
            speaker_entries[(entry['call_id'], entry['speaker'])].append(entry)
        for (call_id, speaker), entries in speaker_entries.items():
            # There should be only one persona.md per speaker per call
            entry = entries[0]
            persona_md = read_persona_md(entry['persona_path'])
            persona_dir = comfyui_images / call_id / speaker
            persona_dir.mkdir(parents=True, exist_ok=True)
            wf_json = update_workflow_json(avatar_workflow, '', avatar_workflow['10']['inputs']['width'], avatar_workflow['10']['inputs']['height'])
            # Insert persona_md text into node 14's text input
            if '14' in wf_json and 'inputs' in wf_json['14']:
                wf_json['14']['inputs']['text'] = persona_md
            else:
                print(f"[WARN] Node 14 not found in workflow JSON for {call_id} {speaker}. Skipping.")
                continue
            # Insert persona_md text into node 1's text input
            if '1' in wf_json and 'inputs' in wf_json['1']:
                wf_json['1']['inputs']['text'] = persona_md
            else:
                print(f"[WARN] Node 1 not found in workflow JSON for {call_id} {speaker}. Skipping.")
                continue
            # Insert system prompt into node 13's system_message
            if '13' in wf_json and 'inputs' in wf_json['13']:
                wf_json['13']['inputs']['system_message'] = SYSTEM_PROMPT
            else:
                print(f"[WARN] Node 13 not found in workflow JSON for {call_id} {speaker}. Skipping.")
                continue
            wf_json = set_random_seed_in_workflow(wf_json)
            # API-compliant: submit and wait for output
            output_files = run_comfyui_workflow(wf_json, api_url=args.api_url)
            if output_files:
                persona_map[f"{call_id}_{speaker}"] = output_files[0]  # Use first output file
            else:
                print(f"[WARN] No persona image generated for {call_id} {speaker}.")
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