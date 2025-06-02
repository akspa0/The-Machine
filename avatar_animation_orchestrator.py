import os
import json
import argparse
from pathlib import Path
import subprocess
import sys
import shutil
import time
import requests
import random

# --- CONFIGURABLE ---
DEFAULT_WORKFLOW_PATH = 'extensions/ComfyUI/avatar/image_to_video_wan.json'
VIDEO_WORKFLOW_TEMPLATE = 'extensions/ComfyUI/avatar/img2wan-avatar.json'
AVATAR_WORKFLOW_TEMPLATE = 'extensions/ComfyUI/avatar/avatar_sdxl_workflow.json'
COMFYUI_RUN_SCRIPT = 'extensions/ComfyUI/run_comfyui_job.py'  # Placeholder for actual ComfyUI invocation
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp']
VIDEO_OUTPUT_DIR = 'comfyui_videos/single_file/'  # Relative to run folder
COMFYUI_URL = "http://localhost:8188"  # Default ComfyUI server port
AVATAR_IMAGE_DIR = 'comfyui_images/avatar/'

# --- UTILS ---
def run_subprocess(cmd, check=True):
    print(f"[INFO] Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(str(c) for c in cmd)}")
        print(result.stdout)
        print(result.stderr)
        if check:
            sys.exit(result.returncode)
    else:
        print(result.stdout)
    return result

def collect_avatar_images(output_root):
    """Collect all avatar images in the output root (recursively)."""
    image_paths = []
    for root, _, files in os.walk(output_root):
        for f in files:
            if any(f.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, f))
    return image_paths

def wait_for_file(filepath, timeout=15):
    """Wait up to timeout seconds for a file to appear."""
    start = time.time()
    while not os.path.exists(filepath) and (time.time() - start) < timeout:
        time.sleep(0.5)
    return os.path.exists(filepath)

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

def wait_for_comfyui_completion(prompt_id, api_url, timeout=3600, max_empty=300):
    start = time.time()
    empty_count = 0
    while time.time() - start < timeout:
        resp = requests.get(f"{api_url.rstrip('/')}/history/{prompt_id}")
        resp.raise_for_status()
        data = resp.json()
        print(f"[DEBUG] Polling ComfyUI for prompt_id {prompt_id}: {data}")
        if not data:
            empty_count += 1
            if empty_count >= max_empty:
                print(f"[ERROR] No history found for prompt_id {prompt_id} after {max_empty} tries. Assuming job is done or lost.")
                return {}
            time.sleep(10)  # Increase poll interval for long jobs
            continue
        empty_count = 0
        return data
    raise TimeoutError("ComfyUI job did not complete in time.")

def get_output_files_from_result(result, file_exts=(".png", ".webp")):
    files = []
    for out in result.get('outputs', []):
        fname = out.get('filename', '')
        if fname.lower().endswith(file_exts):
            files.append(fname)
    return files

def set_random_seed_in_workflow(wf_json):
    for node in wf_json.values():
        if node.get('class_type') == 'KSamplerAdvanced':
            node['inputs']['noise_seed'] = random.randint(0, 2**32 - 1)
    return wf_json

def read_persona_md(persona_path):
    try:
        return Path(persona_path).read_text(encoding='utf-8').strip()
    except Exception:
        return ''

def update_workflow_json(base_json, persona_md):
    wf = json.loads(json.dumps(base_json))  # Deep copy
    # Insert persona_md text into node 14's text input
    if '14' in wf and 'inputs' in wf['14']:
        wf['14']['inputs']['text'] = persona_md
    # Insert persona_md text into node 1's text input
    if '1' in wf and 'inputs' in wf['1']:
        wf['1']['inputs']['text'] = persona_md
    # Insert system prompt into node 13's system_message
    if '13' in wf and 'inputs' in wf['13']:
        wf['13']['inputs']['system_message'] = (
            "You are an expert at writing SDXL image prompts for creative, visually rich character portraits. "
            "Always treat the input as a character or person, even if it is a business, object, or odd name—personify it as needed. "
            "Favor creativity, absurd humor, and surreal situations in your prompt generation. "
            "Your output will be used to generate a unique, memorable, and visually striking image."
        )
    return wf

def copy_comfyui_outputs_to_project(comfyui_files, project_output_dir):
    os.makedirs(project_output_dir, exist_ok=True)
    copied_files = []
    for src in comfyui_files:
        dst = os.path.join(project_output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        copied_files.append(dst)
    return copied_files

def run_comfyui_workflow(workflow_json, api_url, filename_prefix=None, comfyui_output_dir="output", expected_exts=(".png", ".webp"), project_output_dir=None):
    prompt_id = submit_comfyui_workflow(workflow_json, api_url)
    print(f"[INFO] Submitted workflow to ComfyUI, prompt_id: {prompt_id}")
    job_start_time = time.time()
    result = wait_for_comfyui_completion(prompt_id, api_url)
    print(f"[INFO] ComfyUI job completed for prompt_id: {prompt_id}")
    print(f"[DEBUG] Full ComfyUI result: {result}")
    output_files = get_output_files_from_result(result)
    if not output_files:
        # Fallback: scan ComfyUI output dir for new files with the prefix
        if not os.path.isdir(comfyui_output_dir):
            print(f"[ERROR] ComfyUI output directory '{comfyui_output_dir}' does not exist. Please specify the correct path with --comfyui-output-dir.")
            return []
        files = []
        for f in os.listdir(comfyui_output_dir):
            if not f.lower().endswith(expected_exts):
                continue
            if filename_prefix and not f.startswith(filename_prefix):
                continue
            full_path = os.path.join(comfyui_output_dir, f)
            if os.path.getmtime(full_path) >= job_start_time - 2:  # allow 2s clock skew
                files.append(full_path)
        if files:
            print(f"[INFO] Fallback: found files in {comfyui_output_dir} with prefix {filename_prefix}: {files}")
            output_files = files
    if not output_files:
        print(f"[WARN] No output files found for prompt_id: {prompt_id}")
    # Copy files to project output dir if specified
    if output_files and project_output_dir:
        copied_files = copy_comfyui_outputs_to_project(output_files, project_output_dir)
        return copied_files
    return output_files

def set_filename_prefix_in_workflow(wf_json, prefix):
    for node_id, node in wf_json.items():
        if 'inputs' in node and 'filename_prefix' in node['inputs']:
            node['inputs']['filename_prefix'] = prefix
            return True
    print(f"[WARN] No node with 'filename_prefix' found in workflow. Prefix not set.")
    return False

def copy_and_rename_avatar(src, call_id, speaker, project_output_root):
    ext = os.path.splitext(src)[1]
    target_dir = os.path.join(project_output_root, call_id, speaker)
    os.makedirs(target_dir, exist_ok=True)
    dst = os.path.join(target_dir, f"persona_avatar{ext}")
    shutil.copy2(src, dst)
    return dst

def upload_image_to_comfyui_api(image_path, api_url):
    url = f"{api_url.rstrip('/')}/upload/image"
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/png')}
        resp = requests.post(url, files=files)
        resp.raise_for_status()
        data = resp.json()
        return data['name']  # e.g., 'input/filename.png'

def main():
    parser = argparse.ArgumentParser(description='Orchestrate persona, avatar, and animation generation for a run-* folder.\n\nWorkflow:\n1. Run persona generator (character_persona_builder.py) if needed.\n2. Run SDXL avatar generator (sdxl_avatar_generator.py).\n3. (Optional) Run animation/video generation.')
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run-* output folder.')
    parser.add_argument('--skip-persona', action='store_true', help='Skip persona generation step.')
    parser.add_argument('--skip-avatar', action='store_true', help='Skip SDXL avatar image generation step.')
    parser.add_argument('--persona-generator', type=str, default='extensions/character_persona_builder.py', help='Path to persona generator script.')
    parser.add_argument('--avatar-generator', type=str, default='extensions/avatar/sdxl_avatar_generator.py', help='Path to SDXL avatar generator script.')
    parser.add_argument('--persona-manifest', type=str, default=None, help='Path to persona_manifest.json (auto-detect if not set).')
    parser.add_argument('--output-root', type=str, default=None, help='Output root for avatar images (defaults to run-folder).')
    parser.add_argument('--api-url', type=str, default=COMFYUI_URL, help='ComfyUI API URL (default: http://localhost:8188)')
    parser.add_argument('--comfyui-output-dir', type=str, default='output', help='ComfyUI output directory (default: output)')
    parser.add_argument('--comfyui-input-dir', type=str, default='input', help='ComfyUI input directory (default: input)')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        print(f"[ERROR] Run folder not found: {run_folder}")
        sys.exit(1)

    # Step 1: Persona generation
    persona_dir = run_folder / 'characters'
    persona_manifest_path = args.persona_manifest or (persona_dir / 'persona_manifest.json')
    if not persona_manifest_path.exists():
        print(f"[INFO] Persona manifest not found at {persona_manifest_path}, running character_persona_builder.py...")
        cmd = [sys.executable, "extensions/character_persona_builder.py", str(run_folder)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("[ERROR] character_persona_builder.py failed.")
            exit(1)
        if not persona_manifest_path.exists():
            print("[ERROR] Persona manifest still not found after running character_persona_builder.py.")
            exit(1)
        print(f"[INFO] Persona manifest generated at {persona_manifest_path}.")
    else:
        print(f"[INFO] Persona manifest found at {persona_manifest_path}, skipping persona generation.")

    # Step 2: SDXL avatar image generation (direct API, no subprocess)
    output_root = args.output_root or str(run_folder)
    comfyui_images = Path(output_root) / AVATAR_IMAGE_DIR
    comfyui_images.mkdir(parents=True, exist_ok=True)
    try:
        with open(AVATAR_WORKFLOW_TEMPLATE, 'r', encoding='utf-8') as f:
            avatar_workflow = json.load(f)
    except Exception:
        print(f"[ERROR] Avatar workflow not found at {AVATAR_WORKFLOW_TEMPLATE}, aborting avatar image generation.")
        avatar_workflow = None
    persona_map = {}
    avatar_prompt_cache = {}
    if not args.skip_avatar and avatar_workflow:
        from collections import defaultdict
        speaker_entries = defaultdict(list)
        with open(persona_manifest_path, 'r', encoding='utf-8') as f:
            persona_manifest = json.load(f)
        for entry in persona_manifest:
            speaker_entries[(entry['call_id'], entry['speaker'])].append(entry)
        for (call_id, speaker), entries in speaker_entries.items():
            entry = entries[0]
            persona_md = read_persona_md(entry['persona_path'])
            wf_json = update_workflow_json(avatar_workflow, persona_md)
            wf_json = set_random_seed_in_workflow(wf_json)
            output_prefix = f"{call_id}_{speaker}"
            if set_filename_prefix_in_workflow(wf_json, output_prefix):
                output_files = run_comfyui_workflow(wf_json, api_url=args.api_url, filename_prefix=output_prefix, comfyui_output_dir=args.comfyui_output_dir, project_output_dir=str(comfyui_images))
                if output_files:
                    # Copy and rename to speaker folder as persona_avatar
                    avatar_path = copy_and_rename_avatar(output_files[0], call_id, speaker, str(comfyui_images))
                    persona_map[f"{call_id}_{speaker}"] = avatar_path
                    prompt_used = None
                    # Try to extract the prompt from the workflow (e.g., node 14 or 1)
                    if '14' in wf_json and 'inputs' in wf_json['14'] and 'text' in wf_json['14']['inputs']:
                        prompt_used = wf_json['14']['inputs']['text']
                    elif '1' in wf_json and 'inputs' in wf_json['1'] and 'text' in wf_json['1']['inputs']:
                        prompt_used = wf_json['1']['inputs']['text']
                    if prompt_used is not None:
                        avatar_prompt_cache[f"{call_id}_{speaker}"] = prompt_used
                else:
                    print(f"[WARN] No persona image generated for {call_id} {speaker}.")
    else:
        print("[INFO] Skipping SDXL avatar image generation step (per CLI flag or missing workflow).")

    # Write manifest
    manifest = {'personas': persona_map}
    manifest_path = comfyui_images / 'image_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Avatar image manifest written to {manifest_path}")

    # After avatar image generation, save the cache to file:
    with open(os.path.join(output_root, 'avatar_prompt_cache.json'), 'w', encoding='utf-8') as f:
        json.dump(avatar_prompt_cache, f, indent=2)

    # Step 3: Video animation (unchanged)
    print("[INFO] Collecting avatar image outputs...")
    avatar_images = collect_avatar_images(str(comfyui_images))
    print(f"[INFO] Found {len(avatar_images)} avatar images.")
    persona_dirs = [d for d in persona_dir.iterdir() if d.is_dir()]
    video_outputs = []
    # Load the prompt cache and manifest
    with open(os.path.join(output_root, 'avatar_prompt_cache.json'), 'r', encoding='utf-8') as f:
        avatar_prompt_cache = json.load(f)
    manifest_path = comfyui_images / 'image_manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        persona_manifest = json.load(f)['personas']

    for cache_key, prompt in avatar_prompt_cache.items():
        if '_' not in cache_key:
            print(f"[WARN] Unexpected cache key format: {cache_key}, skipping.")
            continue
        call_id, speaker = cache_key.split('_', 1)
        video_prompt = prompt.strip() + " is talking to the camera"
        persona_folder = os.path.join(str(comfyui_images), call_id, speaker)
        avatar_img = None
        for ext in ('.png', '.webp'):
            candidate = os.path.join(persona_folder, f"persona_avatar{ext}")
            if os.path.exists(candidate):
                avatar_img = candidate
                break
        if not avatar_img:
            print(f"[WARN] No persona_avatar image found for {persona_folder}, skipping video workflow.")
            continue
        output_prefix = f"{call_id}_{speaker}_video"
        with open(VIDEO_WORKFLOW_TEMPLATE, 'r', encoding='utf-8') as f:
            video_workflow = json.load(f)
        video_workflow['55']['inputs']['text'] = video_prompt
        video_workflow['28']['inputs']['filename_prefix'] = output_prefix
        # Upload avatar image to ComfyUI API and use the returned path as the image input for the video workflow
        comfyui_image_path = upload_image_to_comfyui_api(avatar_img, args.api_url)
        video_workflow['52']['inputs']['image'] = comfyui_image_path
        video_output_files = run_comfyui_workflow(video_workflow, api_url=args.api_url, filename_prefix=output_prefix, comfyui_output_dir=args.comfyui_output_dir, project_output_dir=persona_folder)
        video_outputs.extend(video_output_files)
    print(f"[INFO] All video outputs: {video_outputs}")
    print("[INFO] Animation/video generation step complete.")
    print("[INFO] All required persona, avatar image, and video outputs should now be present in the run folder.")

if __name__ == '__main__':
    main() 