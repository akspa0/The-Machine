#!/usr/bin/env python3
"""
comfyui_image_generator.py - Extension for The-Machine to generate images using ComfyUI API and a workflow JSON.

Usage (CLI):
    python comfyui_image_generator.py --run-folder outputs/run-YYYYMMDD-HHMMSS \
        --prompt "A cat in a janitor costume eating fish" \
        --workflow extensions/ComfyUI/theMachine_SDXL_Basic.json \
        --output-dir images/ \
        --seed 42

If run from pipeline_orchestrator.py, the script will auto-detect the run folder and use LLM outputs or master transcripts as prompts.

Options:
    --run-folder      Path to a run-YYYYMMDD-HHMMSS output folder (required)
    --prompt         Text prompt for image generation (overrides workflow prompt)
    --prompt-file    Path to a file containing the prompt (e.g., LLM output or transcript)
    --workflow       Path to the ComfyUI workflow JSON (default: extensions/ComfyUI/theMachine_SDXL_Basic.json)
    --output-dir     Output directory for images (default: <run-folder>/comfyui_images)
    --seed           Random seed for generation (optional)
    --batch-size     Number of images to generate (default: 1)
    --api-url        ComfyUI API URL (default: http://127.0.0.1:8188)
    --update-manifest  If set, update manifest.json with image metadata

All outputs and logs are anonymized and PII-free.
"""
import argparse
import json
import os
from pathlib import Path
import requests
from datetime import datetime

def load_prompt(args, run_folder):
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    else:
        # Try to auto-detect a prompt source (LLM output or master transcript)
        llm_dir = run_folder / 'llm'
        prompt = None
        if llm_dir.exists():
            for call_id in sorted(llm_dir.iterdir()):
                if call_id.is_dir():
                    for file in call_id.iterdir():
                        if file.name.endswith('.txt'):
                            with open(file, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                if text:
                                    prompt = text
                                    break
                if prompt:
                    break
        if not prompt:
            # Fallback: master transcript
            soundbites_dir = run_folder / 'soundbites'
            for call_id in sorted(soundbites_dir.iterdir()):
                master_txt = call_id / f"{call_id.name}_master_transcript.txt"
                if master_txt.exists():
                    with open(master_txt, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            prompt = text
                            break
        if not prompt:
            prompt = "A surreal, privacy-safe image."
    # Truncate prompt to 300 characters
    return prompt[:300]

def update_workflow_prompt(workflow, prompt, batch_size=1, seed=None):
    # Find the node with class_type 'CLIPTextEncode' and _meta.title == 'Positive prompt'
    for node in workflow.values():
        if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'Positive prompt':
            node['inputs']['text'] = prompt
    # Set batch size if present
    for node in workflow.values():
        if node.get('class_type') == 'EmptyLatentImage':
            node['inputs']['batch_size'] = batch_size
    # Set seed if present
    for node in workflow.values():
        if node.get('class_type') == 'KSamplerAdvanced' and seed is not None:
            node['inputs']['noise_seed'] = int(seed)
    return workflow

def call_comfyui_api(api_url, workflow):
    url = f"{api_url}/prompt"
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(workflow))
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI API error: {response.status_code} {response.text}")
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="ComfyUI Image Generator Extension for The-Machine")
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run-YYYYMMDD-HHMMSS output folder')
    parser.add_argument('--prompt', type=str, help='Text prompt for image generation')
    parser.add_argument('--prompt-file', type=str, help='File containing prompt text')
    parser.add_argument('--workflow', type=str, default='extensions/ComfyUI/theMachine_SDXL_Basic.json', help='Path to ComfyUI workflow JSON')
    parser.add_argument('--output-dir', type=str, help='Output directory for images (default: <run-folder>/comfyui_images)')
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:8188', help='ComfyUI API URL')
    parser.add_argument('--update-manifest', action='store_true', help='Update manifest.json with image metadata')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder not found: {run_folder}")
    output_dir = Path(args.output_dir) if args.output_dir else run_folder / 'comfyui_images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load workflow
    with open(args.workflow, 'r', encoding='utf-8') as f:
        workflow = json.load(f)

    # Load prompt
    prompt = load_prompt(args, run_folder)
    print(f"[DEBUG] Prompt to be used (truncated to 300 chars): {repr(prompt)}")

    # Update workflow with prompt, batch size, seed
    workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)

    # DEBUG: Print the Positive prompt node after update
    for node_id, node in workflow.items():
        if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'Positive prompt':
            print(f"[DEBUG] Positive prompt node after update: {json.dumps(node, indent=2)}")
            if not node['inputs'].get('text'):
                print("[ERROR] No prompt set in Positive prompt node! Aborting.")
                exit(1)

    # DEBUG: Print workflow to verify prompt is set
    print("[DEBUG] Workflow to be sent to ComfyUI API:")
    print(json.dumps(workflow, indent=2))

    # Call ComfyUI API
    print(f"[INFO] Sending workflow to ComfyUI API at {args.api_url} ...")
    result = call_comfyui_api(args.api_url, workflow)
    print(f"[INFO] ComfyUI API response: {result}")

    # Save metadata
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    meta_path = output_dir / f'comfyui_metadata_{timestamp}.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] Metadata saved to {meta_path}")

    # Optionally update manifest
    if args.update_manifest:
        manifest_path = run_folder / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
        else:
            manifest = []
        manifest.append({
            'stage': 'comfyui_image_generation',
            'timestamp': timestamp,
            'prompt': prompt,
            'workflow': str(args.workflow),
            'output_dir': str(output_dir),
            'result': result
        })
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f"[INFO] Manifest updated at {manifest_path}")

if __name__ == '__main__':
    main() 