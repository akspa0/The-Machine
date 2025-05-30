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

Behavior:
    - Always generates a new SDXL image prompt using the LLM for every run, regardless of whether a previous prompt file exists.
    - Loads the transcript, truncates to 300 tokens, and calls the LLM to generate a new prompt, saving the result and seed.
    - If the LLM call fails, falls back to a default privacy-safe prompt and logs a warning.

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
    --master-transcript Path to a master transcript file to use for LLM prompt generation (overrides automatic search)
    --segmentation-mode Scene segmentation mode: time (default) or utterance
    --window-seconds Time window size in seconds for scene segmentation (default: 30)

All outputs and logs are anonymized and PII-free.
"""
import argparse
import json
import os
from pathlib import Path
import requests
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))
from llm_utils import run_llm_task
from glob import glob
import difflib
import re

# Tokenization utility
try:
    import tiktoken
    def truncate_to_tokens(text, max_tokens=512, model="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            print(f"[WARN] Transcript truncated from {len(tokens)} to {max_tokens} tokens for LLM prompt generation.")
            tokens = tokens[:max_tokens]
            return enc.decode(tokens)
        return text
except ImportError:
    def truncate_to_tokens(text, max_tokens=512, model=None):
        # Fallback: 1 token ≈ 4 chars
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            print(f"[WARN] Transcript truncated from {len(text)} to {max_chars} chars (approx {max_tokens} tokens) for LLM prompt generation.")
            return text[:max_chars]
        return text

def find_llm_image_prompt(run_folder):
    llm_dir = run_folder / 'llm'
    if llm_dir.exists():
        for call_id in sorted(llm_dir.iterdir()):
            if call_id.is_dir():
                prompt_file = call_id / 'sdxl_image_prompt.txt'
                if prompt_file.exists():
                    return prompt_file.read_text(encoding='utf-8').strip(), prompt_file
        # Single-file mode: look for llm/sdxl_image_prompt.txt
        prompt_file = llm_dir / 'sdxl_image_prompt.txt'
        if prompt_file.exists():
            return prompt_file.read_text(encoding='utf-8').strip(), prompt_file
    return None, None

def generate_image_prompt_with_llm(transcript, config, output_path=None):
    # Find the sdxl_image_prompt task
    sdxl_task = None
    for task in config.get('llm_tasks', []):
        if task.get('name') == 'sdxl_image_prompt':
            sdxl_task = task
            break
    if not sdxl_task:
        print("[ERROR] No sdxl_image_prompt task found in llm_tasks.json. Using fallback.")
        return "A surreal, privacy-safe image.", None
    prompt_template = sdxl_task.get('prompt_template')
    prompt = prompt_template.replace('{transcript}', transcript)
    # Use run_llm_task from llm_utils.py
    print("[INFO] Generating SDXL image prompt using LLM...")
    seed = None
    result = run_llm_task(prompt, config, output_path=output_path, seed=seed)
    return result, seed

def postprocess_prompt(prompt, max_chars=512):
    import re
    # Remove leading 'Prompt:' or similar headers
    prompt = re.sub(r'^\s*Prompt:\s*', '', prompt, flags=re.IGNORECASE)
    # Collapse newlines to spaces
    prompt = prompt.replace('\n', ' ')
    # Strip leading/trailing whitespace
    prompt = prompt.strip()
    # Truncate to max_chars
    if len(prompt) > max_chars:
        print(f"[WARN] Final prompt truncated from {len(prompt)} to {max_chars} characters for ComfyUI.")
        prompt = prompt[:max_chars]
    return prompt

def is_llm_error_prompt(prompt: str) -> bool:
    """Detect if the prompt is an LLM error message or context overflow."""
    if not prompt:
        return True
    error_patterns = [
        'LLM API error',
        'context the overflows',
        'error 400',
        'not enough',
        'API error',
        'context length',
        'Try to load the model with a larger context length',
        'provide a shorter input',
    ]
    prompt_lower = prompt.lower()
    return any(pat.lower() in prompt_lower for pat in error_patterns)

def safe_load_prompt_file(prompt_file, fallback_prompt="A surreal, privacy-safe image."):
    """Load prompt from file, fallback if error detected."""
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    if is_llm_error_prompt(prompt):
        print(f"[WARN] Detected LLM error in {prompt_file}. Using fallback prompt.")
        return fallback_prompt
    return prompt

def find_transcript_for_call(run_folder, call_id_or_name):
    # Look for finalized/soundbites/*/<call_id_or_name>_master_transcript.txt
    finalized_soundbites = run_folder / 'finalized' / 'soundbites'
    print(f"[DEBUG] Looking for transcript for call '{call_id_or_name}' in: {finalized_soundbites}")
    if finalized_soundbites.exists():
        # Search all subfolders for <call_id_or_name>_master_transcript.txt
        for subdir in finalized_soundbites.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{call_id_or_name}_master_transcript.txt"
                print(f"[DEBUG] Checking: {candidate}")
                if candidate.exists():
                    print(f"[DEBUG] Found transcript: {candidate}")
                    return candidate.read_text(encoding='utf-8').strip()
    # Fallback: search all subfolders for a single *_master_transcript.txt
    all_transcripts = list(finalized_soundbites.glob('*/'*1 + '*_master_transcript.txt'))
    print(f"[DEBUG] Fallback transcript search found: {all_transcripts}")
    if len(all_transcripts) == 1:
        print(f"[DEBUG] Using fallback transcript: {all_transcripts[0]}")
        return all_transcripts[0].read_text(encoding='utf-8').strip()
    print(f"[WARN] No transcript found for call '{call_id_or_name}'.")
    return None

def get_supplied_transcript(args):
    if args.master_transcript:
        transcript_path = Path(args.master_transcript)
        if transcript_path.exists():
            text = transcript_path.read_text(encoding='utf-8').strip()
            if text:
                print(f"[INFO] Using supplied master transcript: {transcript_path}")
                return text
            else:
                print(f"[WARN] Supplied master transcript {transcript_path} is empty. Falling back.")
        else:
            print(f"[WARN] Supplied master transcript {transcript_path} does not exist. Falling back.")
    return None

def chunk_text(text, max_chars=512):
    """Split text into chunks of <= max_chars, preserving word boundaries."""
    words = text.split()
    chunks = []
    current = ''
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
            current = word
        else:
            current += (' ' if current else '') + word
    if current:
        chunks.append(current.strip())
    return chunks

def deduplicate_and_concatenate(responses, max_chars=300):
    """Deduplicate (exact and near-duplicates), concatenate, and truncate."""
    unique = []
    for resp in responses:
        resp_clean = resp.strip()
        if not resp_clean:
            continue
        # Check for exact or near-duplicate (fuzzy match > 0.85)
        if any(resp_clean == u or difflib.SequenceMatcher(None, resp_clean, u).ratio() > 0.85 for u in unique):
            continue
        unique.append(resp_clean)
    result = ' '.join(unique)
    if len(result) > max_chars:
        print(f"[WARN] Final deduplicated prompt truncated from {len(result)} to {max_chars} characters.")
        result = result[:max_chars]
    return result.strip()

def parse_transcript_time_chunks(transcript, window_seconds=30):
    """Group transcript lines into 30-second windows based on timestamps."""
    # Example line: [CONVERSATION][Speaker02][6.11-8.55]: Hi there, ...
    time_line_re = re.compile(r'\[.*?\]\[.*?\]\[(\d+\.\d+)-(\d+\.\d+)\]: (.*)')
    windows = {}
    for line in transcript.splitlines():
        m = time_line_re.match(line)
        if not m:
            continue
        start, end, text = float(m.group(1)), float(m.group(2)), m.group(3)
        window_idx = int(start // window_seconds)
        if window_idx not in windows:
            windows[window_idx] = []
        windows[window_idx].append(text)
    # Return list of (start_time, end_time, chunk_text)
    result = []
    for idx in sorted(windows.keys()):
        start_time = idx * window_seconds
        end_time = (idx + 1) * window_seconds
        chunk_text = ' '.join(windows[idx])
        result.append((start_time, end_time, chunk_text))
    return result

def segment_transcript(transcript, mode='time', window_seconds=30):
    """Segment transcript by time window or by utterance."""
    if mode == 'utterance':
        # Each line is a segment
        time_line_re = re.compile(r'\[.*?\]\[.*?\]\[(\d+\.\d+)-(\d+\.\d+)\]: (.*)')
        segments = []
        for line in transcript.splitlines():
            m = time_line_re.match(line)
            if not m:
                continue
            start, end, text = float(m.group(1)), float(m.group(2)), m.group(3)
            segments.append({'start': start, 'end': end, 'text': text})
        return segments
    else:
        # Default: time window
        time_chunks = parse_transcript_time_chunks(transcript, window_seconds=window_seconds)
        segments = []
        for start, end, chunk_text in time_chunks:
            if chunk_text.strip():
                segments.append({'start': start, 'end': end, 'text': chunk_text})
        return segments

def generate_prompts_for_segments(segments, llm_config, prompt_type='scene', max_chars=100):
    prompts = []
    for i, seg in enumerate(segments):
        if prompt_type == 'scene':
            prompt = (
                f"Summarize the following conversation segment (from {seg['start']:.2f}s to {seg['end']:.2f}s) into a concise, visually descriptive phrase for an SDXL image or video prompt. "
                f"Avoid repetition and keep it under {max_chars} characters.\n\n{seg['text']}\n\nPrompt:"
            )
        else:
            prompt = seg['text']
        print(f"[INFO] Running LLM for segment {i+1}/{len(segments)}: {seg['start']:.2f}-{seg['end']:.2f}s, length: {len(seg['text'])}")
        result = run_llm_task(prompt, llm_config, output_path=None, chunking=False, single_output=True)
        if result is None or result.strip() == '' or is_llm_error_prompt(result):
            print(f"[WARN] LLM failed on segment {i+1}. Skipping.")
            continue
        prompt_text = result.strip()
        if len(prompt_text) > max_chars:
            prompt_text = prompt_text[:max_chars]
        prompts.append({'start': seg['start'], 'end': seg['end'], 'prompt': prompt_text})
    return prompts

def save_scene_prompts_json(prompts, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2)
    print(f"[INFO] Scene prompts saved to {output_path}")

def run_llm_scene_based(transcript, llm_config, output_path=None, segmentation_mode='time', window_seconds=30):
    segments = segment_transcript(transcript, mode=segmentation_mode, window_seconds=window_seconds)
    if not segments:
        print("[WARN] No valid segments found in transcript.")
        if output_path is not None:
            Path(output_path).write_text("A surreal, privacy-safe image.", encoding='utf-8')
        return "A surreal, privacy-safe image."
    prompts = generate_prompts_for_segments(segments, llm_config, prompt_type='scene', max_chars=100)
    if not prompts:
        print("[WARN] No prompts generated from segments.")
        if output_path is not None:
            Path(output_path).write_text("A surreal, privacy-safe image.", encoding='utf-8')
        return "A surreal, privacy-safe image."
    # Save all prompts as JSON for downstream workflow
    if output_path is not None:
        save_scene_prompts_json(prompts, output_path.with_suffix('.scene_prompts.json'))
    # Aggregate all prompts for a final summary prompt
    synopses_text = ' '.join([p['prompt'] for p in prompts])
    print(f"[INFO] Aggregated scene prompts: {repr(synopses_text)}")
    final_prompt_input = (
        "Generate a single, concise, visually descriptive SDXL image or video prompt for the following scene synopses. "
        "Avoid repetition and keep it under 300 characters.\n\n{}\n\nPrompt:".format(synopses_text)
    )
    print(f"[INFO] Running LLM on aggregated scene prompts to generate final prompt...")
    final_prompt = run_llm_task(final_prompt_input, llm_config, output_path=None, chunking=False, single_output=True)
    if final_prompt is None or final_prompt.strip() == '' or is_llm_error_prompt(final_prompt):
        print(f"[WARN] LLM failed to generate a final prompt from scene prompts. Using synopses as fallback.")
        final_prompt = synopses_text
    if len(final_prompt) > 300:
        print(f"[WARN] Final prompt truncated from {len(final_prompt)} to 300 characters.")
        final_prompt = final_prompt[:300]
    if output_path is not None:
        Path(output_path).write_text(final_prompt, encoding='utf-8')
    return final_prompt

def get_or_generate_prompt(call_folder, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=None):
    prompt_file = call_folder / 'sdxl_image_prompt.txt'
    call_id_or_name = call_folder.name
    transcript = None
    if args is not None:
        transcript = get_supplied_transcript(args)
    if prompt_file.exists() and not transcript:
        prompt = prompt_file.read_text(encoding='utf-8').strip()
        if not is_llm_error_prompt(prompt):
            return prompt
        else:
            print(f"[WARN] Detected LLM error in {prompt_file}. Regenerating prompt using LLM...")
    else:
        if not transcript:
            print(f"[INFO] No sdxl_image_prompt.txt found for {call_id_or_name}, generating prompt using LLM...")
    if not transcript:
        transcript = find_transcript_for_call(run_folder, call_id_or_name)
    if not transcript:
        print(f"[WARN] No transcript found for {call_id_or_name}. Using fallback prompt.")
        prompt = fallback_prompt
    else:
        llm_config_path = Path('workflows/llm_tasks.json')
        if not llm_config_path.exists():
            print("[ERROR] workflows/llm_tasks.json not found. Using fallback prompt.")
            prompt = fallback_prompt
        else:
            with open(llm_config_path, 'r', encoding='utf-8') as f:
                llm_config = json.load(f)
            llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'
            segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
            window_seconds = getattr(args, 'window_seconds', 30) if args else 30
            prompt_result = run_llm_scene_based(transcript, llm_config, output_path=prompt_file, segmentation_mode=segmentation_mode, window_seconds=window_seconds)
            if prompt_result is None or prompt_result.strip() == '' or is_llm_error_prompt(prompt_result):
                print(f"[WARN] LLM failed to generate a valid prompt for {call_id_or_name}. Using fallback.")
                prompt = fallback_prompt
                prompt_file.write_text(prompt, encoding='utf-8')
            else:
                prompt = prompt_result
                prompt_file.write_text(prompt, encoding='utf-8')
    return prompt

def get_or_generate_singlefile_prompt(llm_dir, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=None):
    prompt_file = llm_dir / 'sdxl_image_prompt.txt'
    transcript = None
    if args is not None:
        transcript = get_supplied_transcript(args)
    if prompt_file.exists() and not transcript:
        prompt = prompt_file.read_text(encoding='utf-8').strip()
        if not is_llm_error_prompt(prompt):
            return prompt
        else:
            print(f"[WARN] Detected LLM error in {prompt_file}. Regenerating prompt using LLM...")
    else:
        if not transcript:
            print(f"[INFO] No sdxl_image_prompt.txt found in llm/. Generating prompt using LLM...")
    if not transcript:
        finalized_soundbites = run_folder / 'finalized' / 'soundbites'
        all_transcripts = list(finalized_soundbites.glob('*/'*1 + '*_master_transcript.txt'))
        if len(all_transcripts) == 1:
            transcript = all_transcripts[0].read_text(encoding='utf-8').strip()
        elif len(all_transcripts) > 1:
            print(f"[WARN] Multiple transcripts found in single-file mode. Using the first one: {all_transcripts[0]}")
            transcript = all_transcripts[0].read_text(encoding='utf-8').strip()
    if not transcript:
        print(f"[WARN] No transcript found. Using fallback prompt.")
        prompt = fallback_prompt
    else:
        llm_config_path = Path('workflows/llm_tasks.json')
        if not llm_config_path.exists():
            print("[ERROR] workflows/llm_tasks.json not found. Using fallback prompt.")
            prompt = fallback_prompt
        else:
            with open(llm_config_path, 'r', encoding='utf-8') as f:
                llm_config = json.load(f)
            llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'
            segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
            window_seconds = getattr(args, 'window_seconds', 30) if args else 30
            prompt_result = run_llm_scene_based(transcript, llm_config, output_path=prompt_file, segmentation_mode=segmentation_mode, window_seconds=window_seconds)
            if prompt_result is None or prompt_result.strip() == '' or is_llm_error_prompt(prompt_result):
                print(f"[WARN] LLM failed to generate a valid prompt for single-file mode. Using fallback.")
                prompt = fallback_prompt
                prompt_file.write_text(prompt, encoding='utf-8')
            else:
                prompt = prompt_result
                prompt_file.write_text(prompt, encoding='utf-8')
    return prompt

def update_workflow_prompt(workflow, prompt, batch_size=1, seed=None):
    # For this workflow, the 'text' field must be a string, not a list
    for node in workflow.values():
        if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'Positive prompt':
            node['inputs']['text'] = prompt  # Must be a string
        if node.get('class_type') == 'CLIPTextEncode' and node.get('_meta', {}).get('title') == 'Negative prompt':
            neg = node['inputs'].get('text', '')
            if isinstance(neg, list) and neg:
                node['inputs']['text'] = neg[0]
            elif isinstance(neg, str):
                node['inputs']['text'] = neg
            else:
                node['inputs']['text'] = ''
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
    payload = {"prompt": workflow}  # Wrap workflow as required by ComfyUI API
    print("[DEBUG] Final API payload:")
    print(json.dumps(payload, indent=2))
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI API error: {response.status_code} {response.text}")
    return response.json()

def main():
    parser = argparse.ArgumentParser(description="ComfyUI Image Generator Extension for The-Machine")
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run-YYYYMMDD-HHMMSS output folder')
    parser.add_argument('--workflow', type=str, default='extensions/ComfyUI/theMachine_SDXL_Basic.json', help='Path to ComfyUI workflow JSON')
    parser.add_argument('--output-dir', type=str, help='Output directory for images (default: <run-folder>/comfyui_images)')
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of images to generate')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:8188', help='ComfyUI API URL')
    parser.add_argument('--update-manifest', action='store_true', help='Update manifest.json with image metadata')
    parser.add_argument('--master-transcript', type=str, help='Path to a master transcript file to use for LLM prompt generation (overrides automatic search)')
    parser.add_argument('--segmentation-mode', type=str, choices=['time', 'utterance'], default='time', help='Scene segmentation mode: time (default) or utterance')
    parser.add_argument('--window-seconds', type=int, default=30, help='Time window size in seconds for scene segmentation (default: 30)')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder not found: {run_folder}")
    output_dir = Path(args.output_dir) if args.output_dir else run_folder / 'comfyui_images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load workflow
    with open(args.workflow, 'r', encoding='utf-8') as f:
        workflow_template = json.load(f)

    llm_dir = run_folder / 'llm'
    if not llm_dir.exists():
        print(f"[ERROR] llm/ folder not found in {run_folder}")
        return

    # Find all call subfolders in llm/
    call_folders = [d for d in llm_dir.iterdir() if d.is_dir()]
    single_file_prompt = llm_dir / 'sdxl_image_prompt.txt'
    if call_folders:
        # Multi-call mode
        for call_folder in call_folders:
            call_id = call_folder.name
            prompt = get_or_generate_prompt(call_folder, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=args)
            prompt = postprocess_prompt(prompt, max_chars=300)
            print(f"[INFO] Processing call {call_id} with prompt: {repr(prompt)}")
            # Prepare workflow for this call
            workflow = json.loads(json.dumps(workflow_template))  # Deep copy
            workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)
            call_output_dir = output_dir / call_id
            call_output_dir.mkdir(parents=True, exist_ok=True)
            # Call ComfyUI API
            print(f"[INFO] Sending workflow for call {call_id} to ComfyUI API at {args.api_url} ...")
            result = call_comfyui_api(args.api_url, workflow)
            print(f"[INFO] ComfyUI API response for call {call_id}: {result}")
            # TODO: Move/copy images from ComfyUI output dir to call_output_dir
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
                    'call_id': call_id,
                    'prompt': prompt,
                    'workflow': str(args.workflow),
                    'output_dir': str(call_output_dir),
                    'result': result
                })
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Manifest updated at {manifest_path}")
    elif single_file_prompt.exists():
        # Single-file mode
        prompt = get_or_generate_singlefile_prompt(llm_dir, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=args)
        prompt = postprocess_prompt(prompt, max_chars=300)
        print(f"[INFO] Processing single-file prompt: {repr(prompt)}")
        workflow = update_workflow_prompt(workflow_template, prompt, batch_size=args.batch_size, seed=args.seed)
        # Call ComfyUI API
        print(f"[INFO] Sending workflow to ComfyUI API at {args.api_url} ...")
        result = call_comfyui_api(args.api_url, workflow)
        print(f"[INFO] ComfyUI API response: {result}")
        # TODO: Move/copy images from ComfyUI output dir to output_dir
        if args.update_manifest:
            manifest_path = run_folder / 'manifest.json'
            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            else:
                manifest = []
            manifest.append({
                'stage': 'comfyui_image_generation',
                'prompt': prompt,
                'workflow': str(args.workflow),
                'output_dir': str(output_dir),
                'result': result
            })
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            print(f"[INFO] Manifest updated at {manifest_path}")
    else:
        print("[ERROR] No sdxl_image_prompt.txt found in llm/ or any subfolder. Nothing to process.")

if __name__ == '__main__':
    main() 