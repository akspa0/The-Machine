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
    --llm-model      Override the LLM model used for prompt generation (default: auto per mode)
    --pause-after-prompts Pause after prompt generation to allow manual LLM unload before ComfyUI jobs
    --video          Use video workflow and output video segments
    --image          Use image workflow and output images
    --image-workflow Path to ComfyUI image workflow JSON (default: theMachine_SDXL_Basic.json)
    --max-tokens     Maximum number of tokens for segment generation (default: 4096, max: 16384, hard cap: 23000)
    --force          Force regeneration of all prompts and scene prompt JSONs, even if they already exist.
    --lms-load-model Path to LM Studio model to load before running jobs (overrides workflow model).
    --lms-context-length Context length to use when loading LM Studio model (default: 4096).
    --llm-list-models List available LM Studio models using lms ls and exit.

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
from llm_utils import run_llm_task, load_lm_studio_model, split_into_chunks_advanced, recursive_summarize, default_llm_summarize_fn
from glob import glob
import difflib
import re
import subprocess
import tiktoken

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

def find_transcript_for_call_any(run_folder, call_id_or_name):
    """Find transcript for a call by searching finalized/soundbites/*/<call_id>_master_transcript.txt or <call_name>_master_transcript.txt."""
    finalized_soundbites = run_folder / 'finalized' / 'soundbites'
    if finalized_soundbites.exists():
        # Try by call_id
        for subdir in finalized_soundbites.iterdir():
            if subdir.is_dir():
                candidate = subdir / f"{call_id_or_name}_master_transcript.txt"
                if candidate.exists():
                    print(f"[DEBUG] Found transcript for call '{call_id_or_name}': {candidate}")
                    return candidate.read_text(encoding='utf-8').strip()
        # Try by folder name
        call_dir = finalized_soundbites / call_id_or_name
        if call_dir.exists() and call_dir.is_dir():
            for f in call_dir.iterdir():
                if f.name.endswith('_master_transcript.txt'):
                    print(f"[DEBUG] Found transcript for call '{call_id_or_name}' by folder: {f}")
                    return f.read_text(encoding='utf-8').strip()
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

def batch_run_llm(prompts, llm_config):
    """Batch LLM call: prompts is a list of prompt strings. Returns list of responses."""
    # If your LLM API supports batching, implement here. Otherwise, process sequentially.
    print(f"[INFO] Batch LLM prompt generation: {len(prompts)} prompts, model: {llm_config.get('lm_studio_model_identifier', 'default')}")
    responses = []
    for i, prompt in enumerate(prompts):
        print(f"[INFO] LLM batch item {i+1}/{len(prompts)}")
        result = run_llm_task(prompt, llm_config, output_path=None, chunking=False, single_output=True)
        if result is None or result.strip() == '' or is_llm_error_prompt(result):
            print(f"[WARN] LLM failed on batch item {i+1}. Skipping.")
            responses.append("")
        else:
            responses.append(result.strip())
    return responses

def segment_transcript_token_aware(transcript, window_seconds=90, max_tokens=4096, model="gpt-3.5-turbo"): 
    """
    Segment transcript into windows of up to window_seconds, ensuring each segment does not exceed max_tokens.
    If a segment would exceed the token limit, backtrack to the last sentence/utterance boundary before the limit.
    Returns a list of segments, each with start, end, text, and lines (list of dicts with start, end, speaker, text).
    """
    enc = tiktoken.encoding_for_model(model)
    # Parse transcript lines with timestamps and speaker info
    time_line_re = re.compile(r'\[(.*?)\]\[(.*?)\]\[(\d+\.\d+)-(\d+\.\d+)\]: (.*)')
    lines = []
    for line in transcript.splitlines():
        m = time_line_re.match(line)
        if not m:
            continue
        channel, speaker, start, end, text = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), m.group(5)
        lines.append({'start': start, 'end': end, 'speaker': speaker, 'channel': channel, 'text': text, 'raw': line})
    if not lines:
        return []
    segments = []
    current = []
    current_tokens = 0
    current_start = lines[0]['start']
    current_end = lines[0]['end']
    for i, line in enumerate(lines):
        line_tokens = len(enc.encode(line['text']))
        if (current and (current_tokens + line_tokens > max_tokens or line['end'] - current_start > window_seconds)):
            segment_text = '\n'.join([l['raw'] for l in current])
            segments.append({'start': current_start, 'end': current_end, 'text': segment_text, 'lines': current.copy()})
            current = []
            current_tokens = 0
            current_start = line['start']
        current.append(line)
        current_tokens += line_tokens
        current_end = line['end']
    if current:
        segment_text = '\n'.join([l['raw'] for l in current])
        segments.append({'start': current_start, 'end': current_end, 'text': segment_text, 'lines': current.copy()})
    for seg in segments:
        seg_tokens = len(enc.encode(seg['text']))
        if seg_tokens > max_tokens:
            print(f"[WARN] Segment from {seg['start']}s to {seg['end']}s truncated to fit token limit ({max_tokens}).")
            sentences = re.split(r'(?<=[.!?]) +', seg['text'])
            acc = ''
            acc_tokens = 0
            for sent in sentences:
                sent_tokens = len(enc.encode(sent))
                if acc_tokens + sent_tokens > max_tokens:
                    break
                acc += sent + ' '
                acc_tokens += sent_tokens
            seg['text'] = acc.strip()
    return segments

def generate_prompts_for_segments_token_aware(transcript, llm_config, window_seconds=90, max_tokens=4096, model="gpt-3.5-turbo", prompt_type='scene', max_chars=100):
    segments = segment_transcript_token_aware(transcript, window_seconds, max_tokens, model)
    prompts = []
    prompt_texts = []
    for seg in segments:
        # Log number of lines and preview
        print(f"[INFO] Scene {seg['start']:.2f}-{seg['end']:.2f}s: {len(seg['lines'])} lines. Preview: {seg['lines'][0]['raw'] if seg['lines'] else '[empty]'}")
        if prompt_type == 'scene':
            prompt = (
                f"Given the following scene transcript (multiple lines, speakers, and timestamps), generate a concise, visually descriptive, dynamic phrase for an SDXL image or video prompt. "
                f"Capture the key action, mood, and context. Avoid repetition and keep it under {max_chars} characters.\n\n{seg['text']}\n\nPrompt:"
            )
        else:
            prompt = seg['text']
        prompt_texts.append(prompt)
    responses = batch_run_llm(prompt_texts, llm_config)
    for i, seg in enumerate(segments):
        prompt_text = responses[i] if i < len(responses) else ""
        if len(prompt_text) > max_chars:
            prompt_text = prompt_text[:max_chars]
        prompts.append({'start': seg['start'], 'end': seg['end'], 'prompt': prompt_text, 'lines': seg['lines']})
    return prompts

def save_scene_prompts_json(prompts, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2)
    print(f"[INFO] Scene prompts saved to {output_path}")

def run_llm_scene_based_token_aware(transcript, llm_config, output_path=None, window_seconds=90, max_tokens=4096, model="gpt-3.5-turbo", prompt_type='scene', max_chars=100):
    prompts = generate_prompts_for_segments_token_aware(transcript, llm_config, window_seconds, max_tokens, model, prompt_type, max_chars)
    if not prompts:
        print("[WARN] No prompts generated from segments.")
        if output_path is not None:
            Path(output_path).write_text("A surreal, privacy-safe image.", encoding='utf-8')
        return "A surreal, privacy-safe image."
    if output_path is not None:
        save_scene_prompts_json(prompts, Path(output_path).with_suffix('.scene_prompts.json'))
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
        transcript = find_transcript_for_call_any(run_folder, call_id_or_name)
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
            # Model override logic
            if args and args.llm_model:
                llm_config['lm_studio_model_identifier'] = args.llm_model
            elif args and args.video:
                llm_config['lm_studio_model_identifier'] = 'wan-video-llm-model'  # Example default for video
            else:
                llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'  # Default for image
            segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
            window_seconds = getattr(args, 'window_seconds', 30) if args else 30
            prompt_result = run_llm_scene_based_token_aware(transcript, llm_config, output_path=prompt_file, window_seconds=window_seconds, max_tokens=args.max_tokens if args else 4096, model=args.llm_model if args else "gpt-3.5-turbo", prompt_type=segmentation_mode, max_chars=100)
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
            # Model override logic
            if args and args.llm_model:
                llm_config['lm_studio_model_identifier'] = args.llm_model
            elif args and args.video:
                llm_config['lm_studio_model_identifier'] = 'wan-video-llm-model'  # Example default for video
            else:
                llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'  # Default for image
            segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
            window_seconds = getattr(args, 'window_seconds', 30) if args else 30
            prompt_result = run_llm_scene_based_token_aware(transcript, llm_config, output_path=prompt_file, window_seconds=window_seconds, max_tokens=args.max_tokens if args else 4096, model=args.llm_model if args else "gpt-3.5-turbo", prompt_type=segmentation_mode, max_chars=100)
            if prompt_result is None or prompt_result.strip() == '' or is_llm_error_prompt(prompt_result):
                print(f"[WARN] LLM failed to generate a valid prompt for single-file mode. Using fallback.")
                prompt = fallback_prompt
                prompt_file.write_text(prompt, encoding='utf-8')
            else:
                prompt = prompt_result
                prompt_file.write_text(prompt, encoding='utf-8')
    return prompt

def update_workflow_prompt(workflow, prompt, batch_size=1, seed=None):
    # For both image and video workflows, the 'text' field in the positive prompt node must be a string
    updated = False
    for node_id, node in workflow.items():
        if node.get('class_type') == 'CLIPTextEncode':
            meta = node.get('_meta', {})
            title = meta.get('title', '').lower()
            # Only update positive prompt nodes
            if 'positive' in title or ('prompt' in title and 'negative' not in title) or ('text encode' in title and 'negative' not in title):
                old = node['inputs'].get('text', '')
                node['inputs']['text'] = prompt  # Must be a string
                print(f"[DEBUG] Replaced positive prompt in node {node_id} (title: {title})\n  Old: {old}\n  New: {prompt}")
                updated = True
    # Set batch size if present
    for node in workflow.values():
        if node.get('class_type') == 'EmptyLatentImage' or node.get('class_type') == 'EmptyHunyuanLatentVideo':
            node['inputs']['batch_size'] = batch_size
    # Set seed if present
    for node in workflow.values():
        if node.get('class_type') == 'KSamplerAdvanced' and seed is not None:
            node['inputs']['noise_seed'] = int(seed)
    if not updated:
        print("[WARN] No positive prompt field was updated in the workflow. Please check workflow structure.")
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

def ensure_scene_prompts_json(prompt_file, transcript, llm_config, segmentation_mode, window_seconds, force=False):
    scene_json = prompt_file.with_suffix('.scene_prompts.json')
    if scene_json.exists() and not force:
        with open(scene_json, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Generate scene prompts JSON
    if force and scene_json.exists():
        print(f"[INFO] --force specified: Regenerating scene prompts JSON at {scene_json} ...")
    else:
        print(f"[INFO] Generating scene prompts JSON at {scene_json} ...")
    # Use token-aware segmentation and prompt generation
    max_tokens = getattr(args, 'max_tokens', 4096)
    model = getattr(args, 'llm_model', None)
    if not model:
        model = 'gpt-3.5-turbo'
        print(f"[INFO] No --llm-model specified, using default model for tiktoken: {model}")
    else:
        print(f"[INFO] Using model for tiktoken: {model}")
    prompts = generate_prompts_for_segments_token_aware(
        transcript, llm_config, window_seconds=window_seconds, max_tokens=max_tokens, model=model, prompt_type='scene', max_chars=100)
    if not prompts:
        print(f"[WARN] No prompts generated for scene prompts JSON at {scene_json}.")
        return None
    save_scene_prompts_json(prompts, scene_json)
    return prompts

def unload_llm_model(llm_config=None):
    cmd = ['lms', 'unload', '--all']
    # Optionally add host/port from config
    if llm_config:
        host = llm_config.get('host')
        port = llm_config.get('port')
        if host:
            cmd += ['--host', str(host)]
        if port:
            cmd += ['--port', str(port)]
    print(f"[INFO] Unloading LLM model with command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(f"[INFO] LLM unload stdout: {result.stdout.strip()}")
        if result.stderr:
            print(f"[WARN] LLM unload stderr: {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"[WARN] LLM unload command exited with code {result.returncode}")
    except Exception as e:
        print(f"[WARN] Failed to unload LLM model: {e}")

def main():
    # --- Early argument parsing for --llm-list-models ---
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--llm-list-models', action='store_true')
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.llm_list_models:
        print("[INFO] Listing available LM Studio models (lms ls):")
        try:
            result = subprocess.run(["lms", "ls"], capture_output=True, text=True, check=False)
            print(result.stdout)
            if result.stderr:
                print(f"[WARN] lms ls stderr: {result.stderr.strip()}")
        except Exception as e:
            print(f"[ERROR] Failed to list LM Studio models: {e}")
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ComfyUI Image Generator Extension for The-Machine")
    # --- CLI Argument Definitions ---
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run-YYYYMMDD-HHMMSS output folder')
    parser.add_argument('--workflow', type=str, default='extensions/ComfyUI/theMachine_SDXL_Basic.json', help='Path to ComfyUI workflow JSON (for video mode)')
    parser.add_argument('--output-dir', type=str, help='Output directory for images/videos (default: <run-folder>/comfyui_images or comfyui_videos)')
    parser.add_argument('--seed', type=int, help='Random seed for generation')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of images to generate (default: 1)')
    parser.add_argument('--api-url', type=str, default='http://127.0.0.1:8188', help='ComfyUI API URL')
    parser.add_argument('--update-manifest', action='store_true', help='Update manifest.json with output metadata')
    parser.add_argument('--master-transcript', type=str, help='Path to a master transcript file to use for LLM prompt generation (overrides automatic search)')
    parser.add_argument('--segmentation-mode', type=str, choices=['time', 'utterance'], default='time', help='Scene segmentation mode: time (default) or utterance')
    parser.add_argument('--window-seconds', type=int, default=90, help='Time window size in seconds for scene segmentation (default: 90)')
    parser.add_argument('--llm-model', type=str, help='Override the LLM model used for prompt generation (default: auto per mode)')
    parser.add_argument('--pause-after-prompts', action='store_true', help='Pause after prompt generation to allow manual LLM unload before ComfyUI jobs')
    parser.add_argument('--video', action='store_true', help='Use video workflow and output video segments')
    parser.add_argument('--image', action='store_true', help='Use image workflow and output images')
    parser.add_argument('--image-workflow', type=str, default='extensions/ComfyUI/theMachine_SDXL_Basic.json', help='Path to ComfyUI image workflow JSON (default: theMachine_SDXL_Basic.json)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum number of tokens for segment generation (default: 4096, max: 16384, hard cap: 23000)')
    parser.add_argument('--force', action='store_true', help='Force regeneration of all prompts and scene prompt JSONs, even if they already exist.')
    parser.add_argument('--lms-load-model', type=str, help='Path to LM Studio model to load before running jobs (overrides workflow model).')
    parser.add_argument('--lms-context-length', type=int, default=4096, help='Context length to use when loading LM Studio model (default: 4096).')
    # --- End CLI Argument Definitions ---
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        raise FileNotFoundError(f"Run folder not found: {run_folder}")
    video_output_dir = Path(args.output_dir) if args.output_dir and args.video else run_folder / 'comfyui_videos'
    image_output_dir = Path(args.output_dir) if args.output_dir and args.image and not args.video else run_folder / 'comfyui_images'
    if args.video:
        video_output_dir.mkdir(parents=True, exist_ok=True)
    if args.image:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    # Load workflow templates
    workflow_template_video = None
    workflow_template_image = None
    if args.video:
        with open(args.workflow, 'r', encoding='utf-8') as f:
            workflow_template_video = json.load(f)
    if args.image:
        with open(args.image_workflow, 'r', encoding='utf-8') as f:
            workflow_template_image = json.load(f)

    llm_dir = run_folder / 'llm'
    if not llm_dir.exists():
        print(f"[ERROR] llm/ folder not found in {run_folder}")
        return

    # Find all call subfolders in llm/
    call_folders = [d for d in llm_dir.iterdir() if d.is_dir()]
    single_file_prompt = llm_dir / 'sdxl_image_prompt.txt'

    # Helper to get call name/id for output naming
    def get_call_name_or_id(call_folder):
        return call_folder.name

    # Helper to load scene prompts JSON, generating if missing
    def ensure_scene_prompts_json(prompt_file, transcript, llm_config, segmentation_mode, window_seconds, force=False):
        scene_json = prompt_file.with_suffix('.scene_prompts.json')
        if scene_json.exists() and not force:
            with open(scene_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        # Generate scene prompts JSON
        if force and scene_json.exists():
            print(f"[INFO] --force specified: Regenerating scene prompts JSON at {scene_json} ...")
        else:
            print(f"[INFO] Generating scene prompts JSON at {scene_json} ...")
        # Use token-aware segmentation and prompt generation
        max_tokens = getattr(args, 'max_tokens', 4096)
        model = getattr(args, 'llm_model', None)
        if not model:
            model = 'gpt-3.5-turbo'
            print(f"[INFO] No --llm-model specified, using default model for tiktoken: {model}")
        else:
            print(f"[INFO] Using model for tiktoken: {model}")
        prompts = generate_prompts_for_segments_token_aware(
            transcript, llm_config, window_seconds=window_seconds, max_tokens=max_tokens, model=model, prompt_type='scene', max_chars=100)
        if not prompts:
            print(f"[WARN] No prompts generated for scene prompts JSON at {scene_json}.")
            return None
        save_scene_prompts_json(prompts, scene_json)
        return prompts

    video_jobs = []  # (call_name, call_output_dir, scene_idx, prompt)
    image_jobs = []  # (call_name, call_output_dir, prompt)

    if call_folders:
        # Multi-call mode
        for call_folder in call_folders:
            call_id = call_folder.name
            call_name = get_call_name_or_id(call_folder)
            if args.video:
                call_output_dir = video_output_dir / call_name
                call_output_dir.mkdir(parents=True, exist_ok=True)
                # Video mode: ensure scene prompts JSON exists
                prompt_file = call_folder / 'sdxl_image_prompt.txt'
                # Robust transcript lookup
                transcript = None
                if args.master_transcript:
                    transcript_path = Path(args.master_transcript)
                    if transcript_path.exists():
                        transcript = transcript_path.read_text(encoding='utf-8').strip()
                if not transcript:
                    transcript = find_transcript_for_call_any(run_folder, call_name)
                if not transcript:
                    transcript = find_transcript_for_call_any(run_folder, call_id)
                llm_config_path = Path('workflows/llm_tasks.json')
                if not llm_config_path.exists():
                    print("[ERROR] workflows/llm_tasks.json not found. Skipping.")
                    continue
                with open(llm_config_path, 'r', encoding='utf-8') as f:
                    llm_config = json.load(f)
                # Always use default LLM model for prompt generation
                llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'
                segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
                window_seconds = getattr(args, 'window_seconds', 30) if args else 30
                scene_prompts = ensure_scene_prompts_json(prompt_file, transcript, llm_config, segmentation_mode, window_seconds, force=args.force)
                if not scene_prompts:
                    print(f"[ERROR] No scene prompts JSON found or generated for call {call_name}. Skipping.")
                    continue
                for idx, scene in enumerate(scene_prompts):
                    prompt = postprocess_prompt(scene['prompt'], max_chars=300)
                    video_jobs.append((call_name, call_output_dir, idx, prompt))
            if args.image:
                call_output_dir = image_output_dir / call_name
                call_output_dir.mkdir(parents=True, exist_ok=True)
                prompt = get_or_generate_prompt(call_folder, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=args)
                prompt = postprocess_prompt(prompt, max_chars=300)
                image_jobs.append((call_name, call_output_dir, prompt))

    elif single_file_prompt.exists():
        # Single-file mode
        call_name = 'single_file'
        if args.video:
            call_output_dir = video_output_dir / call_name
            call_output_dir.mkdir(parents=True, exist_ok=True)
            # Video mode: ensure scene prompts JSON exists
            transcript = None
            if args.master_transcript:
                transcript_path = Path(args.master_transcript)
                if transcript_path.exists():
                    transcript = transcript_path.read_text(encoding='utf-8').strip()
            if not transcript:
                finalized_soundbites = run_folder / 'finalized' / 'soundbites'
                all_transcripts = list(finalized_soundbites.glob('*/'*1 + '*_master_transcript.txt'))
                if len(all_transcripts) == 1:
                    transcript = all_transcripts[0].read_text(encoding='utf-8').strip()
                elif len(all_transcripts) > 1:
                    print(f"[WARN] Multiple transcripts found in single-file mode. Using the first one: {all_transcripts[0]}")
                    transcript = all_transcripts[0].read_text(encoding='utf-8').strip()
            llm_config_path = Path('workflows/llm_tasks.json')
            if not llm_config_path.exists():
                print("[ERROR] workflows/llm_tasks.json not found. Skipping.")
            else:
                with open(llm_config_path, 'r', encoding='utf-8') as f:
                    llm_config = json.load(f)
                llm_config['lm_studio_model_identifier'] = 'l3-grand-horror-ii-darkest-hour-uncensored-ed2.15-15b'
                segmentation_mode = getattr(args, 'segmentation_mode', 'time') if args else 'time'
                window_seconds = getattr(args, 'window_seconds', 30) if args else 30
                scene_prompts = ensure_scene_prompts_json(single_file_prompt, transcript, llm_config, segmentation_mode, window_seconds, force=args.force)
                if not scene_prompts:
                    print(f"[ERROR] No scene prompts JSON found or generated for single-file mode. Skipping.")
                else:
                    for idx, scene in enumerate(scene_prompts):
                        prompt = postprocess_prompt(scene['prompt'], max_chars=300)
                        video_jobs.append((call_name, call_output_dir, idx, prompt))
        if args.image:
            call_output_dir = image_output_dir / call_name
            call_output_dir.mkdir(parents=True, exist_ok=True)
            prompt = get_or_generate_singlefile_prompt(llm_dir, run_folder, fallback_prompt="A surreal, privacy-safe image.", args=args)
            prompt = postprocess_prompt(prompt, max_chars=300)
            image_jobs.append((call_name, call_output_dir, prompt))

    # PHASE 1 COMPLETE: All prompts and ComfyUI jobs are now defined
    print("\n[INFO] All prompts and ComfyUI job configs generated.")
    print(f"[INFO] Total video jobs to run: {len(video_jobs)}")
    print(f"[INFO] Total image jobs to run: {len(image_jobs)}")

    # Attempt to unload LLM model before ComfyUI jobs
    llm_config_path = Path('workflows/llm_tasks.json')
    llm_config = None
    if llm_config_path.exists():
        with open(llm_config_path, 'r', encoding='utf-8') as f:
            try:
                llm_config = json.load(f)
            except Exception:
                llm_config = None
    unload_llm_model(llm_config)

    if args.pause_after_prompts:
        input("[INFO] You may now unload the LLM model. Press Enter to begin ComfyUI jobs...")
    else:
        print("[INFO] Proceeding to ComfyUI job execution...")

    # Optionally load LM Studio model if requested
    if args.lms_load_model:
        print(f"[INFO] Loading LM Studio model: {args.lms_load_model} with context length {args.lms_context_length}")
        ok = load_lm_studio_model(path=args.lms_load_model, context_length=args.lms_context_length)
        if not ok:
            print(f"[ERROR] Failed to load LM Studio model {args.lms_load_model} with context length {args.lms_context_length}")
            return

    # PHASE 2: Execute jobs according to mode
    if args.video and not args.image:
        print("[INFO] Running in VIDEO-ONLY mode. Only video jobs will be sent to ComfyUI.")
        for job in video_jobs:
            call_name, call_output_dir, idx, prompt = job
            print(f"[INFO] Processing video scene {idx+1} for call {call_name} with prompt: {repr(prompt)}")
            workflow = json.loads(json.dumps(workflow_template_video))  # Deep copy
            print(f"[INFO] Using workflow: {args.workflow}")
            workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)
            print(f"[INFO] Sending video workflow for call {call_name} scene {idx+1} to ComfyUI API at {args.api_url} ...")
            result = call_comfyui_api(args.api_url, workflow)
            print(f"[INFO] ComfyUI API response for call {call_name} scene {idx+1}: {result}")
            if args.update_manifest:
                manifest_path = run_folder / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                else:
                    manifest = []
                manifest.append({
                    'stage': 'comfyui_video_generation',
                    'call_id': call_name,
                    'scene_index': idx,
                    'prompt': prompt,
                    'workflow': str(args.workflow),
                    'output_dir': str(call_output_dir),
                    'result': result
                })
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Manifest updated at {manifest_path}")
    elif args.image and not args.video:
        print("[INFO] Running in IMAGE-ONLY mode. Only image jobs will be sent to ComfyUI.")
        for job in image_jobs:
            call_name, call_output_dir, prompt = job
            print(f"[INFO] Processing image job for call {call_name} with prompt: {repr(prompt)}")
            workflow = json.loads(json.dumps(workflow_template_image))  # Deep copy
            print(f"[INFO] Using workflow: {args.image_workflow}")
            workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)
            print(f"[INFO] Sending image workflow for call {call_name} to ComfyUI API at {args.api_url} ...")
            result = call_comfyui_api(args.api_url, workflow)
            print(f"[INFO] ComfyUI API response for call {call_name}: {result}")
            if args.update_manifest:
                manifest_path = run_folder / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                else:
                    manifest = []
                manifest.append({
                    'stage': 'comfyui_image_generation',
                    'call_id': call_name,
                    'prompt': prompt,
                    'workflow': str(args.image_workflow),
                    'output_dir': str(call_output_dir),
                    'result': result
                })
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Manifest updated at {manifest_path}")
    elif args.video and args.image:
        print("[INFO] Running in BOTH VIDEO and IMAGE mode. Video jobs will be sent first, then image jobs.")
        for job in video_jobs:
            call_name, call_output_dir, idx, prompt = job
            print(f"[INFO] Processing video scene {idx+1} for call {call_name} with prompt: {repr(prompt)}")
            workflow = json.loads(json.dumps(workflow_template_video))  # Deep copy
            print(f"[INFO] Using workflow: {args.workflow}")
            workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)
            print(f"[INFO] Sending video workflow for call {call_name} scene {idx+1} to ComfyUI API at {args.api_url} ...")
            result = call_comfyui_api(args.api_url, workflow)
            print(f"[INFO] ComfyUI API response for call {call_name} scene {idx+1}: {result}")
            if args.update_manifest:
                manifest_path = run_folder / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                else:
                    manifest = []
                manifest.append({
                    'stage': 'comfyui_video_generation',
                    'call_id': call_name,
                    'scene_index': idx,
                    'prompt': prompt,
                    'workflow': str(args.workflow),
                    'output_dir': str(call_output_dir),
                    'result': result
                })
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Manifest updated at {manifest_path}")
        for job in image_jobs:
            call_name, call_output_dir, prompt = job
            print(f"[INFO] Processing image job for call {call_name} with prompt: {repr(prompt)}")
            workflow = json.loads(json.dumps(workflow_template_image))  # Deep copy
            print(f"[INFO] Using workflow: {args.image_workflow}")
            workflow = update_workflow_prompt(workflow, prompt, batch_size=args.batch_size, seed=args.seed)
            print(f"[INFO] Sending image workflow for call {call_name} to ComfyUI API at {args.api_url} ...")
            result = call_comfyui_api(args.api_url, workflow)
            print(f"[INFO] ComfyUI API response for call {call_name}: {result}")
            if args.update_manifest:
                manifest_path = run_folder / 'manifest.json'
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                else:
                    manifest = []
                manifest.append({
                    'stage': 'comfyui_image_generation',
                    'call_id': call_name,
                    'prompt': prompt,
                    'workflow': str(args.image_workflow),
                    'output_dir': str(call_output_dir),
                    'result': result
                })
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2)
                print(f"[INFO] Manifest updated at {manifest_path}")
    else:
        print("[ERROR] No jobs to run. Please specify --video and/or --image.")

if __name__ == '__main__':
    main() 