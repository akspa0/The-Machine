import os
import json
import argparse
from pathlib import Path
import subprocess
import sys

# --- CONFIGURABLE ---
DEFAULT_WORKFLOW_PATH = 'extensions/ComfyUI/avatar/image_to_video_wan.json'

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

def main():
    parser = argparse.ArgumentParser(description='Orchestrate persona, avatar, and animation generation for a run-* folder.\n\nWorkflow:\n1. Run persona generator (character_persona_builder.py) if needed.\n2. Run SDXL avatar generator (sdxl_avatar_generator.py).\n3. (Optional) Run animation/video generation.')
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run-* output folder.')
    parser.add_argument('--skip-persona', action='store_true', help='Skip persona generation step.')
    parser.add_argument('--skip-avatar', action='store_true', help='Skip SDXL avatar image generation step.')
    parser.add_argument('--persona-generator', type=str, default='extensions/character_persona_builder.py', help='Path to persona generator script.')
    parser.add_argument('--avatar-generator', type=str, default='extensions/avatar/sdxl_avatar_generator.py', help='Path to SDXL avatar generator script.')
    parser.add_argument('--persona-manifest', type=str, default=None, help='Path to persona_manifest.json (auto-detect if not set).')
    parser.add_argument('--output-root', type=str, default=None, help='Output root for avatar images (defaults to run-folder).')
    # Placeholder for future video/animation args
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    if not run_folder.exists():
        print(f"[ERROR] Run folder not found: {run_folder}")
        sys.exit(1)

    # Step 1: Persona generation
    persona_dir = run_folder / 'characters'
    persona_manifest = args.persona_manifest or (persona_dir / 'persona_manifest.json')
    persona_manifest_exists = persona_manifest.exists()
    if not args.skip_persona:
        if not persona_manifest_exists:
            print(f"[INFO] Persona manifest not found at {persona_manifest}, running persona generator...")
            cmd = [sys.executable, args.persona_generator, str(run_folder)]
            run_subprocess(cmd)
        else:
            print(f"[INFO] Persona manifest found at {persona_manifest}, skipping persona generation.")
    else:
        print("[INFO] Skipping persona generation step (per CLI flag).")

    # Step 2: SDXL avatar image generation
    output_root = args.output_root or str(run_folder)
    if not args.skip_avatar:
        print(f"[INFO] Running SDXL avatar generator...")
        cmd = [sys.executable, args.avatar_generator,
               '--persona-manifest', str(persona_manifest),
               '--output-root', output_root]
        run_subprocess(cmd)
    else:
        print("[INFO] Skipping SDXL avatar image generation step (per CLI flag).")

    # Step 3: (Optional) Animation/video generation (placeholder)
    print("[INFO] Animation/video generation step not yet implemented in orchestrator.")
    print("[INFO] All required persona and avatar image outputs should now be present in the run folder.")

if __name__ == '__main__':
    main() 