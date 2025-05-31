import subprocess
from pathlib import Path
import argparse
import json

def run_persona_builder(output_root, llm_config=None):
    cmd = ["python", "character_persona_builder.py", str(output_root)]
    if llm_config:
        cmd += ["--llm-config", llm_config]
    print(f"[INFO] Running persona builder: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def generate_avatars(persona_dir, soundbites, avatar_output_dir, sdxl_workflow):
    # TODO: For each persona.md and soundbite, generate images for each shot type
    # Set node 1 (prompt) and node 12 (filename_prefix) in sdxl_workflow
    # Save images as avatars/callid_SPEAKER0x_[start-end]_[shot].png
    pass

def animate_avatars(avatar_dir, animation_output_dir, framepack_workflow):
    # TODO: For each avatar image, run framepack animation
    # Set node 19 (image), node 47 (prompt), node 23 (filename_prefix)
    # Save videos as avatar_videos/callid_SPEAKER0x_[start-end]_[shot]_animated.mp4
    pass

def lipsync_videos(animation_dir, soundbites_dir, lipsync_output_dir, latentsync_workflow):
    # TODO: For each animated video and soundbite, run lipsync
    # Set node 40 (video), node 37 (audio), node 41 (filename_prefix)
    # Save as lipsynced_videos/callid_SPEAKER0x_[start-end]_[shot]_lipsynced.mp4
    pass

def assemble_final_video(lipsynced_dir, transcript_path, final_output_path):
    # TODO: Stitch together lipsynced videos according to transcript timing
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Talking Head Pipeline Orchestrator")
    parser.add_argument('--run-folder', type=str, required=True, help='Path to run output folder')
    parser.add_argument('--persona-workflow', type=str, default=None, help='Path to persona builder workflow/config (optional)')
    parser.add_argument('--sdxl-workflow', type=str, required=True, help='Path to SDXL image generation workflow JSON')
    parser.add_argument('--framepack-workflow', type=str, required=True, help='Path to FramePack animation workflow JSON')
    parser.add_argument('--latentsync-workflow', type=str, required=True, help='Path to LatentSync lipsync workflow JSON')
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    persona_dir = run_folder / 'characters'
    soundbites_dir = run_folder / 'finalized' / 'soundbites'
    avatar_output_dir = run_folder / 'avatars'
    animation_output_dir = run_folder / 'avatar_videos'
    lipsync_output_dir = run_folder / 'lipsynced_videos'
    final_output_dir = run_folder / 'final_video'

    # 1. Run persona builder
    run_persona_builder(run_folder, llm_config=args.persona_workflow)

    # 2. Generate avatars (SDXL)
    generate_avatars(persona_dir, soundbites_dir, avatar_output_dir, args.sdxl_workflow)

    # 3. Animate avatars (FramePack)
    animate_avatars(avatar_output_dir, animation_output_dir, args.framepack_workflow)

    # 4. Lipsync (LatentSync)
    lipsync_videos(animation_output_dir, soundbites_dir, lipsync_output_dir, args.latentsync_workflow)

    # 5. Assemble final video (future)
    # assemble_final_video(lipsync_output_dir, transcript_path, final_output_dir / 'scene_final.mp4') 