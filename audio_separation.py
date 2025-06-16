import os
import subprocess
import json
from pathlib import Path
from typing import List, Dict
import shutil
import re

def find_latest_run_folder(output_root='output'):
    output_root = Path(output_root)
    run_folders = [f for f in output_root.iterdir() if f.is_dir() and f.name[:8].isdigit()]
    if not run_folders:
        raise FileNotFoundError('No run folders found in output/')
    return max(run_folders, key=lambda f: f.name)

def is_left_or_right(filename):
    return any(x in filename for x in ['-left-', '-right-'])

def parse_anonymized_filename(filename):
    # Example: 0000-left-20250511-221253.wav
    m = re.match(r'(\d{4})-(left|right)-([\d-]+)\.wav', filename)
    if not m:
        return None, None, None
    call_id, channel, timestamp = m.groups()
    return call_id, channel, timestamp

def separate_audio_file(input_file: Path, output_root: Path, model_path: str) -> Dict:
    """
    Run audio-separator on a single file. Moves/renames outputs to a clean structure.
    Returns a dict with status, output stems, and errors.
    If the filename matches the anonymized pattern, use pipeline logic.
    If not, treat as a single-file input: use base name as call_id, 'out' as channel.
    """
    call_id, channel, timestamp = parse_anonymized_filename(input_file.name)
    single_file_mode = False
    if not call_id or not channel:
        # Fallback: single-file mode
        single_file_mode = True
        call_id = input_file.stem  # base name without extension
        channel = 'out'
        timestamp = None
    if single_file_mode:
        call_dir = output_root
    else:
        call_dir = output_root / call_id
    call_dir.mkdir(exist_ok=True)
    # Use a temp subdir for audio-separator output
    temp_out = call_dir / (f'_tmp_{channel}' if not single_file_mode else '_tmp_out')
    temp_out.mkdir(exist_ok=True)
    cmd = [
        'audio-separator',
        str(input_file),
        '-m', model_path,
        '--output_dir', str(temp_out),
        '--output_format', 'WAV',
        '--mdx_enable_denoise',
        '--mdx_overlap', '0.5',
        '--invert_spect'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stems = []
    # Map from audio-separator output to our desired names
    if single_file_mode:
        stem_map = {
            'Vocals': f'{call_id}-vocals.wav',
            'Instrumental': f'{call_id}-instrumental.wav'
        }
    else:
        stem_map = {
            'Vocals': f'{channel}-vocals.wav',
            'Instrumental': f'{channel}-instrumental.wav'
        }
    found = 0
    for stem_type, out_name in stem_map.items():
        # Find the file matching the pattern
        pattern = list(temp_out.glob(f'*({stem_type})*.wav'))
        if pattern:
            src = pattern[0]
            dest = call_dir / out_name
            shutil.move(str(src), str(dest))
            stems.append({
                'call_id': call_id,
                'channel': channel,
                'stem_type': stem_type.lower(),
                'output_path': str(dest)
            })
            found += 1
    # Clean up temp dir
    shutil.rmtree(temp_out)
    return {
        'input_name': input_file.name,
        'output_stems': stems,
        'separation_status': 'success' if found == 2 else 'partial' if found > 0 else 'failed',
        'stderr': result.stderr if result.returncode != 0 else '',
        'returncode': result.returncode
    }

def separate_audio_files(files: List[Path], output_root: Path, model_path: str) -> List[Dict]:
    results = []
    for file in files:
        # Accept any regular WAV file; channel suffix check removed to support single-file inputs
        if not file.is_file():
            continue
        result = separate_audio_file(file, output_root, model_path)
        results.append(result)
    return results

def main():
    # Find latest run folder
    run_folder = find_latest_run_folder()
    renamed_dir = run_folder / 'renamed'
    separated_dir = run_folder / 'separated'
    separated_dir.mkdir(exist_ok=True)
    model_path = 'mel_band_roformer_vocals_fv4_gabox.ckpt'
    manifest = []
    for file in renamed_dir.iterdir():
        if not file.is_file():
            continue
        input_basename = file.stem
        out_subdir = separated_dir / input_basename
        out_subdir.mkdir(exist_ok=True)
        # Run audio-separator
        result = separate_audio_file(file, out_subdir, model_path)
        manifest.append(result)
    # Write manifest
    manifest_path = separated_dir / 'separation_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"Audio separation complete. Manifest written to {manifest_path}.")

if __name__ == '__main__':
    main() 