import argparse
from pathlib import Path
import sys
import json
from audio_separation import separate_audio_file

def main():
    parser = argparse.ArgumentParser(description="Separate vocals from a single audio file using the pipeline's model.")
    parser.add_argument('input_file', type=str, help='Path to the input audio file (wav, mp3, etc.)')
    parser.add_argument('output_dir', type=str, help='Directory to write the separated stems')
    parser.add_argument('--model-path', type=str, default='mel_band_roformer_vocals_fv4_gabox.ckpt', help='Path to separation model checkpoint (default: pipeline model)')
    parser.add_argument('--log-json', type=str, default=None, help='Optional: path to write a JSON log of outputs')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = separate_audio_file(input_file, output_dir, args.model_path)

    if result['separation_status'] == 'success':
        print(f"[SUCCESS] Separated: {input_file} -> {output_dir}")
        for stem in result['output_stems']:
            print(f"  - {stem['stem_type']}: {stem['output_path']}")
        if args.log_json:
            with open(args.log_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
    else:
        print(f"[ERROR] Separation failed: {input_file}")
        print(result.get('stderr', 'No error details'))
        sys.exit(1)

if __name__ == '__main__':
    main() 