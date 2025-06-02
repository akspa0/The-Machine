import argparse
import sys
import os
from pathlib import Path
import shutil
import json

def download_audio(url, output_dir):
    import yt_dlp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / 'input_from_url.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'writethumbnail': False,
        'writeinfojson': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            yt_metadata = {
                'source_url': url,
                'title': info.get('title', 'downloaded'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'duration': info.get('duration'),
                'original_ext': info.get('ext'),
                'id': info.get('id'),
                'info_dict': info
            }
            with open(output_dir / 'yt_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(yt_metadata, f, indent=2)
        wav_files = list(output_dir.glob('input_from_url.*.wav'))
        if not wav_files:
            wav_files = list(output_dir.glob('input_from_url.wav'))
        if wav_files:
            url_wav = wav_files[0]
            canonical_wav = output_dir / 'input_from_url.wav'
            if url_wav != canonical_wav:
                shutil.copy2(url_wav, canonical_wav)
            print(f"[INFO] Downloaded and prepared {canonical_wav}")
            return canonical_wav
        else:
            print(f"[ERROR] No .wav file found after yt-dlp download.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] yt-dlp download failed for {url}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download audio from URL using yt-dlp.")
    parser.add_argument('--url', type=str, required=True, help='URL to download audio from')
    parser.add_argument('--output', type=str, required=True, help='Output folder for downloaded file')
    args = parser.parse_args()
    download_audio(args.url, args.output)

if __name__ == '__main__':
    main() 