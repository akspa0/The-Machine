from __future__ import annotations

"""Show Fix-up Extension – Rebuild a completed show inserting tones between calls.

This extension is designed for runs where the original *show* stage wrote a
single large audio file without call-separating tones.  It reuses the remixed
call WAVs already present in ``call/<call_id>/remixed_call.wav`` and
concatenates them with ``tones.wav`` into a new ``show_fixed.wav`` / MP3 pair
plus updated timeline JSON/TXT.  Other finished assets remain untouched.
"""

import json
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf

# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _format_ts(seconds: float) -> str:
    """Return HH:MM:SS string given seconds float."""
    import math
    sec = int(math.floor(seconds))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02}:{s:02}"


def _clean_title(raw: str) -> str:
    """Return a concise, single-line title.

    Heuristics:
        • keep only the first line
        • strip unmatched quotes
        • if >120 chars, cut at first period/colon/dash
    """
    if not raw:
        return ''

    title = raw.strip()
    # first line only
    title = title.splitlines()[0].strip()

    # Strip leading/trailing quotes
    title = title.strip('"').strip("'")

    # Truncate overly long explanations that slipped in
    if len(title) > 120:
        for sep in ['.', ' - ', ': ']:
            idx = title.find(sep)
            if 0 <= idx < 120:
                title = title[:idx].strip()
                break
        if len(title) > 120:
            title = title[:120].rstrip()
    return title


from extension_base import ExtensionBase
from finalization_stage import wav_to_mp3, sanitize_filename

TONES_WAV = Path('tones.wav')


class ShowFixupExtension(ExtensionBase):
    NAME = 'show_fixup'

    def __init__(self, run_folder: Path):
        super().__init__(run_folder)
        self.run_folder = Path(run_folder)
        self.call_dir = self.run_folder / 'call'
        self.show_dir = self.run_folder / 'show'
        self.show_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    def execute(self, *, insert_final_tone: bool = False):  # noqa: D401
        if not self.call_dir.exists():
            self.log('call/ directory missing – nothing to fix')
            return

        # Collect call audio paths
        call_files = []
        for call_id in sorted(self.call_dir.iterdir()):
            remixed = call_id / 'remixed_call.wav'
            if remixed.exists():
                call_files.append((call_id.name, remixed))
        if not call_files:
            self.log('No remixed_call.wav files found – aborting fix-up.')
            return

        sr = 44100
        out_tmp = self.show_dir / 'show_fixed_tmp.wav'
        sf_handle = sf.SoundFile(str(out_tmp), mode='w', samplerate=sr,
                                 channels=2, subtype='PCM_16')

        timeline = []
        cur_time = 0.0

        for idx, (call_id, wav_path) in enumerate(call_files):
            audio, file_sr = sf.read(str(wav_path))
            if file_sr != sr:
                import librosa
                audio = librosa.resample(audio.T, orig_sr=file_sr, target_sr=sr).T
            start = cur_time
            end = start + audio.shape[0] / sr
            sf_handle.write(audio.astype('float32'))
            timeline.append({
                'call_id': call_id,
                'start': start,
                'end': end,
            })
            cur_time = end
            # Insert tone between calls (except after last)
            if idx < len(call_files) - 1 and TONES_WAV.exists():
                tones, t_sr = sf.read(str(TONES_WAV))
                if t_sr != sr:
                    import librosa
                    tones = librosa.resample(tones.T, orig_sr=t_sr, target_sr=sr).T
                t_start = cur_time
                t_end = t_start + tones.shape[0] / sr
                sf_handle.write(tones.astype('float32'))
                timeline.append({'tones': True, 'start': t_start, 'end': t_end})
                cur_time = t_end

        # Optionally append tone at end
        if insert_final_tone and TONES_WAV.exists():
            tones, t_sr = sf.read(str(TONES_WAV))
            if t_sr != sr:
                import librosa
                tones = librosa.resample(tones.T, orig_sr=t_sr, target_sr=sr).T
            t_start = cur_time
            t_end = t_start + tones.shape[0] / sr
            sf_handle.write(tones.astype('float32'))
            timeline.append({'tones': True, 'start': t_start, 'end': t_end})

        sf_handle.close()

        # Move tmp to final
        show_wav = self.show_dir / 'show_fixed.wav'
        shutil.move(out_tmp, show_wav)
        wav_to_mp3(show_wav, self.show_dir / 'show_fixed.mp3')

        # Timeline JSON/TXT
        json_path = self.show_dir / 'show_fixed.json'
        json_path.write_text(json.dumps(timeline, indent=2), encoding='utf-8')

        # Load original titles if available
        title_map: dict[str, str] = {}
        orig_timeline_json = self.show_dir / 'show.json'
        if orig_timeline_json.exists():
            try:
                for item in json.loads(orig_timeline_json.read_text(encoding='utf-8')):
                    if isinstance(item, dict) and 'call_id' in item and 'call_title' in item:
                        title_map[item['call_id']] = item['call_title']
            except Exception:
                pass  # ignore malformed json

        txt_path = self.show_dir / 'show_fixed.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            for entry in timeline:
                if 'call_id' in entry:
                    cid = entry['call_id']
                    title = _clean_title(title_map.get(cid, ''))
                    ts_start = entry['start']
                    ts_end = entry['end']
                    if title:
                        f.write(f"🎙️ {title} ({_format_ts(ts_start)} - {_format_ts(ts_end)})\n")
                    else:
                        f.write(f"CALL {cid}  {ts_start:.2f}-{ts_end:.2f}\n")
                else:
                    f.write(f"[TONE] {entry['start']:.2f}-{entry['end']:.2f}\n")

        # Manifest append
        manifest_path = self.show_dir / 'manifest.json'
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            except Exception:
                manifest = {}
        else:
            manifest = {}
        manifest.setdefault('enhancements', []).append({
            'type': self.NAME,
            'output_wav': str(show_wav.relative_to(self.run_folder)),
            'output_mp3': str(show_wav.with_suffix('.mp3').relative_to(self.run_folder)),
            'timeline': timeline,
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

        self.log(f'Show fix-up complete – new show saved at {show_wav}')


# ----------------------------------------------------------------------
# hook for extension_runner & standalone usage
# ----------------------------------------------------------------------

def run(run_folder: str | Path):
    ShowFixupExtension(Path(run_folder)).execute()


if __name__ == '__main__':
    import argparse, sys
    ap = argparse.ArgumentParser(description='Fix-up a show by re-inserting tones.')
    ap.add_argument('run_folder', type=str, help='Path to outputs/run-YYYY… folder')
    ap.add_argument('--final-tone', action='store_true', help='Insert tones.wav at end of show as well')
    args = ap.parse_args()
    ext = ShowFixupExtension(Path(args.run_folder))
    ext.execute(insert_final_tone=args.final_tone) 