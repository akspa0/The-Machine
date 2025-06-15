"""video_bites_extension.py – Cut per-speaker video and audio clips using diarization metadata.

Assumptions
-----------
1. A renamed, privacy-safe video file exists in the tuple folder, e.g. `0003_call.mp4`.
2. Diarization JSON lives under `diarized/<tuple>_segments.json` with fields:
       [
         {"speaker": "S01", "start": 12.34, "end": 17.89, "transcription": "hello"},
         ...
       ]
3. FFmpeg is installed and on PATH.

Outputs
-------
Tuple_root/
└── speakers/
    ├── S01/0000_greeting.mp4 + .wav + .json
    └── S02/…

Privacy rule respected – all paths are already anonymised.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from extension_base import ExtensionBase
from utils.video_utils import cut_video_segment


class VideoBitesExtension(ExtensionBase):
    name = "video_bites"
    stage = "video.bites"
    description = "Cut speaker-specific video clips based on diarization segments."

    # ------------------------------------------------------------------

    def _load_segments(self, tuple_idx: str) -> List[dict]:
        dia_dir = self.output_root / "diarized"
        json_path = dia_dir / f"{tuple_idx}_segments.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Diarization JSON not found: {json_path}")
        return json.loads(json_path.read_text(encoding="utf-8"))

    def _find_video(self) -> Path:
        for ext in (".mp4", ".mov", ".mkv", ".webm"):
            vids = list(self.output_root.glob(f"*{ext}"))
            if vids:
                return vids[0]
        raise FileNotFoundError("No video found in tuple folder.")

    # ------------------------------------------------------------------

    def run(self):  # noqa: D401
        try:
            video_path = self._find_video()
        except FileNotFoundError as exc:
            self.log(str(exc))
            return

        tuple_idx = video_path.stem.split("_")[0]  # assumes <0003>_call.mp4
        try:
            segments = self._load_segments(tuple_idx)
        except FileNotFoundError as exc:
            self.log(str(exc))
            return

        speakers_root = self.output_root / "speakers"
        speakers_root.mkdir(exist_ok=True)

        speaker_counters: Dict[str, int] = {}

        for seg in segments:
            speaker = seg.get("speaker", "UNK")
            start = float(seg["start"])
            end = float(seg["end"])
            transcription = seg.get("transcription", "").strip()

            if end - start < 1.0:
                continue  # skip very short clips

            count = speaker_counters.get(speaker, 0)
            speaker_counters[speaker] = count + 1
            index_str = f"{count:04d}"

            # Short slug from transcription (≤48 chars alnum)
            slug = "_".join(transcription.split()[:5])[:48] or "clip"
            safe_slug = "".join(c for c in slug if c.isalnum() or c == "_")

            speaker_dir = speakers_root / speaker
            speaker_dir.mkdir(exist_ok=True)

            base_name = f"{index_str}_{safe_slug}"
            video_out = speaker_dir / f"{base_name}.mp4"
            audio_out = speaker_dir / f"{base_name}.wav"
            meta_out = speaker_dir / f"{base_name}.json"

            # Cut video and audio
            try:
                cut_video_segment(video_path, video_out, start=start, end=end, codec_copy=True)
                cut_video_segment(video_path, audio_out, start=start, end=end, audio_only=True)
            except Exception as exc:
                self.log(f"[WARN] Cutting failed for {speaker} {start}-{end}: {exc}")
                continue

            meta = {
                "speaker": speaker,
                "start": start,
                "end": end,
                "duration": end - start,
                "transcription": transcription,
                "video": video_out.name,
                "audio": audio_out.name,
            }
            meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        self.log(f"VideoBitesExtension complete – created clips for {len(speaker_counters)} speaker(s).")


# CLI entry ---------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video speaker bites cutter – stand-alone")
    parser.add_argument("--output-root", type=str, required=True, help="Tuple folder containing video + diarization data")
    args = parser.parse_args()

    VideoBitesExtension(args.output_root).run() 