import re
import json
from pathlib import Path
from typing import List, Tuple

import yaml
from pydub import AudioSegment
from pydub.generators import Sine
from better_profanity import profanity

from extension_base import ExtensionBase


DEFAULT_CONFIG = {
    "curse_words": [
        "fuck",
        "shit",
        "bitch",
        "asshole",
        "bastard",
        "cunt",
        "dick",
        "piss",
        "bollocks",
        "wanker",
        "motherfucker",
    ],
    "max_seconds": 180,
    "mode": "beep",        # beep or mute
    "beep_frequency": 1000,  # Hz
    "beep_volume_db": -3,    # relative gain
}


class BleeperExtension(ExtensionBase):
    """Censor curse words in the first *max_seconds* of the show WAV.

    It relies on transcripts + timeline (show.json) to localise profanity so
    only the relevant speech segment is censored.
    """

    def __init__(self, output_root: str | Path, config_path: str | Path | None = None):
        super().__init__(output_root)
        self.output_root = Path(output_root)
        self.config_path = Path(config_path or "config/bleeper.yaml")
        self.config = self._load_config()
        profanity.load_censor_words(self.config["curse_words"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_config(self):
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f)
            return DEFAULT_CONFIG.copy()
        with open(self.config_path, "r", encoding="utf-8") as f:
            try:
                cfg = yaml.safe_load(f) or {}
            except yaml.YAMLError:
                cfg = {}
        # Fill defaults for missing keys
        merged = DEFAULT_CONFIG.copy()
        merged.update(cfg)
        return merged

    def _parse_transcript_segments(self, transcript_path: Path) -> List[Tuple[float, float, str]]:
        """Return list of (start_sec, end_sec, text) from a master transcript."""
        segments = []
        if not transcript_path.exists():
            return segments
        segment_re = re.compile(r"\[(?:[^\]]*)\]\s*(?:\[[^\]]*\])?\s*\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)")
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                m = segment_re.match(line.strip())
                if m:
                    start, end, text = m.groups()
                    segments.append((float(start), float(end), text))
        return segments

    def _collect_curse_segments(self, run_folder: Path) -> List[Tuple[int, int]]:
        """Return list of (start_ms, end_ms) intervals (absolute in show) that contain profanity."""
        show_json = run_folder / "show" / "show.json"
        soundbites_root = run_folder / "soundbites"
        if not show_json.exists() or not soundbites_root.exists():
            return []

        max_seconds = float(self.config["max_seconds"])
        intervals_ms: List[Tuple[int, int]] = []

        with open(show_json, "r", encoding="utf-8") as f:
            timeline = json.load(f)

        for entry in timeline:
            if "call_id" not in entry or "start" not in entry:
                continue
            call_id = entry["call_id"]
            call_start = float(entry["start"])  # in show seconds
            if call_start >= max_seconds:
                break  # timeline is ordered chronologically; we can stop early
            # Path to master transcript
            transcript_path = soundbites_root / call_id / "master_transcript.txt"
            if not transcript_path.exists():
                continue
            # Parse transcript lines
            for rel_start, rel_end, text in self._parse_transcript_segments(transcript_path):
                abs_start = call_start + rel_start
                abs_end = call_start + rel_end
                if abs_start > max_seconds:
                    continue  # outside censor window
                if profanity.contains_profanity(text.lower()):
                    # convert to ms and clip to max_seconds
                    s_ms = int(abs_start * 1000)
                    e_ms = int(min(abs_end, max_seconds) * 1000)
                    intervals_ms.append((s_ms, e_ms))
        # Merge overlapping intervals
        intervals_ms.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in intervals_ms:
            if not merged:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))
        return merged

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def run(self):  # noqa: D401
        self.log("Starting bleeper extension …")
        # Detect run_folder and show.wav
        run_folder = self.output_root.parent.parent  # finalized/show -> finalized -> run_folder
        show_wav = self.output_root / "show.wav"
        if not show_wav.exists():
            self.log("show.wav not found, skipping bleeper.")
            return

        segments = self._collect_curse_segments(run_folder)
        if not segments:
            self.log("No profanity detected in first %s seconds – no bleeping needed", self.config["max_seconds"])
            return

        audio = AudioSegment.from_wav(show_wav)
        mode = self.config["mode"]
        if mode == "beep":
            beep_freq = int(self.config["beep_frequency"])
            beep_volume = float(self.config["beep_volume_db"])
        else:
            beep_freq = None  # silence mode
        for start_ms, end_ms in segments:
            duration_ms = end_ms - start_ms
            if duration_ms <= 0:
                continue
            if mode == "mute":
                censor = AudioSegment.silent(duration=duration_ms)
            else:  # beep
                censor = Sine(beep_freq).to_audio_segment(duration=duration_ms).apply_gain(beep_volume)
            # Overlay censor
            audio = audio.overlay(censor, position=start_ms)

        out_path = show_wav.with_name("show_bleeped.wav")
        audio.export(out_path, format="wav")
        self.log(f"Bleeped version saved to {out_path.name}")

        # Update manifest
        self._update_manifest(show_wav.name, out_path.name)

    # ------------------------------------------------------------------
    def _update_manifest(self, src_name: str, dst_name: str):
        if self.manifest is None:
            self.manifest = {"enhancements": []}
        if "enhancements" not in self.manifest:
            self.manifest["enhancements"] = []
        self.manifest["enhancements"].append(
            {
                "type": "bleeper",
                "source": src_name,
                "output_name": dst_name,
                "output_path": dst_name,
            }
        )
        manifest_path = self.output_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2) 