# Bleeper Extension – Early-Profanity Censor

## Purpose
Automatically mutes / beeps curse words that occur in the **first 180 seconds** of the show's final WAV so your uploads stay advertiser-friendly on YouTube.

---
## How It Works
1. Reads `show/show.json` timeline to map each call's offset inside the compiled show.
2. Loads each call's `master_transcript.txt` and, for lines inside the first 3 minutes, checks for profanity via `better_profanity`.
3. Builds a list of offending time ranges, merges overlaps.
4. Opens `show.wav` (pydub), overlays a 1 kHz tone (or silence) over each range.
5. Outputs `show_bleeped.wav` alongside the original and records a manifest entry.
6. Finalization converts both the clean and censored versions to MP3.

---
## Configuration (`config/bleeper.yaml`)
```yaml
curse_words:
  - fuck
  - shit
  - bitch
  # … add more
max_seconds: 180          # Window to censor
mode: beep                # beep | mute
beep_frequency: 1000      # Hz for beep tone
beep_volume_db: -3        # Relative gain in dB
```
The file is created with sensible defaults on first run; edit any field to suit your needs.

---
## Stand-Alone Usage (optional)
```bash
python -m extensions.bleeper_extension --input outputs/run-20250611-103012/finalized/show
```
When run directly it accepts `--input <show_dir>` and uses the same YAML config.

---
## Integration
`finalization_stage.py` now invokes the extension automatically, so no action is required when running the normal pipeline.  Both versions are kept:
```
/finalized/show/
  show.wav              # original
  show_bleeped.wav      # censored
  completed-show.mp3    # original MP3
  completed-show_bleeped.mp3
```

---
## Dependencies
* `pydub` (already in requirements.txt)
* `better_profanity` (added to requirements.txt)
* `PyYAML` (already included)

---
## License
Follows The-Machine licensing (see root `LICENSE`). 