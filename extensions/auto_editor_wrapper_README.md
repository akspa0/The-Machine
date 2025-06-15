# Auto-Editor Wrapper (`tm-auto-editor`)

Standalone CLI tool that trims long silences (or whatever Auto-Editor can do)
from an **already-anonymised** WAV/MP3 file produced by The-Machine's
finalisation stage.

The wrapper lives at `tools/auto_editor_cli.py` and is *not* an in-process
extension – it is executed as an external command so that the heavy Auto-Editor
logic (and its many dependencies) stay out of the core pipeline.

---
## Why?

Call recordings often contain dead air, ringing tones, hold music, or other
quiet stretches that reduce listenability. Auto-Editor can automatically cut
(or speed-up) these regions. Wrapping it in a tiny script gives us:

*  Exact parameter control (margin, silent speed, threshold, …).
*  Privacy-safe JSON report for manifest merging.
*  Output naming that honours our tuple index scheme (`<stem>_ae.<ext>`).
*  Zero global installs – everything is cloned in `external_apps/`.

---
## 1. One-time setup

```bash
# 1) Clone the upstream project next to the codebase
$ git clone --depth 1 https://github.com/WyattBlue/auto-editor external_apps/auto_editor

# 2) Install it in editable mode (same venv as The-Machine)
$ pip install -e external_apps/auto_editor
```

`ffmpeg` must be discoverable on your PATH (Auto-Editor relies on it).  Most
developers already have this for other pipeline stages.

---
## 2. Basic usage

```bash
python tools/auto_editor_cli.py <input.wav|mp3> [OPTIONS]
```

Key flags (all have sensible defaults):

* `--margin 0.2sec`      – neighbouring loud sections are kept.
* `--silent-speed 99999` – effectively *cuts* silence instead of speeding-up.
* `--video-speed 1`      – playback speed for kept audio (1 × real time).
* `--threshold 0.04`     – loudness threshold for what counts as "audio".
* `--output PATH`        – custom output path; default is `<stem>_ae.wav|mp3`.
* `--audio-codec CODEC`  – override inferred codec (`pcm_s16le` for WAV,
                           `libmp3lame` for MP3).
* `--report PATH`        – where to write JSON stats (default: nearest `run-*`).
* `--verbose`            – print the underlying command & report location.

Example:

```bash
python tools/auto_editor_cli.py outputs/run-20250611-000850/finalized/show/completed-show.mp3 ---verbose
```

Produces:

* `completed-show_ae.mp3` (trimmed)
* `run-20250611-000850/completed-show_ae.json` (duration before/after, params).

---
## 3. Integrating with PipelineOrchestrator

1.  Add an optional post-finalisation step that shells out:
    ```python
    subprocess.run([
        sys.executable,
        'tools/auto_editor_cli.py',
        str(path_to_final_audio),
        '--margin', '0.2sec',
        '--silent-speed', '99999',
        '--threshold', '0.04',
    ], check=True)
    ```
2.  Merge the generated JSON into the call manifest so lineage is preserved.
3.  Use orchestrator config to toggle this step (e.g., `--use-auto-editor`).

Because the wrapper touches **only** files that are already anonymised, it
complies with our privacy rules (nothing sensitive is logged).

---
## 4. Troubleshooting

* `Error! Unknown option …` – ensure the local Auto-Editor clone is recent
  (≥ 28.x); pull upstream and reinstall.
* `Auto-Editor is not installed` – forgot the *pip install ‑e* step.
* Output WAV sounds garbled – pick the correct codec: pass
  `--audio-codec pcm_s16le` if you force a WAV extension.

---
## 5. Dev notes

The wrapper is ~250 LOC, single-file, zero external logging, and keeps
Auto-Editor pinned to the project tree so upgrades are trivial: just `git pull`
in *external_apps/auto_editor*.

Feel free to extend CLI flags or expose more Auto-Editor options as needs
evolve. 