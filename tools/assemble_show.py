#!/usr/bin/env python3
"""assemble_show.py

Assemble a mini-show from a folder of processed call audio files.

Features
---------
• Sort calls chronologically using ID3 `tracknumber` tag (primary) or 4-digit filename prefix.
• Concatenate calls with an optional tone between each.
• Generate a track-listing text file with timestamps identical to existing show format.
• Use `extensions.llm_utils.run_llm_task` to create three candidate show names;
  user selects one (or specify --auto N to choose programmatically).

This utility is **PII-safe** — it never logs or writes original input paths.
"""

from __future__ import annotations

import argparse
import re
import sys
import json
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple
import subprocess
import shlex
import tempfile
import shutil

from pydub import AudioSegment  # type: ignore
from mutagen import File  # type: ignore

# Import LLM helper lazily to avoid heavy deps when --llm-config omitted
try:
    from extensions.llm_utils import run_llm_task  # noqa: E402
except Exception:  # pragma: no cover
    run_llm_task = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ms_to_hms(ms: int) -> str:
    """Convert milliseconds to H:MM:SS string (floor to seconds)."""
    total_seconds = ms // 1000
    return str(timedelta(seconds=total_seconds))


def parse_track_number(audio_path: Path) -> int | None:
    """Return the integer track number for ordering, or None."""
    try:
        tags = File(audio_path)
        if tags is not None:
            # Common for ID3: TRCK — may be str like "5/26"
            trck = tags.tags.get("TRCK") or tags.tags.get("TRK")  # type: ignore[attr-defined]
            if trck:
                value = str(trck[0] if isinstance(trck, list) else trck)
                match = re.match(r"(\d+)", value)
                if match:
                    return int(match.group(1))
    except Exception:
        pass
    # Fallback: leading 4-digit prefix
    match = re.match(r"^(\d{4})[_-]", audio_path.stem)
    if match:
        return int(match.group(1))
    return None


def get_title(audio_path: Path) -> str:
    """Return display title (filename stem fallback)."""
    try:
        tags = File(audio_path)
        if tags is not None:
            title_tag = tags.tags.get("TIT2")  # type: ignore[attr-defined]
            if title_tag:
                title_val = str(title_tag[0] if isinstance(title_tag, list) else title_tag)
                title_val = title_val.strip()
                if title_val:
                    return title_val
    except Exception:
        pass
    # Fallback: prettify filename
    title = audio_path.stem
    title = re.sub(r"^[0-9]{4}[_-]", "", title)  # remove index prefix if present
    title = title.replace("_", " ").replace("-", " ")
    return " ".join(title.split())  # collapse whitespace


def discover_calls(calls_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Return list of audio files sorted chronologically per spec."""
    calls_dir = calls_dir.resolve()
    files = [p.resolve() for p in calls_dir.iterdir() if p.suffix.lower() in extensions and p.is_file()]
    if not files:
        raise SystemExit(f"[ERROR] No audio files with extensions {extensions} in {calls_dir}")

    sortable: List[Tuple[int, Path]] = []
    for p in files:
        track_no = parse_track_number(p)
        if track_no is None:
            # Use mtime as tie-breaker; add 1e6 to avoid collision with real track numbers
            track_no = int(p.stat().st_mtime)
        sortable.append((track_no, p))
    sortable.sort(key=lambda x: x[0])
    return [p for _, p in sortable]


# ---------------------------------------------------------------------------
# Utility: get audio duration efficiently (ms)
# ---------------------------------------------------------------------------

def get_duration_ms(path: Path) -> int:
    """Return duration in milliseconds using mutagen (fallback to pydub)."""
    try:
        mf = File(path)
        if mf and mf.info and hasattr(mf.info, "length"):
            return int(mf.info.length * 1000)
    except Exception:
        pass
    # Fallback (should rarely allocate large memory)
    try:
        seg = AudioSegment.from_file(path)
        return len(seg)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# LLM show-title helper
# ---------------------------------------------------------------------------

def generate_show_titles(call_titles: List[str], llm_config: Path | None, variant_count: int = 3) -> List[str]:
    if llm_config is None:
        return ["Untitled Show"]
    if run_llm_task is None:
        print("[WARN] llm_utils not available; using fallback title.")
        return ["Untitled Show"]

    config = json.loads(Path(llm_config).read_text(encoding="utf-8"))
    sample_titles = call_titles[:30]  # keep prompt compact
    prompt = (
        "Given these prank-call titles, craft THREE creative, brief show titles.\n"
        "Avoid profanity or PII.\n"
    )
    prompt += "\n".join(f"{idx+1}. {t}" for idx, t in enumerate(sample_titles))
    prompt += "\n\nOutput each candidate on its own line, no numbering, no extra text.\n"

    response = run_llm_task(prompt, config, single_output=True, chunking=False)
    candidates = [line.strip(" -•*") for line in response.splitlines() if line.strip()]
    # Deduplicate & trim to desired count
    seen = set()
    uniq: List[str] = []
    for c in candidates:
        if c.lower() not in seen:
            uniq.append(c)
            seen.add(c.lower())
        if len(uniq) >= variant_count:
            break
    if not uniq:
        uniq = ["Untitled Show"]
    return uniq


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def assemble_show(calls_dir: Path, output_audio: Path, tone_path: Path, tracklist_path: Path,
                   llm_config: Path | None, auto_choice: int | None, audio_ext: str | None, max_duration: int = 3600, min_fill: float = 0.8):
    extensions = (".wav", ".mp3") if audio_ext is None else (audio_ext.lower(),)

    # Ensure output has extension early (before any part handling)
    if output_audio.suffix == "":
        output_audio = output_audio.with_suffix(".mp3")

    calls = discover_calls(calls_dir, extensions)

    print(f"[INFO] Found {len(calls)} calls to assemble.")

    # Tone duration (ms)
    tone_duration_ms = get_duration_ms(tone_path) if tone_path.exists() else 0

    timeline_entries: List[str] = []
    cursor_ms = 0
    call_titles: List[str] = []

    # Create temporary concat list for ffmpeg
    tmp_dir = Path(tempfile.mkdtemp(prefix="show_concat_"))
    part_index = 1
    concat_path = tmp_dir / f"filelist_{part_index:02d}.txt"
    tmp_lst = open(concat_path, "w", encoding="utf-8")

    # helpers
    def flush_current_part():
        nonlocal part_index, concat_path, tmp_lst, cursor_ms
        tmp_lst.close()

        if cursor_ms == 0:
            return  # nothing written

        part_output = output_audio if part_index == 1 else output_audio.with_name(f"{output_audio.stem}_{part_index:02d}{output_audio.suffix}")
        part_tracklist = tracklist_path if part_index == 1 else tracklist_path.with_name(f"{tracklist_path.stem}_{part_index:02d}{tracklist_path.suffix}")

        # run ffmpeg for this part (reuse existing code via inner function)
        run_ffmpeg(concat_path, part_output)
        write_tracklist(part_tracklist)

        # prepare for next part
        part_index += 1
        concat_path = tmp_dir / f"filelist_{part_index:02d}.txt"
        tmp_lst = open(concat_path, "w", encoding="utf-8")
        timeline_entries.clear()
        cursor_ms = 0

    def run_ffmpeg(list_path: Path, out_path: Path):
        codec = "libmp3lame" if out_path.suffix.lower() == ".mp3" else "pcm_s16le"
        ffmpeg_cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error","-f","concat","-safe","0","-i",str(list_path),"-c:a",codec,str(out_path)
        ]
        log_path = out_path.parent / f"ffmpeg_assemble_show_{part_index:02d}.log"
        res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_path.write_text(res.stdout, encoding="utf-8")
        if res.returncode != 0:
            raise SystemExit(f"[ERROR] FFmpeg part {part_index} failed; see {log_path}")

    show_title: str | None = None  # will be resolved on first flush

    def resolve_show_title() -> str:
        nonlocal show_title
        if show_title is None:
            show_title = generate_show_titles(call_titles, llm_config)[0]
        return show_title

    def write_tracklist(tl_path: Path):
        tl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tl_path, "w", encoding="utf-8") as fp:
            fp.write(f"# Show Title: {resolve_show_title()}\n\n")
            for line in timeline_entries:
                fp.write(line + "\n")

    for idx, call_path in enumerate(calls):
        title = get_title(call_path)
        call_titles.append(title)

        # Determine alias path if header indicates WAV but extension wrong
        alias_path = call_path.resolve()
        try:
            with open(call_path, "rb") as fp_head:
                head4 = fp_head.read(4)
            if head4 == b"RIFF" and call_path.suffix.lower() != ".wav":
                alias_path = tmp_dir / (call_path.stem + ".wav")
                if not alias_path.exists():
                    shutil.copyfile(call_path, alias_path)
        except Exception:
            pass

        # Duration of call (use alias)
        call_dur_ms = get_duration_ms(alias_path)

        start_hms = ms_to_hms(cursor_ms)
        end_ms = cursor_ms + call_dur_ms
        end_hms = ms_to_hms(end_ms)
        timeline_entries.append(f"🎙️ {title} ({start_hms} - {end_hms})")

        tmp_lst.write(f"file '{alias_path.as_posix()}'\n")

        cursor_ms = end_ms

        # Handle tone between calls
        if idx < len(calls) - 1 and tone_duration_ms > 0:
            tone_start = cursor_ms / 1000.0
            tone_end_ms = cursor_ms + tone_duration_ms
            timeline_entries.append(f"[TONE] {tone_start:.2f}-{tone_end_ms/1000.0:.2f}")
            tmp_lst.write(f"file '{tone_path.resolve().as_posix()}'\n")
            cursor_ms = tone_end_ms

        # Check duration cap
        if max_duration > 0 and cursor_ms/1000 >= max_duration:
            flush_current_part()

    # flush remaining part
    flush_current_part()

    # After all parts built, optionally allow interactive title choice
    if auto_choice is None:
        candidates = generate_show_titles(call_titles, llm_config)
        if len(candidates) > 1:
            print("\nCandidate show titles:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            try:
                sel = int(input("Select title number: ").strip())
                if 0 <= sel < len(candidates):
                    show_title = candidates[sel]
            except (ValueError, EOFError):
                pass
        if show_title is None:
            show_title = candidates[0]
    else:
        show_title = generate_show_titles(call_titles, llm_config)[0]

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    # Ensure output has extension (default .mp3 for size safety)
    if output_audio.suffix == "":
        output_audio = output_audio.with_suffix(".mp3")

    # Estimate output size if user insists on .wav
    if output_audio.suffix.lower() == ".wav":
        bytes_per_sec = 44100 * 2 * 2  # 44.1 kHz, 16-bit stereo assumption
        est_size = int(cursor_ms / 1000 * bytes_per_sec)
        if est_size > 3_500_000_000:
            print("[WARN] Estimated WAV >3.5 GiB (RIFF limit). Switching to MP3 to avoid corruption.")
            output_audio = output_audio.with_suffix(".mp3")

    output_audio.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run ffmpeg concat to build show audio with minimal RAM
    # ------------------------------------------------------------------
    codec = "libmp3lame" if output_audio.suffix.lower() == ".mp3" else "pcm_s16le"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c:a",
        codec,
        str(output_audio),
    ]
    print("[INFO] Running FFmpeg to build final show…")
    log_path = output_audio.parent / "ffmpeg_assemble_show.log"
    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed (exit {result.returncode}). See log → {log_path}")
        raise SystemExit(1)
    else:
        print(f"[INFO] FFmpeg completed. Full log saved to {log_path}")

    print(f"[INFO] Wrote show audio → {output_audio.relative_to(Path.cwd())}")

    tracklist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tracklist_path, "w", encoding="utf-8") as fp:
        fp.write(f"# Show Title: {show_title}\n\n")
        for line in timeline_entries:
            fp.write(line + "\n")
    print(f"[INFO] Wrote tracklist  → {tracklist_path.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assemble a mini-show from processed calls.")
    parser.add_argument("--calls-dir", type=Path, required=True, help="Folder containing processed call audio files.")
    parser.add_argument("--output", type=Path, required=True, help="Output audio path (e.g., show.wav).")
    parser.add_argument("--tone", type=Path, default=Path("tones.wav"), help="Tone WAV/MP3 to insert between calls.")
    parser.add_argument("--tracklist", type=Path, required=True, help="Path for track-listing text file.")
    parser.add_argument("--llm-config", type=Path, help="Path to LLM config JSON for show-title generation.")
    parser.add_argument("--auto", type=int, help="Pick Nth candidate title automatically (non-interactive).")
    parser.add_argument("--ext", type=str, help="Audio file extension to search for (.wav or .mp3).")
    parser.add_argument("--max-duration", type=int, default=3600, help="Max duration per show part in seconds (0 = no limit, default 3600).")
    parser.add_argument("--min-fill", type=float, default=0.8, help="Fraction of ideal part length to reach before starting next part (default 0.8).")

    args = parser.parse_args()
    assemble_show(
        calls_dir=args.calls_dir,
        output_audio=args.output,
        tone_path=args.tone,
        tracklist_path=args.tracklist,
        llm_config=args.llm_config,
        auto_choice=args.auto,
        audio_ext=args.ext,
        max_duration=args.max_duration,
        min_fill=args.min_fill,
    )


if __name__ == "__main__":
    main() 