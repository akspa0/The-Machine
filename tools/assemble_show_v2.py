#!/usr/bin/env python3
"""assemble_show_v2.py

Balanced show assembler that splits a folder of processed call audio into
≈1-hour parts, inserts tones, applies optional compression, and auto-generates
per-part titles via extensions.llm_utils.

CLI example:
python tools/assemble_show_v2.py \
  --calls-dir outputs/.../calls \
  --output    outputs/.../show.mp3 \
  --tracklist outputs/.../show_notes.txt \
  --tone tones.wav --tone-level -9 \
  --target 3600 --min-fill 0.8 --compress
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple
import re

from mutagen import File  # type: ignore
from pydub import AudioSegment  # type: ignore

# LLM util (optional)
try:
    from extensions.llm_utils import run_llm_task
except Exception:
    run_llm_task = None  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_duration_ms(path: Path) -> int:
    try:
        mf = File(path)
        if mf and mf.info and hasattr(mf.info, "length"):
            return int(mf.info.length * 1000)
    except Exception:
        pass
    try:
        return len(AudioSegment.from_file(path))
    except Exception:
        return 0


def discover_calls(folder: Path, exts: Tuple[str, ...]) -> List[Path]:
    files = [p.resolve() for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]

    def parse_track_number(path: Path) -> int | None:
        try:
            tags = File(path)
            if tags and tags.tags:
                trk = tags.tags.get("TRCK") or tags.tags.get("TRK")  # type: ignore[attr-defined]
                if trk:
                    val = str(trk[0] if isinstance(trk, list) else trk)
                    m = re.match(r"(\d+)", val)
                    if m:
                        return int(m.group(1))
        except Exception:
            pass
        m = re.match(r"^(\d{4})[_-]", path.stem)
        if m:
            return int(m.group(1))
        return None

    sortable: List[Tuple[int, Path]] = []
    for p in files:
        tn = parse_track_number(p)
        if tn is None:
            tn = int(p.stat().st_mtime)
        sortable.append((tn, p))
    sortable.sort(key=lambda x: x[0])
    return [p for _, p in sortable]


def balanced_parts(call_info: List[Tuple[Path, int]], tone_ms: int, target_sec: int, min_fill: float) -> List[List[Tuple[Path, int]]]:
    if target_sec <= 0:
        return [call_info]
    total_ms = sum(d for _, d in call_info)
    total_ms += tone_ms * (len(call_info) - 1)
    part_count = max(1, round(total_ms / (target_sec * 1000)))
    parts: List[List[Tuple[Path, int]]] = []
    idx = 0
    remaining_calls = len(call_info)
    remaining_ms = total_ms
    while idx < len(call_info):
        ideal_ms = remaining_ms // max(1, part_count - len(parts))
        cur_ms = 0
        part: List[Tuple[Path, int]] = []
        while idx < len(call_info):
            path, dur = call_info[idx]
            add_ms = dur + (tone_ms if part else 0)
            if part and cur_ms >= ideal_ms * min_fill and cur_ms + add_ms > ideal_ms:
                break
            part.append((path, dur))
            cur_ms += add_ms
            idx += 1
        remaining_ms -= cur_ms
        remaining_calls -= len(part)
        parts.append(part)
    return parts


def gain_tone(tone_path: Path, db_change: float) -> Path:
    if abs(db_change) < 0.01:
        return tone_path
    seg = AudioSegment.from_file(tone_path) + db_change
    tmp = Path(tempfile.mkdtemp()) / f"tone_{db_change:+.0f}.wav"
    seg.export(tmp, format="wav")
    return tmp


def gen_title(titles: List[str], llm_cfg: Path | None) -> str:
    if not titles:
        return "Untitled Show"
    if run_llm_task is None or llm_cfg is None:
        base = titles[0][:120]
        return (base + "…") if len(base) > 120 else base
    prompt = (
        "Create ONE witty, engaging show title (<=128 chars, no quotes)\n"
        "for the following prank-call segments:\n" + "\n".join(titles[:40])
    )
    cfg = json.loads(llm_cfg.read_text())
    resp = run_llm_task(prompt, cfg, single_output=True, chunking=False)
    resp = resp.splitlines()[0].strip("- *\t\n")
    if len(resp) > 128:
        resp = resp[:125] + "…"
    return resp or "Untitled Show"


def run_ffmpeg(list_path: Path, out_path: Path, filter_chain: str | None, log_path: Path):
    codec = "libmp3lame" if out_path.suffix.lower() == ".mp3" else "pcm_s16le"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_path),
    ]
    if filter_chain:
        cmd += ["-filter:a", filter_chain]
    cmd += ["-c:a", codec, str(out_path)]
    # ensure directories exist
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(res.stdout, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"FFmpeg failed. See {log_path}")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def hms(ms: int) -> str:
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"

# New pretty-printing helpers -------------------------------------------------

def fmt_call(idx: int, start_ms: int, title: str) -> str:
    """Return a call line like ``01  00:00:00  Some Title``"""
    return f"{idx:02d}  {hms(start_ms)}  {title}"


def fmt_tone(start_ms: int, end_ms: int) -> str:
    """Return a tone line like ``[TONE] 00:42:15 – 00:42:18``"""
    return f"[TONE] {hms(start_ms)} – {hms(end_ms)}"


def part_header(title: str, part_idx: int, total_parts: int, duration_ms: int) -> List[str]:
    """Return the two header lines for a part."""
    return [
        f"# Show Title: {title}",
        f"# Part {part_idx:02d} of {total_parts:02d}   |  Duration: {hms(duration_ms)}",
        "",
    ]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Balanced show assembler v2")
    ap.add_argument("--calls-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="Base output (with or without extension)")
    ap.add_argument("--tracklist", type=Path, required=True, help="Base tracklist txt path")
    ap.add_argument("--tone", type=Path, help="Tone WAV/MP3 to insert between calls")
    ap.add_argument("--tone-level", type=float, default=0.0, help="Gain (dB) applied to tone, default 0")
    ap.add_argument("--no-tone", action="store_true")
    ap.add_argument("--compress", dest="compress", action="store_true", default=True)
    ap.add_argument("--no-compress", dest="compress", action="store_false")
    ap.add_argument("--target", type=int, default=3600, help="Target seconds per part (default 3600)")
    ap.add_argument("--min-fill", type=float, default=0.8)
    ap.add_argument("--llm-config", type=Path)
    ap.add_argument("--no-tail-tone", action="store_true", help="Do not append tone after the final call in each part")
    args = ap.parse_args()

    if args.llm_config and not args.llm_config.exists():
        raise SystemExit("--llm-config path does not exist.")

    exts = (".wav", ".mp3")
    calls = discover_calls(args.calls_dir, exts)
    if not calls:
        raise SystemExit("No call files found.")

    tone_path: Path | None = None
    tone_ms = 0
    if not args.no_tone and args.tone and args.tone.exists():
        tone_path = gain_tone(args.tone, args.tone_level)
        # Ensure tone matches final output codec (mp3 vs wav)
        desired_ext = ".mp3" if (args.output.suffix.lower() in ("", ".mp3")) else ".wav"
        if tone_path.suffix.lower() != desired_ext:
            conv_dir = Path(tempfile.mkdtemp(prefix="tone_conv_"))
            conv_path = conv_dir / (tone_path.stem + desired_ext)
            codec = "libmp3lame" if desired_ext == ".mp3" else "pcm_s16le"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(tone_path), "-c:a", codec, str(conv_path)
            ]
            subprocess.run(cmd, check=True)
            tone_path = conv_path
        tone_ms = get_duration_ms(tone_path)

    call_info = [(p, get_duration_ms(p)) for p in calls]

    parts = balanced_parts(call_info, tone_ms, args.target, args.min_fill)
    print(f"[INFO] Building {len(parts)} show parts ≈{args.target/60:.0f} min each")

    used_titles: set[str] = set()
    for idx, part in enumerate(parts, 1):
        part_output = args.output
        if part_output.suffix == "":
            part_output = part_output.with_suffix(".mp3")
        if len(parts) > 1:
            part_output = part_output.with_name(f"{part_output.stem}_{idx:02d}{part_output.suffix}")
        part_track = args.tracklist.with_name(f"{args.tracklist.stem}_{idx:02d}{args.tracklist.suffix}")
        log_path = part_output.with_suffix(".log")

        tmp_dir = Path(tempfile.mkdtemp(prefix="showv2_part_") )
        list_path = tmp_dir / "concat.txt"
        tl_entries: List[str] = []
        cur_ms = 0
        with list_path.open("w", encoding="utf-8") as fp:
            for j, (path, dur) in enumerate(part):
                start = cur_ms
                end = cur_ms + dur
                display_title = path.stem.replace("_", " ")
                # use new call formatting
                tl_entries.append(fmt_call(j + 1, start, display_title))
                fp.write(f"file '{path.as_posix()}'\n")
                cur_ms = end
                if tone_path and not (args.no_tail_tone and j == len(part)-1):
                    # tone segment (including after last call unless --no-tail-tone)
                    tone_start_ms = cur_ms
                    fp.write(f"file '{tone_path.as_posix()}'\n")
                    cur_ms += tone_ms
                    tl_entries.append(fmt_tone(tone_start_ms, cur_ms))
        # filter chain
        filt = None
        if args.compress:
            filt = "acompressor=threshold=-18dB:ratio=4:attack=20:release=200,dynaudnorm=f=75"
        run_ffmpeg(list_path, part_output, filt, log_path)

        # title
        try:
            title = gen_title([p.stem for p, _ in part], args.llm_config)
        except Exception as e:
            print(f"[WARN] LLM title generation failed ({e}); falling back to first call title.")
            title = part[0][0].stem.replace("_", " ")
        if title in used_titles:
            title = f"{title} – Part {idx}"
        used_titles.add(title)

        # write tracklist
        part_track.parent.mkdir(parents=True, exist_ok=True)
        with part_track.open("w", encoding="utf-8") as ft:
            # header lines
            for line in part_header(title, idx, len(parts), cur_ms):
                ft.write(line + "\n")
            # entries
            for line in tl_entries:
                ft.write(line + "\n")
        print(f"[OK] Part {idx}: {part_output.name}  title → {title}")

if __name__ == "__main__":
    main() 