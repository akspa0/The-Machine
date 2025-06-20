"""show_builder.py

Refactored balanced show assembler previously implemented in tools/assemble_show_v2.py
so that it can be imported as a normal Python module **and** still be invoked via CLI.

It exposes one high-level function::

    assemble_show(
        calls_dir: Path,
        output: Path,
        tracklist: Path,
        *,
        tone: Path | None = None,
        tone_level: float = 0.0,
        no_tone: bool = False,
        compress: bool = True,
        target: int = 3600,
        min_fill: float = 0.8,
        llm_config: Path | None = None,
        no_tail_tone: bool = False,
        llm_batch: bool = False,
        call_titles: dict[str, str] | None = None,
    ) -> list[Path]

The CLI entry-point simply parses args and delegates to this function.
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
from tempfile import TemporaryDirectory

# ---------------------------------------------------------------------------
# Optional LLM
# ---------------------------------------------------------------------------
try:
    from extensions.llm_utils import run_llm_task, LLMTaskManager
except Exception:  # pragma: no cover
    run_llm_task = None  # type: ignore
    LLMTaskManager = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper functions (unchanged from original script)
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


def discover_calls(folder: Path, exts: tuple[str, ...]) -> List[Path]:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(res.stdout, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed. See {log_path}")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def hms(ms: int) -> str:
    s = ms // 1000
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"


def fmt_call(idx: int, start_ms: int, title: str) -> str:
    return f"{idx:02d}  {hms(start_ms)}  {title}"


def fmt_tone(start_ms: int, end_ms: int) -> str:
    return f"[TONE] {hms(start_ms)} – {hms(end_ms)}"


def part_header(title: str, part_idx: int, total_parts: int, duration_ms: int) -> List[str]:
    return [
        f"# Show Title: {title}",
        f"# Part {part_idx:02d} of {total_parts:02d}   |  Duration: {hms(duration_ms)}",
        "",
    ]

# ---------------------------------------------------------------------------
# Core high-level function
# ---------------------------------------------------------------------------

def assemble_show(
    *,
    calls_dir: Path,
    output: Path,
    tracklist: Path,
    tone: Path | None = None,
    tone_level: float = 0.0,
    no_tone: bool = False,
    compress: bool = True,
    target: int = 3600,
    min_fill: float = 0.8,
    llm_config: Path | None = None,
    no_tail_tone: bool = False,
    llm_batch: bool = False,
    call_titles: dict[str, str] | None = None,
) -> List[Path]:
    """Build the show and return list of part output paths."""

    exts = (".wav", ".mp3")
    calls = discover_calls(calls_dir, exts)
    if not calls:
        raise FileNotFoundError("No call files found in calls_dir")

    tone_path: Path | None = None
    tone_ms = 0
    if not no_tone and tone and tone.exists():
        tone_path = gain_tone(tone, tone_level)
        desired_ext = ".mp3" if (output.suffix.lower() in ("", ".mp3")) else ".wav"
        if tone_path.suffix.lower() != desired_ext:
            conv_dir = Path(tempfile.mkdtemp(prefix="tone_conv_"))
            conv_path = conv_dir / (tone_path.stem + desired_ext)
            codec = "libmp3lame" if desired_ext == ".mp3" else "pcm_s16le"
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(tone_path), "-c:a", codec, str(conv_path)
            ], check=True)
            tone_path = conv_path
        tone_ms = get_duration_ms(tone_path)

    call_info = [(p, get_duration_ms(p)) for p in calls]
    parts = balanced_parts(call_info, tone_ms, target, min_fill)

    # ----------------------------- LLM batching -----------------------------
    batched_titles: List[str] | None = None
    if llm_batch and LLMTaskManager and llm_config and llm_config.exists():
        try:
            cfg = json.loads(llm_config.read_text())
            mgr = LLMTaskManager(cfg)
            for part in parts:
                prompt = (
                    "Create ONE witty, engaging show title (<=128 chars, no quotes)\n"
                    "for the following prank-call segments:\n" + "\n".join([p.stem for p, _ in part][:40])
                )
                mgr.add(prompt, single_output=True, chunking=False)
            batched_titles = mgr.run_all()
        except Exception as e:
            print(f"[WARN] Batched LLM title generation failed: {e}")

    used_titles: set[str] = set()
    produced_parts: List[Path] = []
    used_call_paths: List[Path] = []

    for idx, part in enumerate(parts, 1):
        part_output = output
        if part_output.suffix == "":
            part_output = part_output.with_suffix(".mp3")
        if len(parts) > 1:
            part_output = part_output.with_name(f"{part_output.stem}_{idx:02d}{part_output.suffix}")
        part_track = tracklist.with_name(f"{tracklist.stem}_{idx:02d}{tracklist.suffix}")
        log_path = part_output.with_suffix(".log")

        with TemporaryDirectory(prefix="showv2_part_") as _tmpdir:
            tmp_dir = Path(_tmpdir)
            list_path = tmp_dir / "concat.txt"
            tl_entries: List[str] = []
            cur_ms = 0
            with list_path.open("w", encoding="utf-8") as fp:
                for j, (path, dur) in enumerate(part):
                    start = cur_ms
                    end = cur_ms + dur
                    cid = path.parent.name if path.parent.name.isdigit() else path.stem
                    display_title = call_titles.get(cid, path.stem.replace("_", " ")) if call_titles else path.stem.replace("_", " ")
                    tl_entries.append(fmt_call(j + 1, start, display_title))
                    fp.write(f"file '{path.as_posix()}'\n")
                    cur_ms = end
                    if tone_path and not (no_tail_tone and j == len(part)-1):
                        tone_start_ms = cur_ms
                        fp.write(f"file '{tone_path.as_posix()}'\n")
                        cur_ms += tone_ms
                        tl_entries.append(fmt_tone(tone_start_ms, cur_ms))
            filt = None
            if compress:
                filt = "acompressor=threshold=-18dB:ratio=4:attack=20:release=200,dynaudnorm=f=75"
            run_ffmpeg(list_path, part_output, filt, log_path)

            if batched_titles is not None and idx-1 < len(batched_titles):
                title = batched_titles[idx-1] or part[0][0].stem.replace("_", " ")
            else:
                try:
                    title = gen_title([p.stem for p, _ in part], llm_config)
                except Exception as e:
                    print(f"[WARN] LLM title generation failed ({e}); falling back to first call title.")
                    title = part[0][0].stem.replace("_", " ")
            if title in used_titles:
                title = f"{title} – Part {idx}"
            used_titles.add(title)

            part_track.parent.mkdir(parents=True, exist_ok=True)
            with part_track.open("w", encoding="utf-8") as ft:
                for line in part_header(title, idx, len(parts), cur_ms):
                    ft.write(line + "\n")
                for line in tl_entries:
                    ft.write(line + "\n")
            print(f"[OK] Part {idx}: {part_output.name}  title → {title}")
            produced_parts.append(part_output)
            used_call_paths.extend([p for p, _ in part])

    # -------------------------------------------------------------------
    # Copy assets (remixed calls, master transcripts, soundbites) into a
    # single folder next to the final show for easy access.
    # -------------------------------------------------------------------

    show_dir = output.parent  # finalized/show
    assets_dir = show_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Derive run_folder to locate soundbites – assumes calls_dir == <run_folder>/call
    run_folder = calls_dir.parent
    soundbites_root = run_folder / "soundbites"
    finalized_calls_dir = run_folder / "finalized" / "calls"
    rename_map_path = finalized_calls_dir / "rename_map.json"
    rename_map = {}
    if rename_map_path.exists():
        try:
            rename_map = json.loads(rename_map_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read rename_map.json: {e}")

    for src_call_path in used_call_paths:
        call_id = src_call_path.parent.name

        # Determine renamed base if available
        sanitized_title = rename_map.get(call_id)

        # Copy remixed call audio – prefer finalized renamed mp3 if available
        if sanitized_title:
            renamed_audio_src = finalized_calls_dir / f"{sanitized_title}.mp3"
            audio_src = renamed_audio_src if renamed_audio_src.exists() else src_call_path
            dest_audio_name = f"{call_id}_{sanitized_title}.mp3"
        else:
            audio_src = src_call_path
            dest_audio_name = f"{call_id}_{src_call_path.name}"

        dest_audio = assets_dir / dest_audio_name
        if not dest_audio.exists() and audio_src.exists():
            try:
                shutil.copy2(audio_src, dest_audio)
            except Exception as e:
                print(f"[WARN] Failed to copy call audio {audio_src}: {e}")

        # Copy master transcript – prefer finalized
        if sanitized_title:
            transcript_src = finalized_calls_dir / f"{sanitized_title}_transcript.txt"
        else:
            transcript_src = soundbites_root / call_id / f"{call_id}_master_transcript.txt"

        if transcript_src.exists():
            dest_transcript = assets_dir / (f"{call_id}_master_transcript.txt" if not sanitized_title else f"{call_id}_{sanitized_title}_transcript.txt")
            if not dest_transcript.exists():
                try:
                    shutil.copy2(transcript_src, dest_transcript)
                except Exception as e:
                    print(f"[WARN] Failed to copy transcript {transcript_src}: {e}")

        # Copy per-soundbite directory (optional)
        sb_src_dir = soundbites_root / call_id
        if sb_src_dir.exists():
            dest_sb_dir = assets_dir / f"{call_id}_soundbites"
            if not dest_sb_dir.exists():
                try:
                    shutil.copytree(sb_src_dir, dest_sb_dir)
                except Exception as e:
                    print(f"[WARN] Failed to copy soundbites dir {sb_src_dir}: {e}")

    return produced_parts

# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def cli():
    """Entry-point used by tools/assemble_show_v2.py and direct CLI."""
    ap = argparse.ArgumentParser(description="Balanced show assembler (module version)")
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
    ap.add_argument("--no-tail-tone", action="store_true")
    ap.add_argument("--llm-batch", action="store_true", help="Batch LLM title generation via LLMTaskManager")
    ap.add_argument("--call-titles", type=json.loads)
    args = ap.parse_args()

    assemble_show(
        calls_dir=args.calls_dir,
        output=args.output,
        tracklist=args.tracklist,
        tone=args.tone,
        tone_level=args.tone_level,
        no_tone=args.no_tone,
        compress=args.compress,
        target=args.target,
        min_fill=args.min_fill,
        llm_config=args.llm_config,
        no_tail_tone=args.no_tail_tone,
        llm_batch=args.llm_batch,
        call_titles=args.call_titles,
    )


if __name__ == "__main__":
    cli() 