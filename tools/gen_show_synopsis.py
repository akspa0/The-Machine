#!/usr/bin/env python3
"""gen_show_synopsis.py
Generate ≤N-char synopsis for a show track-list using LLM.

Usage:
  python tools/gen_show_synopsis.py shownotes_01.txt [shownotes_02.txt …] \
      --llm-config config/llm.json --max-chars 255 [--out-dir synopses]

If --out-dir is omitted the synopsis is printed only; otherwise a
<basename>.syn.txt file is written in that directory for each input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import importlib.util, sys

# LLM helper (optional)
run_llm_task = None
try:
    from extensions.llm_utils import run_llm_task  # type: ignore
except Exception:
    # dynamic fallback when extensions is not a package
    llm_path = Path(__file__).resolve().parent.parent / "extensions" / "llm_utils.py"
    if llm_path.exists():
        spec = importlib.util.spec_from_file_location("llm_utils", llm_path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            run_llm_task = getattr(mod, "run_llm_task", None)
            summarize_if_long = getattr(mod, "summarize_if_long", None)
            split_into_chunks = getattr(mod, "split_into_chunks", None)

SYS_PROMPT_TEMPLATE = (
    "You are a calm copywriter. "
    "Given a chronological list of prank-call segment titles, "
    "write a concise, neutral summary in chronological order. "
    "• Use plain PG-13 language and avoid sensational terms (e.g. 'emergency', 'disaster'). "
    "• Do not mention hosts or identities—refer simply to 'callers' or 'people'. "
    "• No profanity, no timestamps, no mention that this is a show. "
    "• Limit the response to {max_paragraphs} short paragraphs."
)


def extract_titles(path: Path) -> List[str]:
    titles: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("🎙️ "):
            # strip emoji and timestamp
            try:
                title = line.split("(", 1)[0].replace("🎙️", "").strip()
            except Exception:
                title = line.replace("🎙️", "").strip()
            titles.append(title)
    return titles


def build_prompt(titles: List[str], max_paragraphs: int) -> str:
    bullet_list = "\n".join(f"- {t}" for t in titles)
    return (
        f"Write ONE concise synopsis (≤{max_paragraphs} paragraphs) summarising these show segments in chronological order:\n" + bullet_list
    )


def generate_synopsis(titles: List[str], llm_cfg: Path | None, max_paragraphs: int) -> str:
    if run_llm_task is None:
        raise RuntimeError("extensions.llm_utils not available")
    cfg = json.loads(llm_cfg.read_text()) if llm_cfg else {}
    # If titles list is huge, condense with llm_utils.summarize_if_long
    if 'summarize_if_long' in globals() and summarize_if_long is not None:
        condensed = summarize_if_long("\n".join(titles), cfg)
        titles_for_prompt = condensed.split("\n") if condensed else titles
    else:
        titles_for_prompt = titles

    # Build bullet list string and chunk if necessary
    bullet_str = "\n".join(f"- {t}" for t in titles_for_prompt)
    max_tokens = cfg.get("max_tokens", 4096) - 500  # leave headroom
    if split_into_chunks is not None:
        chunks = split_into_chunks(bullet_str, max_tokens=max_tokens)
    else:
        chunks = [bullet_str]

    partials: List[str] = []
    sys_prompt = SYS_PROMPT_TEMPLATE.format(max_paragraphs=1)
    for chunk in chunks:
        prompt = (
            "Summarise these prank-call segments in chronological order as ONE short paragraph:\n" + chunk
        )
        part = run_llm_task(prompt, cfg, single_output=True, chunking=False, system_prompt=sys_prompt)
        partials.append(part.strip())

    combined_prompt = (
        "Combine the following partial synopses into a concise synopsis in chronological order "
        f"of no more than {max_paragraphs} paragraphs:\n" + "\n".join(partials)
    )
    sys_prompt_final = SYS_PROMPT_TEMPLATE.format(max_paragraphs=max_paragraphs)
    final_syn = run_llm_task(combined_prompt, cfg, single_output=True, chunking=False, system_prompt=sys_prompt_final)

    paras = [p.strip() for p in final_syn.strip().split("\n") if p.strip()]
    return "\n".join(paras[:max_paragraphs])


def main():
    ap = argparse.ArgumentParser(description="Generate show synopsis via LLM")
    ap.add_argument("tracklists", nargs="+", type=Path, help="Track-list txt files (🎙️ lines)")
    ap.add_argument("--llm-config", type=Path, help="LLM config JSON (optional, falls back to defaults)")
    ap.add_argument("--max-paragraphs", type=int, default=2, help="Maximum paragraphs in synopsis (default 2)")
    ap.add_argument("--out-dir", type=Path, help="Directory to write <basename>.syn.txt files")
    args = ap.parse_args()

    files: List[Path] = []
    for p in args.tracklists:
        if p.is_dir():
            files.extend(sorted(p.glob("*.txt")))
        else:
            files.append(p)

    if not files:
        print("[ERROR] No tracklist files found.")
        return

    for tl in files:
        if not tl.exists():
            print(f"[WARN] tracklist {tl} not found; skipping")
            continue
        if tl.name.endswith(".syn.txt"):
            continue  # skip previously generated synopsis files
        titles = extract_titles(tl)
        if not titles:
            print(f"[WARN] no titles found in {tl}; skipping")
            continue
        try:
            synopsis = generate_synopsis(titles, args.llm_config, args.max_paragraphs)
        except Exception as e:
            print(f"[ERROR] Failed to generate synopsis for {tl.name}: {e}")
            continue
        print(f"\n{tl.name} synopsis:\n{synopsis}\n")
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.out_dir / f"{tl.stem}.syn.txt"
            out_path.write_text(synopsis, encoding="utf-8")
            print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main() 