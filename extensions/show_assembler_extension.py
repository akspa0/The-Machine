from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from .extension_base import ExtensionBase


# ---------------------------------------------------------------------------
#  Show Assembler Extension
# ---------------------------------------------------------------------------


class ShowAssemblerExtension(ExtensionBase):
    """Extension that wraps tools/assemble_show_v2.py to build the final show.

    Stage: finalise.F4 (global scope)
    """

    name = "show_assembler"
    stage = "finalise.F4"
    description = "Balanced show assembler using assemble_show_v2 logic"

    # Accept delegated flags
    def __init__(self, output_root, *, call_tones: bool = False):
        super().__init__(output_root)
        self.call_tones = call_tones

    def run(self):
        # Paths
        calls_dir = self.output_root / "call"
        if not calls_dir.exists():
            self.log(f"calls_dir not found: {calls_dir}, skipping show assembly")
            return

        finalized_show_dir = self.output_root / "finalized" / "show"
        finalized_show_dir.mkdir(parents=True, exist_ok=True)

        output_base = finalized_show_dir / "show"
        tracklist_base = finalized_show_dir / "show_notes.txt"

        # Direct import avoids subprocess overhead
        from . import show_builder  # local import to avoid dependency if unused

        tone_path = None
        if self.call_tones:
            p = Path("tones.wav")
            if p.exists():
                tone_path = p

        llm_cfg = Path("workflows/llm_tasks.json") if Path("workflows/llm_tasks.json").exists() else None

        # Build mapping call_id -> title from llm outputs if present
        call_titles: dict[str, str] = {}
        llm_dir = self.output_root / "llm"
        if llm_dir.exists():
            for call_sub in llm_dir.iterdir():
                title_path = call_sub / "call_title.txt"
                if title_path.exists():
                    title = title_path.read_text(encoding="utf-8").splitlines()[0].strip().strip('"')
                    if title:
                        call_titles[call_sub.name] = title

        self.log("Assembling show via show_builder.assemble_show()")
        try:
            show_builder.assemble_show(
                calls_dir=calls_dir,
                output=output_base,
                tracklist=tracklist_base,
                tone=tone_path,
                tone_level=-9.0 if tone_path else 0.0,
                compress=True,
                target=3600,
                min_fill=0.8,
                llm_config=llm_cfg,
                llm_batch=True if llm_cfg else False,
                call_titles=call_titles,
            )
            self.log("Show assembly completed successfully")
        except Exception as e:
            self.log(f"Show assembly failed: {e}") 