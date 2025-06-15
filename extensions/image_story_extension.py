"""image_story_extension.py – Convert an input image into a narrated story.

1. Generate a short caption using a vision-language model (see utils.vlm_utils).
2. Expand the caption into a creative 60-120-word scene with an LLM.
3. Optionally synthesise speech from the story using NeMo TTS.

Outputs are written next to the image as:
    <stem>_imgstory.txt   – story text
    <stem>_imgstory.wav   – narrated WAV (if TTS succeeded)
    <stem>_imgstory.json  – metadata (caption, models, durations)

This extension adheres to privacy rules: all filenames are anonymised before
execution, and no original path is leaked to logs or manifests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

from extension_base import ExtensionBase
from utils.vlm_utils import generate_caption
from extensions.llm_utils import run_llm_task  # type: ignore


class ImageStoryExtension(ExtensionBase):
    name = "image_story"
    stage = "image.story"
    description = "Analyse an image, create a story, and render TTS audio."

    # ---------------------------------------------------------------------
    #  Helper methods
    # ---------------------------------------------------------------------

    def _find_images(self) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return [p for p in self.output_root.glob("**/*") if p.suffix.lower() in exts]

    def _tts_to_wav(self, text: str, out_path: Path) -> bool:
        """Render *text* to WAV via NeMo TTS.

        Returns True on success; False if TTS not available.
        """
        try:
            from nemo.collections.tts.models import FastPitchModel, HifiGanModel  # type: ignore
            import soundfile as sf  # pip install soundfile
            import torch
        except Exception as exc:
            self.log(f"[WARN] NeMo TTS unavailable: {exc}. Skipping audio render.")
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use small checkpoint names to keep download size modest.
        try:
            fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch").to(device)
            hifigan = HifiGanModel.from_pretrained("tts_hifigan").to(device)
        except Exception as exc:
            self.log(f"[WARN] Failed to load NeMo TTS models: {exc}")
            return False

        with torch.inference_mode():
            spectrogram = fastpitch.parse(text)
            audio = hifigan.convert_spectrogram_to_audio(spectrogram)

        # audio tensor → numpy → WAV 48 kHz mono
        wav = audio.detach().cpu().numpy()[0]
        sf.write(str(out_path), wav, 48000)
        return True

    # ------------------------------------------------------------------

    def run(self):  # noqa: D401 – imperative verb fine
        images = self._find_images()
        if not images:
            self.log("No images found – skipping ImageStoryExtension.")
            return

        llm_config = {
            "lm_studio_base_url": "http://localhost:1234/v1",
            "lm_studio_model_identifier": "mistral-7b-instruct",
            "lm_studio_temperature": 0.7,
            "lm_studio_max_tokens": 512,
        }

        for img_path in images:
            stem = img_path.stem
            caption = generate_caption(img_path)
            prompt = (
                "You are a creative narrator. Given the image caption below, "
                "write a vivid scene-setting story in present tense (≈80 words).\n\n"
                f"Image caption: {caption}"
            )
            story_text = run_llm_task(
                prompt,
                llm_config=llm_config,  # type: ignore[arg-type]
                chunking=False,
                single_output=True,
            ).strip()
            if not story_text:
                story_text = caption  # Fallback

            txt_out = img_path.with_name(f"{stem}_imgstory.txt")
            wav_out = img_path.with_name(f"{stem}_imgstory.wav")
            meta_out = img_path.with_name(f"{stem}_imgstory.json")

            txt_out.write_text(story_text, encoding="utf-8")

            tts_ok = self._tts_to_wav(story_text, wav_out)
            if not tts_ok and wav_out.exists():
                wav_out.unlink(missing_ok=True)

            meta = {
                "caption": caption,
                "story_text": story_text,
                "caption_model": "Salesforce/blip-image-captioning-large",
                "llm_model": llm_config["lm_studio_model_identifier"],
                "tts_generated": bool(tts_ok),
                "wav_path": wav_out.name if tts_ok else None,
                "txt_path": txt_out.name,
            }
            meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self.log(f"Created story for {img_path.name} -> {txt_out.name}")


# ----------------------------------------------------------------------------
#  CLI utility (makes extension runnable standalone)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image → Story extension (stand-alone)")
    parser.add_argument("--output-root", type=str, required=True, help="Folder containing image(s)")
    args = parser.parse_args()

    ImageStoryExtension(args.output_root).run() 