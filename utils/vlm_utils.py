from __future__ import annotations

"""vlm_utils.py – lightweight helper for image captioning / VLM tasks.

This aims for *zero* global side-effects: heavyweight libraries are imported
inside functions so that extensions that do not need VLM remain fast to load.

Currently we rely on the 🤗 `transformers` pipeline with a default BLIP
captioning model.  The interface is intentionally generic so future upgrades
(LLava, NeMo vision-language, etc.) only need internal tweaks.
"""

from pathlib import Path
from typing import Optional

__all__ = [
    "generate_caption",
]


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def generate_caption(
    image_path: str | Path,
    *,
    prompt: str | None = None,
    model_name: str = "Salesforce/blip-image-captioning-large",
    device: str | None = None,
    max_new_tokens: int = 64,
) -> str:
    """Return a caption / description for *image_path* using a VLM.

    Parameters
    ----------
    image_path : str | Path
        Local path to an image file (PNG/JPG/…).
    prompt : str | None, optional
        Some models (e.g. LLaVA) support additional user prompts.  If *None*,
        the pipeline's default prompt is used.
    model_name : str, default "Salesforce/blip-image-captioning-large"
        HuggingFace model identifier.  The default BLIP captioner is ~350 MB
        and fast on CPU.  For multi-modal chat models (e.g. llava-v1.5) you can
        pass their repo ID – the function detects the required pipeline type
        automatically.
    device : str | None
        "cpu", "cuda", or index like "cuda:1".  If *None* we auto-pick GPU if
        available.
    max_new_tokens : int, default 64
        Upper bound to keep responses short.
    """
    from PIL import Image  # pillow is lightweight
    from transformers import pipeline, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor  # noqa
    import torch

    img = Image.open(Path(image_path)).convert("RGB")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Heuristic: BLIP uses pipeline("image-to-text"), LLaVA needs
    # pipeline("image-to-text", model=..., processor=...).
    # We attempt BLIP first and fall back.
    try:
        generate = pipeline(
            "image-to-text",
            model=model_name,
            device=0 if device.startswith("cuda") else -1,
            max_new_tokens=max_new_tokens,
        )
        result = generate(img)
        caption = result[0]["generated_text"].strip()
        return caption
    except Exception:
        # Fallback: manual forward for more complex VLMs like LLaVA
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
        inputs = processor(images=img, text=prompt or "Describe the image", return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return caption 