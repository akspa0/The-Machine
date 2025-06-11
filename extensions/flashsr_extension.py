import os
import argparse
from pathlib import Path
import json

import torch
import torchaudio
import soundfile as sf
from huggingface_hub import snapshot_download

# Attempt to import the official FlashSR inference wrapper. If unavailable we will gracefully
# degrade to a simple resampler implementation so the extension still works in a minimal form.
try:
    # The upstream repo installs as `flashsr_inference`, but we wrap the import in a try/except
    # because users might install from a different fork.
    from flashsr_inference.inference import FlashSRProcessor  # type: ignore
except ImportError:  # pragma: no cover – environment may not have FlashSR installed
    FlashSRProcessor = None  # fallback handled later

from extension_base import ExtensionBase


class FlashSRRunner:
    """Wrapper that holds a FlashSR model on the requested device.

    If FlashSR is not available, falls back to a basic torchaudio Resample so that the
    extension remains functional (albeit with lower quality results).
    """

    def __init__(self, model_dir: str | None = None, device: str = "cuda") -> None:
        # Resolve device preference
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.using_flashsr = FlashSRProcessor is not None

        # ------------------------------------------------------------------
        # Resolve model directory – automatically download if not supplied
        # ------------------------------------------------------------------
        if self.using_flashsr:
            if model_dir is None or not Path(model_dir).exists():
                # We will download the weights from HuggingFace and cache them locally.
                # This avoids requiring git-lfs to be installed and keeps everything
                # fully automated. The snapshot_download call is idempotent: it will
                # reuse the local cache on subsequent runs.
                model_dir = snapshot_download(
                    repo_id="jakeoneijk/FlashSR_weights", repo_type="dataset", local_dir=".cache/flashsr_weights", local_dir_use_symlinks=False
                )

            # The FlashSR API expects a folder path containing the checkpoint .pth files.
            self.processor = FlashSRProcessor(model_dir=model_dir, device=self.device)
        else:
            # Pre-construct a resampler that upsamples to 48 kHz for the fallback path.
            self.processor = None
            self._fallback_target_sr = 48_000

    def enhance(self, wav_path: Path, out_path: Path) -> Path:
        """Enhance *wav_path* and write the result to *out_path*.

        Returns the output path for convenience.
        """
        waveform, sr = torchaudio.load(str(wav_path))
        waveform = waveform.to(self.device)

        if self.using_flashsr:
            # FlashSR takes numpy or torch input depending on implementation.
            with torch.inference_mode():
                enhanced = self.processor.enhance(waveform, sr)  # type: ignore[attr-defined]
                target_sr = sr  # FlashSR keeps original unless configured otherwise
        else:
            # Simple high-quality sinc resampling fallback
            if sr >= self._fallback_target_sr:
                # Nothing to do – already high sample rate.
                enhanced = waveform
                target_sr = sr
            else:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self._fallback_target_sr).to(self.device)
                enhanced = resampler(waveform)
                target_sr = self._fallback_target_sr

        # torchaudio.save expects CPU tensor
        enhanced = enhanced.cpu()
        torchaudio.save(str(out_path), enhanced, target_sr)
        return out_path


class FlashSRExtension(ExtensionBase):
    """Extension that applies FlashSR super-resolution to low-quality audio files."""

    def __init__(self, output_root: str | Path, model_dir: str | None = None, device: str = "cuda") -> None:
        super().__init__(output_root)
        self.runner = FlashSRRunner(model_dir=model_dir, device=device)

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    def _is_low_quality(self, wav_path: Path, thresh_sr: int = 32_000) -> bool:
        """Return True if *wav_path* is considered low quality (sample rate below *thresh_sr*)."""
        try:
            info = torchaudio.info(str(wav_path))
            return info.sample_rate < thresh_sr
        except Exception:
            # If we cannot inspect, assume low quality to be safe.
            return True

    def _update_manifest(self, src: Path, dst: Path) -> None:
        if self.manifest is None:
            self.manifest = {"enhancements": []}
        if "enhancements" not in self.manifest:
            self.manifest["enhancements"] = []

        self.manifest["enhancements"].append(
            {
                "type": "flashsr",
                "source": src.name,
                "output_name": dst.name,
                "output_path": str(dst.relative_to(self.output_root)),
            }
        )

        manifest_path = self.output_root / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Core extension logic
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401  – simple imperative verb accepted
        self.log("Starting FlashSR enhancement pass …")

        wav_files = list(self.output_root.rglob("*.wav"))
        if not wav_files:
            self.log("No WAV files found – nothing to enhance.")
            return

        for wav_path in wav_files:
            # Decide whether to enhance: low quality OR explicit flag (runner is always invoked here)
            if self._is_low_quality(wav_path):
                out_path = wav_path.with_name(f"{wav_path.stem}_flashsr.wav")
                self.log(f"Enhancing {wav_path.name} → {out_path.name}")
                self.runner.enhance(wav_path, out_path)
                self._update_manifest(wav_path, out_path)
            else:
                self.log(f"Skipping {wav_path.name} (already high quality)")

        self.log("FlashSR enhancement pass complete.")


# -------------------------------------------------------------------------
# Stand-alone CLI entry point
# -------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover – manual invocation utility
    parser = argparse.ArgumentParser(description="Apply FlashSR super-resolution to audio files.")
    parser.add_argument("--input", required=True, help="Input file or directory (processed output root)")
    parser.add_argument("--model_dir", default=None, help="Directory containing FlashSR checkpoints")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda, cuda:0, cpu)")
    args = parser.parse_args()

    root = Path(args.input)
    if root.is_file():
        root = root.parent

    ext = FlashSRExtension(output_root=root, model_dir=args.model_dir, device=args.device)
    ext.run()


if __name__ == "__main__":
    _cli() 