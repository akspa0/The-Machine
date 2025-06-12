import os
import argparse
from pathlib import Path
import json

import torch
import torchaudio
import soundfile as sf
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Try a variety of known FlashSR import paths so installation on Windows/Linux
# works regardless of the package name (flashsr_inference, FlashSR, etc.).
# ---------------------------------------------------------------------------
FlashSRProcessor = None  # type: ignore[assignment]
_flashsr_import_err = None

try:  # Common pip name (flashsr_inference)
    from flashsr_inference.inference import FlashSRProcessor  # type: ignore
except Exception as _e:
    _flashsr_import_err = _e

if FlashSRProcessor is None:
    try:  # Upstream module layout: FlashSR.FlashSR.FlashSR class
        from FlashSR.FlashSR import FlashSR as FlashSRProcessor  # type: ignore
    except Exception as _e:
        _flashsr_import_err = _e

# If still None, we'll fall back to resampler later.

from extension_base import ExtensionBase


class FlashSRRunner:
    """Wrapper that holds a FlashSR model on the requested device.

    If FlashSR is not available, falls back to a basic torchaudio Resample so that the
    extension still works in a minimal form.
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
            # Default cache directory under user home so every run (including orchestrator)
            # resolves the same absolute path regardless of cwd.
            default_cache_dir = Path.home() / ".cache" / "flashsr_weights"

            if model_dir is None:
                model_dir = default_cache_dir

            if not Path(model_dir).exists():
                # Download weights into the specified directory.
                model_dir = snapshot_download(
                    repo_id="jakeoneijk/FlashSR_weights",
                    repo_type="dataset",
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False,
                )

            # Instantiate depending on interface signature.
            if hasattr(FlashSRProcessor, "__call__") and not hasattr(FlashSRProcessor, "enhance"):
                # Interface matches upstream FlashSR class: FlashSR(student_ckpt, vocoder_ckpt, vae_ckpt)
                ckpt_dir = Path(model_dir)
                self.processor = FlashSRProcessor(
                    ckpt_dir / "student_ldm.pth",
                    ckpt_dir / "sr_vocoder.pth",
                    ckpt_dir / "vae.pth",
                )
                # Move to target device if possible (GPU acceleration)
                if self.device.type == "cuda" and hasattr(self.processor, "to"):
                    try:
                        self.processor = self.processor.to(self.device)
                    except Exception:
                        pass
            else:
                # flashsr_inference wrapper provides .enhance()
                self.processor = FlashSRProcessor(model_dir=model_dir, device=self.device)  # type: ignore[arg-type]
        else:
            # Pre-construct a resampler that upsamples to 48 kHz for the fallback path.
            self.processor = None
            self._fallback_target_sr = 48_000

    def _prepare_waveform(self, waveform: torch.Tensor, sr: int) -> tuple[torch.Tensor, int]:
        """Prepare waveform for upstream callable FlashSR (stereo 48 kHz 245760 samples).

        Returns processed waveform and new sample rate.
        """
        if sr != 48_000:
            resampler = torchaudio.transforms.Resample(sr, 48_000)
            waveform = resampler(waveform)
            sr = 48_000

        # Ensure stereo
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform, sr

    def enhance(self, wav_path: Path, out_path: Path) -> Path:
        """Enhance *wav_path* and write the result to *out_path*.

        For the upstream callable FlashSR implementation, the audio is processed in
        5.12-second (245 760 sample) chunks, stitched back together.
        """
        waveform, sr = torchaudio.load(str(wav_path))

        waveform, sr = self._prepare_waveform(waveform, sr)

        n_channels = waveform.shape[0]

        if self.using_flashsr:
            with torch.inference_mode():
                if hasattr(self.processor, "enhance"):
                    # Wrapper path – supports arbitrary length, no chunking needed
                    enhanced = self.processor.enhance(waveform.to(self.device), sr)  # type: ignore[attr-defined]
                    target_sr = sr
                else:
                    # Callable path – process each channel separately, preserving stereo
                    chunk_len = 245_760  # samples @48k
                    enhanced_channels = []
                    for ch in range(n_channels):
                        ch_wave = waveform[ch, :]
                        total_len = ch_wave.shape[0]
                        chunks = []
                        for start in range(0, total_len, chunk_len):
                            end = min(start + chunk_len, total_len)
                            chunk = ch_wave[start:end]
                            orig_len = chunk.shape[0]
                            if orig_len < chunk_len:
                                chunk = torch.nn.functional.pad(chunk, (0, chunk_len - orig_len))
                            chunks.append((chunk, orig_len))

                        out_chunks = []
                        from tqdm import tqdm
                        for b_start in tqdm(range(0, len(chunks), self.batch_size), desc=f"FlashSR ch{ch}", leave=False):
                            batch_items = chunks[b_start:b_start + self.batch_size]
                            batch_tensor = torch.stack([ci[0] for ci in batch_items]).to(self.device)
                            batch_lengths = [ci[1] for ci in batch_items]

                            try:
                                out_batch = self.processor(batch_tensor, lowpass_input=False)  # type: ignore[misc]
                            except Exception:
                                out_list = []
                                for single in batch_tensor:
                                    out_list.append(self.processor(single.unsqueeze(0), lowpass_input=False))  # type: ignore[misc]
                                out_batch = torch.cat(out_list, dim=0)

                            out_batch = out_batch.detach().cpu()
                            for i, orig_len in enumerate(batch_lengths):
                                out_chunks.append(out_batch[i, :orig_len])

                        enhanced_ch = torch.cat(out_chunks, dim=0)
                        enhanced_channels.append(enhanced_ch)

                    enhanced = torch.stack(enhanced_channels, dim=0)  # (C, T)
                    target_sr = 48_000

                    # Resample back to original SR if needed
                    orig_sr = torchaudio.info(str(wav_path)).sample_rate
                    if orig_sr != 48_000:
                        resampler_back = torchaudio.transforms.Resample(48_000, orig_sr)
                        enhanced = resampler_back(enhanced)
                        target_sr = orig_sr
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

        # torchaudio.save expects 2-D (channels, time)
        if enhanced.ndim == 1:
            enhanced = enhanced.unsqueeze(0)
        enhanced = enhanced.cpu()
        torchaudio.save(str(out_path), enhanced, target_sr)
        return out_path


class FlashSRExtension(ExtensionBase):
    """Extension that applies FlashSR super-resolution to one or more WAV files.

    If *target_wav* is supplied, only that file is enhanced; otherwise every *.wav
    under *output_root* is processed.
    """

    def __init__(
        self,
        input_root: str | Path,
        *,
        output_root: str | Path | None = None,
        model_dir: str | None = None,
        device: str = "cuda",
        target_wav: Path | None = None,
        chunk_seconds: float = 5.12,
        batch_size: int = 4,
    ) -> None:
        self.input_root = Path(input_root)
        self.actual_output_root = Path(output_root) if output_root else self.input_root
        self.actual_output_root.mkdir(parents=True, exist_ok=True)
        super().__init__(self.actual_output_root)
        self.runner = FlashSRRunner(model_dir=model_dir, device=device)
        self._target_wav = Path(target_wav) if target_wav else None
        self.chunk_seconds = chunk_seconds
        self.batch_size = max(1, batch_size)

        # Propagate settings to runner for internal use
        self.runner.batch_size = self.batch_size  # type: ignore[attr-defined]
        self.runner.chunk_len_samples = int(self.chunk_seconds * 48_000)  # type: ignore[attr-defined]

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

        if self._target_wav is not None:
            wav_files = [self._target_wav]
        else:
            wav_files = list(self.output_root.rglob("*.wav"))

        if not wav_files:
            self.log("No WAV files found – nothing to enhance.")
            return

        from tqdm import tqdm
        iterable = tqdm(wav_files, desc="FlashSR", unit="file") if len(wav_files) > 1 else wav_files

        for wav_path in iterable:
            # Determine output path inside output_root, preserving relative structure
            try:
                rel = wav_path.relative_to(self.input_root)
            except ValueError:
                rel = wav_path.name
            out_dir = (self.actual_output_root / rel).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{Path(wav_path).stem}_flashsr.wav"
            if out_path.exists():
                self.log(f"{out_path.name} already exists – skipping re-enhancement.")
                continue

            self.log(f"Enhancing {wav_path.name} → {out_path.name}")
            self.runner.enhance(wav_path, out_path)
            self._update_manifest(wav_path, out_path)

        self.log("FlashSR enhancement pass complete.")

        if FlashSRProcessor is None:
            print("[WARN] FlashSR library could not be imported (", _flashsr_import_err, ") – falling back to resampler. Install instructions are in the README.")


# -------------------------------------------------------------------------
# Stand-alone CLI entry point
# -------------------------------------------------------------------------

def _cli() -> None:  # pragma: no cover – manual invocation utility
    parser = argparse.ArgumentParser(description="Apply FlashSR super-resolution to audio files.")
    parser.add_argument("--input", required=True, help="Input file or directory (processed output root)")
    parser.add_argument("--output", help="Output folder for enhanced files (default: alongside input)")
    parser.add_argument("--model_dir", default=None, help="Directory containing FlashSR checkpoints")
    parser.add_argument("--device", default="cuda", help="Device to run inference on (cuda, cuda:0, cpu)")
    parser.add_argument("--chunk_seconds", type=float, default=5.12, help="Chunk length in seconds for processing (callable mode)")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of chunks to process simultaneously on GPU")
    args = parser.parse_args()

    input_path = Path(args.input)
    single_file: Path | None = None
    if input_path.is_file():
        single_file = input_path
        root = input_path.parent
    else:
        root = input_path

    ext = FlashSRExtension(
        input_root=root,
        output_root=Path(args.output) if args.output else None,
        model_dir=args.model_dir,
        device=args.device,
        target_wav=single_file,
        chunk_seconds=args.chunk_seconds,
        batch_size=args.batch_size,
    )
    ext.run()


if __name__ == "__main__":
    _cli() 