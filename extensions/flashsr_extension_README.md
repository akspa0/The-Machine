# FlashSR Audio Super-Resolution Extension for The-Machine

## Purpose
This extension (`flashsr_extension.py`) upgrades low-quality or narrow-band audio files using **FlashSR** super-resolution.  
It integrates seamlessly with The-Machine's librarian orchestrator and respects all privacy / traceability rules.

---
## Key Features
* **Automatic model download** – Weights are fetched from the Hugging Face dataset [`jakeoneijk/FlashSR_weights`](https://huggingface.co/datasets/jakeoneijk/FlashSR_weights/tree/main) on first run and cached in `.cache/flashsr_weights/`.
* **Smart auto-trigger** – Enhances any WAV whose sample-rate < 32 kHz, or whenever the orchestrator is invoked with the `--flashsr` flag.
* **GPU / CPU friendly** – Defaults to CUDA but falls back to CPU if no GPU is available.
* **Graceful fallback** – If FlashSR cannot be imported, the extension upsamples via high-quality sinc resampling so your pipeline never breaks.
* **Manifest integration** – Adds anonymised enhancement records to `manifest.json` for every processed file.

---
## Installation
1. Ensure you have an up-to-date environment with PyTorch + torchaudio (GPU or CPU).  
2. Install / update project requirements:
   ```bash
   pip install -r requirements.txt
   ```
   The FlashSR wrapper library and `huggingface_hub` are already included.

---
## Stand-Alone Usage
```bash
python extensions/flashsr_extension.py \
       --input outputs/0003_vocals_only     \
       --device cuda                        # or cpu
```
Arguments:
* `--input`  – Path to a single WAV **or** a directory that represents the processed-root for a call.
* `--model_dir`  – Optional path to checkpoints; omit to auto-download.
* `--device`  – `cuda`, `cuda:0`, `cpu`, etc.

For each low-quality WAV the extension writes `<stem>_flashsr.wav` in the same folder.

---
## Orchestrator Integration (Preview)
Add a CLI flag to `pipeline_orchestrator.py`:
```python
if args.flashsr:
    FlashSRExtension(output_root=stage_path, device=args.device).run()
```
Then call the full pipeline:
```bash
python pipeline_orchestrator.py --input mycall.wav --flashsr --device cuda
```

---
## How It Works
1. **Model Resolution**  
   – If `model_dir` missing, weights are downloaded via `huggingface_hub.snapshot_download` (idempotent).  
2. **Quality Detection**  
   – Any file with `sample_rate < 32000` triggers enhancement.  
3. **FlashSR Inference**  
   – Audio is processed on the chosen device; output retains the original SR (or 48 kHz in fallback mode).  
4. **Manifest Update**  
   – Each enhancement appends an entry under `"enhancements"` with anonymised fields.

---
## Troubleshooting
| Issue | Solution |
|-------|----------|
| `flashsr_inference` import error | Confirm that the dependency installed correctly; run `pip install git+https://github.com/jakeoneijk/FlashSR_Inference`. The extension will automatically fall back to resampling if necessary. |
| Out-of-memory on GPU | Use `--device cpu` or a smaller GPU. |
| Slow inference on CPU | Pre-download weights so first-run overhead is removed; consider running on GPU. |
| No `_flashsr.wav` produced | Check that the input file's sample-rate is below 32 kHz or override by manually calling the extension in a script. |

---
## Example Manifest Snippet
```jsonc
{
  "enhancements": [
    {
      "type": "flashsr",
      "source": "call_0003.wav",
      "output_name": "call_0003_flashsr.wav",
      "output_path": "call_0003_flashsr.wav"
    }
  ]
}
```

---
## File Structure Example
```
outputs/run-20250611-103012/
  0003_vocals_only/
    call_0003.wav
    call_0003_flashsr.wav   # ← enhanced version
    manifest.json           # ← updated with enhancement entry
```

---
## License
FlashSR weights are distributed under their respective license.  
This extension follows The-Machine licensing (see root `LICENSE`). 