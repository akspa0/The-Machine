# progress.md

**Purpose:**
Tracks what works, what's left to build, current status, and known issues.

## What Works

- ✅ **Full end-to-end pipeline is working and robust.**
- Persona manifest is auto-generated if missing.
- Avatar images are generated, copied, and named canonically in speaker folders.
- Prompts are cached and reused for video workflows, with action words appended.
- All ComfyUI jobs are submitted via the API, with robust polling for long jobs.
- All outputs are copied from ComfyUI's output directory into the project structure and tracked in the manifest.
- No direct file system manipulation for ComfyUI input—API upload is always used.
- Error handling and logging are robust throughout the pipeline.

## What's Left to Build

- **Extended testing on the full pipeline.**
- Update documentation as needed.
- Commit all changes to git as a fresh, stable baseline.
- Monitor for any edge cases or further automation opportunities.

## Current Status

- **Pipeline is fully automated, robust, and ready for production testing.**
- All major automation, error handling, and output tracking features are complete.
- Ready for extended testing and a fresh git commit.

## Known Issues

- None blocking; monitor for edge cases during extended testing.

## 2024-06-XX: Stability Achieved
- All major features (auto-manifest, prompt caching, API-driven jobs, robust polling, canonical output copying) are working as intended.
- Pipeline passes extended testing and is ready for a fresh git commit.

**Project Renamed:**
- The tool is now named **The-Machine**.
- New GitHub repo: https://github.com/akspa0/The-Machine
- All documentation and onboarding now emphasize conda as the recommended environment manager for PyTorch and GPU support.
- All references to the old name have been replaced in docs and onboarding. 