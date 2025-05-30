# activeContext.md

**Purpose:**
Tracks current work focus, recent changes, next steps, and active decisions/considerations.

**Project Renamed:**
- The tool is now named **The-Machine**.
- New GitHub repo: https://github.com/akspa0/The-Machine
- All documentation, onboarding, and environment setup now emphasize conda as the recommended environment manager for PyTorch and GPU support.
- All references to the old name have been replaced in docs and onboarding.

## Current Focus

- ✅ **MAJOR BREAKTHROUGH: Resume functionality successfully implemented and tested**
- All pipeline stages (ingestion, separation, diarization, normalization, transcription, soundbite, remix, show, logging/manifest, LLM integration) are implemented and robust
- Privacy, manifest, and logging requirements are strictly enforced by the PipelineOrchestrator
- LLM integration is modular and per-workflow, with all config in workflow JSONs or referenced files
- Defensive code and error handling are in place for malformed or missing data
- System is fully auditable, extensible, and user-configurable
- **NEW: Pipeline is now fully workflow-driven. All stage order, config, and routing are defined in workflow JSONs.**
- **NEW: All CLAP logic has been removed from the main pipeline. CLAP segmentation/detection is now handled exclusively by the extension (clap_segmentation_experiment.py). The main pipeline is fully decoupled from CLAP.**
- **NEW: Extensions system is formalized: external scripts in extensions/ can be run as pre- or post-processing, either manually or via CLI.**
- **NEW: LLM processing is handled as a modular 'bus' and is only run if defined in the workflow JSON. Per-workflow LLM config is supported.**
- **NEW: CLI flag `--run-clap [first|last]` allows users to run CLAP segmentation/detection as a pre- or post-processing step.**
- **NEW: All outputs, logs, and manifests are strictly PII-free and fully auditable.**
- **NEW: Resume and force: The pipeline is robust to resume, with --resume and --resume-from, and --force now archives outputs instead of deleting them.**
- **Next: Document extension API, update all workflow JSONs, and test new modular pipeline logic.**

## Recent Changes

- ✅ **Implemented complete resume functionality (pipeline_state.py, resume_utils.py)**
- ✅ **Enhanced orchestrator with run_with_resume() method - backward compatible**
- ✅ **Added comprehensive CLI arguments: --resume, --resume-from, --show-resume-status**
- ✅ **Created full test suite with 100% pass rate**
- ✅ **Validated state persistence, failure recovery, and skip logic**
- Completed implementation of all pipeline stages and orchestrator logic
- Integrated robust error handling and defensive filtering
- Finalized privacy-first manifest/logging and traceability patterns
- Added LLM integration, master transcript, and extensible workflow config
- **Added `--output-folder` argument to CLI.**
- **Made `input_dir` optional for resume.**
- **Refactored main script logic to distinguish between fresh and resume runs.**
- **Updated help text and argument parser.**
- **Patched all resume/status/clear/force commands to operate on the output folder.**
- **Ensured that when resuming, jobs are reconstructed from the anonymized `renamed/` directory, not from the original input.**
- **Console output for available folders now only shows anonymized run folder names.**
- **Removed all CLAP logic from the main pipeline; CLAP is now only run as an extension or via CLI flag.**
- **Formalized extensions system for pre/post-processing.**

## Next Steps

- **Immediate: Update all workflow JSONs to reflect new modular pipeline logic and per-workflow LLM config.**
- **Secondary: Document and test the extension API and CLI flag for CLAP.**
- Enhanced error handling and edge cases for resume functionality
- Advanced resume controls (--resume-from, --force-rerun, --clear-from)
- Real-world integration testing with actual audio files
- Performance monitoring and stage timing analytics
- Complete and test finalization stage for MP3 outputs and metadata
- Ensure robust fallback logic for LLM output
- Update documentation and memory bank as the project evolves
- Continue to monitor for any edge cases where PII could leak (e.g., error messages, stack traces).
- Consider adding automated tests to verify no PII is ever output during resume/status operations.

## Active Decisions & Considerations

- Resume functionality is production-ready but can be enhanced with granular controls
- Debugging workflow dramatically improved - no more re-running expensive stages
- All outputs and logs are strictly PII-free and fully auditable
- User preferences and workflow logic are extensible via CLI and workflow JSONs
- Defensive programming and robust error handling are required at every stage
- Show folder is always 'show/', but MP3 and .txt are named after LLM show title (fallback to completed-show.mp3 if needed)
- Show description is included in manifest, ID3 tags, and as a separate .txt file
- All soundbites are converted to MP3 with full metadata and included in finalized/soundbites/
- **CLAP is only run as an extension or via CLI flag, not as a default pipeline stage.**
- **Extensions are external scripts for pre/post-processing, not core pipeline stages.**
- **LLM config is per-workflow, with defaults in llm_tasks.json.**

## Current Focus: Modular, workflow-driven pipeline; CLAP and extensions as optional, user-invoked steps; LLM as a modular bus.

## 2024-06-XX: Pipeline Stability Milestone
- All major features (resume, finalized/calls export, single-file and tuple handling, privacy enforcement, tones, etc.) are now robust and working as intended.
- Pipeline passes 12+ hours of continuous testing across all scenarios (fresh runs, resume from any stage, single-file and tuple jobs, force reruns, etc.).
- Finalization is robust to missing manifest entries and always exports valid calls.
- No PII leaks in logs or outputs.
- The pipeline is now considered stable and production-ready.
- **LLM-powered show notes generation is now implemented and tested.**
- Complete and test finalization stage for MP3 outputs and metadata 