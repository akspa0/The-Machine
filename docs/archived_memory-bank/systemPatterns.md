# systemPatterns.md

**Purpose:**
Documents system architecture, key technical decisions, design patterns, and component relationships.

## New Patterns (2024-06-XX)

- **Auto-generation of persona manifests:**
  - If the persona manifest is missing, the pipeline automatically runs character_persona_builder.py to generate it before proceeding.
- **Prompt caching and reuse:**
  - Prompts used for avatar image generation are cached and reused for video workflows, with action words appended for animation context.
- **API-based file upload to ComfyUI:**
  - All image inputs for ComfyUI are uploaded via the /upload/image API endpoint; no direct file system manipulation is used for ComfyUI input.
- **Robust polling for long-running jobs:**
  - The orchestrator polls the ComfyUI API for up to 1 hour (10s interval, 300 tries) to support long-running video jobs.
- **Canonical output copying:**
  - All outputs (avatar images, videos) are copied from ComfyUI's output directory into the project structure, with canonical naming (persona_avatar, etc.) and tracked in the manifest.
- **Fully automated, API-driven pipeline:**
  - The entire pipeline is now robust, automated, and API-driven, with all major error handling and output tracking features implemented.

## Architecture Overview

- Batch-oriented pipeline for audio file ingestion, renaming, and processing
- Modular steps for PII removal, separation, normalization, diarization, metadata propagation, remixing, and show creation
- Explicit handling of sample rates, channel assignments, and recursive folder processing throughout the pipeline
- Strict chronological tracking of calls and tuples using a unique, zero-padded index (e.g., <0000>, <0001>, ...)
- **Pipeline is now fully workflow-driven: all stage order, config, and routing are defined in workflow JSONs.**
- **All CLAP logic has been removed from the main pipeline. CLAP segmentation/detection is now handled exclusively by the extension (clap_segmentation_experiment.py). The main pipeline is fully decoupled from CLAP.**
- **Extensions system is formalized: external scripts in extensions/ can be run as pre- or post-processing, either manually or via CLI.**
- **LLM processing is handled as a modular 'bus' and is only run if defined in the workflow JSON. Per-workflow LLM config is supported.**
- **NEW: ComfyUI extension enables LLM-driven, scene-based prompt generation for both image and video workflows, with hierarchical summarization and batching.**
- **NEW: Two-phase extension execution: all prompts and ComfyUI job configs are generated first, then LLM is unloaded before jobs are sent.**
- **NEW: Robust workflow selection and prompt insertion logic: only positive prompt nodes are updated in ComfyUI workflows, never negative.**
- **NEW: Extension system is now the standard for all non-core pipeline logic, including image/video generation and CLAP segmentation.**

## Key Technical Decisions

- Assign a unique, zero-padded chronological index to each call tuple, based on timestamp
- Use the index as a prefix for all output folders, files, and manifest entries
- Remove PII from all filenames at ingestion
- Maintain traceability by prefixing all outputs with tuple index
- Use open-source, state-of-the-art models for each processing step
- Propagate and update audio metadata (ID3 tags) at every step using mutagen
- All input and output files are 44.1kHz stereo; resample to 16kHz mono only for model inference (pyannote, parakeet)
- Soundbites and final outputs are always cut from the highest quality, normalized split vocal audio (not from 16kHz model input)
- Remixing: vocals and instrumentals are mixed per channel (instrumentals at 50% volume), then channels are combined into a stereo file with 20% separation per channel (40% total center separation)
- Output directory structure mirrors input, with each call in its own indexed folder containing /soundbites, /call, /transcripts, /llm_outputs
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- **CLAP is only run as an extension (clap_segmentation_experiment.py) and is fully decoupled from the main pipeline.**
- **Extensions are external scripts for pre/post-processing, not core pipeline stages.**
- **LLM config is per-workflow, with defaults in llm_tasks.json.**
- Integrate CLAP annotation as a core step for all out- prefixed files and other audio types
- Use a configurable set of prompts for CLAP to detect contextual audio events (e.g., dogs barking, DTMF tones, ringing, yelling, etc.)
- Set a confidence threshold of 0.6 for accepting CLAP annotations
- Merge CLAP annotations into the master transcript and manifest for each call, providing context for downstream LLM processing
- Diarization (pyannote) is performed on the highest quality, normalized vocals (not 16kHz model input)
- Diarization output is used to segment audio into per-speaker, per-segment files
- Each speaker's segments are stored in a folder structure: speakers/S00/, speakers/S01/, etc.
- Each segment file is named with a chronological index and a short, meaningful name (≤ 488 characters)
- For each segment, a transcription is generated (parakeet) and saved as a .txt file in the same folder
- Segment files are renamed to include the index and a short version of the transcription
- Manifest records original input file, timestamp range, speaker ID, index, and transcription for each segment
- In later steps, all segments are converted to MP3 and tagged with full lineage (input file, timestamps, speaker, transcription, etc.)
- LLM task management is driven by configurable workflow presets (JSON/YAML) that define llm_tasks, model parameters, and output mapping
- Each llm_task specifies a name, prompt_template (with placeholders like {transcript}), and output_file
- The LLM module (e.g., llm_module.py) executes all tasks per call, saving outputs to the call's output folder and returning output paths for manifest integration
- Workflow presets are fully extensible: users can add, remove, or modify tasks and prompts without code changes
- For long audio files (e.g., ≥10 minutes), enable CLAP-driven segmentation mode via CLI/config
- Use CLAP prompts (e.g., 'telephone ring tones', 'hang-up tones') to detect call boundaries and segment audio into individual calls
- Each segment is processed as a single-file call (mono vocals, no left/right distinction), following the standard call pipeline
- CLI/config allows users to enable/disable segmentation mode and specify segmentation prompts
- Manifest records original file, segment boundaries, and full processing lineage for each segmented call
- This segmentation approach is extensible to other use cases (e.g., segmenting by other sound events)
- All workflows are defined as JSON files and stored in a workflows/ folder at the project root
- Each workflow JSON contains routing/configuration for audio processing (LLM tasks, CLAP prompts, segmentation, etc.)
- A separate config/ folder stores user-specific data, such as HuggingFace token (set via huggingface-cli) and other credentials or preferences
- The program reads from config/ for user data and from workflows/ for pipeline logic
- **CLAP-based segmentation logic for --call-cutter: flag is present, segmentation logic is pending implementation.**

## Design Patterns

- File tuple identification: out-, trans_out-, recv_out- prefix matching
- Filename parsing and reformatting to <0000>-<prefix>-YYYYMMDD-HHMMSS.wav
- Output file/folder naming: always prefix with chronological index
- Modular pipeline: each step operates on tracked files, outputs to next step
- Metadata pattern: At each step, read, update, and write metadata to preserve lineage and context
- Resampling pattern: Only resample to 16kHz for model input; all other processing uses highest available quality
- Manifest pattern: Track all file versions, sample rates, indices, and processing lineage
- Recursive folder processing: scan input directories and subfolders, mirror structure in output
- Show pattern: concatenate finalized calls in order, insert tones, and document order in a text file
- CLAP annotation pattern: After normalization, run CLAP on the combined mono (out-) file using relevant prompts; filter results by confidence; add accepted annotations to the transcript and manifest
- Diarization/transcription pattern: Segment normalized vocals per speaker, store in speakers/SXX/ folders, transcribe each segment, save .txt alongside audio, and rename files with index and short transcription (≤ 488 chars)
- Manifest pattern: Track all segment lineage, including input file, timestamps, speaker, index, and transcription
- LLM workflow pattern: For each call, run all llm_tasks as defined in the workflow preset, render prompts, call the LLM API, save outputs, and update the manifest
- **Segmentation pattern: For long audio, use CLAP to detect call boundaries, segment into single-file calls, process each as a call tuple, and track all lineage in the manifest.**
- CLAP-driven segmentation pattern: For long audio, use CLAP to detect boundaries, segment into calls, process each as a single-file call, and track all lineage in the manifest
- Workflow/config separation pattern: workflows/ contains pipeline logic (JSON), config/ contains user/environment data; program loads both at runtime

## Component Relationships

- Ingestion → PII removal/renaming → Separation/Annotation/Normalization/Metadata/Remixing → Diarization prep → Segmentation → Transcription → Soundbite extraction → Show creation
- Each processing step updates file tracking, metadata, and outputs for next step
- Manifest and metadata ensure traceability and quality at every stage
- After remixing, mix vocals and instrumentals for trans_out and recv_out pairs into left and right channels, with 50% volume on instrumental track
- Produce new call files with chronological call index as filename and LLM-generated call title, sanitized of all punctuation from LLM call title response
- Optionally apply tones to call files (if not in show-mode)
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- All workflows are JSON files in workflows/ (routing, prompts, tasks, etc.), user-specific data in config/

# System Patterns

- **NEW (2024-06): Soundbites folders are now named after sanitized call titles, not just call IDs.**
- **NEW: SFW (safe-for-work) call titles and show summaries are generated and included in outputs.**
- **NEW: Show notes are generated by the LLM and appended to show summaries.**
- **NEW: Extensions system: users can add scripts (like character_ai_description_builder.py) to an extensions/ folder, which run after the main pipeline and can use all outputs.**
- **NEW: CLAP annotation is now optional and disabled by default; out- files are only processed for CLAP unless --process-out-files is set.**
- **NEW: All outputs, logs, and manifests are strictly PII-free and fully auditable.**
- **NEW: Timestamps in show summaries are now in HH:MM:SS format, with emoji for calls/tones.**
- **NEW: Resume and force: The pipeline is robust to resume, with --resume and --resume-from, and --force now archives outputs instead of deleting them.**
- **Next: Build out extension API and document extension interface.**

# systemPatterns.md

**Purpose:**
Documents system architecture, key technical decisions, design patterns, and component relationships.

## Architecture Overview

- Batch-oriented pipeline for audio file ingestion, renaming, and processing
- Modular steps for PII removal, separation, normalization, diarization, metadata propagation, remixing, and show creation
- Explicit handling of sample rates, channel assignments, and recursive folder processing throughout the pipeline
- Strict chronological tracking of calls and tuples using a unique, zero-padded index (e.g., <0000>, <0001>, ...)

## Key Technical Decisions

- Assign a unique, zero-padded chronological index to each call tuple, based on timestamp
- Use the index as a prefix for all output folders, files, and manifest entries
- Remove PII from all filenames at ingestion
- Maintain traceability by prefixing all outputs with tuple index
- Use open-source, state-of-the-art models for each processing step
- Propagate and update audio metadata (ID3 tags) at every step using mutagen
- All input and output files are 44.1kHz stereo; resample to 16kHz mono only for model inference (pyannote, parakeet)
- Soundbites and final outputs are always cut from the highest quality, normalized split vocal audio (not from 16kHz model input)
- Remixing: vocals and instrumentals are mixed per channel (instrumentals at 50% volume), then channels are combined into a stereo file with 20% separation per channel (40% total center separation)
- Output directory structure mirrors input, with each call in its own indexed folder containing /soundbites, /call, /transcripts, /llm_outputs
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- **CLAP is only run as an extension or via CLI flag, not as a default pipeline stage.**
- **Extensions are external scripts for pre/post-processing, not core pipeline stages.**
- **LLM config is per-workflow, with defaults in llm_tasks.json.**
- Integrate CLAP annotation as a core step for all out- prefixed files and other audio types
- Use a configurable set of prompts for CLAP to detect contextual audio events (e.g., dogs barking, DTMF tones, ringing, yelling, etc.)
- Set a confidence threshold of 0.6 for accepting CLAP annotations
- Merge CLAP annotations into the master transcript and manifest for each call, providing context for downstream LLM processing
- Diarization (pyannote) is performed on the highest quality, normalized vocals (not 16kHz model input)
- Diarization output is used to segment audio into per-speaker, per-segment files
- Each speaker's segments are stored in a folder structure: speakers/S00/, speakers/S01/, etc.
- Each segment file is named with a chronological index and a short, meaningful name (≤ 488 characters)
- For each segment, a transcription is generated (parakeet) and saved as a .txt file in the same folder
- Segment files are renamed to include the index and a short version of the transcription
- Manifest records original input file, timestamp range, speaker ID, index, and transcription for each segment
- In later steps, all segments are converted to MP3 and tagged with full lineage (input file, timestamps, speaker, transcription, etc.)
- LLM task management is driven by configurable workflow presets (JSON/YAML) that define llm_tasks, model parameters, and output mapping
- Each llm_task specifies a name, prompt_template (with placeholders like {transcript}), and output_file
- The LLM module (e.g., llm_module.py) executes all tasks per call, saving outputs to the call's output folder and returning output paths for manifest integration
- Workflow presets are fully extensible: users can add, remove, or modify tasks and prompts without code changes
- For long audio files (e.g., ≥10 minutes), enable CLAP-driven segmentation mode via CLI/config
- Use CLAP prompts (e.g., 'telephone ring tones', 'hang-up tones') to detect call boundaries and segment audio into individual calls
- Each segment is processed as a single-file call (mono vocals, no left/right distinction), following the standard call pipeline
- CLI/config allows users to enable/disable segmentation mode and specify segmentation prompts
- Manifest records original file, segment boundaries, and full processing lineage for each segmented call
- This segmentation approach is extensible to other use cases (e.g., segmenting by other sound events)
- All workflows are defined as JSON files and stored in a workflows/ folder at the project root
- Each workflow JSON contains routing/configuration for audio processing (LLM tasks, CLAP prompts, segmentation, etc.)
- A separate config/ folder stores user-specific data, such as HuggingFace token (set via huggingface-cli) and other credentials or preferences
- The program reads from config/ for user data and from workflows/ for pipeline logic

## Design Patterns

- File tuple identification: out-, trans_out-, recv_out- prefix matching
- Filename parsing and reformatting to <0000>-<prefix>-YYYYMMDD-HHMMSS.wav
- Output file/folder naming: always prefix with chronological index
- Modular pipeline: each step operates on tracked files, outputs to next step
- Metadata pattern: At each step, read, update, and write metadata to preserve lineage and context
- Resampling pattern: Only resample to 16kHz for model input; all other processing uses highest available quality
- Manifest pattern: Track all file versions, sample rates, indices, and processing lineage
- Recursive folder processing: scan input directories and subfolders, mirror structure in output
- Show pattern: concatenate finalized calls in order, insert tones, and document order in a text file
- CLAP annotation pattern: After normalization, run CLAP on the combined mono (out-) file using relevant prompts; filter results by confidence; add accepted annotations to the transcript and manifest
- Diarization/transcription pattern: Segment normalized vocals per speaker, store in speakers/SXX/ folders, transcribe each segment, save .txt alongside audio, and rename files with index and short transcription (≤ 488 chars)
- Manifest pattern: Track all segment lineage, including input file, timestamps, speaker, index, and transcription
- LLM workflow pattern: For each call, run all llm_tasks as defined in the workflow preset, render prompts, call the LLM API, save outputs, and update the manifest
- CLAP-driven segmentation pattern: For long audio, use CLAP to detect boundaries, segment into calls, process each as a single-file call, and track all lineage in the manifest
- Workflow/config separation pattern: workflows/ contains pipeline logic (JSON), config/ contains user/environment data; program loads both at runtime

## Component Relationships

- Ingestion → PII removal/renaming → Separation/Annotation/Normalization/Metadata/Remixing → Diarization prep → Segmentation → Transcription → Soundbite extraction → Show creation
- Each processing step updates file tracking, metadata, and outputs for next step
- Manifest and metadata ensure traceability and quality at every stage
- After remixing, mix vocals and instrumentals for trans_out and recv_out pairs into left and right channels, with 50% volume on instrumental track
- Produce new call files with chronological call index as filename and LLM-generated call title, sanitized of all punctuation from LLM call title response
- Optionally apply tones to call files (if not in show-mode)
- Show output: concatenate all valid calls (>10s, error-free) into a single WAV file, insert tones between calls, and include a text file listing call order, names, timestamps, and metadata
- All workflows are JSON files in workflows/ (routing, prompts, tasks, etc.), user-specific data in config/

# System Patterns

- Resume logic is now output-folder-centric, not input-centric.
- All job creation and file discovery for resume is based on the anonymized `renamed/` directory.
- Console output for available folders is sanitized and only shows anonymized run folder names.

## 2024-06-XX: Robustness Patterns
- Finalization stage falls back to scanning call/ for remixed calls if manifest is missing remix entries, ensuring export always works.
- All transcript and output pathing uses mapping logic (call_id_to_folder) to avoid folder naming mismatches.
- CLI flags (e.g., --call-tones) are robustly propagated through all run and resume modes.
- Strict privacy enforcement: no PII in logs or outputs at any stage.
- **Finalization stage:**
  - All valid soundbites and show audio are converted to 192kbps VBR MP3s in a finalized/ output folder.
  - ID3 tags are embedded in all MP3s, including call index, channel, speaker, segment index, timestamps, transcript, LLM-generated call title, and full lineage.
  - LLM-driven show title and description are generated via a secondary LLM task using all call titles and synopses; show title is a short, family-friendly, comedic sentence or phrase.
  - Show MP3 and .txt description are named after the sanitized LLM show title; fallback to completed-show.mp3 if LLM output is missing/invalid.
  - Show description is included in manifest, ID3 tags, and as a separate .txt file.
  - Two-stage LLM workflow: per-call titles/synopses, then show-level title/description.
  - Manifest and logs are updated with all finalized outputs and metadata.
  - **LLM-powered show notes generation: For each completed show, an LLM task generates a privacy-safe, compelling paragraph ("show notes") summarizing the show and enticing listeners. Show notes are saved in finalized/show/, referenced in the manifest, and appended to show description files. Show notes must never include PII and should be family-friendly and enticing.**

**Project Renamed:**
- The tool is now named **The-Machine**.
- New GitHub repo: https://github.com/akspa0/The-Machine
- All documentation and onboarding now emphasize conda as the recommended environment manager for PyTorch and GPU support.
- All references to the old name have been replaced in docs and onboarding. 