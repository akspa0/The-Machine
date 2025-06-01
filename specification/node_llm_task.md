# Node Specification: LLM Task Node

## Purpose
This node runs LLM tasks as defined in workflow presets (JSON/YAML), renders prompts from transcript/context, saves outputs to the call's output folder, and updates the manifest with LLM outputs and references.

## Inputs
- Manifest and transcript from the Soundbite Generation Node
- Workflow preset (JSON/YAML) defining LLM tasks, prompt templates, output files, and model parameters
- Audio and transcript context for each call/segment

## Outputs
- LLM output files (e.g., call title, synopsis, categories, image prompt, song, etc.) saved in /llm_outputs/<tuple_index>/
- Updated manifest with LLM task results and lineage

## Core Responsibilities
- For each call/segment as defined in the workflow preset:
  - Render prompt using transcript/context and preset template
  - Run LLM task (local or API, e.g., OpenAI, LM Studio)
  - Save output to /llm_outputs/<tuple_index>/
  - Update manifest with:
    - llm_task_name
    - prompt_template
    - rendered_prompt
    - llm_output_file (filename/path)
    - llm_model (name, version, parameters)
    - llm_task_timestamp
    - lineage (add LLM step)
- Ensure all outputs and manifest updates are PII-free

## Workflow Preset Support
- Workflow presets are JSON/YAML files in /workflows/
- Each preset defines:
  - Task names
  - Prompt templates (with variables for transcript/context)
  - Output file mapping
  - Model parameters (e.g., temperature, max tokens)
- Presets are user-extensible: users can add, remove, or modify tasks and prompts without code changes

## Privacy & PII Logic
- No original filenames, paths, or PII in output files or manifest
- Validate that all outputs are anonymized before updating manifest
- All manifest updates must be privacy-compliant

## Batch & Single-File Support
- Batch: Process all calls/segments in the batch as defined in the workflow preset
- Single-file: Process one call/segment at a time
- Manifest must support both modes (list of LLM results for batch, single result for single-file)

## Error Handling
- Flag LLM tasks that fail (e.g., API error, invalid prompt) in the manifest
- Log only anonymized, non-PII information about errors
- Manifest should include a 'llm_errors' section for any failures

## UI/UX Notes
- Allow user to select workflow preset and model parameters (config panel)
- Display rendered prompts, LLM outputs, and task status for each call/segment
- Teal-themed UI elements for workflow selection, progress bars, and output display
- Show manifest summary (number of LLM tasks, outputs, and errors)

## Example Manifest LLM Entry
```json
{
  "tuple_index": "0000",
  "llm_task_name": "call_title",
  "prompt_template": "Summarize the following call transcript...",
  "rendered_prompt": "Summarize the following call transcript: ...",
  "llm_output_file": "llm_outputs/0000/call_title.txt",
  "llm_model": {
    "name": "openai-gpt-4",
    "version": "2024-06-01",
    "parameters": {"temperature": 0.7, "max_tokens": 128}
  },
  "llm_task_timestamp": "2025-06-01T00:35:00Z",
  "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task"]
}
```

## Validation
- Unit tests for LLM task execution, manifest updates, and error handling
- Integration tests for batch and single-file workflows
- UI tests for workflow selection, prompt rendering, and output display 