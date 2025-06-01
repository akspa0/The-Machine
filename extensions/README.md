# Extension System for The-Machine

> **Note:** This extension system is for [The-Machine](https://github.com/akspa0/The-Machine). All extensions must follow the project's extension-driven, API-first, librarian-orchestrator architecture. See the main README for global philosophy and best practices.

## Overview
Extensions allow you to add post-processing, context generation, and derivative art/analytics to the finalized outputs of the pipeline. Extensions run after the main pipeline and operate only on anonymized, finalized data. All new features and improvements should be implemented as modular extensions ("stacks"), not as monolithic pipeline logic.

## Character Persona Builder Extension

- The `character_persona_builder.py` extension generates advanced Character.AI persona definitions for each call/channel/speaker using transcripts and LLMs.
- **Channel folders may have run-specific prefixes** (e.g., `0000-conversation`). The extension normalizes these for output and is robust to naming.
- **For conversation-only calls:** Generates a separate persona for each detected speaker (no merging).
- **For left/right calls:** Merges all speakers per channel and generates one persona per channel.
- **System prompt and persona style are embedded** for best results (no external files needed).
- **Usage example:**
  ```sh
  python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
  ```
- Outputs are written to `characters/<call_title or call_id>/<channel or conversation_speaker>/` with transcript, persona, and audio clips.

## Best Practices for Extension Authors
- Be robust to folder naming (handle run-specific prefixes, normalize for output).
- Log only anonymized, PII-free information.
- Support both batch and single-file workflows.
- Document your extension's purpose and usage.
- Follow privacy and traceability rules.

## General Extension Workflow
- Place your extension scripts in the `extensions/` directory.
- Each extension should inherit from `ExtensionBase` (see `extension_base.py`).
- Extensions are run manually or can be invoked automatically after pipeline completion.
- Extensions receive the root output directory as their argument and should only access finalized outputs.

## How Extensions Work
- Place your extension scripts in the `extensions/` directory.
- Each extension should inherit from `ExtensionBase` (see `extension_base.py`).
- Extensions are run manually or can be invoked automatically after pipeline completion.
- Extensions receive the root output directory as their argument and should only access finalized, anonymized outputs.
- All outputs, logs, and manifest updates must be strictly PII-free and fully traceable.

## Authoring Extensions
- **Privacy:** Never access or log original filenames, paths, or PII. Only use anonymized, finalized data.
- **Traceability:** Use the manifest and output folder structure for all data lineage.
- **Idempotence:** Extensions should be safe to run multiple times.
- **Robustness:** Handle missing or partial data gracefully.
- **Documentation:** Clearly document your extension's purpose, usage, and CLI options.
- **Testing:** Extensions should be independently testable and reusable.

## Example Extension Usage
```sh
python extensions/character_persona_builder.py outputs/run-YYYYMMDD-HHMMSS --llm-config workflows/llm_tasks.json
```

## Using LLM Utilities in Extensions
- For large text artifacts (e.g., personas, transcripts), use `llm_tokenize.py` to chunk files for LLM processing.
- Use `llm_summarize.py` to generate creative SDXL prompts or summaries from multiple chunks.
- See the main README for detailed usage examples.

## Best Practices
- Be robust to folder naming (handle run-specific prefixes, normalize for output).
- Log only anonymized, PII-free information.
- Support both batch and single-file workflows.
- Follow privacy, traceability, and idempotence rules.
- Use the manifest and output folder structure for all data lineage.
- Handle missing or partial data gracefully.

## Example Extensions
- `character_persona_builder.py`: Generates advanced Character.AI persona definitions for each call/channel/speaker using transcripts and LLMs.
- `avatar/sdxl_avatar_generator.py`: Generates persona (avatar) and backdrop images for each call/persona using SDXL workflows.
- `comfyui_image_generator.py`: Enables LLM-driven image and video generation from audio transcripts.
- Analytics, visualizations, LLM-based summaries, and more can be added as new extensions.

## Contributing
- Document your extension's purpose and usage.
- Follow the privacy and traceability rules outlined above.
- Reference the main README for global architecture and philosophy. 