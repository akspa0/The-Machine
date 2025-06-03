# 🛠️ Workflow Configurations

---

## Overview

This folder contains **workflow JSONs** that configure the main pipeline, CLAP annotation/segmentation, LLM tasks, and extension workflows. Workflows make The-Machine flexible, extensible, and easy to adapt to new tasks or models.

---

## 📂 Main Workflow Files

- `llm_tasks.json` — Defines LLM tasks, prompt templates, and model parameters for all LLM-driven extensions.
- `clap_annotation.json` — Configures CLAP annotation prompts, thresholds, and chunking for event detection.
- `clap_segmentation.json` — Configures CLAP-based segmentation for call boundary detection.

---

## 🚀 How to Use & Modify Workflows

- Edit any workflow JSON to add, remove, or modify tasks, prompts, or model settings.
- Add new workflow files for custom extensions or new processing stages.
- Reference workflow files in your extension or pipeline CLI options.
- Share your workflow configs with others for reproducible results.

---

## 🛡️ Best Practices

- Keep workflows modular and focused on a single task or stage.
- Use clear, descriptive names for tasks and prompts.
- Document any custom workflows for reproducibility.
- Version-control your workflow configs for traceability.

---

## 📚 Related Extensions & Docs

- See [`../extensions/README.md`](../extensions/README.md) for the full extension system overview.
- See [`../extensions/llm_utils.py`](../extensions/llm_utils.py) for LLM utility usage.

---

**Workflows make The-Machine flexible, extensible, and ready for any context-driven task.** 