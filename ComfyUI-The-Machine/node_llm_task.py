from .utils import privacy
from .utils import manifest as manifest_utils
import os

class LLMTaskNode:
    """
    LLM Task Node
    Runs LLM tasks as defined in workflow presets, renders prompts, saves outputs, updates manifest.
    """
    @classmethod
    def input_types(cls):
        return {"manifest": "dict", "transcript": "list[str]", "workflow_preset": "dict"}

    @classmethod
    def output_types(cls):
        return {"llm_outputs": "list[dict]", "manifest": "dict"}

    def process(self, manifest, transcript, workflow_preset):
        llm_outputs = []
        errors = []
        tasks = workflow_preset.get("tasks", [
            {"name": "call_title", "prompt_template": "Summarize the following call transcript..."}
        ])
        for idx, task in enumerate(tasks):
            try:
                # --- Render prompt from template and transcript/context ---
                prompt = task["prompt_template"].replace("{transcript}", " ".join(transcript))
                # --- LLM call stub: simulate output ---
                llm_output = f"Simulated output for {task['name']}"
                # --- Save output to file (stub) ---
                output_dir = os.path.join("llm_outputs", f"{idx:04d}")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{task['name']}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(llm_output)
                # --- Manifest update ---
                llm_entry = {
                    "tuple_index": f"{idx:04d}",
                    "llm_task_name": task["name"],
                    "prompt_template": task["prompt_template"],
                    "rendered_prompt": prompt,
                    "llm_output_file": output_file,
                    "llm_model": {
                        "name": workflow_preset.get("model", "openai-gpt-4"),
                        "version": workflow_preset.get("version", "2024-06-01"),
                        "parameters": workflow_preset.get("parameters", {"temperature": 0.7, "max_tokens": 128})
                    },
                    "llm_task_timestamp": "2025-06-01T00:35:00Z",
                    "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task"]
                }
                llm_outputs.append(llm_entry)
            except Exception as e:
                errors.append({"task": task.get("name", str(idx)), "error": str(e)})
        manifest_utils.update_manifest(manifest, llm_outputs)
        if errors:
            manifest["llm_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"llm_outputs": llm_outputs, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for workflow selection, output display
        pass 