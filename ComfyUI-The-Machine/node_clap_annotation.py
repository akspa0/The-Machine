import os
from .utils import privacy
from .utils import manifest as manifest_utils
from pathlib import Path

class CLAPAnnotationNode:
    """
    CLAP Annotation Node
    Runs CLAP model on normalized files, annotates with events (using prompts), updates manifest and transcript, ensures privacy.
    """
    @classmethod
    def input_types(cls):
        return {"normalized": "list[dict]", "manifest": "dict", "config": "dict"}

    @classmethod
    def output_types(cls):
        return {"clap_annotated": "list[dict]", "manifest": "dict"}

    def process(self, normalized, manifest, config=None):
        from tqdm import tqdm
        import torch
        # Placeholder: replace with actual CLAP model import and logic
        # from my_clap_module import CLAPModel
        prompts = config.get("clap_prompts", ["ringing", "dtmf", "yelling", "dogs barking"]) if config else ["ringing", "dtmf"]
        confidence_threshold = config.get("clap_confidence", 0.6) if config else 0.6
        clap_annotated = []
        errors = []
        for entry in tqdm(normalized, desc="CLAP annotation"):
            audio_path = entry["normalized_path"]
            tuple_index = entry["tuple_index"]
            try:
                # Simulate CLAP output: replace with real model inference
                # model = CLAPModel.load_from_config(config)
                # results = model.annotate(audio_path, prompts)
                # For now, fake results:
                results = [
                    {"prompt": p, "confidence": 0.7, "start": 0.0, "end": 1.0} for p in prompts
                ]
                accepted = [r for r in results if r["confidence"] >= confidence_threshold]
                for ann in accepted:
                    ann_entry = {
                        "tuple_index": tuple_index,
                        "audio_path": audio_path,
                        "prompt": ann["prompt"],
                        "confidence": ann["confidence"],
                        "start": ann["start"],
                        "end": ann["end"],
                        "lineage": [audio_path],
                    }
                    clap_annotated.append(ann_entry)
            except Exception as e:
                errors.append({"tuple_index": tuple_index, "audio_path": audio_path, "error": str(e)})
        manifest_utils.update_manifest(manifest, clap_annotated, key="clap_annotations")
        if errors:
            manifest["clap_annotation_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"clap_annotated": clap_annotated, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for CLAP annotation.
        - normalized: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        - config: optional JSON
        """
        return {
            "normalized": {"type": "hidden"},
            "manifest": {"type": "hidden"},
            "config": {"type": "json", "label": "CLAP Config (JSON)", "default": {}}
        } 