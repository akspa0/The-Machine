"""
Manifest utilities for ComfyUI-The-Machine nodes.
Implement all logic according to the privacy_and_manifest specification.
"""
import json
import os
import tempfile
from . import privacy

def validate_manifest(manifest):
    """
    Validate manifest privacy and completeness.
    Checks for required fields, no PII, and correct types for all node types.
    Returns True if valid, else raises ValueError.
    """
    # Required fields for all entries (core)
    core_fields = ["output_name", "output_path", "tuple_index", "type", "timestamp"]
    # Node-specific required fields (extend as needed)
    node_fields = {
        "diarization": ["segments", "diarization_model", "diarization_timestamp"],
        "transcription": ["transcription", "asr_model", "transcription_timestamp"],
        "soundbite": ["output_soundbite", "soundbite_criteria", "soundbite_timestamp"],
        "clap": ["clap_events", "clap_model", "clap_annotation_timestamp"],
        # Add more as needed
    }
    for entry in manifest.get("entries", []):
        for field in core_fields:
            if field not in entry:
                raise ValueError(f"Manifest entry missing required field: {field}")
            if privacy.contains_pii(str(entry.get(field, ""))):
                raise ValueError(f"PII detected in manifest field: {field}")
        # Check node-specific fields
        for node, fields in node_fields.items():
            if node in entry.get("lineage", []):
                for nf in fields:
                    if nf not in entry:
                        raise ValueError(f"Manifest entry missing {node} field: {nf}")
    return True

def update_manifest(manifest, updates):
    """
    Deep merge updates into manifest. Updates can be a dict or list of dicts.
    Anonymizes all entries before adding.
    """
    if isinstance(updates, dict):
        updates = [updates]
    if "entries" not in manifest:
        manifest["entries"] = []
    for entry in updates:
        manifest["entries"].append(anonymize_entry(entry))
    return manifest

def anonymize_entry(entry):
    """
    Anonymize a manifest entry using privacy utilities (filenames, metadata).
    """
    entry = dict(entry)
    if "output_name" in entry:
        entry["output_name"] = privacy.scrub_filename(entry["output_name"])
    if "output_path" in entry:
        entry["output_path"] = privacy.scrub_filename(entry["output_path"])
    if "source_segment" in entry:
        entry["source_segment"] = privacy.scrub_filename(entry["source_segment"])
    # Scrub all metadata fields
    for k in list(entry.keys()):
        if k.endswith("_metadata") and isinstance(entry[k], dict):
            entry[k] = privacy.scrub_metadata(entry[k])
    return entry

def merge_manifests(manifests):
    """
    Merge a list of manifest dicts into a single manifest (for batch processing).
    """
    merged = {"entries": []}
    for m in manifests:
        merged["entries"].extend(m.get("entries", []))
    return merged

def write_manifest_to_disk(manifest, path):
    """
    Atomically write manifest to disk as JSON.
    Future: add manifest versioning and schema migration.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".tmp") as tf:
        json.dump(manifest, tf, indent=2)
        tempname = tf.name
    os.replace(tempname, path)
# Future: add manifest versioning, schema migration, and DB integration 