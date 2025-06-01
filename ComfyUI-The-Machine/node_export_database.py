from .utils import privacy
from .utils import manifest as manifest_utils
import os
import sqlite3
import json
import csv

class ExportDatabaseNode:
    """
    Export/Database Node
    Exports manifest/metadata to a SQLite database, supports CSV/JSON export, updates manifest.
    """
    @classmethod
    def input_types(cls):
        return {"manifest": "dict", "export_dir": "str", "export_csv": "bool", "export_json": "bool"}

    @classmethod
    def output_types(cls):
        return {"export_info": "dict", "manifest": "dict"}

    def process(self, manifest, export_dir="database", export_csv=False, export_json=False):
        errors = []
        os.makedirs(export_dir, exist_ok=True)
        db_path = os.path.join(export_dir, "manifest.db")
        try:
            # --- Create SQLite DB and table ---
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            # Create table with generic columns (expand as needed)
            c.execute("""
                CREATE TABLE IF NOT EXISTS manifest_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    output_name TEXT,
                    output_path TEXT,
                    tuple_index TEXT,
                    type TEXT,
                    timestamp TEXT,
                    lineage TEXT,
                    data JSON
                )
            """)
            # Insert all manifest entries
            for entry in manifest.get("entries", []):
                # Remove PII and ensure privacy compliance
                clean_entry = manifest_utils.anonymize_entry(entry)
                # Store all non-core fields as JSON in 'data'
                core = {k: clean_entry.get(k) for k in ["output_name", "output_path", "tuple_index", "type", "timestamp", "lineage"]}
                extra = {k: v for k, v in clean_entry.items() if k not in core}
                c.execute(
                    "INSERT INTO manifest_entries (output_name, output_path, tuple_index, type, timestamp, lineage, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        core["output_name"],
                        core["output_path"],
                        core["tuple_index"],
                        core["type"],
                        core["timestamp"],
                        json.dumps(core.get("lineage", [])),
                        json.dumps(extra)
                    )
                )
            conn.commit()
            conn.close()
        except Exception as e:
            errors.append({"error": str(e)})
        # --- Optionally export CSV ---
        csv_path = None
        if export_csv:
            try:
                csv_path = os.path.join(export_dir, "manifest.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = list(manifest.get("entries", [{}])[0].keys()) if manifest.get("entries") else []
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for entry in manifest.get("entries", []):
                        writer.writerow(manifest_utils.anonymize_entry(entry))
            except Exception as e:
                errors.append({"csv_error": str(e)})
        # --- Optionally export JSON ---
        json_path = None
        if export_json:
            try:
                json_path = os.path.join(export_dir, "manifest.json")
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(manifest.get("entries", []), jf, indent=2)
            except Exception as e:
                errors.append({"json_error": str(e)})
        # --- Manifest update ---
        export_entry = {
            "database_file": db_path,
            "csv_file": csv_path,
            "json_file": json_path,
            "export_timestamp": "2025-06-01T00:50:00Z",
            "lineage": ["tuple_assembler", "separation", "normalization", "clap", "diarization", "transcription", "soundbite", "llm_task", "remixing", "show_output", "export_database"]
        }
        manifest_utils.update_manifest(manifest, export_entry)
        if errors:
            manifest["export_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"export_info": export_entry, "manifest": manifest}

    def ui(self):
        # Optional: custom UI for export settings, query interface
        pass 