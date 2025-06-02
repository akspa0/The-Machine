from .utils import privacy
from .utils import manifest as manifest_utils
from collections import defaultdict

class TupleAssemblerNode:
    """
    Tuple Assembler Node
    Groups ingested files into tuples (call pairs, out files) based on tuple_index and timestamp. Ensures correct mapping, updates manifest, and maintains privacy.
    """
    @classmethod
    def input_types(cls):
        return {"ingested": "list[dict]", "manifest": "dict"}

    @classmethod
    def output_types(cls):
        return {"tuples": "list[dict]", "manifest": "dict"}

    def process(self, ingested, manifest):
        from tqdm import tqdm
        tuples = []
        errors = []
        # Group by tuple_index
        tuple_map = defaultdict(list)
        for entry in ingested:
            idx = entry.get("tuple_index")
            if idx is not None:
                tuple_map[idx].append(entry)
            else:
                errors.append({"error": f"Missing tuple_index in entry: {entry.get('output_name', '<unknown>')}"})
        # Assemble tuples
        for idx, files in tqdm(sorted(tuple_map.items()), desc="Assembling tuples"):
            tuple_dict = {
                "tuple_index": idx,
                "files": [],
                "left": None,
                "right": None,
                "out": None,
                "timestamp": None,
                "lineage": [],
            }
            for entry in files:
                ftype = entry.get("type")
                if ftype in ("left", "right", "out"):
                    tuple_dict[ftype] = entry["output_path"]
                tuple_dict["files"].append(entry["output_path"])
                tuple_dict["lineage"].append(entry["output_name"])
                if not tuple_dict["timestamp"]:
                    tuple_dict["timestamp"] = entry.get("timestamp")
            tuples.append(tuple_dict)
        # Update manifest
        manifest_utils.update_manifest(manifest, tuples, key="tuples")
        if errors:
            manifest["tuple_assembler_errors"] = errors
        manifest_utils.validate_manifest(manifest)
        return {"tuples": tuples, "manifest": manifest}

    def ui(self):
        """
        ComfyUI node UI definition for tuple assembler.
        - ingested: hidden (auto-passed)
        - manifest: hidden (auto-passed)
        """
        return {
            "ingested": {"type": "hidden"},
            "manifest": {"type": "hidden"}
        } 