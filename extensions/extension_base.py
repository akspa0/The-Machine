import os
import json
from pathlib import Path
from typing import Dict, Type, Optional, List
# If LLM chunking/tokenization is needed, import from llm_utils
# from llm_utils import split_into_chunks_advanced

# ---------------------------------------------------------------------------
#  Global Extension Registry
# ---------------------------------------------------------------------------

EXTENSION_REGISTRY: Dict[str, Type["ExtensionBase"]] = {}


def _register_extension(cls: Type["ExtensionBase"]):
    """Internal helper to register an ExtensionBase subclass.

    Registration key is `cls.name` if defined, otherwise the class name.
    Duplicate keys raise ValueError so we catch accidental collisions early.
    """
    key = getattr(cls, "name", None) or cls.__name__
    if key in EXTENSION_REGISTRY:
        raise ValueError(f"Duplicate extension registered under key '{key}'.")
    EXTENSION_REGISTRY[key] = cls


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class ExtensionBase:
    """
    Base class for all extension scripts. Provides standardized access to finalized outputs and manifest,
    and enforces privacy/logging rules.
    """

    # Optional class-level metadata --------------------------------------------------
    # These can be overridden by subclasses to advertise what they do.
    name: Optional[str] = None          # Unique identifier; defaults to class.__name__
    stage: str = "unspecified"         # Pipeline stage this extension targets (e.g., 'ingestion', 'finalisation')
    description: str = ""              # One-liner description used by task manager / help UIs

    # Automatic registry hook -------------------------------------------------------

    def __init_subclass__(cls, **kwargs):  # type: ignore[override]
        super().__init_subclass__(**kwargs)
        if cls is ExtensionBase:
            return  # Don't register the abstract base itself
        _register_extension(cls)

    def __init__(self, output_root):
        self.output_root = Path(output_root)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        manifest_path = self.output_root / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def log(self, message):
        # Only log anonymized, PII-free information
        print(f"[EXTENSION LOG] {message}")

    def run(self):
        raise NotImplementedError("Extension must implement the run() method.")

# Example usage:
# class MyExtension(ExtensionBase):
#     def run(self):
#         # Your logic here
#         self.log("Running my extension!")

# ---------------------------------------------------------------------------
#  Registry helpers for orchestrator / task manager
# ---------------------------------------------------------------------------


def list_extensions(stage: Optional[str] = None) -> Dict[str, Type[ExtensionBase]]:
    """Return mapping of registered extension key -> class.

    If *stage* is provided, filter to extensions whose `stage` attribute matches.
    """
    if stage is None:
        return dict(EXTENSION_REGISTRY)
    return {k: v for k, v in EXTENSION_REGISTRY.items() if getattr(v, "stage", None) == stage}


def run_extensions_for_stage(output_root: str | Path, stage: str, **kwargs):
    """Instantiate and run all extensions registered for *stage*.

    Any extra **kwargs are forwarded to the extension constructor.
    """
    for key, ext_cls in list_extensions(stage).items():
        inst = ext_cls(output_root, **kwargs)  # type: ignore[arg-type]
        inst.log(f"Running extension '{key}' (stage={stage})")
        inst.run() 