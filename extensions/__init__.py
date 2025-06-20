# Auto-import extension modules so they register themselves
from importlib import import_module as _import_module

# Explicit list for now – add new modules here
for _mod in [
    "extensions.show_assembler_extension",
]:
    try:
        _import_module(_mod)
    except Exception as _e:
        # Soft fail – log but continue; orchestrator will warn later
        print(f"[EXTENSIONS] Failed to import {_mod}: {_e}") 