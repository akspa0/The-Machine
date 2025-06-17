from pathlib import Path
import importlib
import pkgutil


def run_all(run_folder: Path):
    """Discover and execute all extensions that expose a run(run_folder) function.

    Extensions are regular Python modules in the ``extensions`` package.  If a
    module defines a top-level ``run`` callable accepting a ``Path`` argument,
    it will be invoked.  Any exceptions are caught and logged so that a single
    faulty extension cannot break the finalization stage.
    """
    try:
        # Import the parent package first so __path__ is initialised.
        import extensions  # noqa: WPS433 – dynamic import required
    except Exception as exc:  # pragma: no cover – fatal startup error
        print(f"[EXT][ERROR] Cannot import extensions package: {exc}")
        return

    for mod_info in pkgutil.iter_modules(extensions.__path__):
        name = mod_info.name
        # Skip private helpers (underscore-prefixed)
        if name.startswith('_'):
            continue
        try:
            module = importlib.import_module(f"extensions.{name}")
        except Exception as exc:
            print(f"[EXT][ERROR] Failed to import extension '{name}': {exc}")
            continue

        run_callable = getattr(module, 'run', None)
        if callable(run_callable):
            print(f"[EXT] ▶ Running extension: {name}")
            try:
                run_callable(run_folder)
                print(f"[EXT] ✔ Completed: {name}")
            except Exception as exc:  # pragma: no cover – extension failure is isolated
                print(f"[EXT][ERROR] Extension '{name}' raised an exception: {exc}")
        else:
            # Not an error – module simply doesn't implement the hook.
            print(f"[EXT] ⏩ Skipping '{name}' (no run() hook)") 