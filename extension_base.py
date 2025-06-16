"""Shim module so that `import extension_base` resolves from project root.

This simply re-exports the symbols from `extensions.extension_base`, keeping the
single authoritative implementation there while supporting legacy imports.
"""

from importlib import import_module as _import_module

# Import the actual implementation module
_ext_mod = _import_module("extensions.extension_base")

# Re-export everything so `from extension_base import ExtensionBase` works
globals().update(_ext_mod.__dict__) 