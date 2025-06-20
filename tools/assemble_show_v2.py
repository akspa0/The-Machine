#!/usr/bin/env python3
"""Deprecated shim – use extensions.show_builder instead.

This file remains for backward-compatibility; it simply forwards all CLI
arguments to ``extensions.show_builder.cli``.
"""

# Add project root to sys.path to allow 'extensions' package import when
# this script is executed directly via ``python tools/assemble_show_v2.py``.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extensions.show_builder import cli

if __name__ == "__main__":
    cli() 