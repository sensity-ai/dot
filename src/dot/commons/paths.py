#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
import sys
from pathlib import Path


def get_app_base_path() -> Path:
    """
    Root path for configs, data, and saved_models when running as a frozen app
    (PyInstaller/cx_Freeze) or from source.

    - Frozen (PyInstaller): returns the bundle resource directory
      (sys._MEIPASS on Windows/Linux, Contents/Resources on macOS .app).
    - Development: returns current working directory so relative paths still work.
    """
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            # macOS .app: executable is in Contents/MacOS, data in Contents/Resources
            exe = Path(sys.executable).resolve()
            # .../dot.app/Contents/MacOS/dot -> .../dot.app/Contents/Resources
            resources = exe.parents[0].parent / "Resources"
            if resources.is_dir():
                return resources
        return Path(getattr(sys, "_MEIPASS", os.getcwd()))
    return Path(os.getcwd())


def resolve_app_path(relative_path: str) -> str:
    """
    If relative_path is relative, resolve it against the app base path (for frozen builds).
    Otherwise return it unchanged.
    """
    if not relative_path or os.path.isabs(relative_path):
        return relative_path
    base = get_app_base_path()
    resolved = base / relative_path
    return str(resolved)
