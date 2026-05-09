#!/usr/bin/env python3
"""
Setup validation script for the DOT project.
Checks conda environment, dependencies, and model availability.
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def check_conda_env():
    """Check if we're in the 'dot' conda environment."""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            check=True
        )
        if "dot" in result.stdout and "*" in result.stdout:
            print("✓ Conda environment 'dot' is active")
            return True
        else:
            print("✗ Conda environment 'dot' is not active")
            return False
    except Exception as e:
        print(f"✗ Could not check conda environment: {e}")
        return False

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 10):
        print(f"✓ Python {version.major}.{version.minor} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} is too old (need 3.10+)")
        return False

def check_dependencies():
    """Check key dependencies."""
    deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("onnxruntime", "onnxruntime"),
    ]
    failed = []
    for import_name, display_name in deps:
        try:
            __import__(import_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is missing")
            failed.append(display_name)
    return len(failed) == 0

def check_models():
    """Check model files."""
    models = [
        ROOT / "saved_models" / "onnx" / "inswapper_128_fp16.onnx",
        ROOT / "saved_models" / "simswap" / "checkpoints" / "people" / "latest_net_G.pth",
        ROOT / "saved_models" / "simswap" / "arcface_model" / "archive" / "version",
    ]
    all_present = True
    for model in models:
        if model.exists():
            print(f"✓ {model.relative_to(ROOT)} exists")
        else:
            print(f"✗ {model.relative_to(ROOT)} is missing")
            all_present = False
    return all_present

def main():
    print("DOT Setup Check")
    print("=" * 20)

    checks = [
        check_conda_env,
        check_python_version,
        check_dependencies,
        check_models,
    ]

    passed = 0
    for check in checks:
        if check():
            passed += 1
        print()

    print(f"Passed {passed}/{len(checks)} checks")
    if passed == len(checks):
        print("✓ Setup looks good!")
        return 0
    else:
        print("✗ Setup issues detected. Please fix before running.")
        return 1

if __name__ == "__main__":
    sys.exit(main())