#!/usr/bin/env python3
"""Validate the minimal DOT live setup."""

from pathlib import Path
import platform
import sys

import cv2
import mediapipe
import psutil
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def check(name: str, ok: bool, hint: str = "") -> bool:
    mark = "OK" if ok else "FAIL"
    print(f"[{mark}] {name}")
    if not ok and hint:
        print(f"       {hint}")
    return ok


def main() -> int:
    ok = True
    ok &= check("macOS", platform.system() == "Darwin", "This setup is tuned for macOS M-series.")
    ok &= check("Python 3.10+", sys.version_info >= (3, 10), "Use the conda env named dot.")
    ok &= check("PyTorch MPS", torch.backends.mps.is_available(), "Install PyTorch with Apple Silicon support.")
    ok &= check("OpenCV", cv2.__version__ is not None)
    ok &= check(
        "MediaPipe FaceMesh",
        hasattr(mediapipe, "solutions") and hasattr(mediapipe.solutions, "face_mesh"),
        "Install a MediaPipe build that includes mediapipe.solutions.face_mesh.",
    )
    ok &= check("live.py", (ROOT / "live.py").exists())
    ok &= check("run.sh", (ROOT / "run.sh").exists())

    required = [
        ROOT / "configs" / "simswap.yaml",
        ROOT / "configs" / "m2_8gb_balanced.yaml",
        ROOT / "configs" / "m2_8gb_natural.yaml",
        ROOT / "configs" / "m2_8gb_natural_max.yaml",
        ROOT / "saved_models" / "simswap" / "checkpoints" / "512" / "550000_net_G.pth",
        ROOT / "saved_models" / "simswap" / "parsing_model" / "checkpoint" / "79999_iter.pth",
        ROOT / "saved_models" / "simswap" / "arcface_model" / "arcface_checkpoint.tar",
        ROOT / "data" / "source_face.webm",
        ROOT / "data" / "source_face.jpg",
    ]
    for path in required:
        ok &= check(str(path.relative_to(ROOT)), path.exists(), "Run download_models.py or restore the sample.")

    for preset in ("simswap.yaml", "m2_8gb_balanced.yaml", "m2_8gb_natural.yaml", "m2_8gb_natural_max.yaml"):
        with (ROOT / "configs" / preset).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        ok &= check(f"{preset} is SimSwap", config.get("swap_type", "simswap") == "simswap")

    available = psutil.virtual_memory().available / (1024**3)
    print(f"[INFO] Available memory: {available:.1f}GB")
    if available < 4:
        print("[WARN] Close other apps before running the 512px preset.")

    if ok:
        print("\nReady: ./run.sh --source data/source_face.webm --camera 1 --preset natural")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
