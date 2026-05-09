#!/usr/bin/env python3
"""
Download models from VPS using API key.
Usage: python vps_downloader.py --key=YOUR_API_KEY [--output=saved_models/]
"""

import argparse
import sys
import os
import hashlib
from pathlib import Path
import requests

VPS_BASE = os.environ.get("DOT_VPS_URL", "http://your-vps-ip-here")

def parse_args():
    p = argparse.ArgumentParser(description="Download DOT models from VPS")
    p.add_argument("--key", required=True, help="Your API key")
    p.add_argument("--output", default="saved_models/", help="Output directory")
    p.add_argument("--base-url", default=VPS_BASE, help="VPS base URL")
    return p.parse_args()

def download_model(base_url, key, filename, output_dir):
    """Download a single model file with API key authentication."""
    url = f"{base_url}/download/{filename}"
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {filename}...")

    headers = {"X-API-Key": key}
    r = requests.get(url, headers=headers, stream=True)
    if r.status_code != 200:
        print(f"  ✗ Failed: {r.status_code} {r.reason}")
        return False

    total_size = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"  {pct:5.1f}%", end="\r")
    print(f"  ✓ Saved to {output_path}")
    return True

def verify_checksum(path, expected=None):
    """Verify SHA256 checksum of downloaded file."""
    if not path.exists():
        return False
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if expected:
        return actual == expected
    print(f"  SHA256: {actual}")
    return True

def main():
    args = parse_args()
    base = args.base_url.rstrip("/")
    key = args.key
    out = args.output

    print(f"DOT VPS Downloader")
    print(f"  VPS: {base}")
    print(f"  Output: {out}\n")

    # List of models to download
    models = [
        "onnx/inswapper_128_fp16.onnx",
        "simswap/checkpoints/512/550000_net_G.pth",
        "simswap/parsing_model/checkpoint/79999_iter.pth",
        "simswap/arcface_model/arcface_checkpoint.tar",
    ]

    success = 0
    for model in models:
        if download_model(base, key, model, out):
            verify_checksum(Path(out) / model)
            success += 1

    print(f"\nDone: {success}/{len(models)} models downloaded.")
    return 0 if success == len(models) else 1

if __name__ == "__main__":
    raise SystemExit(main())
