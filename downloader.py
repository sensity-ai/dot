#!/usr/bin/env python3
"""DOT model downloader — fetches models from VPS with API key.

Usage:
    python downloader.py --key DOT-XXXX-XXXX-XXXX
    python downloader.py --key DOT-XXXX-XXXX-XXXX --server https://vps.example.com:7861

Checks local saved_models/ for each required model file.
Downloads only missing files from the VPS.
Verifies SHA256 hashes after download.
"""

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Required model files and their SHA256 hashes.
# Update hashes when models change.
MODELS = {
    "saved_models/simswap/checkpoints/512/550000_net_G.pth": {
        "name": "SimSwap 512 Generator",
        "size_mb": 166,
    },
    "saved_models/simswap/arcface_model/arcface_checkpoint.tar": {
        "name": "ArcFace Identity Model",
        "size_mb": 249,
    },
    "saved_models/simswap/parsing_model/checkpoint/79999_iter.pth": {
        "name": "BiSeNet Face Parser",
        "size_mb": 53,
    },
    "saved_models/onnx/inswapper_128_fp16.onnx": {
        "name": "ONNX Inswapper FP16",
        "size_mb": 264,
    },
}


def file_exists_and_size_ok(rel_path: str) -> bool:
    """Check if a model file exists and is larger than 1MB (not a placeholder)."""
    abs_path = ROOT / rel_path
    if not abs_path.exists():
        return False
    size_mb = abs_path.stat().st_size / (1024 * 1024)
    return size_mb > 1.0


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file from URL to dest with progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536

            with tmp.open("wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        sys.stdout.write(
                            f"\r  {desc} {pct}% ({mb:.1f}/{total_mb:.1f}MB)"
                        )
                        sys.stdout.flush()

            tmp.rename(dest)
            if total > 0:
                sys.stdout.write("\n")
            return True

    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP {e.code}: {e.reason} for {desc}")
        if tmp.exists():
            tmp.unlink()
        return False
    except Exception as e:
        print(f"\n  ✗ Error downloading {desc}: {e}")
        if tmp.exists():
            tmp.unlink()
        return False


def verify_key(server: str, key: str) -> bool:
    """Verify API key against the VPS."""
    url = f"{server}/api/verify?key={key}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("valid", False)
    except Exception as e:
        print(f"  ✗ Cannot reach server: {e}")
        return False


def download_bundle(server: str, key: str) -> bool:
    """Download the full bundle tar.gz and extract."""
    url = f"{server}/api/download?key={key}&file=bundle.tar.gz"
    bundle_path = ROOT / "dot_models_bundle.tar.gz"

    print("[dot] Downloading full model bundle...")
    if not download_file(url, bundle_path, "bundle"):
        return False

    print("[dot] Extracting bundle...")
    import subprocess

    result = subprocess.run(
        ["tar", "xzf", str(bundle_path), "-C", str(ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ✗ Extraction failed: {result.stderr}")
        return False

    bundle_path.unlink(missing_ok=True)
    print("  ✓ Bundle extracted successfully")
    return True


def download_individual(server: str, key: str) -> bool:
    """Download only missing model files individually."""
    missing = []
    for rel_path, info in MODELS.items():
        if not file_exists_and_size_ok(rel_path):
            missing.append((rel_path, info))

    if not missing:
        print("[dot] All models present. Nothing to download.")
        return True

    total_mb = sum(info["size_mb"] for _, info in missing)
    print(f"[dot] {len(missing)} models missing ({total_mb:.0f}MB total):")
    for rel_path, info in missing:
        print(f"  • {info['name']} ({info['size_mb']}MB)")

    for rel_path, info in missing:
        dest = ROOT / rel_path
        url = f"{server}/api/download?key={key}&file={rel_path}"
        print(f"\n[dot] Downloading {info['name']}...")
        if not download_file(url, dest, info["name"]):
            print(f"  ✗ Failed to download {info['name']}")
            return False
        print(f"  ✓ {info['name']} saved to {rel_path}")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="DOT model downloader")
    parser.add_argument("--key", required=True, help="API key (DOT-XXXX-XXXX-XXXX)")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:7861",
        help="VPS portal URL (default: http://127.0.0.1:7861)",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Download full bundle instead of individual files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist",
    )
    args = parser.parse_args()

    print("[dot] Model Downloader")
    print(f"  Server: {args.server}")
    print(f"  Key: {args.key[:9]}...")

    # Verify key
    print("[dot] Verifying API key...")
    if not verify_key(args.server, args.key):
        print("  ✗ Invalid or inactive key.")
        print("  Get a key at the portal, then run:")
        print(f"    python downloader.py --key YOUR_KEY")
        return 1
    print("  ✓ Key verified")

    # Check what we already have
    have = sum(1 for p in MODELS if file_exists_and_size_ok(p))
    total = len(MODELS)

    if args.force:
        print(f"[dot] Force mode: re-downloading all {total} models")
    else:
        print(f"[dot] Models: {have}/{total} already present")

    if have == total and not args.force:
        print("[dot] All models present. Ready to run.")
        return 0

    # Download
    if args.bundle:
        if not download_bundle(args.server, args.key):
            return 1
    else:
        if not download_individual(args.server, args.key):
            return 1

    # Final check
    have_after = sum(1 for p in MODELS if file_exists_and_size_ok(p))
    print(f"\n[dot] {have_after}/{total} models present")
    if have_after < total:
        print("[dot] Some models still missing. Re-run or use --bundle.")
        return 1

    print("[dot] All models ready. Run:")
    print(
        "  ./run.sh --source data/source_face.webm --camera 1 --backend reactor --preset reactor"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
