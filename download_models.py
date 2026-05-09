import os
import zipfile
from pathlib import Path

import requests

models = {
    "resnet18-5c106cde.pth": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "79000_iter.pth": "https://github.com/sensity-ai/dot/releases/download/v1.0.0/79000_iter.pth",
    "simswap_512.onnx": "https://github.com/sensity-ai/dot/releases/download/v1.0.0/simswap_512.onnx",
    "512.zip": "https://github.com/neuralchen/SimSwap/releases/download/512_beta/512.zip",
}

# The sensity-ai release links can be finicky with requests, 
# so we'll try the HuggingFace mirrors which are more reliable for direct downloads.
hf_models = {
    "79000_iter.pth": "https://huggingface.co/ezioruan/SimSwap/resolve/main/parsing_model/79000_iter.pth",
    "simswap_512.onnx": "https://huggingface.co/ezioruan/SimSwap/resolve/main/simswap_512.onnx",
}

MODEL_DIR = Path("saved_models/simswap")
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
ONNX_MODEL_DIR = Path("saved_models/onnx")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def download(name, url):
    path = MODEL_DIR / name
    # Remove corrupt files first
    if path.exists() and path.stat().st_size < 1000:
        path.unlink()
        
    if path.exists():
        print(f"✅ {name} already exists.")
        return True

    print(f"Downloading {name}...")
    try:
        response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if response.status_code == 200:
            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Successfully downloaded {name} ({path.stat().st_size} bytes)")
            return True
        else:
            print(f"❌ Failed {name} (Status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error {name}: {e}")
        return False


def download_to(path: Path, url: str, min_bytes: int = 1000) -> bool:
    if path.exists() and path.stat().st_size >= min_bytes:
        print(f"✅ {path} already exists.")
        return True
    if path.exists():
        path.unlink()

    print(f"Downloading {path.name}...")
    try:
        with requests.get(
            url,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=60,
        ) as response:
            if response.status_code != 200:
                print(f"❌ Failed {path.name} (Status {response.status_code})")
                return False
            with path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        if path.stat().st_size < min_bytes:
            print(f"❌ Downloaded file is too small: {path} ({path.stat().st_size} bytes)")
            path.unlink(missing_ok=True)
            return False
        print(f"✅ Successfully downloaded {path} ({path.stat().st_size} bytes)")
        return True
    except Exception as e:
        print(f"❌ Error {path.name}: {e}")
        return False


def ensure_simswap_512_checkpoint():
    checkpoint = CHECKPOINT_DIR / "512" / "550000_net_G.pth"
    if checkpoint.exists() and checkpoint.stat().st_size > 100_000_000:
        print(f"✅ SimSwap 512 checkpoint already exists: {checkpoint}")
        return True

    if not download("512.zip", models["512.zip"]):
        return False

    zip_path = MODEL_DIR / "512.zip"
    print("Extracting SimSwap 512 checkpoint...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(CHECKPOINT_DIR)

    if checkpoint.exists() and checkpoint.stat().st_size > 100_000_000:
        print(f"✅ SimSwap 512 checkpoint ready: {checkpoint}")
        return True

    print(f"❌ SimSwap 512 checkpoint not found after extraction: {checkpoint}")
    return False

# Download ResNet from PyTorch
download("resnet18-5c106cde.pth", models["resnet18-5c106cde.pth"])

# Download others from HuggingFace mirror
for name, url in hf_models.items():
    download(name, url)

ensure_simswap_512_checkpoint()

download_to(
    ONNX_MODEL_DIR / "inswapper_128_fp16.onnx",
    "https://huggingface.co/fofr/comfyui/resolve/main/insightface/inswapper_128_fp16.onnx",
    min_bytes=250_000_000,
)
