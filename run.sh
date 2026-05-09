#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
ENV_NAME="${DOT_ENV:-dot}"
DOT_SERVER="${DOT_SERVER:-http://127.0.0.1:7861}"
DOT_KEY="${DOT_KEY:-}"
DOT_MODE="${DOT_MODE:-live}"

cd "$ROOT_DIR"

if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

# Required model files for reactor backend
REQUIRED_MODELS=(
  "saved_models/simswap/checkpoints/512/550000_net_G.pth"
  "saved_models/simswap/arcface_model/arcface_checkpoint.tar"
  "saved_models/simswap/parsing_model/checkpoint/79999_iter.pth"
)

check_models() {
  local missing=0
  for model in "${REQUIRED_MODELS[@]}"; do
    if [[ ! -f "$model" ]] || [[ $(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null) -lt 1048576 ]]; then
      missing=$((missing + 1))
    fi
  done
  return $missing
}

if ! check_models; then
  echo "[dot] Some required models are missing."
  if [[ -n "$DOT_KEY" ]]; then
    echo "[dot] Downloading models with key ${DOT_KEY:0:9}..."
    python downloader.py --key "$DOT_KEY" --server "$DOT_SERVER" || {
      echo "[dot] Download failed. Get a key at the portal and run:"
      echo "  python downloader.py --key YOUR_KEY"
      exit 1
    }
  else
    echo "[dot] Set DOT_KEY to download automatically, or run manually:"
    echo "  python downloader.py --key YOUR_KEY"
    echo ""
    echo "[dot] Continuing with available models..."
  fi
fi

if [[ "$DOT_MODE" == "liveness-video" ]]; then
  DOT_SOURCE="${DOT_SOURCE:-data/source_face.jpg}"
  DOT_DRIVER_VIDEO="${DOT_DRIVER_VIDEO:-data/source_face.webm}"
  if [[ -z "${DOT_BACKEND:-}" ]]; then
    if [[ -f "saved_models/onnx/inswapper_128_fp16.onnx" ]]; then
      DOT_BACKEND="onnx"
    else
      DOT_BACKEND="simswap"
    fi
  fi
  DOT_PRESET="${DOT_PRESET:-natural}"
  DOT_OUTPUT="${DOT_OUTPUT:-window}"

  args=(
    --backend "$DOT_BACKEND"
    --preset "$DOT_PRESET"
    --source "$DOT_SOURCE"
    --driver-video "$DOT_DRIVER_VIDEO"
    --output "$DOT_OUTPUT"
  )
  if [[ -n "${DOT_RECORD_OUTPUT:-}" ]]; then
    args+=(--record-output "$DOT_RECORD_OUTPUT")
  fi
  if [[ "${DOT_LOOP_DRIVER:-}" == "1" || "${DOT_LOOP_DRIVER:-}" == "true" ]]; then
    args+=(--loop-driver)
  fi
  exec python live.py "${args[@]}" "$@"
fi

exec python live.py "$@"
