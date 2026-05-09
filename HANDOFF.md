# DOT — Project Handoff

**Date:** 2026-05-09
**Project:** DOT live face swap for macOS M2 8GB
**Scope:** Local face swap from source image/video plus webcam/liveness input, with optional virtual camera output, browser control surface, and an optional GFPGAN restoration stage.  Shipping focus: a stable local app for macOS M-series.

---

## Executive Summary

DOT is a hardware-tuned, local live face swap application built for Apple Silicon M2 with 8GB RAM. It supports three backends:

- `simswap`: default production path using GAN-based SimSwap with quality presets
- `onnx`: ONNX-based inswapper path for faster model inference when ONNX runtime is available
- `reactor`: lightweight motion-field warping and face tracking for preview/debug use

A minimal golden path exists: use `simswap` + `natural` preset + webcam driver. Optional quality features such as GFPGAN restoration and temporal smoothing are available behind flags.

---

## Current Status

### Working

- `python live.py --backend simswap --preset natural --source data/source_face.jpg --driver-camera 1`
- `run.sh` wrapper for local launch
- `python web.py` local browser control panel at `http://127.0.0.1:7860`
- `health_check.py` validates runtime assets, configs, and FaceMesh availability
- `setup_check.py` validates conda env, Python, key dependencies, and model files
- `onnxruntime` added to `requirements-apple-m2.txt`
- Shared MediaPipe face detector consolidated via `src/dot/commons/shared_detector.py`
- Optional restore stage implemented in `src/dot/restore.py`
- Temporal smoothing implemented in `src/dot/temporal.py`

### Optional / feature-ready but not packaged

- GFPGAN restoration is optional and disabled by default. It is wired into the ONNX swap loop and can be enabled with `--restore gfpgan`.
- GFPGAN weights are not included in the repo; the expected path is `saved_models/gfpgan/GFPGANv1.4.pth`.
- The restore stage will auto-disable if GFPGAN is not installed or if the weights are missing.

### Remaining issues / current blockers

- Health check currently flags `MediaPipe FaceMesh` failure unless the installed MediaPipe build exposes `mediapipe.solutions.face_mesh`.
- `pyvirtualcam` is not part of the default dependency list; virtual camera output remains a manual optional install.
- `saved_models/gfpgan/` is absent from repo, so restoration cannot be enabled without additional downloads.
- `CodeFormer` support is intentionally not wired; only `gfpgan` is available.

---

## Key Files and Architecture

### Entry points

- `live.py`: CLI entry point, backend selection, driver input, frame loop, output routing
- `run.sh`: shell wrapper to activate the conda env and start `live.py`
- `web.py`: local browser control interface for starting/stopping runs and uploading sources
- `health_check.py`: runtime validation for app dependencies and assets
- `setup_check.py`: conda env and dependency sanity checker

### Backend and pipeline

- `src/dot/conditioning.py`: source image/video normalization, face crop preparation, source scoring
- `src/dot/reactor_backend.py`: reactor warping backend
- `src/dot/temporal.py`: EMA smoothing for affine matrices and landmarks
- `src/dot/restore.py`: optional GFPGAN restoration stage, frame skipping, error auto-disable
- `src/dot/commons/shared_detector.py`: shared MediaPipe detector factory for static/live modes
- `src/dot/simswap/`: SimSwap-specific model loader and pipeline

### Important SimSwap pieces

- `src/dot/simswap/fs_model.py`: SimSwap model loading wrapper
- `src/dot/simswap/fs_networks.py`: generator architecture
- `src/dot/simswap/option.py`: config-driven SimSwap pipeline and MediaPipe detector wiring
- `src/dot/simswap/util/reverse2original.py`: face crop reverse transform and blending

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sensity-ai/dot.git
   cd dot
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f envs/environment-apple-m2.yaml
   conda activate dot
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements-apple-m2.txt
   ```

4. Validate setup:
   ```bash
   python setup_check.py
   python health_check.py
   ```

5. Download models if necessary:
   ```bash
   python download_models.py
   ```

---

## Runtime Usage

### Recommended golden path

```bash
python live.py --backend simswap --preset natural --source data/source_face.jpg --driver-camera 1
```

### ONNX path

```bash
python live.py --backend onnx --preset natural --source data/source_face.jpg --driver-camera 1
```

Requires:

- `saved_models/onnx/inswapper_128_fp16.onnx`
- `saved_models/simswap/arcface_model/arcface_checkpoint.tar`
- `onnxruntime` installed
- `onnx` installed for source embedding extraction

### Restoration options

```bash
python live.py --backend onnx --preset natural --source data/source_face.jpg --driver-camera 1 --restore gfpgan --restore-strength 0.4 --restore-every 3
```

Flags:

- `--restore`: `none` or `gfpgan` (default: `none`)
- `--restore-strength`: float `0.0-1.0` (default: `0.4`)
- `--restore-every`: integer frame interval (default: `3`)

### Temporal smoothing

```bash
python live.py --temporal --temporal-alpha 0.35
```

### Output modes

- `--output window` (default)
- `--output virtualcam`
- `--output both`
- `--output none`

### Web UI

```bash
python web.py
```

Open:

```text
http://127.0.0.1:7860
```

The browser UI is a local control surface only; it starts and stops the local process and does not expose the app as a cloud service.

---

## Health and Diagnostics

### setup_check.py

`python setup_check.py` validates:

- `conda` env activation for `dot`
- Python 3.10+
- `torch`, `torchvision`, `opencv-python`, `mediapipe`, `numpy`, `onnxruntime`
- presence of critical model files

### health_check.py

`python health_check.py` validates:

- macOS environment
- Python version
- PyTorch MPS support
- OpenCV availability
- `mediapipe.solutions.face_mesh`
- required config files and sample assets
- expected SimSwap config values

Current note: health_check will fail if the installed MediaPipe package does not expose `mediapipe.solutions.face_mesh`.

---

## Known Issues and Risks

- `MediaPipe FaceMesh` may fail with some pip builds. The health check specifically verifies `mediapipe.solutions.face_mesh`.
- `pyvirtualcam` is still optional. Virtual camera output on macOS requires a supported virtual camera provider such as OBS virtual camera.
- GFPGAN weights are not in the repo. To enable restoration, place `GFPGANv1.4.pth` in `saved_models/gfpgan/`.
- The code path for `CodeFormer` is not implemented; only `gfpgan` is available.
- `onnxruntime` and `onnx` are required for the ONNX backend. If missing, `live.py` will raise a clear runtime error.
- `run.sh` and manual command usage are the intended launch paths; `web.py` is a convenience wrapper.

---

## Current Implementation Notes

### Shared MediaPipe detector

`src/dot/commons/shared_detector.py` centralizes static vs live FaceMesh instantiation, which avoids duplicate detector creation across backends.

### Optional GFPGAN restoration

`src/dot/restore.py` implements the optional face restoration stage. It is wired into the ONNX backend loop in `OnnxBackend.process()` immediately after the face crop is synthesized and before blending back into the full frame.

Restoration is safe by default:

- `mode=none` disables it
- missing GFPGAN import disables it with an explanatory message
- missing weights disable it and fall back to no restoration
- repeated runtime exceptions are counted and restoration auto-disables after 3 failures

### Temporal smoothing

`src/dot/temporal.py` provides EMA smoothing for both landmark-aware face matrices and frame-to-frame transformations. It is disabled unless `--temporal` is enabled.

### ONNX backend

`src/dot/live.py` contains the `OnnxBackend` class. It:

- loads `inswapper_128_fp16.onnx`
- loads arcface weights for source identity extraction
- constructs `onnxruntime` inference session
- uses shared MediaPipe detector for face crops
- optionally restores the swapped crop with GFPGAN
- performs quality adjustments and soft paste blending

---

## Recommended Next Actions

1. Document the optional GFPGAN model download flow and include it in the packaging docs.
2. Add `gfpgan` to an optional requirements file or install guide, not the base `requirements-apple-m2.txt`.
3. Harden MediaPipe installation guidance: specify the working pip package or wheel for `face_mesh`.
4. Add a simple `saved_models/gfpgan/` downloader or model acquisition helper.
5. Verify the actual `web.py` port and add a note about not exposing it beyond localhost.

---

## Packaging and v1 Delivery Notes

For a runnable v1 ZIP distribution, the recommended approach is:

- Package the app code and `setup_check.py`/`health_check.py`
- Exclude large model files from the ZIP
- Require first-run model download via `download_models.py` or a remote VPS endpoint
- Use `python live.py --backend simswap --preset natural --source data/source_face.jpg --driver-camera 1` as the documented quickstart

This keeps the ZIP size manageable and avoids shipping multi-hundred-megabyte model files directly.

---

## Contact

For follow-up or maintenance, use the project repository issue tracker or reach out to the team managing DOT.

---

*This handoff has been updated to reflect the current code and build state.*
