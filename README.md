# DOT Live Face Swap

Minimal local workflow:

```text
source image/video + physical webcam id -> swapped frames -> window or optional virtual camera
```

## Run

```bash
./run.sh --source data/source_face.webm --camera 1 --preset natural
```

Default output is the OpenCV window named `DOT - Live Deepfake`.

Useful options:

```bash
./run.sh --help
./run.sh --source data/source_face.jpg --camera 1 --preset fast
./run.sh --source data/source_face.webm --camera 1 --preset natural --output both
./run.sh --source data/source_face.webm --camera 1 --backend onnx
```

`--backend simswap` is the installed default. `--backend onnx` is reserved for `saved_models/onnx/inswapper_128_fp16.onnx` and fails clearly until that model path is wired.

`--output virtualcam` and `--output both` require `pyvirtualcam` plus a supported virtual camera provider. On macOS 13+, pyvirtualcam expects OBS 30+ virtual camera support to be initialized once.

## Local Web Control

```bash
conda run -n dot python web.py
```

Open:

```text
http://127.0.0.1:7860
```

The page starts and stops the same local runner. It is a control surface, not a remote GPU service.

## Check

```bash
bash -n run.sh
conda run -n dot python -m compileall -q live.py health_check.py download_models.py src/dot
conda run -n dot python -m pytest -q
conda run -n dot python health_check.py
./run.sh --help
```

## Kept Runtime Assets

```text
data/source_face.webm
data/source_face.jpg
saved_models/simswap/
```
