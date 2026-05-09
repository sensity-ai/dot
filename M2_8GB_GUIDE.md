# M2 8GB Guide

Use the `dot` conda environment and keep other apps closed for the 512px presets.

```bash
./run.sh --source data/source_face.webm --camera 1 --preset natural
```

Presets:

| Preset | Notes |
| --- | --- |
| `fast` | Lowest cost SimSwap path |
| `balanced` | 512px SimSwap without the natural render pass |
| `natural` | Default 512px mask/color/detail alpha blend |
| `natural-max` | Slower Poisson blend experiment |

Outputs:

| Output | Notes |
| --- | --- |
| `window` | Default OpenCV window |
| `virtualcam` | Requires `pyvirtualcam` and a virtual camera provider |
| `both` | Sends the same swapped frame to both outputs |
