
# Diffusion Models

Dot supports several diffusion-based face swap models that provide higher quality results than traditional methods. These models require additional dependencies to be installed.

## Available Diffusion Models

| Swap Type | Description | Install Command | Live Streaming |
|-----------|-------------|-----------------|----------------|
| `reactor` | GAN-based face swap with face restoration | `pip install "dot[reactor]"` | **Recommended** |
| `ipadapter_face` | Diffusion-based face transfer using IP-Adapter | `pip install "dot[ipadapter]"` | Slow |
| `instantid` | Identity-preserving generation with ControlNet | `pip install "dot[instantid]"` | Slow |

> **For live streaming, use ReActor.** IP-Adapter and InstantID run full diffusion inference per frame which is too slow for real-time use.

## ReActor

ReActor is a fast face swap model with optional face restoration.

### Basic Usage

```bash
python -m dot --swap_type reactor --target 1 --source "./data" --use_gpu
```

### With Face Restoration

```bash
python -m dot --swap_type reactor --target 1 --source "./data" --use_gpu \
    --use_face_restore --face_restore_model codeformer
```

### Options

- `--use_face_restore` - Enable face restoration (default: true)
- `--face_restore_model` - Face restore model: `codeformer` or `gfpgan` (default: codeformer)

## IP-Adapter Face

IP-Adapter Face uses diffusion models to transfer facial identity while preserving pose and expression.

### Basic Usage

```bash
python -m dot --swap_type ipadapter_face --target 1 --source "./data" --use_gpu
```

### With Custom Parameters

```bash
python -m dot --swap_type ipadapter_face --target 1 --source "./data" --use_gpu \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --ip_adapter_scale 0.6
```

### With SDXL Model

```bash
python -m dot --swap_type ipadapter_face --target 1 --source "./data" --use_gpu \
    --use_sdxl
```

### Options

- `--num_inference_steps` - Number of diffusion steps (default: 20)
- `--guidance_scale` - Classifier-free guidance scale (default: 7.5)
- `--ip_adapter_scale` - IP-Adapter scale, controls identity strength (default: 0.6)
- `--use_sdxl` - Use SDXL model for higher quality
- `--use_lcm` - Use LCM scheduler for faster inference

## InstantID

InstantID uses ControlNet for identity-preserving face generation with strong identity preservation.

### Basic Usage

```bash
python -m dot --swap_type instantid --target 1 --source "./data" --use_gpu
```

### With Custom Parameters

```bash
python -m dot --swap_type instantid --target 1 --source "./data" --use_gpu \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --controlnet_scale 0.8 \
    --ip_adapter_scale 0.8
```

### With LCM Scheduler (Faster)

```bash
python -m dot --swap_type instantid --target 1 --source "./data" --use_gpu \
    --use_lcm --num_inference_steps 4
```

### Options

- `--num_inference_steps` - Number of diffusion steps (default: 20, use 4-8 with LCM)
- `--guidance_scale` - Classifier-free guidance scale (default: 7.5)
- `--controlnet_scale` - ControlNet conditioning scale (default: 0.8)
- `--ip_adapter_scale` - IP-Adapter scale (default: 0.8)
- `--use_lcm` - Use LCM scheduler for faster inference

## Common Diffusion Options

These options apply to all diffusion-based models:

| Option | Description | Default |
|--------|-------------|---------|
| `--num_inference_steps` | Number of diffusion denoising steps | 20 |
| `--guidance_scale` | Classifier-free guidance scale | 7.5 |
| `--use_lcm` | Use LCM scheduler for faster inference | false |

## Using with Video and Image Pipelines

Diffusion models work with all dot pipelines:

### Camera Feed

```bash
python -m dot --swap_type instantid --target 0 --source "./data" --use_gpu
```

### Video Swap

```bash
python -m dot --swap_type instantid --target "./video.mp4" --source "./data" \
    --use_gpu --use_video --save_folder ./output
```

### Image Swap

```bash
python -m dot --swap_type instantid --target "./images/" --source "./data" \
    --use_gpu --use_image --save_folder ./output
```

## Hardware Requirements

Diffusion models are computationally intensive and benefit significantly from GPU acceleration:

- **GPU recommended**: CUDA-compatible GPU with at least 8GB VRAM
- **MPS support**: Apple Silicon Macs are supported via MPS backend
- **CPU**: Possible but significantly slower
