# Model Commands Reference

Complete command reference for all available face swap models in dot.

## Quick Reference

| Model | Type | Install | Best For |
|-------|------|---------|----------|
| `simswap` | Traditional | Built-in | General use |
| `fomm` | Traditional | Built-in | Expression transfer |
| `faceswap_cv2` | Traditional | Built-in | Lightweight/fast |
| `reactor` | GAN-based | `pip install "dot[reactor]"` | **Live streaming (recommended)** |
| `ipadapter_face` | Diffusion | `pip install "dot[ipadapter]"` | Image generation |
| `instantid` | Diffusion | `pip install "dot[instantid]"` | High-quality image generation |

## Recommended for Live Streaming

**ReActor** is the recommended model for live camera streaming. It provides the best balance of quality and speed:

```bash
python -m dot --swap_type reactor --target 1 --source "./data/your_image.jpg" --use_gpu --use_face_restore --face_restore_model codeformer
```

- Uses GAN-based face swapping (not full diffusion per frame)
- CodeFormer face restoration improves output quality
- Reasonable download size (~500MB vs 5-12GB for other diffusion models)

---

## Traditional Models

### SimSwap

High-quality face swap using SimSwap architecture.

**Camera (live feed):**
```bash
python -m dot --swap_type simswap \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/simswap.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type simswap \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type simswap \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**High quality (512px):**
```bash
python -m dot -c ./configs/simswaphq.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --crop_size 512
```

**With GPEN face enhancement:**
```bash
python -m dot --swap_type simswap \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --gpen_type gpen_256 \
    --gpen_path saved_models/gpen
```

**Options:**
- `--crop_size` - Crop size: 224 (default) or 512 for HQ
- `--gpen_type` - Face enhancement: `gpen_256` or `gpen_512`
- `--gpen_path` - Path to GPEN models
- `--parsing_model_path` - Path to parsing model
- `--arcface_model_path` - Path to ArcFace model
- `--checkpoints_dir` - Path to checkpoints

---

### FOMM (First Order Motion Model)

Motion-based face reenactment that transfers expressions.

**Camera (live feed):**
```bash
python -m dot --swap_type fomm \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/fomm.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type fomm \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type fomm \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**Options:**
- `--model_path` - Path to FOMM checkpoint (vox-adv-cpk.pth.tar)
- `--crop_size` - Crop size (default: 256)

**Keyboard controls (camera mode):**
- `w/s` - Adjust frame proportion
- `h/k` - Adjust horizontal offset
- `j/u` - Adjust vertical offset
- `x` - Calibrate/reset
- `r` - Toggle preview flip
- `t` - Toggle output flip
- `f` - Toggle keyframe finding
- `o` - Toggle landmarks display

---

### FaceSwap CV2

Lightweight OpenCV-based face swap using dlib landmarks.

**Camera (live feed):**
```bash
python -m dot --swap_type faceswap_cv2 \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/faceswap_cv2.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type faceswap_cv2 \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type faceswap_cv2 \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**Options:**
- `--model_path` - Path to dlib 68-point landmark predictor
- `--crop_size` - Crop size (default: 224)

---

## Diffusion Models

### ReActor (Recommended for Live Streaming)

Fast GAN-based face swap with optional face restoration. Best balance of speed and quality for real-time use.

**Install:**
```bash
pip install "dot[reactor]"
```

**Camera (live feed) - Recommended:**
```bash
python -m dot --swap_type reactor \
    --target 1 \
    --source "./data/your_image.jpg" \
    --use_gpu \
    --use_face_restore \
    --face_restore_model codeformer
```

**Camera (basic):**
```bash
python -m dot --swap_type reactor \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/reactor.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type reactor \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type reactor \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**With CodeFormer face restoration:**
```bash
python -m dot --swap_type reactor \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --use_face_restore \
    --face_restore_model codeformer
```

**With GFPGAN face restoration:**
```bash
python -m dot --swap_type reactor \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --use_face_restore \
    --face_restore_model gfpgan
```

**Options:**
- `--use_face_restore` - Enable face restoration (default: true)
- `--face_restore_model` - Model: `codeformer` or `gfpgan`
- `--crop_size` - Crop size (default: 512)

---

### IP-Adapter Face

Diffusion-based face transfer using IP-Adapter. Generates new images with source identity.

> **Note:** This model runs full diffusion inference per frame. It's slow for live streaming - use ReActor instead for real-time use.

**Install:**
```bash
pip install "dot[ipadapter]"
```

**Camera (live feed):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/ipadapter_face.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type ipadapter_face \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type ipadapter_face \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**With custom parameters:**
```bash
python -m dot --swap_type ipadapter_face \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --ip_adapter_scale 0.6
```

**With SDXL (higher quality):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --use_sdxl
```

**With LCM scheduler (faster):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --use_lcm \
    --num_inference_steps 4
```

**Options:**
- `--num_inference_steps` - Diffusion steps (default: 20, use 4-8 with LCM)
- `--guidance_scale` - CFG scale (default: 7.5)
- `--ip_adapter_scale` - Identity strength 0.0-1.0 (default: 0.6)
- `--use_sdxl` - Use SDXL model for higher quality
- `--use_lcm` - Use LCM scheduler for faster inference
- `--crop_size` - Crop size (default: 512)

---

### InstantID

Identity-preserving generation with ControlNet. Best identity preservation.

> **Note:** This model downloads ~12GB of models and runs full diffusion inference per frame. It's slow for live streaming - use ReActor instead for real-time use.

**Install:**
```bash
pip install "dot[instantid]"
```

**Camera (live feed):**
```bash
python -m dot --swap_type instantid \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/instantid.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Video swap:**
```bash
python -m dot --swap_type instantid \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output
```

**Image swap:**
```bash
python -m dot --swap_type instantid \
    --target "./images/" \
    --source "./data" \
    --use_gpu \
    --use_image \
    --save_folder ./output
```

**With custom parameters:**
```bash
python -m dot --swap_type instantid \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --controlnet_scale 0.8 \
    --ip_adapter_scale 0.8
```

**With LCM scheduler (faster):**
```bash
python -m dot --swap_type instantid \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --use_lcm \
    --num_inference_steps 4 \
    --guidance_scale 1.5
```

**Using LCM config:**
```bash
python -m dot -c ./configs/instantid_lcm.yaml \
    --target 1 \
    --source "./data" \
    --use_gpu
```

**Options:**
- `--num_inference_steps` - Diffusion steps (default: 20, use 4-8 with LCM)
- `--guidance_scale` - CFG scale (default: 7.5, use 1.5 with LCM)
- `--controlnet_scale` - ControlNet strength (default: 0.8)
- `--ip_adapter_scale` - IP-Adapter strength (default: 0.8)
- `--use_lcm` - Use LCM scheduler for faster inference
- `--crop_size` - Crop size (default: 512)

---

## Common Options

These options work with all models:

| Option | Description | Default |
|--------|-------------|---------|
| `--list-cameras` | List available camera devices (index and resolution) then exit | — |
| `--target` | Camera ID (int) or path to video/image | Required |
| `--source` | Path to source image(s) | Required |
| `--use_gpu` | Enable GPU acceleration | false |
| `--use_video` | Video swap mode | false |
| `--use_image` | Image swap mode | false |
| `--save_folder` | Output directory | None |
| `--show_fps` | Display FPS in camera mode | false |
| `--virtual_cam` | Send output to a virtual camera (FaceTime/Zoom). Requires pyvirtualcam; on macOS install OBS. To see output in OBS: add Source → Window Capture → select the **cam** window (OBS cannot use the virtual camera as an input) | false |
| `--virtual_cam_fps` | Virtual camera FPS when `--virtual_cam` is used | 20 |
| `--limit` | Limit number of frames to process | None |
| `-c, --config` | Path to YAML config file | None |
| `--crop_size` | Image crop size | Model-dependent |
| `--gpen_type` | GPEN enhancement: `gpen_256` or `gpen_512` | None |
| `--gpen_path` | Path to GPEN models | saved_models/gpen |
| `--filter` | Camera/film filter (see below) | None |

---

## Camera/Film Filters

Apply realistic camera degradation effects to make deepfakes look more natural. Filters simulate various camera qualities and add natural imperfections.

### Available Filters

| Filter | Description | Effects Applied |
|--------|-------------|-----------------|
| `webcam_720p` | Typical webcam quality | Slight blur, warm color cast, reduced dynamic range, light grain |
| `webcam_480p` | Low-quality webcam | Downscale, blur, warm color, reduced range, grain, compression |
| `film_grain` | Film camera look | Heavy grain, cool color cast, vignette |
| `security_cam` | CCTV/security camera | Downscale, low contrast, grain, heavy compression |
| `phone_front` | Front-facing phone camera | Light blur, warm tones, subtle grain |

### Usage Examples

**With any model:**
```bash
# SimSwap with webcam filter
python -m dot --swap_type simswap \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --filter webcam_720p

# ReActor with security cam filter
python -m dot --swap_type reactor \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --filter security_cam

# FOMM with film grain
python -m dot --swap_type fomm \
    --target 1 \
    --source "./data" \
    --use_gpu \
    --filter film_grain
```

**Video/image swap with filter:**
```bash
python -m dot --swap_type reactor \
    --target "./video.mp4" \
    --source "./data" \
    --use_gpu \
    --use_video \
    --save_folder ./output \
    --filter webcam_480p
```

### Keyboard Controls (Camera Mode)

Switch filters on-the-fly during live camera mode:

| Key | Action |
|-----|--------|
| `f` | Cycle to next filter |
| `d` | Cycle to previous filter |
| `g` | Disable filter (no filter applied) |

The current filter name is printed to the terminal when you switch.

---

## Config Files

All models have pre-configured YAML files in `configs/`:

```
configs/
├── simswap.yaml          # SimSwap standard
├── simswaphq.yaml        # SimSwap high quality (512px)
├── fomm.yaml             # First Order Motion Model
├── faceswap_cv2.yaml     # FaceSwap CV2
├── reactor.yaml          # ReActor
├── ipadapter_face.yaml   # IP-Adapter Face
├── instantid.yaml        # InstantID standard
└── instantid_lcm.yaml    # InstantID with LCM (fast)
```

Use with `-c` flag:
```bash
python -m dot -c ./configs/simswap.yaml --target 1 --source "./data" --use_gpu
```

---

## Keyboard Controls (Camera Mode)

All keyboard controls available during live camera mode:

### General Controls

| Key | Action |
|-----|--------|
| `c` | Next camera (cycle through camera indices 0–9) |
| `C` (Shift+c) | Previous camera |
| `Up Arrow` | Previous source image |
| `Down Arrow` | Next source image |
| `1-9` | Jump to source image 1-9 |
| `y` | Show original camera feed (passthrough) |
| `q` | Quit |
| `i` | Toggle FPS display |

### Filter Controls

| Key | Action |
|-----|--------|
| `f` | Next filter |
| `d` | Previous filter |
| `g` | Disable filter |

### Enhancement Controls

| Key | Action |
|-----|--------|
| `t` | Toggle temporal smoothing (reduces flickering) |
| `r` | Toggle trails (psychedelic ghosting effect) |
| `b` | Toggle bilateral smoothing (beauty mode - edge-preserving blur) |
| `s` | Toggle sharpening (enhance details) |
| `u` | Toggle 2x upscale (higher resolution display) |
| `o` | Toggle OSD (on-screen display with current settings) |

### FOMM-Specific Controls

| Key | Action |
|-----|--------|
| `w/s` | Adjust frame proportion |
| `h/k` | Adjust horizontal offset |
| `j/u` | Adjust vertical offset |
| `x` | Calibrate/reset |
| `r` | Toggle preview flip |
| `t` | Toggle output flip |
| `o` | Toggle landmarks display |
