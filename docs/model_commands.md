# Model Commands Reference

Complete command reference for all available face swap models in dot.

## Quick Reference

| Model | Type | Install |
|-------|------|---------|
| `simswap` | Traditional | Built-in |
| `fomm` | Traditional | Built-in |
| `faceswap_cv2` | Traditional | Built-in |
| `reactor` | Diffusion | `pip install "dot[reactor]"` |
| `ipadapter_face` | Diffusion | `pip install "dot[ipadapter]"` |
| `instantid` | Diffusion | `pip install "dot[instantid]"` |

---

## Traditional Models

### SimSwap

High-quality face swap using SimSwap architecture.

**Camera (live feed):**
```bash
python -m dot --swap_type simswap \
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/simswap.yaml \
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu \
    --crop_size 512
```

**With GPEN face enhancement:**
```bash
python -m dot --swap_type simswap \
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/fomm.yaml \
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/faceswap_cv2.yaml \
    --target 0 \
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

### ReActor

Fast face swap with optional face restoration. Best balance of speed and quality.

**Install:**
```bash
pip install "dot[reactor]"
```

**Camera (live feed):**
```bash
python -m dot --swap_type reactor \
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/reactor.yaml \
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu \
    --use_face_restore \
    --face_restore_model codeformer
```

**With GFPGAN face restoration:**
```bash
python -m dot --swap_type reactor \
    --target 0 \
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

**Install:**
```bash
pip install "dot[ipadapter]"
```

**Camera (live feed):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/ipadapter_face.yaml \
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu \
    --num_inference_steps 20 \
    --guidance_scale 7.5 \
    --ip_adapter_scale 0.6
```

**With SDXL (higher quality):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 0 \
    --source "./data" \
    --use_gpu \
    --use_sdxl
```

**With LCM scheduler (faster):**
```bash
python -m dot --swap_type ipadapter_face \
    --target 0 \
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

**Install:**
```bash
pip install "dot[instantid]"
```

**Camera (live feed):**
```bash
python -m dot --swap_type instantid \
    --target 0 \
    --source "./data" \
    --use_gpu
```

**With config file:**
```bash
python -m dot -c ./configs/instantid.yaml \
    --target 0 \
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
    --target 0 \
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
    --target 0 \
    --source "./data" \
    --use_gpu \
    --use_lcm \
    --num_inference_steps 4 \
    --guidance_scale 1.5
```

**Using LCM config:**
```bash
python -m dot -c ./configs/instantid_lcm.yaml \
    --target 0 \
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
| `--target` | Camera ID (int) or path to video/image | Required |
| `--source` | Path to source image(s) | Required |
| `--use_gpu` | Enable GPU acceleration | false |
| `--use_video` | Video swap mode | false |
| `--use_image` | Image swap mode | false |
| `--save_folder` | Output directory | None |
| `--show_fps` | Display FPS in camera mode | false |
| `--limit` | Limit number of frames to process | None |
| `-c, --config` | Path to YAML config file | None |
| `--crop_size` | Image crop size | Model-dependent |
| `--gpen_type` | GPEN enhancement: `gpen_256` or `gpen_512` | None |
| `--gpen_path` | Path to GPEN models | saved_models/gpen |

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
python -m dot -c ./configs/simswap.yaml --target 0 --source "./data" --use_gpu
```
