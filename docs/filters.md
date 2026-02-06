# Camera/Film Filters

Apply realistic camera degradation effects to make deepfake outputs look more natural by simulating various camera qualities and adding natural imperfections.

## Quick Start

```bash
# Apply a filter via CLI
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter kodak_portra

# Switch filters during live camera mode
# Press 'f' for next filter, 'd' for previous, 'g' to disable
```

## Available Filters (74 total)

### Webcam (3)
| Filter | Description |
|--------|-------------|
| `webcam_720p` | Typical webcam quality |
| `webcam_480p` | Low-quality webcam with compression |
| `webcam_old` | Old/cheap webcam look |

### Phone Cameras (3)
| Filter | Description |
|--------|-------------|
| `phone_front` | Front-facing selfie camera |
| `phone_back` | Rear phone camera |
| `phone_night` | Phone camera in low light |

### Security/Surveillance (6)
| Filter | Description |
|--------|-------------|
| `security_cam` | Standard CCTV |
| `security_cam_bw` | Black & white security with timestamp |
| `security_cam_night` | Night mode security camera |
| `doorbell_cam` | Ring-style doorbell camera |
| `dashcam` | Dashboard camera with timestamp |
| `bodycam` | Police bodycam style |

### Film/Analog (6)
| Filter | Description |
|--------|-------------|
| `film_grain` | Classic film grain look |
| `film_8mm` | 8mm home movie style |
| `film_vhs` | VHS tape look |
| `film_vhs_worn` | Heavily degraded VHS |
| `film_polaroid` | Polaroid instant camera |
| `damaged_film` | Scratched/dusty film |

### Film Stocks (8)
| Filter | Description |
|--------|-------------|
| `kodak_portra` | Kodak Portra - warm skin tones, low contrast |
| `kodak_gold` | Kodak Gold - warm, saturated |
| `kodak_ultramax` | Kodak Ultramax - vibrant colors |
| `fuji_superia` | Fuji Superia - cool greens |
| `fuji_c200` | Fuji C200 - natural colors |
| `cinestill_800t` | CineStill 800T - tungsten, halation |
| `ilford_hp5` | Ilford HP5 - B&W, high contrast |
| `kodak_trix` | Kodak Tri-X - B&W, classic grain |

### Analog Style (4)
| Filter | Description |
|--------|-------------|
| `analog_warm` | Warm analog aesthetic |
| `analog_cool` | Cool analog aesthetic |
| `analog_faded` | Faded/lifted blacks |
| `analog_vibrant` | Punchy analog colors |

### CRT/Monitor (2)
| Filter | Description |
|--------|-------------|
| `crt_monitor` | CRT computer monitor |
| `crt_tv` | Old CRT television |

### Video Call/Streaming (4)
| Filter | Description |
|--------|-------------|
| `zoom_call` | Typical Zoom call quality |
| `zoom_call_bad` | Poor connection Zoom call |
| `stream_720p` | 720p streaming quality |
| `stream_480p` | 480p streaming quality |

### Night Vision/Thermal (3)
| Filter | Description |
|--------|-------------|
| `night_vision` | Green night vision |
| `night_vision_ir` | Infrared night vision |
| `thermal` | Thermal camera look |

### Instagram/Social (5)
| Filter | Description |
|--------|-------------|
| `instagram_vintage` | Vintage Instagram filter |
| `instagram_fade` | Faded Instagram look |
| `instagram_cool` | Cool-toned Instagram |
| `instagram_warm` | Warm-toned Instagram |
| `instagram_compressed` | Instagram compression |

### Light Leaks/Damage (3)
| Filter | Description |
|--------|-------------|
| `light_leak_warm` | Warm light leak on right |
| `light_leak_cool` | Cool light leak on left |
| `found_footage` | Found footage / damaged tape |

### Camera Brands (5)
| Filter | Description |
|--------|-------------|
| `dslr_canon` | Canon DSLR color science |
| `dslr_nikon` | Nikon DSLR color science |
| `dslr_sony` | Sony DSLR color science |
| `gopro` | GoPro action camera |
| `drone_footage` | Aerial drone footage |

### Depth of Field (2)
| Filter | Description |
|--------|-------------|
| `shallow_dof` | Shallow depth of field |
| `portrait_bokeh` | Portrait mode bokeh |

### Social Media Compression (4)
| Filter | Description |
|--------|-------------|
| `twitter_compressed` | Twitter image compression |
| `tiktok_compressed` | TikTok video compression |
| `whatsapp_compressed` | WhatsApp heavy compression |
| `instagram_compressed` | Instagram compression |

### Professional/Broadcast (4)
| Filter | Description |
|--------|-------------|
| `broadcast_tv` | Broadcast TV quality |
| `news_footage` | News camera footage |
| `old_footage` | Aged/old footage |
| `corrupted` | Corrupted/glitched video |

### Psychedelic/Trippy (13) - Animated with Trails!
| Filter | Description |
|--------|-------------|
| `acid_trip` | Full psychedelic effect with trails, hue shift, waves |
| `acid_mild` | Subtle psychedelic with light trails |
| `acid_intense` | Intense psychedelic with heavy trails |
| `rainbow_wave` | Rainbow hue cycling with wave distortion + trails |
| `color_storm` | Swirling color overlay with trails |
| `kaleidoscope` | Kaleidoscope mirror effect (no trails) |
| `rgb_glitch` | Animated RGB channel splitting with trails |
| `pixel_acid` | Pulsating pixelation with hue shift + trails |
| `dream_sequence` | Dreamy waves with soft blur + trails |
| `neon_glow` | Neon colors with glow effect + trails |
| `warp_speed` | Intense wave distortion with RGB split + trails |
| `trails_only` | Just the trailing/ghosting effect |
| `heavy_trails` | Strong trails with hue shift |

## Keyboard Controls (Live Camera Mode)

### Avatar/Source Controls
| Key | Action |
|-----|--------|
| `Up Arrow` | Previous avatar |
| `Down Arrow` | Next avatar |
| `1-9` | Jump to avatar 1-9 |

### Filter Controls
| Key | Action |
|-----|--------|
| `f` | Cycle to next filter |
| `d` | Cycle to previous filter |
| `g` | Disable filter |

### Enhancement Controls
| Key | Action |
|-----|--------|
| `t` | Toggle temporal smoothing |
| `r` | Toggle trails (psychedelic ghosting effect) |
| `b` | Toggle bilateral smoothing (beauty mode) |
| `s` | Toggle sharpening |
| `u` | Toggle 2x upscale |
| `o` | Toggle OSD (on-screen display) |

### Other Controls
| Key | Action |
|-----|--------|
| `y` | Show original camera feed |
| `i` | Toggle FPS display |
| `q` | Quit |

**Important:** Click on the camera window first for keyboard controls to work.

## Usage Examples

### Film Stocks
```bash
# Kodak Portra - great for portraits
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter kodak_portra

# CineStill 800T - cinematic night look with halation
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter cinestill_800t

# Black & white Tri-X
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter kodak_trix
```

### Surveillance/Security
```bash
# CCTV with timestamp
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter security_cam_bw

# Dashcam
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter dashcam
```

### Degraded/Damaged
```bash
# Found footage horror style
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter found_footage

# VHS tape
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter film_vhs_worn
```

### Psychedelic/Trippy (Animated)
```bash
# Full acid trip effect - hue shifting, wave distortion, color cycling
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter acid_trip

# Milder version
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter acid_mild

# Kaleidoscope mirror effect
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter kaleidoscope

# Rainbow wave with hue cycling
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter rainbow_wave

# Dreamy sequence with soft waves
python -m dot --swap_type reactor --target 0 --source "./data" --use_gpu --filter dream_sequence
```

### Video Processing
```bash
python -m dot --swap_type reactor \
    --target "./input.mp4" \
    --source "./data/source.jpg" \
    --use_gpu \
    --use_video \
    --save_folder ./output \
    --filter kodak_portra
```

## Individual Effects (API)

```python
from dot.filters import (
    # Basic effects
    add_grain, add_blur, add_compression_artifacts,
    adjust_color_cast, reduce_dynamic_range, add_vignette,
    downscale_upscale, adjust_brightness, adjust_contrast,
    adjust_saturation,

    # Analog/Film effects
    add_scanlines, add_chromatic_aberration, add_noise_bands,
    add_interlacing, add_light_leak, add_color_fade,
    add_halation, add_dust_scratches, add_film_border,

    # Camera effects
    add_barrel_distortion, add_lens_blur, add_color_tint,
    apply_lut_shift, add_timestamp, convert_grayscale,

    # Apply presets
    apply_filter, AVAILABLE_FILTERS,
)

import cv2

image = cv2.imread("input.jpg")

# Individual effects
image = add_grain(image, intensity=0.3)
image = add_halation(image, intensity=0.25)
image = add_light_leak(image, intensity=0.3, position="right")

# Or apply a preset
image = apply_filter(image, "cinestill_800t")

cv2.imwrite("output.jpg", image)
```

## Effect Reference

| Effect | Parameters | Description |
|--------|------------|-------------|
| `add_grain` | intensity (0-1) | Film grain noise |
| `add_blur` | kernel_size (odd int) | Gaussian blur |
| `add_compression_artifacts` | quality (0-100) | JPEG artifacts |
| `adjust_color_cast` | temperature (-100 to +100) | Warm/cool shift |
| `reduce_dynamic_range` | factor (0-1) | Flatten contrast |
| `add_vignette` | intensity (0-1) | Darken corners |
| `downscale_upscale` | scale_factor (0-1) | Simulate low res |
| `add_scanlines` | intensity, spacing | CRT scanlines |
| `add_chromatic_aberration` | shift (pixels) | RGB channel shift |
| `add_color_tint` | color (BGR), intensity | Color overlay |
| `adjust_brightness` | factor | Brightness multiplier |
| `adjust_contrast` | factor | Contrast multiplier |
| `adjust_saturation` | factor | Saturation multiplier |
| `add_noise_bands` | intensity, band_height | VHS noise bands |
| `add_interlacing` | - | Interlace artifact |
| `add_timestamp` | text (optional) | Timestamp overlay |
| `convert_grayscale` | - | Black & white |
| `add_barrel_distortion` | strength | Fisheye effect |
| `add_light_leak` | intensity, position | Film light leak |
| `add_color_fade` | intensity | Lifted blacks, desaturated |
| `add_halation` | intensity | Glow around highlights |
| `add_dust_scratches` | dust_count, scratch_count | Film damage |
| `add_film_border` | border_size, sprocket | Film frame border |
| `apply_lut_shift` | shadows/mids/highlights RGB | Color grading |
| `add_lens_blur` | blur_amount, focus_point | Depth of field blur |
