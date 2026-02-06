#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

Filter presets - combinations of effects that simulate various camera types.
"""

# Filter presets - each is a list of (effect_name, params) tuples
FILTER_PRESETS = {
    # === Webcam Filters ===
    "webcam_720p": [
        ("blur", {"kernel_size": 3}),
        ("color_cast", {"temperature": 15}),
        ("dynamic_range", {"factor": 0.85}),
        ("grain", {"intensity": 0.1}),
    ],
    "webcam_480p": [
        ("downscale", {"scale_factor": 0.65}),
        ("blur", {"kernel_size": 5}),
        ("color_cast", {"temperature": 10}),
        ("dynamic_range", {"factor": 0.75}),
        ("grain", {"intensity": 0.2}),
        ("compression", {"quality": 60}),
    ],
    "webcam_old": [
        ("downscale", {"scale_factor": 0.5}),
        ("blur", {"kernel_size": 7}),
        ("color_cast", {"temperature": 20}),
        ("dynamic_range", {"factor": 0.65}),
        ("grain", {"intensity": 0.3}),
        ("compression", {"quality": 45}),
        ("saturation", {"factor": 0.8}),
    ],

    # === Phone Cameras ===
    "phone_front": [
        ("blur", {"kernel_size": 3}),
        ("color_cast", {"temperature": 8}),
        ("dynamic_range", {"factor": 0.9}),
        ("grain", {"intensity": 0.05}),
    ],
    "phone_back": [
        ("grain", {"intensity": 0.03}),
        ("saturation", {"factor": 1.1}),
        ("contrast", {"factor": 1.05}),
    ],
    "phone_night": [
        ("grain", {"intensity": 0.35}),
        ("blur", {"kernel_size": 3}),
        ("brightness", {"factor": 1.2}),
        ("dynamic_range", {"factor": 0.7}),
        ("color_cast", {"temperature": -10}),
    ],

    # === Security/Surveillance ===
    "security_cam": [
        ("downscale", {"scale_factor": 0.5}),
        ("dynamic_range", {"factor": 0.6}),
        ("grain", {"intensity": 0.25}),
        ("compression", {"quality": 50}),
    ],
    "security_cam_bw": [
        ("grayscale", {}),
        ("downscale", {"scale_factor": 0.5}),
        ("dynamic_range", {"factor": 0.55}),
        ("grain", {"intensity": 0.3}),
        ("compression", {"quality": 45}),
        ("timestamp", {}),
    ],
    "security_cam_night": [
        ("grayscale", {}),
        ("brightness", {"factor": 1.3}),
        ("grain", {"intensity": 0.4}),
        ("dynamic_range", {"factor": 0.5}),
        ("compression", {"quality": 40}),
    ],
    "doorbell_cam": [
        ("barrel_distortion", {"strength": 0.2}),
        ("downscale", {"scale_factor": 0.6}),
        ("dynamic_range", {"factor": 0.7}),
        ("grain", {"intensity": 0.15}),
        ("compression", {"quality": 55}),
    ],
    "dashcam": [
        ("barrel_distortion", {"strength": 0.15}),
        ("dynamic_range", {"factor": 0.75}),
        ("grain", {"intensity": 0.1}),
        ("compression", {"quality": 65}),
        ("timestamp", {}),
    ],
    "bodycam": [
        ("blur", {"kernel_size": 3}),
        ("dynamic_range", {"factor": 0.7}),
        ("grain", {"intensity": 0.2}),
        ("compression", {"quality": 55}),
        ("barrel_distortion", {"strength": 0.1}),
    ],

    # === Film/Analog ===
    "film_grain": [
        ("grain", {"intensity": 0.4}),
        ("color_cast", {"temperature": -5}),
        ("vignette", {"intensity": 0.25}),
    ],
    "film_8mm": [
        ("grain", {"intensity": 0.5}),
        ("vignette", {"intensity": 0.4}),
        ("color_cast", {"temperature": 25}),
        ("saturation", {"factor": 0.85}),
        ("dynamic_range", {"factor": 0.75}),
        ("blur", {"kernel_size": 3}),
    ],
    "film_vhs": [
        ("chromatic_aberration", {"shift": 2}),
        ("scanlines", {"intensity": 0.15, "spacing": 3}),
        ("noise_bands", {"intensity": 0.15, "band_height": 15}),
        ("blur", {"kernel_size": 3}),
        ("saturation", {"factor": 0.9}),
        ("color_cast", {"temperature": 10}),
    ],
    "film_vhs_worn": [
        ("chromatic_aberration", {"shift": 4}),
        ("scanlines", {"intensity": 0.25, "spacing": 2}),
        ("noise_bands", {"intensity": 0.25, "band_height": 25}),
        ("interlacing", {}),
        ("blur", {"kernel_size": 5}),
        ("saturation", {"factor": 0.75}),
        ("grain", {"intensity": 0.2}),
    ],
    "film_polaroid": [
        ("vignette", {"intensity": 0.35}),
        ("color_cast", {"temperature": 15}),
        ("saturation", {"factor": 0.85}),
        ("contrast", {"factor": 0.9}),
        ("grain", {"intensity": 0.1}),
    ],

    # === CRT/Monitor ===
    "crt_monitor": [
        ("scanlines", {"intensity": 0.2, "spacing": 2}),
        ("chromatic_aberration", {"shift": 1}),
        ("blur", {"kernel_size": 3}),
        ("vignette", {"intensity": 0.2}),
    ],
    "crt_tv": [
        ("scanlines", {"intensity": 0.25, "spacing": 3}),
        ("chromatic_aberration", {"shift": 2}),
        ("blur", {"kernel_size": 3}),
        ("saturation", {"factor": 1.1}),
        ("vignette", {"intensity": 0.3}),
        ("barrel_distortion", {"strength": 0.1}),
    ],

    # === Video Call/Streaming ===
    "zoom_call": [
        ("compression", {"quality": 50}),
        ("blur", {"kernel_size": 3}),
        ("dynamic_range", {"factor": 0.85}),
        ("grain", {"intensity": 0.05}),
    ],
    "zoom_call_bad": [
        ("downscale", {"scale_factor": 0.5}),
        ("compression", {"quality": 30}),
        ("blur", {"kernel_size": 5}),
        ("dynamic_range", {"factor": 0.7}),
        ("grain", {"intensity": 0.15}),
    ],
    "stream_720p": [
        ("compression", {"quality": 60}),
        ("grain", {"intensity": 0.05}),
        ("dynamic_range", {"factor": 0.9}),
    ],
    "stream_480p": [
        ("downscale", {"scale_factor": 0.7}),
        ("compression", {"quality": 45}),
        ("blur", {"kernel_size": 3}),
        ("grain", {"intensity": 0.1}),
    ],

    # === Night Vision ===
    "night_vision": [
        ("color_tint", {"color": (0, 255, 0), "intensity": 0.6}),
        ("grain", {"intensity": 0.35}),
        ("brightness", {"factor": 1.3}),
        ("dynamic_range", {"factor": 0.6}),
    ],
    "night_vision_ir": [
        ("grayscale", {}),
        ("color_tint", {"color": (200, 200, 255), "intensity": 0.2}),
        ("grain", {"intensity": 0.3}),
        ("brightness", {"factor": 1.4}),
        ("dynamic_range", {"factor": 0.55}),
    ],

    # === Thermal/Special ===
    "thermal": [
        ("grayscale", {}),
        ("color_tint", {"color": (0, 100, 255), "intensity": 0.4}),
        ("blur", {"kernel_size": 5}),
        ("dynamic_range", {"factor": 0.5}),
    ],

    # === Artistic/Instagram ===
    "instagram_vintage": [
        ("color_cast", {"temperature": 20}),
        ("saturation", {"factor": 0.8}),
        ("contrast", {"factor": 0.9}),
        ("vignette", {"intensity": 0.3}),
        ("grain", {"intensity": 0.1}),
    ],
    "instagram_fade": [
        ("dynamic_range", {"factor": 0.7}),
        ("saturation", {"factor": 0.7}),
        ("brightness", {"factor": 1.1}),
        ("color_cast", {"temperature": 5}),
    ],
    "instagram_cool": [
        ("color_cast", {"temperature": -20}),
        ("saturation", {"factor": 1.1}),
        ("contrast", {"factor": 1.1}),
        ("vignette", {"intensity": 0.15}),
    ],
    "instagram_warm": [
        ("color_cast", {"temperature": 30}),
        ("saturation", {"factor": 1.05}),
        ("contrast", {"factor": 1.05}),
        ("vignette", {"intensity": 0.15}),
    ],

    # === Degraded/Damaged ===
    "old_footage": [
        ("grain", {"intensity": 0.45}),
        ("vignette", {"intensity": 0.4}),
        ("saturation", {"factor": 0.6}),
        ("dynamic_range", {"factor": 0.65}),
        ("blur", {"kernel_size": 3}),
        ("color_cast", {"temperature": 20}),
    ],
    "corrupted": [
        ("compression", {"quality": 20}),
        ("chromatic_aberration", {"shift": 5}),
        ("noise_bands", {"intensity": 0.3, "band_height": 30}),
        ("grain", {"intensity": 0.3}),
    ],

    # === Professional ===
    "broadcast_tv": [
        ("saturation", {"factor": 1.1}),
        ("contrast", {"factor": 1.05}),
        ("dynamic_range", {"factor": 0.95}),
        ("grain", {"intensity": 0.02}),
    ],
    "news_footage": [
        ("compression", {"quality": 70}),
        ("dynamic_range", {"factor": 0.9}),
        ("grain", {"intensity": 0.05}),
        ("saturation", {"factor": 0.95}),
    ],

    # === Film Stocks (inspired by SD community) ===
    "kodak_portra": [
        ("color_cast", {"temperature": 12}),
        ("saturation", {"factor": 0.9}),
        ("contrast", {"factor": 0.95}),
        ("grain", {"intensity": 0.15}),
        ("color_fade", {"intensity": 0.15}),
        ("lut_shift", {"shadows_rgb": (5, 10, 15), "midtones_rgb": (0, 0, 0), "highlights_rgb": (-5, 0, 5)}),
    ],
    "kodak_gold": [
        ("color_cast", {"temperature": 25}),
        ("saturation", {"factor": 1.1}),
        ("contrast", {"factor": 1.05}),
        ("grain", {"intensity": 0.2}),
        ("vignette", {"intensity": 0.15}),
    ],
    "kodak_ultramax": [
        ("saturation", {"factor": 1.15}),
        ("contrast", {"factor": 1.1}),
        ("color_cast", {"temperature": 15}),
        ("grain", {"intensity": 0.18}),
    ],
    "fuji_superia": [
        ("color_cast", {"temperature": -8}),
        ("saturation", {"factor": 1.05}),
        ("contrast", {"factor": 1.0}),
        ("grain", {"intensity": 0.15}),
        ("lut_shift", {"shadows_rgb": (10, 5, 0), "midtones_rgb": (0, 5, 0), "highlights_rgb": (0, 0, 5)}),
    ],
    "fuji_c200": [
        ("color_cast", {"temperature": 5}),
        ("saturation", {"factor": 0.95}),
        ("grain", {"intensity": 0.2}),
        ("dynamic_range", {"factor": 0.9}),
    ],
    "cinestill_800t": [
        ("color_cast", {"temperature": -15}),
        ("halation", {"intensity": 0.25}),
        ("grain", {"intensity": 0.25}),
        ("saturation", {"factor": 0.9}),
        ("lut_shift", {"shadows_rgb": (15, 5, 0), "midtones_rgb": (0, 0, 5), "highlights_rgb": (-10, 0, 10)}),
    ],
    "ilford_hp5": [
        ("grayscale", {}),
        ("contrast", {"factor": 1.15}),
        ("grain", {"intensity": 0.3}),
        ("dynamic_range", {"factor": 0.85}),
    ],
    "kodak_trix": [
        ("grayscale", {}),
        ("contrast", {"factor": 1.2}),
        ("grain", {"intensity": 0.35}),
        ("dynamic_range", {"factor": 0.8}),
    ],

    # === Analog Style (SD community favorites) ===
    "analog_warm": [
        ("color_cast", {"temperature": 20}),
        ("grain", {"intensity": 0.2}),
        ("vignette", {"intensity": 0.25}),
        ("saturation", {"factor": 0.9}),
        ("contrast", {"factor": 0.95}),
        ("color_fade", {"intensity": 0.1}),
    ],
    "analog_cool": [
        ("color_cast", {"temperature": -15}),
        ("grain", {"intensity": 0.2}),
        ("vignette", {"intensity": 0.2}),
        ("saturation", {"factor": 0.85}),
        ("color_fade", {"intensity": 0.15}),
    ],
    "analog_faded": [
        ("color_fade", {"intensity": 0.35}),
        ("grain", {"intensity": 0.25}),
        ("vignette", {"intensity": 0.3}),
        ("saturation", {"factor": 0.7}),
        ("color_cast", {"temperature": 10}),
    ],
    "analog_vibrant": [
        ("saturation", {"factor": 1.2}),
        ("contrast", {"factor": 1.1}),
        ("grain", {"intensity": 0.15}),
        ("vignette", {"intensity": 0.15}),
        ("color_cast", {"temperature": 8}),
    ],

    # === Light Leaks & Damage ===
    "light_leak_warm": [
        ("light_leak", {"intensity": 0.35, "position": "right"}),
        ("grain", {"intensity": 0.15}),
        ("color_cast", {"temperature": 10}),
    ],
    "light_leak_cool": [
        ("light_leak", {"intensity": 0.3, "position": "left"}),
        ("color_cast", {"temperature": -10}),
        ("grain", {"intensity": 0.15}),
    ],
    "damaged_film": [
        ("dust_scratches", {"dust_count": 80, "scratch_count": 5}),
        ("grain", {"intensity": 0.35}),
        ("color_fade", {"intensity": 0.25}),
        ("vignette", {"intensity": 0.35}),
    ],
    "found_footage": [
        ("dust_scratches", {"dust_count": 100, "scratch_count": 8}),
        ("grain", {"intensity": 0.4}),
        ("interlacing", {}),
        ("color_fade", {"intensity": 0.3}),
        ("noise_bands", {"intensity": 0.15, "band_height": 20}),
        ("compression", {"quality": 40}),
    ],

    # === Camera Brands ===
    "dslr_canon": [
        ("saturation", {"factor": 1.05}),
        ("contrast", {"factor": 1.02}),
        ("color_cast", {"temperature": 5}),
        ("grain", {"intensity": 0.02}),
    ],
    "dslr_nikon": [
        ("saturation", {"factor": 1.0}),
        ("contrast", {"factor": 1.05}),
        ("color_cast", {"temperature": -3}),
        ("grain", {"intensity": 0.02}),
    ],
    "dslr_sony": [
        ("saturation", {"factor": 0.98}),
        ("contrast", {"factor": 1.0}),
        ("color_cast", {"temperature": 0}),
        ("grain", {"intensity": 0.015}),
    ],
    "gopro": [
        ("barrel_distortion", {"strength": 0.25}),
        ("saturation", {"factor": 1.15}),
        ("contrast", {"factor": 1.1}),
        ("grain", {"intensity": 0.05}),
    ],
    "drone_footage": [
        ("saturation", {"factor": 1.1}),
        ("contrast", {"factor": 1.05}),
        ("compression", {"quality": 70}),
        ("grain", {"intensity": 0.03}),
    ],

    # === Depth of Field ===
    "shallow_dof": [
        ("lens_blur", {"blur_amount": 25, "focus_point": (0.5, 0.5)}),
        ("grain", {"intensity": 0.05}),
    ],
    "portrait_bokeh": [
        ("lens_blur", {"blur_amount": 31, "focus_point": (0.5, 0.4)}),
        ("saturation", {"factor": 1.05}),
        ("contrast", {"factor": 1.02}),
    ],

    # === Social Media Compression ===
    "twitter_compressed": [
        ("compression", {"quality": 55}),
        ("downscale", {"scale_factor": 0.8}),
        ("grain", {"intensity": 0.03}),
    ],
    "instagram_compressed": [
        ("compression", {"quality": 60}),
        ("saturation", {"factor": 1.05}),
        ("grain", {"intensity": 0.02}),
    ],
    "tiktok_compressed": [
        ("compression", {"quality": 50}),
        ("downscale", {"scale_factor": 0.75}),
        ("saturation", {"factor": 1.1}),
    ],
    "whatsapp_compressed": [
        ("compression", {"quality": 40}),
        ("downscale", {"scale_factor": 0.6}),
        ("grain", {"intensity": 0.05}),
    ],

    # === Psychedelic/Trippy Effects (Animated) ===
    "acid_trip": [
        ("trails", {"decay": 0.35}),
        ("psychedelic", {"intensity": 0.8}),
    ],
    "acid_mild": [
        ("trails", {"decay": 0.2}),
        ("psychedelic", {"intensity": 0.4}),
    ],
    "acid_intense": [
        ("trails", {"decay": 0.5}),
        ("psychedelic", {"intensity": 1.0}),
    ],
    "rainbow_wave": [
        ("trails", {"decay": 0.25}),
        ("hue_shift", {"animated": True, "speed": 3.0}),
        ("saturation", {"factor": 1.3}),
        ("wave_distortion", {"amplitude": 8.0, "frequency": 0.04, "animated": True}),
    ],
    "color_storm": [
        ("trails", {"decay": 0.3}),
        ("color_cycle", {"intensity": 0.5, "speed": 4.0}),
        ("saturation", {"factor": 1.4}),
        ("chromatic_aberration", {"shift": 3}),
    ],
    "kaleidoscope": [
        ("kaleidoscope", {"segments": 6}),
        ("saturation", {"factor": 1.2}),
        ("hue_shift", {"animated": True, "speed": 1.5}),
    ],
    "rgb_glitch": [
        ("trails", {"decay": 0.2}),
        ("rgb_split", {"offset": 8, "animated": True}),
        ("hue_shift", {"animated": True, "speed": 2.0}),
        ("grain", {"intensity": 0.1}),
    ],
    "pixel_acid": [
        ("trails", {"decay": 0.25}),
        ("pixelate_pulse", {"min_size": 4, "max_size": 12, "speed": 0.15}),
        ("hue_shift", {"animated": True, "speed": 2.5}),
        ("saturation", {"factor": 1.3}),
    ],
    "dream_sequence": [
        ("trails", {"decay": 0.4}),
        ("wave_distortion", {"amplitude": 5.0, "frequency": 0.02, "animated": True, "speed": 0.05}),
        ("blur", {"kernel_size": 3}),
        ("hue_shift", {"animated": True, "speed": 0.5}),
        ("vignette", {"intensity": 0.35}),
        ("color_fade", {"intensity": 0.2}),
    ],
    "neon_glow": [
        ("trails", {"decay": 0.25}),
        ("saturation", {"factor": 1.5}),
        ("contrast", {"factor": 1.2}),
        ("hue_shift", {"animated": True, "speed": 1.0}),
        ("chromatic_aberration", {"shift": 4}),
        ("halation", {"intensity": 0.3}),
    ],
    "warp_speed": [
        ("trails", {"decay": 0.35}),
        ("wave_distortion", {"amplitude": 15.0, "frequency": 0.06, "animated": True, "speed": 0.2}),
        ("rgb_split", {"offset": 5, "animated": True}),
        ("saturation", {"factor": 1.2}),
    ],
    "trails_only": [
        ("trails", {"decay": 0.4}),
        ("saturation", {"factor": 1.1}),
    ],
    "heavy_trails": [
        ("trails", {"decay": 0.6}),
        ("hue_shift", {"animated": True, "speed": 1.0}),
        ("saturation", {"factor": 1.2}),
    ],
}

# List of available filter names
AVAILABLE_FILTERS = list(FILTER_PRESETS.keys())
