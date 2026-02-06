#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

Camera and film filter effects for realistic deepfake degradation.
"""

import numpy as np

from .effects import (
    EFFECTS,
    add_grain,
    add_blur,
    add_compression_artifacts,
    adjust_color_cast,
    reduce_dynamic_range,
    add_vignette,
    downscale_upscale,
    add_scanlines,
    add_chromatic_aberration,
    add_color_tint,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    add_noise_bands,
    add_interlacing,
    add_timestamp,
    convert_grayscale,
    add_barrel_distortion,
    add_light_leak,
    add_color_fade,
    add_halation,
    add_dust_scratches,
    add_film_border,
    apply_lut_shift,
    add_lens_blur,
    # Psychedelic/animated effects
    add_hue_shift,
    add_wave_distortion,
    add_color_cycle,
    add_trails,
    add_kaleidoscope,
    add_rgb_split,
    add_pixelate_pulse,
    add_psychedelic_combo,
    # Frame counter for animated effects
    set_frame_counter,
    get_frame_counter,
    increment_frame_counter,
    # Previous frame for trails effect
    set_previous_frame,
    get_previous_frame,
    clear_previous_frame,
)

from .presets import FILTER_PRESETS, AVAILABLE_FILTERS


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    """Apply a named filter preset to an image.

    Args:
        image: Input image (BGR format).
        filter_name: Name of the filter preset to apply.

    Returns:
        Filtered image, or original if filter not found.
    """
    from .effects import set_previous_frame, get_previous_frame

    if filter_name not in FILTER_PRESETS:
        return image

    for effect_name, params in FILTER_PRESETS[filter_name]:
        if effect_name in EFFECTS:
            image = EFFECTS[effect_name](image, **params)

    # Store this frame for trails effect in next frame
    set_previous_frame(image)

    return image


__all__ = [
    # Main function
    "apply_filter",
    # Presets
    "FILTER_PRESETS",
    "AVAILABLE_FILTERS",
    # Effects
    "EFFECTS",
    "add_grain",
    "add_blur",
    "add_compression_artifacts",
    "adjust_color_cast",
    "reduce_dynamic_range",
    "add_vignette",
    "downscale_upscale",
    "add_scanlines",
    "add_chromatic_aberration",
    "add_color_tint",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_saturation",
    "add_noise_bands",
    "add_interlacing",
    "add_timestamp",
    "convert_grayscale",
    "add_barrel_distortion",
    "add_light_leak",
    "add_color_fade",
    "add_halation",
    "add_dust_scratches",
    "add_film_border",
    "apply_lut_shift",
    "add_lens_blur",
    # Psychedelic/animated effects
    "add_hue_shift",
    "add_wave_distortion",
    "add_color_cycle",
    "add_trails",
    "add_kaleidoscope",
    "add_rgb_split",
    "add_pixelate_pulse",
    "add_psychedelic_combo",
    # Frame counter
    "set_frame_counter",
    "get_frame_counter",
    "increment_frame_counter",
    # Previous frame for trails
    "set_previous_frame",
    "get_previous_frame",
    "clear_previous_frame",
]
