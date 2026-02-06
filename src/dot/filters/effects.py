#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

Individual filter effect functions.
"""

import cv2
import numpy as np


def add_grain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add film grain noise to an image.

    Args:
        image: Input image (BGR format).
        intensity: Noise intensity (0.0 to 1.0).

    Returns:
        Image with grain noise applied.
    """
    h, w = image.shape[:2]
    noise = np.random.normal(0, intensity * 50, (h, w, 3)).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def add_blur(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Add Gaussian blur to an image.

    Args:
        image: Input image (BGR format).
        kernel_size: Size of the blur kernel (must be odd).

    Returns:
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_compression_artifacts(image: np.ndarray, quality: int = 70) -> np.ndarray:
    """Add JPEG compression artifacts to an image.

    Args:
        image: Input image (BGR format).
        quality: JPEG quality (0-100, lower = more artifacts).

    Returns:
        Image with compression artifacts.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def adjust_color_cast(image: np.ndarray, temperature: float = 0) -> np.ndarray:
    """Adjust color temperature of an image.

    Args:
        image: Input image (BGR format).
        temperature: Color shift (-100 to 100, negative=cool/blue, positive=warm/orange).

    Returns:
        Color-adjusted image.
    """
    if temperature == 0:
        return image

    result = image.astype(np.float32)

    if temperature > 0:
        # Warm: increase red, decrease blue
        result[:, :, 2] = np.clip(result[:, :, 2] + temperature * 0.5, 0, 255)  # R
        result[:, :, 0] = np.clip(result[:, :, 0] - temperature * 0.3, 0, 255)  # B
    else:
        # Cool: increase blue, decrease red
        result[:, :, 0] = np.clip(result[:, :, 0] - temperature * 0.5, 0, 255)  # B
        result[:, :, 2] = np.clip(result[:, :, 2] + temperature * 0.3, 0, 255)  # R

    return result.astype(np.uint8)


def reduce_dynamic_range(image: np.ndarray, factor: float = 0.8) -> np.ndarray:
    """Reduce the dynamic range of an image (flatten contrast).

    Args:
        image: Input image (BGR format).
        factor: Range reduction factor (0.0 to 1.0, lower = more reduction).

    Returns:
        Image with reduced dynamic range.
    """
    result = image.astype(np.float32)
    mid = 128.0
    result = mid + (result - mid) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def add_vignette(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add vignette effect (darker corners) to an image.

    Args:
        image: Input image (BGR format).
        intensity: Vignette intensity (0.0 to 1.0).

    Returns:
        Image with vignette applied.
    """
    h, w = image.shape[:2]

    # Create vignette mask
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    radius = np.sqrt(X**2 + Y**2)

    # Create smooth falloff from center
    vignette = 1 - np.clip(radius * intensity, 0, 1)
    vignette = np.stack([vignette] * 3, axis=-1)

    result = image.astype(np.float32) * vignette
    return np.clip(result, 0, 255).astype(np.uint8)


def downscale_upscale(image: np.ndarray, scale_factor: float = 0.5) -> np.ndarray:
    """Simulate low resolution by downscaling then upscaling.

    Args:
        image: Input image (BGR format).
        scale_factor: Scale factor for downscaling (0.0 to 1.0).

    Returns:
        Image with simulated low resolution.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if new_h < 1 or new_w < 1:
        return image

    # Downscale
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Upscale back
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def add_scanlines(image: np.ndarray, intensity: float = 0.3, spacing: int = 2) -> np.ndarray:
    """Add CRT-style scanlines to an image.

    Args:
        image: Input image (BGR format).
        intensity: Scanline darkness (0.0 to 1.0).
        spacing: Pixels between scanlines.

    Returns:
        Image with scanlines.
    """
    result = image.astype(np.float32)
    h, w = image.shape[:2]

    for y in range(0, h, spacing):
        result[y, :] = result[y, :] * (1 - intensity)

    return np.clip(result, 0, 255).astype(np.uint8)


def add_chromatic_aberration(image: np.ndarray, shift: int = 2) -> np.ndarray:
    """Add chromatic aberration (RGB channel shift) to an image.

    Args:
        image: Input image (BGR format).
        shift: Pixel shift amount for color channels.

    Returns:
        Image with chromatic aberration.
    """
    h, w = image.shape[:2]
    result = np.zeros_like(image)

    # Shift red channel right, blue channel left
    result[:, shift:, 2] = image[:, :-shift, 2]  # Red
    result[:, :, 1] = image[:, :, 1]  # Green (no shift)
    result[:, :-shift, 0] = image[:, shift:, 0]  # Blue

    return result


def add_color_tint(image: np.ndarray, color: tuple = (0, 255, 0), intensity: float = 0.3) -> np.ndarray:
    """Add a color tint overlay to an image.

    Args:
        image: Input image (BGR format).
        color: BGR color tuple for the tint.
        intensity: Tint intensity (0.0 to 1.0).

    Returns:
        Tinted image.
    """
    overlay = np.full_like(image, color, dtype=np.uint8)
    return cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)


def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image brightness.

    Args:
        image: Input image (BGR format).
        factor: Brightness multiplier (1.0 = no change, <1 = darker, >1 = brighter).

    Returns:
        Brightness-adjusted image.
    """
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image contrast.

    Args:
        image: Input image (BGR format).
        factor: Contrast multiplier (1.0 = no change, <1 = less contrast, >1 = more).

    Returns:
        Contrast-adjusted image.
    """
    result = image.astype(np.float32)
    mean = np.mean(result)
    result = mean + (result - mean) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Adjust image saturation.

    Args:
        image: Input image (BGR format).
        factor: Saturation multiplier (1.0 = no change, 0 = grayscale, >1 = more saturated).

    Returns:
        Saturation-adjusted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def add_noise_bands(image: np.ndarray, intensity: float = 0.2, band_height: int = 20) -> np.ndarray:
    """Add horizontal noise bands (VHS-style interference).

    Args:
        image: Input image (BGR format).
        intensity: Noise intensity.
        band_height: Height of noise bands.

    Returns:
        Image with noise bands.
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Random band position
    band_y = np.random.randint(0, max(1, h - band_height))

    # Add noise to the band
    noise = np.random.normal(0, intensity * 100, (band_height, w, 3)).astype(np.float32)
    result[band_y:band_y + band_height] = np.clip(
        result[band_y:band_y + band_height].astype(np.float32) + noise, 0, 255
    ).astype(np.uint8)

    return result


def add_interlacing(image: np.ndarray) -> np.ndarray:
    """Add interlacing artifact (blend odd/even lines from shifted frame).

    Args:
        image: Input image (BGR format).

    Returns:
        Interlaced image.
    """
    result = image.copy()
    h = image.shape[0]

    # Shift odd lines slightly
    for y in range(1, h, 2):
        shift = np.random.randint(1, 3)
        result[y] = np.roll(result[y], shift, axis=0)

    return result


def add_timestamp(image: np.ndarray, text: str = None) -> np.ndarray:
    """Add a timestamp overlay to an image.

    Args:
        image: Input image (BGR format).
        text: Custom text (default: current-looking timestamp).

    Returns:
        Image with timestamp.
    """
    from datetime import datetime

    result = image.copy()
    if text is None:
        text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * (w / 640)  # Scale with image size
    thickness = max(1, int(w / 640))

    # Position in bottom right
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = w - text_size[0] - 10
    y = h - 10

    # Draw with background
    cv2.rectangle(result, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), thickness)

    return result


def convert_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale (but keep 3 channels).

    Args:
        image: Input image (BGR format).

    Returns:
        Grayscale image with 3 channels.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def add_barrel_distortion(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Add barrel distortion (fisheye-like effect).

    Args:
        image: Input image (BGR format).
        strength: Distortion strength.

    Returns:
        Distorted image.
    """
    h, w = image.shape[:2]
    K = np.array([[w, 0, w / 2],
                  [0, h, h / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([strength, strength, 0, 0], dtype=np.float32)

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def add_light_leak(image: np.ndarray, intensity: float = 0.3, position: str = "random") -> np.ndarray:
    """Add light leak effect (overexposed areas, common in analog film).

    Args:
        image: Input image (BGR format).
        intensity: Leak intensity (0.0 to 1.0).
        position: Leak position - "left", "right", "top", "bottom", or "random".

    Returns:
        Image with light leak.
    """
    h, w = image.shape[:2]
    result = image.astype(np.float32)

    # Create gradient mask for light leak
    if position == "random":
        position = np.random.choice(["left", "right", "top", "bottom"])

    if position == "left":
        gradient = np.linspace(1, 0, w).reshape(1, w, 1)
        gradient = np.tile(gradient, (h, 1, 3))
    elif position == "right":
        gradient = np.linspace(0, 1, w).reshape(1, w, 1)
        gradient = np.tile(gradient, (h, 1, 3))
    elif position == "top":
        gradient = np.linspace(1, 0, h).reshape(h, 1, 1)
        gradient = np.tile(gradient, (1, w, 3))
    else:  # bottom
        gradient = np.linspace(0, 1, h).reshape(h, 1, 1)
        gradient = np.tile(gradient, (1, w, 3))

    # Apply orange/red tint leak
    leak_color = np.array([50, 100, 255], dtype=np.float32)  # BGR - orange/red
    leak = gradient * leak_color * intensity

    result = result + leak
    return np.clip(result, 0, 255).astype(np.uint8)


def add_color_fade(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add faded color effect (lifted blacks, desaturated).

    Args:
        image: Input image (BGR format).
        intensity: Fade intensity (0.0 to 1.0).

    Returns:
        Faded image.
    """
    result = image.astype(np.float32)

    # Lift blacks (add to dark areas)
    lift = intensity * 40
    result = result + lift

    # Reduce saturation
    hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * (1 - intensity * 0.4)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result


def add_halation(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add halation effect (glow around bright areas, common in film).

    Args:
        image: Input image (BGR format).
        intensity: Halation intensity (0.0 to 1.0).

    Returns:
        Image with halation.
    """
    # Find bright areas
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Blur the bright areas heavily
    kernel_size = max(21, int(image.shape[0] / 10) | 1)  # Ensure odd
    glow = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply red/orange tint to glow
    glow = glow.astype(np.float32)
    glow[:, :, 2] = np.clip(glow[:, :, 2] * 1.3, 0, 255)  # Boost red

    # Blend based on bright mask
    bright_mask_3ch = np.stack([bright_mask] * 3, axis=-1).astype(np.float32) / 255
    result = image.astype(np.float32) * (1 - bright_mask_3ch * intensity) + glow * bright_mask_3ch * intensity

    return np.clip(result, 0, 255).astype(np.uint8)


def add_dust_scratches(image: np.ndarray, dust_count: int = 50, scratch_count: int = 3) -> np.ndarray:
    """Add dust spots and scratches (film damage simulation).

    Args:
        image: Input image (BGR format).
        dust_count: Number of dust particles.
        scratch_count: Number of scratches.

    Returns:
        Image with dust and scratches.
    """
    result = image.copy()
    h, w = image.shape[:2]

    # Add dust particles (small white/gray dots)
    for _ in range(dust_count):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        radius = np.random.randint(1, 3)
        color = np.random.randint(180, 255)
        cv2.circle(result, (x, y), radius, (color, color, color), -1)

    # Add scratches (vertical lines)
    for _ in range(scratch_count):
        x = np.random.randint(0, w)
        length = np.random.randint(h // 4, h)
        y_start = np.random.randint(0, h - length)
        color = np.random.randint(150, 220)
        thickness = 1
        cv2.line(result, (x, y_start), (x + np.random.randint(-2, 2), y_start + length),
                 (color, color, color), thickness)

    return result


def add_film_border(image: np.ndarray, border_size: int = 20, sprocket: bool = True) -> np.ndarray:
    """Add film border with optional sprocket holes.

    Args:
        image: Input image (BGR format).
        border_size: Size of border in pixels.
        sprocket: Whether to add sprocket holes.

    Returns:
        Image with film border.
    """
    h, w = image.shape[:2]

    # Create larger canvas with black border
    new_h = h + border_size * 2
    new_w = w + border_size * 2
    result = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Place image in center
    result[border_size:border_size + h, border_size:border_size + w] = image

    # Add sprocket holes if requested
    if sprocket:
        hole_h = border_size - 4
        hole_w = hole_h // 2
        spacing = hole_h * 2

        for y in range(hole_h // 2, new_h - hole_h // 2, spacing):
            # Left side
            cv2.rectangle(result, (2, y - hole_h // 2), (2 + hole_w, y + hole_h // 2), (40, 40, 40), -1)
            # Right side
            cv2.rectangle(result, (new_w - 2 - hole_w, y - hole_h // 2), (new_w - 2, y + hole_h // 2), (40, 40, 40), -1)

    return result


def apply_lut_shift(image: np.ndarray, shadows_rgb: tuple = (0, 0, 0),
                    midtones_rgb: tuple = (0, 0, 0), highlights_rgb: tuple = (0, 0, 0)) -> np.ndarray:
    """Apply color grading via shadow/midtone/highlight shifts.

    Args:
        image: Input image (BGR format).
        shadows_rgb: RGB shift for shadows (-50 to 50).
        midtones_rgb: RGB shift for midtones (-50 to 50).
        highlights_rgb: RGB shift for highlights (-50 to 50).

    Returns:
        Color graded image.
    """
    result = image.astype(np.float32)

    # Create luminance mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    # Shadow mask (dark areas)
    shadow_mask = 1 - np.clip(gray * 2, 0, 1)
    # Highlight mask (bright areas)
    highlight_mask = np.clip((gray - 0.5) * 2, 0, 1)
    # Midtone mask
    midtone_mask = 1 - shadow_mask - highlight_mask
    midtone_mask = np.clip(midtone_mask, 0, 1)

    # Apply shifts (convert RGB to BGR)
    for i, (s, m, h) in enumerate(zip(shadows_rgb[::-1], midtones_rgb[::-1], highlights_rgb[::-1])):
        result[:, :, i] += shadow_mask * s + midtone_mask * m + highlight_mask * h

    return np.clip(result, 0, 255).astype(np.uint8)


def add_lens_blur(image: np.ndarray, blur_amount: int = 15, focus_point: tuple = None) -> np.ndarray:
    """Add depth-of-field style lens blur (sharp center, blurry edges).

    Args:
        image: Input image (BGR format).
        blur_amount: Amount of blur for edges.
        focus_point: (x, y) focus point as ratio (0-1). Default is center.

    Returns:
        Image with lens blur.
    """
    h, w = image.shape[:2]

    if focus_point is None:
        focus_point = (0.5, 0.5)

    # Create distance map from focus point
    y_coords, x_coords = np.ogrid[:h, :w]
    center_x = int(focus_point[0] * w)
    center_y = int(focus_point[1] * h)

    distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2) / 2
    blur_mask = np.clip(distance / max_dist, 0, 1)

    # Create blurred version
    if blur_amount % 2 == 0:
        blur_amount += 1
    blurred = cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)

    # Blend based on distance
    blur_mask_3ch = np.stack([blur_mask] * 3, axis=-1)
    result = image.astype(np.float32) * (1 - blur_mask_3ch) + blurred.astype(np.float32) * blur_mask_3ch

    return result.astype(np.uint8)


# Frame counter for animated effects (will be incremented by caller)
_frame_counter = 0
# Previous frame storage for trails effect
_previous_frame = None


def set_frame_counter(count: int) -> None:
    """Set the frame counter for animated effects."""
    global _frame_counter
    _frame_counter = count


def get_frame_counter() -> int:
    """Get the current frame counter."""
    global _frame_counter
    return _frame_counter


def increment_frame_counter() -> int:
    """Increment and return the frame counter."""
    global _frame_counter
    _frame_counter += 1
    return _frame_counter


def set_previous_frame(frame: np.ndarray) -> None:
    """Store the previous frame for trails effect."""
    global _previous_frame
    _previous_frame = frame.copy() if frame is not None else None


def get_previous_frame() -> np.ndarray:
    """Get the stored previous frame."""
    global _previous_frame
    return _previous_frame


def clear_previous_frame() -> None:
    """Clear the stored previous frame."""
    global _previous_frame
    _previous_frame = None


def add_hue_shift(image: np.ndarray, shift: int = 0, animated: bool = False, speed: float = 2.0) -> np.ndarray:
    """Shift the hue of an image (color rotation).

    Args:
        image: Input image (BGR format).
        shift: Hue shift amount (0-180). If animated=True, this is the base offset.
        animated: If True, hue shifts over time based on frame counter.
        speed: Animation speed multiplier.

    Returns:
        Hue-shifted image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if animated:
        # Use frame counter for animation
        actual_shift = int((_frame_counter * speed) % 180)
    else:
        actual_shift = shift

    hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + actual_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_wave_distortion(image: np.ndarray, amplitude: float = 10.0, frequency: float = 0.05,
                        animated: bool = True, speed: float = 0.1) -> np.ndarray:
    """Add wavy distortion effect.

    Args:
        image: Input image (BGR format).
        amplitude: Wave amplitude in pixels.
        frequency: Wave frequency.
        animated: If True, waves move over time.
        speed: Animation speed.

    Returns:
        Distorted image.
    """
    h, w = image.shape[:2]

    # Create coordinate maps
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)

    # Time offset for animation
    t = _frame_counter * speed if animated else 0

    # Apply wave distortion
    x_coords = x_coords + amplitude * np.sin(2 * np.pi * frequency * y_coords + t)
    y_coords = y_coords + amplitude * np.sin(2 * np.pi * frequency * x_coords + t * 0.7)

    # Remap image
    return cv2.remap(image, x_coords, y_coords, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def add_color_cycle(image: np.ndarray, intensity: float = 0.5, speed: float = 3.0) -> np.ndarray:
    """Add cycling color overlay effect.

    Args:
        image: Input image (BGR format).
        intensity: Color overlay intensity (0-1).
        speed: Cycle speed.

    Returns:
        Image with color cycling.
    """
    h, w = image.shape[:2]
    t = _frame_counter * speed * 0.1

    # Create RGB cycle based on position and time
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Different phase for each channel creates moving rainbow
    r = (np.sin(x_coords * 0.01 + t) * 127 + 128).astype(np.float32)
    g = (np.sin(y_coords * 0.01 + t + 2.094) * 127 + 128).astype(np.float32)  # 2π/3 offset
    b = (np.sin((x_coords + y_coords) * 0.01 + t + 4.189) * 127 + 128).astype(np.float32)  # 4π/3 offset

    overlay = np.stack([b, g, r], axis=-1).astype(np.uint8)

    return cv2.addWeighted(image, 1 - intensity, overlay, intensity, 0)


def add_trails(image: np.ndarray, decay: float = 0.6) -> np.ndarray:
    """Add color trails/ghosting effect by blending with previous frame.

    Uses the stored previous frame from the effects module.
    Creates that dreamy, trippy trailing effect on movement.

    Args:
        image: Current input image (BGR format).
        decay: Trail decay factor (0-1). Higher = longer/stronger trails.

    Returns:
        Image with trails.
    """
    global _previous_frame

    if _previous_frame is None or _previous_frame.shape != image.shape:
        return image

    # Blend current with previous (trails) - decay controls how much of previous frame shows
    # Higher decay = more of previous frame = longer trails
    result = cv2.addWeighted(image, 1 - decay, _previous_frame, decay, 0)
    return result


def add_kaleidoscope(image: np.ndarray, segments: int = 6) -> np.ndarray:
    """Add kaleidoscope mirror effect.

    Args:
        image: Input image (BGR format).
        segments: Number of mirror segments.

    Returns:
        Kaleidoscope image.
    """
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    result = image.copy()

    # Create polar coordinates
    y_coords, x_coords = np.ogrid[:h, :w]
    x_centered = x_coords - center_x
    y_centered = y_coords - center_y

    # Calculate angle for each pixel
    angles = np.arctan2(y_centered, x_centered)

    # Segment angle
    segment_angle = 2 * np.pi / segments

    # Mirror within each segment
    mirrored_angles = np.abs(((angles + np.pi) % segment_angle) - segment_angle / 2)
    mirrored_angles = mirrored_angles - segment_angle / 2

    # Calculate radius
    radius = np.sqrt(x_centered ** 2 + y_centered ** 2)

    # Convert back to cartesian
    new_x = (radius * np.cos(mirrored_angles) + center_x).astype(np.float32)
    new_y = (radius * np.sin(mirrored_angles) + center_y).astype(np.float32)

    # Clip to valid range
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)

    return cv2.remap(image, new_x, new_y, cv2.INTER_LINEAR)


def add_rgb_split(image: np.ndarray, offset: int = 5, animated: bool = True) -> np.ndarray:
    """Add RGB channel splitting with optional animation.

    Args:
        image: Input image (BGR format).
        offset: Base pixel offset.
        animated: If True, offset oscillates over time.

    Returns:
        Image with RGB split.
    """
    h, w = image.shape[:2]

    if animated:
        t = _frame_counter * 0.15
        actual_offset = int(offset * np.sin(t) + offset)
    else:
        actual_offset = offset

    result = np.zeros_like(image)

    # Offset each channel differently
    if actual_offset > 0:
        result[:, actual_offset:, 2] = image[:, :-actual_offset, 2]  # Red right
        result[:, :, 1] = image[:, :, 1]  # Green center
        result[:, :-actual_offset, 0] = image[:, actual_offset:, 0]  # Blue left
    else:
        result = image.copy()

    return result


def add_pixelate_pulse(image: np.ndarray, min_size: int = 4, max_size: int = 16, speed: float = 0.1) -> np.ndarray:
    """Add pulsating pixelation effect.

    Args:
        image: Input image (BGR format).
        min_size: Minimum pixel block size.
        max_size: Maximum pixel block size.
        speed: Pulse speed.

    Returns:
        Pixelated image.
    """
    h, w = image.shape[:2]

    # Oscillate block size
    t = _frame_counter * speed
    block_size = int((np.sin(t) + 1) / 2 * (max_size - min_size) + min_size)
    block_size = max(1, block_size)

    # Downscale
    small_h, small_w = h // block_size, w // block_size
    if small_h < 1 or small_w < 1:
        return image

    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def add_psychedelic_combo(image: np.ndarray, intensity: float = 0.7) -> np.ndarray:
    """Combined psychedelic effect with hue shift, saturation boost, and waves.

    Args:
        image: Input image (BGR format).
        intensity: Overall effect intensity.

    Returns:
        Psychedelic image.
    """
    result = image.copy()

    # Boost saturation significantly
    result = adjust_saturation(result, factor=1.5 + intensity * 0.5)

    # Animated hue shift
    result = add_hue_shift(result, animated=True, speed=2.0 * intensity)

    # Subtle wave distortion
    result = add_wave_distortion(result, amplitude=5 * intensity, frequency=0.03, animated=True)

    # Color cycling overlay
    result = add_color_cycle(result, intensity=0.2 * intensity, speed=3.0)

    # Slight chromatic aberration
    result = add_chromatic_aberration(result, shift=int(3 * intensity))

    return result


# Mapping of effect names to functions
EFFECTS = {
    "grain": add_grain,
    "blur": add_blur,
    "compression": add_compression_artifacts,
    "color_cast": adjust_color_cast,
    "dynamic_range": reduce_dynamic_range,
    "vignette": add_vignette,
    "downscale": downscale_upscale,
    "scanlines": add_scanlines,
    "chromatic_aberration": add_chromatic_aberration,
    "color_tint": add_color_tint,
    "brightness": adjust_brightness,
    "contrast": adjust_contrast,
    "saturation": adjust_saturation,
    "noise_bands": add_noise_bands,
    "interlacing": add_interlacing,
    "timestamp": add_timestamp,
    "grayscale": convert_grayscale,
    "barrel_distortion": add_barrel_distortion,
    "light_leak": add_light_leak,
    "color_fade": add_color_fade,
    "halation": add_halation,
    "dust_scratches": add_dust_scratches,
    "film_border": add_film_border,
    "lut_shift": apply_lut_shift,
    "lens_blur": add_lens_blur,
    "hue_shift": add_hue_shift,
    "wave_distortion": add_wave_distortion,
    "color_cycle": add_color_cycle,
    "trails": add_trails,
    "kaleidoscope": add_kaleidoscope,
    "rgb_split": add_rgb_split,
    "pixelate_pulse": add_pixelate_pulse,
    "psychedelic": add_psychedelic_combo,
}
