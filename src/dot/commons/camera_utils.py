#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from .cam.cam import draw_fps
from .utils import TicToc, find_images_from_path
from .video.videocaptureasync import VideoCaptureAsync

# Optional: send output to a virtual camera device (e.g. OBS Virtual Camera)
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False

# Import filter names for cycling
try:
    from ..filters import (
        AVAILABLE_FILTERS,
        increment_frame_counter,
        clear_previous_frame,
        set_previous_frame,
        get_previous_frame,
    )
    from ..filters.effects import add_trails
except ImportError:
    AVAILABLE_FILTERS = []
    increment_frame_counter = None
    clear_previous_frame = None
    set_previous_frame = None
    get_previous_frame = None
    add_trails = None


def draw_osd(
    image: np.ndarray,
    avatar_name: str,
    avatar_index: int,
    avatar_total: int,
    filter_name: Optional[str],
    fps: float,
    smoothing_enabled: bool,
    trails_enabled: bool = False,
    bilateral_enabled: bool = False,
    sharpen_enabled: bool = False,
    upscale_enabled: bool = False,
) -> np.ndarray:
    """Draw on-screen display overlay with current settings.

    Args:
        image: Input image (BGR format).
        avatar_name: Current avatar filename.
        avatar_index: Current avatar index (0-based).
        avatar_total: Total number of avatars.
        filter_name: Current filter name or None.
        fps: Current frames per second.
        smoothing_enabled: Whether temporal smoothing is on.
        trails_enabled: Whether trails effect is on.
        bilateral_enabled: Whether bilateral smoothing is on.
        sharpen_enabled: Whether sharpening is on.
        upscale_enabled: Whether 2x upscale is on.

    Returns:
        Image with OSD overlay.
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Semi-transparent background for OSD
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (350, 175), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

    # Text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    # Draw info lines
    y_offset = 30
    line_height = 20

    # Avatar info
    avatar_short = avatar_name.split("/")[-1] if "/" in avatar_name else avatar_name
    if len(avatar_short) > 25:
        avatar_short = avatar_short[:22] + "..."
    cv2.putText(result, f"Avatar: {avatar_index + 1}/{avatar_total} - {avatar_short}",
                (20, y_offset), font, font_scale, color, thickness)

    # Filter info
    y_offset += line_height
    filter_text = filter_name if filter_name else "None"
    cv2.putText(result, f"Filter: {filter_text}",
                (20, y_offset), font, font_scale, color, thickness)

    # FPS
    y_offset += line_height
    cv2.putText(result, f"FPS: {fps:.1f}",
                (20, y_offset), font, font_scale, color, thickness)

    # Effects status line - combine multiple effects
    y_offset += line_height
    effects = []
    if smoothing_enabled:
        effects.append("Smooth")
    if trails_enabled:
        effects.append("Trails")
    if bilateral_enabled:
        effects.append("Beauty")
    if sharpen_enabled:
        effects.append("Sharp")
    if upscale_enabled:
        effects.append("2x")
    effects_text = ", ".join(effects) if effects else "None"
    cv2.putText(result, f"Effects: {effects_text}",
                (20, y_offset), font, font_scale, color, thickness)

    # Keyboard hints
    y_offset += line_height
    cv2.putText(result, "Keys: t/r/b/s/u f/d/g q",
                (20, y_offset), font, font_scale * 0.9, (180, 180, 180), thickness)

    return result


def apply_temporal_smoothing(
    current_frame: np.ndarray,
    previous_frame: Optional[np.ndarray],
    alpha: float = 0.85,
) -> np.ndarray:
    """Apply motion-aware temporal smoothing to reduce flickering.

    Only smooths areas with small changes (stable areas). Areas with large
    changes (movement) use the current frame to avoid motion blur/trails.

    Args:
        current_frame: Current frame (BGR format).
        previous_frame: Previous frame or None.
        alpha: Base weight for current frame (0-1). Higher = less smoothing.

    Returns:
        Smoothed frame without motion blur.
    """
    if previous_frame is None:
        return current_frame

    # Ensure frames have same shape
    if current_frame.shape != previous_frame.shape:
        return current_frame

    # Calculate per-pixel difference
    diff = cv2.absdiff(current_frame, previous_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Create motion mask: high values where there's movement
    # Threshold: pixels with >30 difference are considered "moving"
    _, motion_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    # Dilate motion mask to catch edges of moving areas
    kernel = np.ones((5, 5), np.uint8)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    # Convert to 3-channel float mask (0 = static, 1 = moving)
    motion_mask_3ch = (motion_mask[:, :, np.newaxis] / 255.0).astype(np.float32)
    motion_mask_3ch = np.repeat(motion_mask_3ch, 3, axis=2)

    # For moving areas: use current frame (alpha=1)
    # For static areas: blend with previous (alpha=0.85)
    static_blend = cv2.addWeighted(current_frame, alpha, previous_frame, 1 - alpha, 0)

    # Combine: moving areas get current frame, static areas get blend
    result = (current_frame.astype(np.float32) * motion_mask_3ch +
              static_blend.astype(np.float32) * (1 - motion_mask_3ch))

    return result.astype(np.uint8)


def fetch_camera(target: int, width: int = 1920, height: int = 1080) -> VideoCaptureAsync:
    """Fetches a VideoCaptureAsync object.

    Args:
        target (int): Camera ID descriptor.
        width (int): Requested camera width. Defaults to 1920.
        height (int): Requested camera height. Defaults to 1080.

    Raises:
        ValueError: If camera ID descriptor is not valid.

    Returns:
        VideoCaptureAsync: VideoCaptureAsync object.
    """
    try:
        return VideoCaptureAsync(target, width=width, height=height)
    except RuntimeError:
        raise ValueError(f"Camera {target} does not exist.")


def camera_pipeline(
    cap: VideoCaptureAsync,
    source: str,
    target: int,
    change_option: Callable[[np.ndarray], None],
    process_image: Callable[[np.ndarray], np.ndarray],
    post_process_image: Callable[[np.ndarray], np.ndarray],
    crop_size: int = 224,
    show_fps: bool = False,
    model_option: Optional[Any] = None,
    use_virtual_cam: bool = False,
    virtual_cam_fps: int = 20,
    **kwargs: Dict,
) -> None:
    """Open a webcam stream `target` and performs face-swap based on `source` image by frame.

    Args:
        cap (VideoCaptureAsync): VideoCaptureAsync object.
        source (str): Path to source image folder.
        target (int): Camera ID descriptor.
        change_option (Callable[[np.ndarray], None]): Set `source` arg as faceswap source image.
        process_image (Callable[[np.ndarray], np.ndarray]): Performs actual face swap.
        post_process_image (Callable[[np.ndarray], np.ndarray]): Applies face restoration GPEN to result image.
        crop_size (int, optional): Face crop size. Defaults to 224.
        show_fps (bool, optional): Display FPS. Defaults to False.
        model_option (Optional[Any], optional): Model option for filter control. Defaults to None.
        use_virtual_cam (bool, optional): If True, send output to a virtual camera (requires pyvirtualcam + OBS on macOS). Defaults to False.
        virtual_cam_fps (int, optional): Virtual camera FPS when use_virtual_cam is True. Defaults to 20.
    """
    source_path = source
    source = find_images_from_path(source)
    if not source:
        raise ValueError(
            f"No images found in source path: {source_path!r}. "
            "Use a folder or file path that contains at least one image (e.g. .jpg, .png)."
        )
    print("=== Control keys (click camera window first!) ===")
    print("c / C: Next / previous camera")
    print("Up/Down arrows: Change avatar")
    print("1-9: Jump to avatar")
    for i, fname in enumerate(source):
        print(str(i + 1) + ": " + fname)
    print("q: Quit")
    print("y: Show original feed")
    print("i: Toggle FPS")

    # Filter controls
    if AVAILABLE_FILTERS:
        print("f: Next filter")
        print("d: Previous filter")
        print("g: Disable filter")
        print("Available filters:", ", ".join(AVAILABLE_FILTERS))

    # Enhancement controls
    print("t: Toggle temporal smoothing")
    print("r: Toggle trails (psychedelic effect)")
    print("b: Toggle bilateral smoothing (beauty mode)")
    print("s: Toggle sharpening")
    print("u: Toggle 2x upscale")
    print("o: Toggle OSD (on-screen display)")

    print("\n>>> Click on the camera window to enable keyboard controls <<<\n")

    # Find first source image that contains a detectable face
    pic_a = None
    img_a_whole = None
    for path in source:
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            change_option(img, **kwargs)
            pic_a = path
            img_a_whole = img
            break
        except ValueError as e:
            if "No face detected" in str(e):
                continue
            raise
    if pic_a is None or img_a_whole is None:
        raise ValueError(
            "No face detected in any source image. Use photos that clearly show a single face (front-facing, good lighting)."
        )

    current_source_index = source.index(pic_a)

    img_a_align_crop = process_image(img_a_whole, **kwargs)
    img_a_align_crop = post_process_image(img_a_align_crop, **kwargs)

    cap.start()
    ret, frame = cap.read()
    cv2.namedWindow("cam", cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("cam", 500, 250)
    # Show first frame immediately so the window appears and is visible (e.g. on macOS)
    if ret:
        cv2.imshow("cam", cv2.flip(frame, 1))
        cv2.waitKey(1)

    # Optional: virtual camera device (macOS: needs OBS installed; Windows: OBS or other backend)
    vcam = None
    if use_virtual_cam and ret:
        if PYVIRTUALCAM_AVAILABLE:
            h, w = frame.shape[:2]
            try:
                vcam = pyvirtualcam.Camera(
                    width=w, height=h, fps=virtual_cam_fps, fmt=PixelFormat.BGR
                )
                print(
                    f"Virtual camera active: {vcam.device} ({w}x{h} @ {virtual_cam_fps} fps). "
                    "Use this device in FaceTime/Zoom. In OBS: add Source → Window Capture → choose the 'cam' window (OBS cannot capture the virtual camera itself)."
                )
            except Exception as e:
                print(f"Virtual camera failed: {e}. On macOS install OBS for virtual camera support.")
                vcam = None
        else:
            print("Virtual camera skipped: pip install pyvirtualcam (and on macOS install OBS)")

    frame_index = -1
    fps_hist: List = []
    fps: Union[Any, float] = 0

    # Track current filter index (-1 means no filter)
    current_filter_index = -1
    if model_option and hasattr(model_option, "filter_type") and model_option.filter_type:
        try:
            current_filter_index = AVAILABLE_FILTERS.index(model_option.filter_type)
        except ValueError:
            current_filter_index = -1

    # OSD and temporal smoothing state
    show_osd = False
    smoothing_enabled = False
    previous_frame: Optional[np.ndarray] = None
    smoothing_alpha = 0.7  # Higher = less smoothing (more responsive)

    # Trails effect state (independent of filter)
    trails_enabled = False
    trails_decay = 0.35  # How much of previous frame to blend
    trails_frame: Optional[np.ndarray] = None

    # Smoothing and upscale state
    bilateral_enabled = False
    sharpen_enabled = False
    upscale_enabled = False

    show_self = False
    current_camera_id = target
    try:
        while True:
            frame_index += 1
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if ret:
                # Process GUI events so Live Settings sliders stay responsive when run from GUI
                gui_update = kwargs.get("gui_update_callback")
                if callable(gui_update):
                    gui_update()

                tt = TicToc()

                timing = {"preproc": 0, "predict": 0, "postproc": 0}

                tt.tic()

                # Use & 0xFF so key codes are correct on macOS (waitKey can return 32-bit values)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # 255 means no key pressed
                    print(f"Key pressed: {key} ({chr(key) if 32 <= key < 127 else 'special'})")
                if 48 < key < 58:
                    show_self = False
                    idx = min(key - 49, len(source) - 1)
                    img_a_whole = cv2.imread(source[idx])
                    try:
                        change_option(img_a_whole, **kwargs)
                        current_source_index = idx
                        pic_a = source[idx]
                        print(f"Avatar: {current_source_index + 1}/{len(source)} - {pic_a}")
                    except ValueError as e:
                        if "No face detected" in str(e):
                            print(f"No face in {source[idx]}; try another image.")
                        else:
                            raise
                # Up arrow: macOS=0, Linux=82
                elif key in (0, 82) and len(source) > 1:
                    show_self = False
                    idx = (current_source_index - 1) % len(source)
                    img_a_whole = cv2.imread(source[idx])
                    try:
                        change_option(img_a_whole, **kwargs)
                        current_source_index = idx
                        pic_a = source[idx]
                        print(f"Avatar: {current_source_index + 1}/{len(source)} - {pic_a}")
                    except ValueError as e:
                        if "No face detected" in str(e):
                            print(f"No face in {source[idx]}; try another image.")
                        else:
                            raise
                # Down arrow: macOS=1, Linux=84
                elif key in (1, 84) and len(source) > 1:
                    show_self = False
                    idx = (current_source_index + 1) % len(source)
                    img_a_whole = cv2.imread(source[idx])
                    try:
                        change_option(img_a_whole, **kwargs)
                        current_source_index = idx
                        pic_a = source[idx]
                        print(f"Avatar: {current_source_index + 1}/{len(source)} - {pic_a}")
                    except ValueError as e:
                        if "No face detected" in str(e):
                            print(f"No face in {source[idx]}; try another image.")
                        else:
                            raise
                elif key == ord("y"):
                    show_self = True
                elif key == ord("c"):
                    # Next camera (try current+1, current+2, ... up to 10)
                    for new_id in range(current_camera_id + 1, 10):
                        try:
                            new_cap = VideoCaptureAsync(new_id)
                            new_cap.start()
                            cap.stop()
                            cap = new_cap
                            current_camera_id = new_id
                            print(f"Camera: {current_camera_id}")
                            break
                        except (RuntimeError, ValueError):
                            continue
                    else:
                        print(f"No next camera found (tried up to 9). Current: {current_camera_id}")
                elif key == ord("C"):
                    # Previous camera (try current-1 down to 0)
                    for new_id in range(current_camera_id - 1, -1, -1):
                        try:
                            new_cap = VideoCaptureAsync(new_id)
                            new_cap.start()
                            cap.stop()
                            cap = new_cap
                            current_camera_id = new_id
                            print(f"Camera: {current_camera_id}")
                            break
                        except (RuntimeError, ValueError):
                            continue
                    else:
                        print(f"No previous camera. Current: {current_camera_id}")
                elif key == ord("q"):
                    break
                elif key == ord("i"):
                    show_fps = not show_fps
                # Filter controls
                elif key == ord("f") and AVAILABLE_FILTERS and model_option:
                    # Next filter
                    current_filter_index = (current_filter_index + 1) % len(AVAILABLE_FILTERS)
                    model_option.filter_type = AVAILABLE_FILTERS[current_filter_index]
                    if clear_previous_frame is not None:
                        clear_previous_frame()  # Reset trails when switching
                    print(f"Filter: {model_option.filter_type}")
                elif key == ord("d") and AVAILABLE_FILTERS and model_option:
                    # Previous filter
                    current_filter_index = (current_filter_index - 1) % len(AVAILABLE_FILTERS)
                    model_option.filter_type = AVAILABLE_FILTERS[current_filter_index]
                    if clear_previous_frame is not None:
                        clear_previous_frame()  # Reset trails when switching
                    print(f"Filter: {model_option.filter_type}")
                elif key == ord("g") and model_option:
                    # Disable filter
                    current_filter_index = -1
                    model_option.filter_type = None
                    if clear_previous_frame is not None:
                        clear_previous_frame()  # Reset trails when disabling
                    print("Filter: disabled")
                elif key == ord("t"):
                    # Toggle temporal smoothing
                    smoothing_enabled = not smoothing_enabled
                    if not smoothing_enabled:
                        previous_frame = None  # Reset when disabled
                    print(f"Temporal smoothing: {'ON' if smoothing_enabled else 'OFF'}")
                elif key == ord("o"):
                    # Toggle OSD
                    show_osd = not show_osd
                    print(f"OSD: {'ON' if show_osd else 'OFF'}")
                elif key == ord("r"):
                    # Toggle trails
                    trails_enabled = not trails_enabled
                    if not trails_enabled:
                        trails_frame = None  # Reset when disabled
                    print(f"Trails: {'ON' if trails_enabled else 'OFF'}")
                elif key == ord("b"):
                    # Toggle bilateral smoothing
                    bilateral_enabled = not bilateral_enabled
                    print(f"Bilateral smoothing: {'ON' if bilateral_enabled else 'OFF'}")
                elif key == ord("s"):
                    # Toggle sharpening
                    sharpen_enabled = not sharpen_enabled
                    print(f"Sharpening: {'ON' if sharpen_enabled else 'OFF'}")
                elif key == ord("u"):
                    # Toggle upscale
                    upscale_enabled = not upscale_enabled
                    print(f"2x Upscale: {'ON' if upscale_enabled else 'OFF'}")

                if not show_self:
                    # Increment frame counter for animated filter effects
                    if increment_frame_counter is not None:
                        increment_frame_counter()

                    # Let swap model store its face region mask for face-only effects
                    kwargs["_face_mask"] = {}
                    result_frame = process_image(frame, crop_size=crop_size, **kwargs)  # type: ignore
                    timing["postproc"] = tt.toc()
                    result_frame = post_process_image(result_frame, **kwargs)

                    # Apply bilateral smoothing if enabled (edge-preserving blur)
                    if bilateral_enabled:
                        result_frame = cv2.bilateralFilter(result_frame, d=9, sigmaColor=75, sigmaSpace=75)

                    # Apply sharpening if enabled
                    if sharpen_enabled:
                        # Unsharp mask: sharpen = original + (original - blurred) * amount
                        blurred = cv2.GaussianBlur(result_frame, (0, 0), 3)
                        result_frame = cv2.addWeighted(result_frame, 1.5, blurred, -0.5, 0)

                    # Apply trails effect if enabled (independent of filter)
                    if trails_enabled and add_trails is not None:
                        if trails_frame is not None and trails_frame.shape == result_frame.shape:
                            result_frame = cv2.addWeighted(
                                result_frame, 1 - trails_decay,
                                trails_frame, trails_decay, 0
                            )
                        trails_frame = result_frame.copy()

                    # Apply temporal smoothing if enabled
                    if smoothing_enabled:
                        result_frame = apply_temporal_smoothing(
                            result_frame, previous_frame, smoothing_alpha
                        )
                        previous_frame = result_frame.copy()

                    if show_fps:
                        result_frame = draw_fps(np.array(result_frame), fps, timing)

                    # Draw OSD if enabled
                    if show_osd:
                        filter_name = model_option.filter_type if model_option else None
                        result_frame = draw_osd(
                            result_frame,
                            pic_a,
                            current_source_index,
                            len(source),
                            filter_name,
                            fps,
                            smoothing_enabled,
                            trails_enabled,
                            bilateral_enabled,
                            sharpen_enabled,
                            upscale_enabled,
                        )

                    fps_hist.append(tt.toc(total=True))
                    if len(fps_hist) == 10:
                        fps = 10 / (sum(fps_hist) / 1000)
                        fps_hist = []

                    # Send original resolution to virtual cam (before upscale)
                    if vcam is not None:
                        vcam_frame = result_frame
                        if (vcam_frame.shape[0], vcam_frame.shape[1]) != (vcam.height, vcam.width):
                            vcam_frame = cv2.resize(
                                vcam_frame, (vcam.width, vcam.height), interpolation=cv2.INTER_LINEAR
                            )
                        vcam.send(vcam_frame)
                        vcam.sleep_until_next_frame()

                    # Apply 2x upscale if enabled (for display only)
                    if upscale_enabled:
                        h, w = result_frame.shape[:2]
                        result_frame = cv2.resize(
                            result_frame, (w * 2, h * 2),
                            interpolation=cv2.INTER_LANCZOS4
                        )

                    cv2.imshow("cam", result_frame)

                else:
                    cv2.imshow("cam", frame)
                    if vcam is not None:
                        vcam_frame = frame
                        if (vcam_frame.shape[0], vcam_frame.shape[1]) != (vcam.height, vcam.width):
                            vcam_frame = cv2.resize(
                                vcam_frame, (vcam.width, vcam.height), interpolation=cv2.INTER_LINEAR
                            )
                        vcam.send(vcam_frame)
                        vcam.sleep_until_next_frame()

            else:
                break
    finally:
        if vcam is not None:
            vcam.close()
    cap.stop()
    cv2.destroyAllWindows()
