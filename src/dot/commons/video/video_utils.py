#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import math
import os
import random
import urllib.request
from typing import Callable, Dict, Optional, Tuple, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ..pose.head_pose import pose_estimation
from ..utils import expand_bbox, find_files_from_path

# Model file path and download URL for face detection
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "saved_models", "mediapipe")
_FACE_DETECTOR_MODEL_PATH = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")
_FACE_DETECTOR_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"


def _ensure_face_detector_model_downloaded():
    """Download the face detector model if not present."""
    if os.path.exists(_FACE_DETECTOR_MODEL_PATH):
        return
    os.makedirs(_MODEL_DIR, exist_ok=True)
    print(f"Downloading face detector model to {_FACE_DETECTOR_MODEL_PATH}...")
    urllib.request.urlretrieve(_FACE_DETECTOR_MODEL_URL, _FACE_DETECTOR_MODEL_PATH)
    print("Download complete.")


def _get_face_detector():
    """Create and return a FaceDetector instance."""
    _ensure_face_detector_model_downloaded()
    base_options = python.BaseOptions(model_asset_path=_FACE_DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


# Lazy initialization of face detector
_face_detector = None


def _get_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = _get_face_detector()
    return _face_detector


def _normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates (MediaPipe-compatible)."""
    if not (0 <= normalized_x <= 1 and 0 <= normalized_y <= 1):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return (x_px, y_px)


def _crop_and_pose(
    image: np.array, estimate_pose: bool = False
) -> Union[np.array, int]:
    """Crops face of `image` and estimates head pose.

    Args:
        image (np.array): Image to be cropped and estimate pose.
        estimate_pose (Boolean, optional): Enables pose estimation. Defaults to False.

    Returns:
        Union[np.array,int]: Cropped image or -1.
    """
    detector = _get_detector()

    image_rows, image_cols, _ = image.shape

    # Convert BGR to RGB for mediapipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    results = detector.detect(mp_image)
    if not results.detections:
        return -1

    detection = results.detections[0]
    bbox = detection.bounding_box

    # Convert pixel coordinates to normalized, then back to pixel
    # The new API returns pixel coordinates directly in bounding_box
    xmin = bbox.origin_x / image_cols
    ymin = bbox.origin_y / image_rows
    width = bbox.width / image_cols
    height = bbox.height / image_rows

    rect_start_point = _normalized_to_pixel_coordinates(
        xmin, ymin, image_cols, image_rows
    )
    rect_end_point = _normalized_to_pixel_coordinates(
        min(xmin + width, 1.0),
        min(ymin + height, 1.0),
        image_cols,
        image_rows,
    )

    xleft, ytop = rect_start_point
    xright, ybot = rect_end_point

    xleft, ytop, xright, ybot = expand_bbox(
        (xleft, ytop, xright, ybot), image_rows, image_cols, 2.0
    )

    try:
        crop_image = image[ytop:ybot, xleft:xright]
        if estimate_pose:
            if pose_estimation(image=crop_image, roll=3, pitch=3, yaw=3) != 0:
                return -1

        return cv2.flip(crop_image, 1)
    except Exception as e:
        print(e)
        return -1


def video_pipeline(
    source: str,
    target: str,
    save_folder: str,
    duration: int,
    change_option: Callable[[np.ndarray], None],
    process_image: Callable[[np.ndarray], np.ndarray],
    post_process_image: Callable[[np.ndarray], np.ndarray],
    crop_size: int = 224,
    limit: int = None,
    **kwargs: Dict,
) -> None:
    """Process input video file `target` by frame and performs face-swap based on first image
    found in `source` path folder. Uses cv2.VideoWriter to flush the resulted video on disk.
    Trimming video is done as: trimmed = fps * duration.

    Args:
        source (str): Path to source image folder.
        target (str): Path to target video folder.
        save_folder (str): Output folder path.
        duration (int): Crop target video in seconds.
        change_option (Callable[[np.ndarray], None]): Set `source` arg as faceswap source image.
        process_image (Callable[[np.ndarray], np.ndarray]): Performs actual face swap.
        post_process_image (Callable[[np.ndarray], np.ndarray]): Applies face restoration GPEN to result image.
        head_pose (bool): Estimates head pose before swap. Used by Avatarify.
        crop_size (int, optional): Face crop size. Defaults to 224.
        limit (int, optional): Limit number of video-swaps. Defaults to None.
    """
    head_pose = kwargs.get("head_pose", False)
    source_imgs = find_files_from_path(source, ["jpg", "png", "jpeg"], filter=None)
    target_videos = find_files_from_path(target, ["avi", "mp4", "mov", "MOV"])
    if not source_imgs or not target_videos:
        print("Could not find any source/target files")
        return

    # unique combinations of source/target
    swaps_combination = [(im, vi) for im in source_imgs for vi in target_videos]
    # randomize list
    random.shuffle(swaps_combination)
    if limit:
        swaps_combination = swaps_combination[:limit]

    print("Total source images: ", len(source_imgs))
    print("Total target videos: ", len(target_videos))
    print("Total number of face-swaps: ", len(swaps_combination))

    # iterate on each source-target pair
    for (source, target) in swaps_combination:
        img_a_whole = cv2.imread(source)
        img_a_whole = _crop_and_pose(img_a_whole, estimate_pose=head_pose)
        if isinstance(img_a_whole, int):
            print(
                f"Image {source} failed on face detection or pose estimation requirements haven't met."
            )
            continue

        change_option(img_a_whole)

        img_a_align_crop = process_image(img_a_whole)
        img_a_align_crop = post_process_image(img_a_align_crop)

        # video handle
        cap = cv2.VideoCapture(target)

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # trim original video length
        if duration and (fps * int(duration)) < total_frames:
            total_frames = fps * int(duration)

        # result video is saved in `save_folder` with name combining source/target files.
        source_base_name = os.path.basename(source)
        target_base_name = os.path.basename(target)
        output_file = f"{os.path.splitext(source_base_name)[0]}_{os.path.splitext(target_base_name)[0]}.mp4"
        output_file = os.path.join(save_folder, output_file)

        fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
        video_writer = cv2.VideoWriter(
            output_file, fourcc, fps, (frame_width, frame_height), True
        )
        print(
            f"Source: {source} \nTarget: {target} \nOutput: {output_file} \nFPS: {fps} \nTotal frames: {total_frames}"
        )

        # process each frame individually
        for _ in range(total_frames):
            ret, frame = cap.read()
            if ret is True:
                frame = cv2.flip(frame, 1)
                result_frame = process_image(frame, use_cam=False, crop_size=crop_size, **kwargs)  # type: ignore
                result_frame = post_process_image(result_frame, **kwargs)
                video_writer.write(result_frame)
            else:
                break

        cap.release()
        video_writer.release()
