#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
import random
from typing import Callable, Dict, Union

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from tqdm import tqdm

from ..pose.head_pose import pose_estimation
from ..utils import expand_bbox, find_files_from_path

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,  # model selection
    min_detection_confidence=0.5,  # confidence threshold
)


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

    image_rows, image_cols, _ = image.shape
    results = mp_face.process(image)
    if results.detections is None:
        return -1

    detection = results.detections[0]
    location = detection.location_data
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = _normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows
    )
    rect_end_point = _normalized_to_pixel_coordinates(
        min(relative_bounding_box.xmin + relative_bounding_box.width, 1.0),
        min(relative_bounding_box.ymin + relative_bounding_box.height, 1.0),
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
    for (source, target) in tqdm(swaps_combination):
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
        if crop_size == 256:  # fomm
            frame_width = frame_height = crop_size
        else:
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
        for _ in tqdm(range(total_frames)):
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
