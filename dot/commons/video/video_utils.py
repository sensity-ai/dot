#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
import random
from typing import Callable, Dict

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import find_files_from_path


def video_pipeline(
    source: str,
    target: str,
    save_folder: str,
    duration: int,
    change_option: Callable[[np.ndarray], None],
    process_image: Callable[[np.ndarray], np.ndarray],
    post_process_image: Callable[[np.ndarray], np.ndarray],
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
        limit (int, optional): Limit number of video-swaps. Defaults to None.
    """
    frame_width = kwargs.get("frame_width", None)
    frame_height = kwargs.get("frame_height", None)
    crop_size = kwargs["opt_crop_size"]

    source_imgs = find_files_from_path(source, ["jpg", "png", "jpeg"])
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
        change_option(img_a_whole)

        img_a_align_crop = process_image(img_a_whole)
        img_a_align_crop = post_process_image(img_a_align_crop)

        # video handle
        cap = cv2.VideoCapture(target)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if frame_width is None or frame_height is None:
            frame_width = int(cap.get(3))  # type: ignore
            frame_height = int(cap.get(4))  # type: ignore

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
        video_writer = cv2.VideoWriter(
            output_file, fourcc, fps, (frame_width, frame_height), True
        )
        print(
            f"Source: {source} \nTarget: {target} \nOutput: {output_file} \nFPS: {fps} \nTotal frames: {total_frames}"
        )

        # process each frame individually
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if ret is True:
                result_frame = process_image(frame, use_cam=False, crop_size=crop_size, **kwargs)  # type: ignore
                result_frame = post_process_image(result_frame, **kwargs)
                video_writer.write(result_frame)
            else:
                break

        cap.release()
        video_writer.release()
