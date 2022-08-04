#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

from typing import Any, Callable, Dict, List, Union

import cv2
import numpy as np

from .cam.cam import draw_fps
from .utils import TicToc, find_images_from_path
from .video.videocaptureasync import VideoCaptureAsync


def fetch_camera(target: int) -> VideoCaptureAsync:
    """Fetches a VideoCaptureAsync object.

    Args:
        target (int): Camera ID descriptor.

    Raises:
        ValueError: If camera ID descriptor is not valid.

    Returns:
        VideoCaptureAsync: VideoCaptureAsync object.
    """
    try:
        return VideoCaptureAsync(target)
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
    """
    source = find_images_from_path(source)
    print("=== Control keys ===")
    print("1-9: Change avatar")
    for i, fname in enumerate(source):
        print(str(i + 1) + ": " + fname)

    # Todo describe controls available

    pic_a = source[0]

    img_a_whole = cv2.imread(pic_a)
    change_option(img_a_whole)

    img_a_align_crop = process_image(img_a_whole)
    img_a_align_crop = post_process_image(img_a_align_crop)

    cap.start()
    ret, frame = cap.read()
    cv2.namedWindow("cam", cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("cam", 500, 250)

    frame_index = -1
    fps_hist: List = []
    fps: Union[Any, float] = 0

    show_self = False
    while True:
        frame_index += 1
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            tt = TicToc()

            timing = {"preproc": 0, "predict": 0, "postproc": 0}

            tt.tic()

            key = cv2.waitKey(1)
            if 48 < key < 58:
                show_self = False
                source_image_i = min(key - 49, len(source) - 1)
                pic_a = source[source_image_i]
                img_a_whole = cv2.imread(pic_a)
                change_option(img_a_whole, **kwargs)
            elif key == ord("y"):
                show_self = True

            elif key == ord("q"):
                break
            elif key == ord("i"):
                show_fps = not show_fps

            if not show_self:
                result_frame = process_image(frame, crop_size=crop_size, **kwargs)  # type: ignore
                timing["postproc"] = tt.toc()
                result_frame = post_process_image(result_frame, **kwargs)

                if show_fps:
                    result_frame = draw_fps(np.array(result_frame), fps, timing)

                fps_hist.append(tt.toc(total=True))
                if len(fps_hist) == 10:
                    fps = 10 / (sum(fps_hist) / 1000)
                    fps_hist = []

                cv2.imshow("cam", result_frame)

            else:
                cv2.imshow("cam", frame)

        else:
            break
    cap.stop()
    cv2.destroyAllWindows()
