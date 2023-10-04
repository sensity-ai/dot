#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import torch

from ..gpen.face_enhancement import FaceEnhancement
from .camera_utils import camera_pipeline, fetch_camera
from .utils import find_images_from_path, generate_random_file_idx, rand_idx_tuple
from .video.video_utils import video_pipeline


class ModelOption(ABC):
    def __init__(
        self,
        gpen_type=None,
        gpen_path="saved_models/gpen",
        use_gpu=True,
        crop_size=256,
    ):

        self.gpen_type = gpen_type
        self.use_gpu = use_gpu
        self.crop_size = crop_size

        if gpen_type:
            if gpen_type == "gpen_512":
                model = {
                    "name": "GPEN-BFR-512",
                    "size": 512,
                    "channel_multiplier": 2,
                    "narrow": 1,
                }
            else:
                model = {
                    "name": "GPEN-BFR-256",
                    "size": 256,
                    "channel_multiplier": 1,
                    "narrow": 0.5,
                }

            self.face_enhancer = FaceEnhancement(
                size=model["size"],
                model=model["name"],
                channel_multiplier=model["channel_multiplier"],
                narrow=model["narrow"],
                use_gpu=self.use_gpu,
                base_dir=gpen_path,
            )

    def generate_from_image(
        self,
        source: Union[str, List],
        target: Union[str, List],
        save_folder: str,
        limit: Optional[int] = None,
        swap_case_idx: Optional[Tuple] = (0, 0),
        **kwargs,
    ) -> Optional[List[Dict]]:
        """_summary_

        Args:
            source (Union[str, List]): A list with source images filepaths, or single image filepath.
            target (Union[str, List]): A list with target images filepaths, or single image filepath.
            save_folder (str): Output path.
            limit (Optional[int], optional): Total number of face-swaps. If None,
            is set to `len(souce)` * `len(target)`. Defaults to None.
            swap_case_idx (Optional[Tuple], optional): Used as keyword among multiple swaps. Defaults to (0, 0).

        Returns:
            List[Dict]: Array of successful and rejected metadata dictionaries
        """

        if not save_folder:
            print("Need to define output folder... Skipping")
            return None

        # source/target can be single file
        if not isinstance(source, list):
            source = find_images_from_path(source)
            target = find_images_from_path(target)

        if not limit:
            # allow all possible swaps
            limit = len(source) * len(target)

        swappedDict = {}
        rejectedDict = {}
        count = 0
        rejected_count = 0
        seen_swaps = []
        source_len = len(source)
        target_len = len(target)
        with torch.no_grad():
            profiler = kwargs.get("profiler", False)
            if not profiler:
                self.create_model(**kwargs)

            while count < limit:
                rand_swap = rand_idx_tuple(source_len, target_len)
                while rand_swap in seen_swaps:
                    rand_swap = rand_idx_tuple(source_len, target_len)

                src_idx = rand_swap[0]
                tar_idx = rand_swap[1]
                src_img = source[src_idx]
                tar_img = target[tar_idx]

                # check if files exits
                if not os.path.exists(src_img) or not os.path.exists(tar_img):
                    print("source/image file does not exist", src_img, tar_img)
                    continue

                # read source image
                source_image = cv2.imread(src_img)
                frame = cv2.imread(tar_img)

                try:
                    self.change_option(source_image)
                    frame = self.process_image(frame, use_cam=False, ignore_error=False)

                    # check if frame == target_image, if it does, image rejected
                    frame = self.post_process_image(frame)

                    # flush image to disk
                    file_idx = generate_random_file_idx(6)
                    file_name = os.path.join(save_folder, f"{file_idx:0>6}.jpg")
                    while os.path.exists(file_name):
                        print(f"Swap id: {file_idx} already exists, generating again.")
                        file_idx = generate_random_file_idx(6)
                        file_name = os.path.join(save_folder, f"{file_idx:0>6}.jpg")

                    cv2.imwrite(file_name, frame)

                    # keep track metadata
                    key = f"{swap_case_idx[1]}{file_idx:0>6}.jpg"
                    swappedDict[key] = {
                        "target": {"path": tar_img, "size": frame.shape},
                        "source": {"path": src_img, "size": source_image.shape},
                    }

                    print(
                        f"{count}: Performed face swap {src_img, tar_img} saved to {file_name}"
                    )

                    # keep track of previous swaps
                    seen_swaps.append(rand_swap)
                    count += 1
                except Exception as e:
                    rejectedDict[rejected_count] = {
                        "target": {"path": tar_img, "size": frame.shape},
                        "source": {"path": src_img, "size": source_image.shape},
                    }
                    rejected_count += 1
                    print(f"Cannot perform face swap {src_img, tar_img}")
                    print(e)
            return [swappedDict, rejectedDict]

    def generate_from_camera(
        self,
        source: str,
        target: int,
        opt_crop_size: int = 224,
        show_fps: bool = False,
        **kwargs: Dict,
    ) -> None:
        """Invokes `camera_pipeline` main-loop.

        Args:
            source (str): Source image filepath.
            target (int): Camera descriptor/ID.
            opt_crop_size (int, optional): Crop size. Defaults to 224.
            show_fps (bool, optional): Show FPS. Defaults to False.
        """
        with torch.no_grad():
            cap = fetch_camera(target)
            self.create_model(opt_crop_size=opt_crop_size, **kwargs)
            camera_pipeline(
                cap,
                source,
                target,
                self.change_option,
                self.process_image,
                self.post_process_image,
                crop_size=opt_crop_size,
                show_fps=show_fps,
            )

    def generate_from_video(
        self,
        source: str,
        target: str,
        save_folder: str,
        duration: int,
        limit: int = None,
        **kwargs: Dict,
    ) -> None:
        """Invokes `video_pipeline` main-loop.

        Args:
            source (str): Source image filepath.
            target (str): Target video filepath.
            save_folder (str): Output folder.
            duration (int): Trim target video in seconds.
            limit (int, optional): Limit number of video-swaps. Defaults to None.
        """
        with torch.no_grad():
            self.create_model(**kwargs)
            video_pipeline(
                source,
                target,
                save_folder,
                duration,
                self.change_option,
                self.process_image,
                self.post_process_image,
                self.crop_size,
                limit,
                **kwargs,
            )

    def post_process_image(self, image, **kwargs):
        if self.gpen_type:
            image, orig_faces, enhanced_faces = self.face_enhancer.process(
                img=image, use_gpu=self.use_gpu
            )

        return image

    @abstractmethod
    def change_option(self, image, **kwargs):
        pass

    @abstractmethod
    def process_image(self, image, **kwargs):
        pass

    @abstractmethod
    def create_model(self, source, target, limit=None, swap_case_idx=0, **kwargs):
        pass
