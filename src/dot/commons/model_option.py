#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from ..gpen.face_enhancement import FaceEnhancement
from .camera_utils import camera_pipeline, fetch_camera
from .utils import find_images_from_path, generate_random_file_idx, rand_idx_tuple
from .video.video_utils import video_pipeline

# Lazy-loaded face detector for "effects on face only" (OpenCV cascade)
_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = os.path.join(
            os.path.dirname(cv2.__file__),
            "data",
            "haarcascade_frontalface_default.xml",
        )
        if not os.path.isfile(path):
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def _get_face_only_mask(image: np.ndarray, expand: float = 1.15, blur: int = 35) -> Optional[np.ndarray]:
    """
    Detect face(s) and return a soft mask (0-1 float) that is 1 on the face region.
    Used to apply tint/effects only to the face. Returns None if no face detected.

    Args:
        image: BGR image (H, W, 3).
        expand: Scale factor for face bbox (slightly larger than detection).
        blur: Kernel size for soft mask edges (odd integer).

    Returns:
        Mask (H, W, 1) float32 in [0, 1], or None if no face.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    if len(faces) == 0:
        return None
    mask = np.zeros((h, w), dtype=np.float32)
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw / 2, y + fh / 2
        ax = (fw / 2) * expand
        ay = (fh / 2) * expand
        cv2.ellipse(
            mask,
            (int(cx), int(cy)),
            (int(ax), int(ay)),
            0, 0, 360, 1.0, -1,
        )
    if blur > 0 and blur % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask = np.clip(mask, 0, 1)
    return mask[:, :, np.newaxis].astype(np.float32)


def _mask_from_bbox(
    image_shape: Tuple[int, ...],
    bbox: Union[List[float], np.ndarray],
    expand: float = 1.15,
    blur: int = 35,
) -> np.ndarray:
    """
    Build a soft face mask from a bounding box [x1, y1, x2, y2].
    Used so tint/effects can follow the face region from the swap model.

    Args:
        image_shape: (H, W) or (H, W, C).
        bbox: [x1, y1, x2, y2] in image coordinates.
        expand: Scale factor for ellipse axes (slightly larger than bbox).
        blur: Gaussian blur kernel size for soft edges (odd).

    Returns:
        Mask (H, W, 1) float32 in [0, 1].
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ax = ((x2 - x1) / 2) * expand
    ay = ((y2 - y1) / 2) * expand
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        mask,
        (int(round(cx)), int(round(cy))),
        (max(1, int(round(ax))), max(1, int(round(ay)))),
        0, 0, 360, 1.0, -1,
    )
    if blur > 0 and blur % 2 == 1:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    mask = np.clip(mask, 0, 1)
    return mask[:, :, np.newaxis].astype(np.float32)


class ModelOption(ABC):
    def __init__(
        self,
        gpen_type=None,
        gpen_path="saved_models/gpen",
        use_gpu=True,
        crop_size=256,
        filter_type=None,
    ):

        self.gpen_type = gpen_type
        self.use_gpu = use_gpu
        self.crop_size = crop_size
        self.filter_type = filter_type

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
            cam_width = kwargs.pop("cam_width", 1920)
            cam_height = kwargs.pop("cam_height", 1080)
            cap = fetch_camera(target, width=cam_width, height=cam_height)
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
                model_option=self,
                **kwargs,
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

        # Apply camera/film filter
        if self.filter_type:
            from ..filters import apply_filter

            image = apply_filter(image, self.filter_type)

        # Live settings (tint, brightness, saturation, contrast) from GUI
        live = kwargs.get("live_settings")
        if live:
            from ..filters.effects import (
                add_color_tint,
                adjust_brightness,
                adjust_contrast,
                adjust_saturation,
            )

            effects_face_only = live.get("effects_face_only", False)
            original = image.copy()

            b = live.get("brightness", 1.0)
            if b != 1.0:
                image = adjust_brightness(image, factor=b)
            c = live.get("contrast", 1.0)
            if c != 1.0:
                image = adjust_contrast(image, factor=c)
            s = live.get("saturation", 1.0)
            if s != 1.0:
                image = adjust_saturation(image, factor=s)
            ti = live.get("tint_intensity", 0.0)
            if ti > 0:
                color = (
                    live.get("tint_b", 128),
                    live.get("tint_g", 128),
                    live.get("tint_r", 128),
                )
                image = add_color_tint(image, color=color, intensity=ti)

            if effects_face_only:
                # Prefer mask from the face-swap model (same region as the swap)
                face_mask = kwargs.get("_face_mask", {}).get("mask")
                if face_mask is None or face_mask.shape[:2] != image.shape[:2]:
                    face_mask = _get_face_only_mask(image)
                if face_mask is not None:
                    image = (
                        image.astype(np.float32) * face_mask
                        + original.astype(np.float32) * (1.0 - face_mask)
                    ).astype(np.uint8)
                # If no face detected, keep full-frame effects

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
