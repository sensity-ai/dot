#!/usr/bin/env python3

from typing import Any, Dict

import cv2
import dlib
import numpy as np
from PIL import Image

from .generic import (
    apply_mask,
    correct_colours,
    mask_from_points,
    transformation_from_points,
    warp_image_2d,
    warp_image_3d,
)

# define globals
CACHED_PREDICTOR_PATH = "saved_models/faceswap_cv/shape_predictor_68_face_landmarks.dat"


class Swap:
    def __init__(
        self,
        predictor_path: str = None,
        warp_2d: bool = True,
        correct_color: bool = True,
        end: int = 48,
    ):
        """
        Face Swap.
        @description:
            perform face swapping using Poisson blending
        @arguments:
            predictor_path: (str) path to 68-point facial landmark detector
            warp_2d: (bool) if True, perform 2d warping for swapping
            correct_color: (bool) if True, color correct swap output image
            end: (int) last facial landmark point for face swap
        """
        if not predictor_path:
            predictor_path = CACHED_PREDICTOR_PATH

        # init
        self.predictor_path = predictor_path
        self.warp_2d = warp_2d
        self.correct_color = correct_color
        self.end = end

        # Load dlib models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def apply_face_swap(self, source_image, target_image, save_path=None, **kwargs):
        """
        apply face swapping from source to target image
        @arguments:
            source_image: (PIL or str) source PIL image or path to source image
            target_image: (PIL or str) target PIL image or path to target image
            save_path: (str) path to save face swap output image (optional)
            **kwargs: Extra argument for specifying the source and target landmarks, shape and face
        @returns:
            faceswap_output_image: (PIL) face swap output image
        """
        # load image if path given, else convert to cv2 format
        if isinstance(source_image, str):
            source_image_cv2 = cv2.imread(source_image)
        else:
            source_image_cv2 = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
        if isinstance(target_image, str):
            target_image_cv2 = cv2.imread(target_image)
        else:
            target_image_cv2 = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)

        # process source image
        try:
            src_landmarks = kwargs["src_landmarks"]
            src_shape = kwargs["src_shape"]
            src_face = kwargs["src_face"]
        except Exception as e:
            print(e)
            src_landmarks, src_shape, src_face = self._process_face(source_image_cv2)

        # process target image
        trg_landmarks, trg_shape, trg_face = self._process_face(target_image_cv2)

        # get target face dimensions
        h, w = trg_face.shape[:2]

        # 3d warp
        warped_src_face = warp_image_3d(
            src_face, src_landmarks[: self.end], trg_landmarks[: self.end], (h, w)
        )

        # Mask for blending
        mask = mask_from_points((h, w), trg_landmarks)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)

        # Correct color
        if self.correct_color:
            warped_src_face = apply_mask(warped_src_face, mask)
            dst_face_masked = apply_mask(trg_face, mask)
            warped_src_face = correct_colours(
                dst_face_masked, warped_src_face, trg_landmarks
            )

        # 2d warp
        if self.warp_2d:
            unwarped_src_face = warp_image_3d(
                warped_src_face,
                trg_landmarks[: self.end],
                src_landmarks[: self.end],
                src_face.shape[:2],
            )
            warped_src_face = warp_image_2d(
                unwarped_src_face,
                transformation_from_points(trg_landmarks, src_landmarks),
                (h, w, 3),
            )

            mask = mask_from_points((h, w), trg_landmarks)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask * mask_src, dtype=np.uint8)

        # perform base blending operation
        faceswap_output_cv2 = self._perform_base_blending(
            mask, trg_face, warped_src_face
        )

        x, y, w, h = trg_shape
        target_faceswap_img = target_image_cv2.copy()
        target_faceswap_img[y : y + h, x : x + w] = faceswap_output_cv2

        faceswap_output_image = Image.fromarray(
            cv2.cvtColor(target_faceswap_img, cv2.COLOR_BGR2RGB)
        )

        if save_path:
            faceswap_output_image.save(save_path, compress_level=0)

        return faceswap_output_image

    def _face_and_landmark_detection(self, image):
        """perform face detection and get facial landmarks"""
        # get face bounding box
        faces = self.detector(image)
        idx = np.argmax(
            [
                (face.right() - face.left()) * (face.bottom() - face.top())
                for face in faces
            ]
        )
        bbox = faces[idx]

        # predict landmarks
        landmarks_dlib = self.predictor(image=image, box=bbox)
        face_landmarks = np.array([[p.x, p.y] for p in landmarks_dlib.parts()])

        return face_landmarks

    def _process_face(self, image, r=10):
        """process detected face and landmarks"""
        # get landmarks
        landmarks = self._face_and_landmark_detection(image)

        # get image dimensions
        im_w, im_h = image.shape[:2]

        # get face edges
        left, top = np.min(landmarks, 0)
        right, bottom = np.max(landmarks, 0)

        # scale landmarks and face edges
        x, y = max(0, left - r), max(0, top - r)
        w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

        return (
            landmarks - np.asarray([[x, y]]),
            (x, y, w, h),
            image[y : y + h, x : x + w],
        )

    @staticmethod
    def _perform_base_blending(mask, trg_face, warped_src_face):
        """perform Poisson blending using mask"""

        # Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        # Poisson Blending
        r = cv2.boundingRect(mask)
        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))

        output_cv2 = cv2.seamlessClone(
            warped_src_face, trg_face, mask, center, cv2.NORMAL_CLONE
        )
        return output_cv2

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Swap":
        """
        Instantiates a Swap from a configuration.
        Args:
            config: A configuration for a Swap.
        Returns:
            A Swap instance.
        """
        # get config
        swap_config = config.get("swap")

        # return instance
        return cls(
            predictor_path=swap_config.get("predictor_path", CACHED_PREDICTOR_PATH),
            warp_2d=swap_config.get("warp_2d", True),
            correct_color=swap_config.get("correct_color", True),
            end=swap_config.get("end", 48),
        )
