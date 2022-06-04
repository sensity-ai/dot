#!/usr/bin/env python3

import cv2
import dlib
import numpy as np

from ..commons import ModelOption
from ..commons.utils import crop, resize
from ..faceswap_cv2.swap import Swap


class FaceswapCVOption(ModelOption):
    def __init__(
        self,
        use_gpu=True,
        use_mask=False,
        crop_size=224,
        gpen_type=None,
        gpen_path=None,
    ):
        super(FaceswapCVOption, self).__init__(
            gpen_type=gpen_type,
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_path=gpen_path,
        )
        self.frame_proportion = 0.9
        self.frame_offset_x = 0
        self.frame_offset_y = 0

    def create_model(self, model_path, **kwargs) -> None:  # type: ignore

        self.model = Swap(
            predictor_path=model_path, end=68, warp_2d=False, correct_color=True
        )

        self.detector = dlib.get_frontal_face_detector()

    def change_option(self, image, **kwargs):
        self.source_image = image
        self.src_landmarks, self.src_shape, self.src_face = self.model._process_face(
            image
        )

    def process_image(
        self, image, use_cam=True, ignore_error=True, **kwargs
    ) -> np.array:
        frame = image[..., ::-1]

        if use_cam:
            frame, (self.frame_offset_x, self.frame_offset_y) = crop(
                frame,
                p=self.frame_proportion,
                offset_x=self.frame_offset_x,
                offset_y=self.frame_offset_y,
            )
            frame = resize(frame, (self.crop_size, self.crop_size))[..., :3]
            frame = cv2.flip(frame, 1)

        faces = self.detector(frame[..., ::-1])
        if len(faces) > 0:
            try:
                swapped_img = self.model.apply_face_swap(
                    source_image=self.source_image,
                    target_image=frame,
                    save_path=None,
                    src_landmarks=self.src_landmarks,
                    src_shape=self.src_shape,
                    src_face=self.src_face,
                )

                swapped_img = np.array(swapped_img)[..., ::-1].copy()
            except Exception as e:
                if ignore_error:
                    print(e)
                    swapped_img = frame[..., ::-1].copy()
                else:
                    raise e
        else:
            swapped_img = frame[..., ::-1].copy()

        return swapped_img
