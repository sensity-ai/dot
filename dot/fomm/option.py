#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np

from ..commons import ModelOption
from ..commons.cam.cam import (
    draw_calib_text,
    draw_face_landmarks,
    draw_landmark_text,
    draw_rect,
    is_new_frame_better,
)
from ..commons.utils import crop, log, pad_img, resize
from .predictor_local import PredictorLocal


def determine_path():
    """
    Find the script path
    """
    try:
        root = __file__
        if os.path.islink(root):
            root = os.path.realpath(root)

        return os.path.dirname(os.path.abspath(root))
    except Exception as e:
        print(e)
        print("I'm sorry, but something is wrong.")
        print("There is no __file__ variable. Please contact the author.")
        sys.exit()


class FOMMOption(ModelOption):
    def __init__(
        self,
        use_gpu: bool = True,
        use_mask: bool = False,
        crop_size: int = 256,
        gpen_type: str = None,
        gpen_path: str = None,
        offline: bool = False,
    ):
        super(FOMMOption, self).__init__(
            gpen_type=gpen_type,
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_path=gpen_path,
        )
        # use FOMM offline, video or image file
        self.offline = offline
        self.frame_proportion = 0.9
        self.frame_offset_x = 0
        self.frame_offset_y = 0

        self.overlay_alpha = 0.0
        self.preview_flip = False
        self.output_flip = False
        self.find_keyframe = False
        self.is_calibrated = True if self.offline else False

        self.show_landmarks = False
        self.passthrough = False
        self.green_overlay = False
        self.opt_relative = True
        self.opt_adapt_scale = True
        self.opt_enc_downscale = 1
        self.opt_no_pad = True
        self.opt_in_port = 5557
        self.opt_out_port = 5558
        self.opt_hide_rect = False
        self.opt_in_addr = None
        self.opt_out_addr = None
        self.LANDMARK_SLICE_ARRAY = np.array([17, 22, 27, 31, 36, 42, 48, 60])
        self.display_string = ""

    def create_model(self, model_path, **kwargs) -> None:  # type: ignore
        opt_config = determine_path() + "/config/vox-adv-256.yaml"
        opt_checkpoint = model_path

        predictor_args = {
            "config_path": opt_config,
            "checkpoint_path": opt_checkpoint,
            "relative": self.opt_relative,
            "adapt_movement_scale": self.opt_adapt_scale,
            "enc_downscale": self.opt_enc_downscale,
        }

        self.predictor = PredictorLocal(**predictor_args)

    def change_option(self, image, **kwargs):
        if image.ndim == 2:
            image = np.tile(image[..., None], [1, 1, 3])
        image = image[..., :3][..., ::-1]
        image = resize(image, (self.crop_size, self.crop_size))
        print("Image shape ", image.shape)
        self.source_kp = self.predictor.get_frame_kp(image)
        self.kp_source = None
        self.predictor.set_source_image(image)
        self.source_image = image

    def handle_keyboard_input(self):
        key = cv2.waitKey(1)

        if key == ord("w"):
            self.frame_proportion -= 0.05
            self.frame_proportion = max(self.frame_proportion, 0.1)
        elif key == ord("s"):
            self.frame_proportion += 0.05
            self.frame_proportion = min(self.frame_proportion, 1.0)
        elif key == ord("H"):
            self.frame_offset_x -= 1
        elif key == ord("h"):
            self.frame_offset_x -= 5
        elif key == ord("K"):
            self.frame_offset_x += 1
        elif key == ord("k"):
            self.frame_offset_x += 5
        elif key == ord("J"):
            self.frame_offset_y -= 1
        elif key == ord("j"):
            self.frame_offset_y -= 5
        elif key == ord("U"):
            self.frame_offset_y += 1
        elif key == ord("u"):
            self.frame_offset_y += 5
        elif key == ord("Z"):
            self.frame_offset_x = 0
            self.frame_offset_y = 0
            self.frame_proportion = 0.9
        elif key == ord("x"):
            self.predictor.reset_frames()

            if not self.is_calibrated:
                cv2.namedWindow("FOMM", cv2.WINDOW_GUI_NORMAL)
                cv2.moveWindow("FOMM", 600, 250)

            self.is_calibrated = True
            self.show_landmarks = False
        elif key == ord("z"):
            self.overlay_alpha = max(self.overlay_alpha - 0.1, 0.0)
        elif key == ord("c"):
            self.overlay_alpha = min(self.overlay_alpha + 0.1, 1.0)
        elif key == ord("r"):
            self.preview_flip = not self.preview_flip
        elif key == ord("t"):
            self.output_flip = not self.output_flip
        elif key == ord("f"):
            self.find_keyframe = not self.find_keyframe
        elif key == ord("o"):
            self.show_landmarks = not self.show_landmarks
        elif key == 48:
            self.passthrough = not self.passthrough
        elif key != -1:
            log(key)

    def process_image(self, image, use_gpu=True, **kwargs) -> np.array:
        if not self.offline:
            self.handle_keyboard_input()

        stream_img_size = image.shape[1], image.shape[0]

        frame = image[..., ::-1]

        frame, (frame_offset_x, frame_offset_y) = crop(
            frame,
            p=self.frame_proportion,
            offset_x=self.frame_offset_x,
            offset_y=self.frame_offset_y,
        )

        frame = resize(frame, (self.crop_size, self.crop_size))[..., :3]

        if self.find_keyframe:
            if is_new_frame_better(log, self.source_image, frame, self.predictor):
                log("Taking new frame!")
                self.green_overlay = True
                self.predictor.reset_frames()

        if self.passthrough:
            out = frame
        elif self.is_calibrated:
            out = self.predictor.predict(frame)
            if out is None:
                log("predict returned None")
        else:
            out = None

        if self.overlay_alpha > 0:
            preview_frame = cv2.addWeighted(
                self.source_image,
                self.overlay_alpha,
                frame,
                1.0 - self.overlay_alpha,
                0.0,
            )
        else:
            preview_frame = frame.copy()

        if self.show_landmarks:
            # Dim the background to make it easier to see the landmarks
            preview_frame = cv2.convertScaleAbs(preview_frame, alpha=0.5, beta=0.0)

            draw_face_landmarks(
                self.LANDMARK_SLICE_ARRAY, preview_frame, self.source_kp, (200, 20, 10)
            )

            frame_kp = self.predictor.get_frame_kp(frame)
            draw_face_landmarks(self.LANDMARK_SLICE_ARRAY, preview_frame, frame_kp)

        preview_frame = cv2.flip(preview_frame, 1)

        if self.green_overlay:
            green_alpha = 0.8
            overlay = preview_frame.copy()
            overlay[:] = (0, 255, 0)
            preview_frame = cv2.addWeighted(
                preview_frame, green_alpha, overlay, 1.0 - green_alpha, 0.0
            )

        if self.find_keyframe:
            preview_frame = cv2.putText(
                preview_frame,
                self.display_string,
                (10, 220),
                0,
                0.5 * self.crop_size / 256,
                (255, 255, 255),
                1,
            )

        if not self.is_calibrated:
            preview_frame = draw_calib_text(preview_frame)

        elif self.show_landmarks:
            preview_frame = draw_landmark_text(preview_frame)

        if not self.opt_hide_rect:
            draw_rect(preview_frame)

        if not self.offline:
            cv2.imshow("FOMM", preview_frame[..., ::-1])

        if out is not None:
            if not self.opt_no_pad:
                out = pad_img(out, stream_img_size)

            if self.output_flip:
                out = cv2.flip(out, 1)

            return out[..., ::-1]
        else:
            return preview_frame[..., ::-1]
