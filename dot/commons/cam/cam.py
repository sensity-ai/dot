#!/usr/bin/env python3

import glob
import os

import cv2
import numpy as np
import requests
import yaml

from ..utils import info, resize
from .camera_selector import query_cameras


def is_new_frame_better(log, source, driving, predictor):
    global avatar_kp
    global display_string

    if avatar_kp is None:
        display_string = "No face detected in avatar."
        return False

    if predictor.get_start_frame() is None:
        display_string = "No frame to compare to."
        return True

    _ = resize(driving, (128, 128))[..., :3]
    new_kp = predictor.get_frame_kp(driving)

    if new_kp is not None:
        new_norm = (np.abs(avatar_kp - new_kp) ** 2).sum()
        old_norm = (np.abs(avatar_kp - predictor.get_start_frame_kp()) ** 2).sum()

        out_string = "{0} : {1}".format(int(new_norm * 100), int(old_norm * 100))
        display_string = out_string
        log(out_string)

        return new_norm < old_norm
    else:
        display_string = "No face found!"
        return False


def load_stylegan_avatar(IMG_SIZE=256):

    url = "https://thispersondoesnotexist.com/image"
    r = requests.get(url, headers={"User-Agent": "My User Agent 1.0"}).content

    image = np.frombuffer(r, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = resize(image, (IMG_SIZE, IMG_SIZE))

    return image


def load_images(log, opt_avatars, IMG_SIZE=256):
    avatars = []
    filenames = []
    images_list = sorted(glob.glob(f"{opt_avatars}/*"))
    for i, f in enumerate(images_list):
        if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
            img = cv2.imread(f)
            if img is None:
                log("Failed to open image: {}".format(f))
                continue

            if img.ndim == 2:
                img = np.tile(img[..., None], [1, 1, 3])
            img = img[..., :3][..., ::-1]
            img = resize(img, (IMG_SIZE, IMG_SIZE))
            avatars.append(img)
            filenames.append(f)
    return avatars, filenames


def draw_rect(img, rw=0.6, rh=0.8, color=(255, 0, 0), thickness=2):
    h, w = img.shape[:2]
    _l = w * (1 - rw) // 2
    r = w - _l
    u = h * (1 - rh) // 2
    d = h - u
    img = cv2.rectangle(img, (int(_l), int(u)), (int(r), int(d)), color, thickness)


def kp_to_pixels(arr):
    """Convert normalized landmark locations to screen pixels"""
    return ((arr + 1) * 127).astype(np.int32)


def draw_face_landmarks(LANDMARK_SLICE_ARRAY, img, face_kp, color=(20, 80, 255)):

    if face_kp is not None:
        img = cv2.polylines(
            img, np.split(kp_to_pixels(face_kp), LANDMARK_SLICE_ARRAY), False, color
        )


def print_help(avatar_names):
    info("\n\n=== Control keys ===")
    info("1-9: Change avatar")
    for i, fname in enumerate(avatar_names):
        key = i + 1
        name = fname.split("/")[-1]
        info(f"{key}: {name}")
    info("W: Zoom camera in")
    info("S: Zoom camera out")
    info("A: Previous avatar in folder")
    info("D: Next avatar in folder")
    info("Q: Get random avatar")
    info("X: Calibrate face pose")
    info("I: Show FPS")
    info("ESC: Quit")
    info("\nFull key list: https://github.com/alievk/avatarify#controls")
    info("\n\n")


def draw_fps(
    frame,
    fps,
    timing,
    x0=10,
    y0=20,
    ystep=30,
    fontsz=0.5,
    color=(255, 255, 255),
    IMG_SIZE=256,
):

    frame = frame.copy()
    black = (0, 0, 0)
    black_thick = 2

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (x0, y0 + ystep * 0),
        0,
        fontsz * IMG_SIZE / 256,
        (0, 0, 0),
        black_thick,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (x0, y0 + ystep * 0),
        0,
        fontsz * IMG_SIZE / 256,
        color,
        1,
    )
    cv2.putText(
        frame,
        f"Model time (ms): {timing['predict']:.1f}",
        (x0, y0 + ystep * 1),
        0,
        fontsz * IMG_SIZE / 256,
        black,
        black_thick,
    )
    cv2.putText(
        frame,
        f"Model time (ms): {timing['predict']:.1f}",
        (x0, y0 + ystep * 1),
        0,
        fontsz * IMG_SIZE / 256,
        color,
        1,
    )
    cv2.putText(
        frame,
        f"Preproc time (ms): {timing['preproc']:.1f}",
        (x0, y0 + ystep * 2),
        0,
        fontsz * IMG_SIZE / 256,
        black,
        black_thick,
    )
    cv2.putText(
        frame,
        f"Preproc time (ms): {timing['preproc']:.1f}",
        (x0, y0 + ystep * 2),
        0,
        fontsz * IMG_SIZE / 256,
        color,
        1,
    )
    cv2.putText(
        frame,
        f"Postproc time (ms): {timing['postproc']:.1f}",
        (x0, y0 + ystep * 3),
        0,
        fontsz * IMG_SIZE / 256,
        black,
        black_thick,
    )
    cv2.putText(
        frame,
        f"Postproc time (ms): {timing['postproc']:.1f}",
        (x0, y0 + ystep * 3),
        0,
        fontsz * IMG_SIZE / 256,
        color,
        1,
    )
    return frame


def draw_landmark_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255), IMG_SIZE=256):

    frame = frame.copy()
    cv2.putText(frame, "ALIGN FACES", (60, 20), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(
        frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk
    )
    return frame


def draw_calib_text(frame, thk=2, fontsz=0.5, color=(0, 0, 255), IMG_SIZE=256):
    frame = frame.copy()
    cv2.putText(
        frame, "FIT FACE IN RECTANGLE", (40, 20), 0, fontsz * IMG_SIZE / 255, color, thk
    )
    cv2.putText(frame, "W - ZOOM IN", (60, 40), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(frame, "S - ZOOM OUT", (60, 60), 0, fontsz * IMG_SIZE / 255, color, thk)
    cv2.putText(
        frame, "THEN PRESS X", (60, 245), 0, fontsz * IMG_SIZE / 255, color, thk
    )
    return frame


def select_camera(log, config):
    cam_config = config["cam_config"]
    cam_id = None

    if os.path.isfile(cam_config):
        with open(cam_config, "r") as f:
            cam_config = yaml.load(f, Loader=yaml.FullLoader)
            cam_id = cam_config["cam_id"]
    else:
        cam_frames = query_cameras(config["query_n_cams"])

        if cam_frames:
            if len(cam_frames) == 1:
                cam_id = list(cam_frames)[0]
            else:
                cam_id = select_camera(cam_frames, window="CLICK ON YOUR CAMERA")
            log(f"Selected camera {cam_id}")

            with open(cam_config, "w") as f:
                yaml.dump({"cam_id": cam_id}, f)
        else:
            log("No cameras are available")

    return cam_id
