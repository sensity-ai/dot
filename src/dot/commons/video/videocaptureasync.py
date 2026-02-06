#!/usr/bin/env python3
# https://github.com/gilbertfrancois/video-capture-async

import platform
import threading
import time

import cv2

WARMUP_TIMEOUT = 10.0


def camera_backend():
    """Use AVFoundation on macOS for reliable camera frames (avoids black screen)."""
    if platform.system() == "Darwin":
        api = getattr(cv2, "CAP_AVFOUNDATION", None)
        if api is not None:
            return api
    return cv2.CAP_ANY


def list_cameras(max_index=10):
    """Probe for available camera indices. Returns list of (index, resolution_str or None)."""
    backend = camera_backend()
    result = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        # Try to read one frame to confirm the device delivers frames
        ret, frame = cap.read()
        res = None
        if ret and frame is not None:
            h, w = frame.shape[:2]
            res = f"{w}x{h}"
        cap.release()
        result.append((idx, res))
    return result


class VideoCaptureAsync:
    def __init__(self, src=0, width=1920, height=1080):
        self.src = src
        backend = camera_backend()
        self.cap = cv2.VideoCapture(self.src, backend)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def isOpened(self):
        return self.cap.isOpened()

    def start(self):
        if self.started:
            print("[!] Asynchronous video capturing has already been started.")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

        # (warmup) wait for the first successfully grabbed frame
        warmup_start_time = time.time()
        while not self.grabbed:
            warmup_elapsed_time = time.time() - warmup_start_time
            if warmup_elapsed_time > WARMUP_TIMEOUT:
                raise RuntimeError(
                    f"Failed to succesfully grab frame from "
                    f"the camera (timeout={WARMUP_TIMEOUT}s). "
                    f"Try to restart."
                )

            time.sleep(0.5)

        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed or frame is None or frame.size == 0:
                continue
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        while True:
            with self.read_lock:
                frame = self.frame.copy()
                grabbed = self.grabbed
            break
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
