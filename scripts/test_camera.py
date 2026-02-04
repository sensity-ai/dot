#!/usr/bin/env python3
"""
Test script: display camera feed only (no face swap).
Use this to verify the camera and OpenCV window work on your system.
On macOS, the app uses the AVFoundation backend to avoid a black feed.

Usage:
    python scripts/test_camera.py
    python scripts/test_camera.py --camera 0
    python scripts/test_camera.py --camera 1   # if 0 is black (e.g. IR camera)
"""

import sys
import time

import cv2

from dot.commons.video.videocaptureasync import VideoCaptureAsync


def main(camera_id: int = 0):
    print(f"Opening camera {camera_id}...")
    try:
        cap = VideoCaptureAsync(camera_id)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cap.start()
    # Discard first few frames; some cameras need warmup before non-black frames
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            time.sleep(0.05)
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read first frame from camera.")
        cap.stop()
        sys.exit(1)

    cv2.namedWindow("cam", cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("cam", 500, 250)
    cv2.imshow("cam", cv2.flip(frame, 1))
    cv2.waitKey(1)

    print("Camera feed displayed. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break
            cv2.imshow("cam", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Display camera feed only (no face swap).")
    p.add_argument(
    "--camera", "-c", type=int, default=0,
    help="Camera device ID (default: 0). Try 1 if feed is black (e.g. IR camera at 0).",
)
    args = p.parse_args()
    main(camera_id=args.camera)
