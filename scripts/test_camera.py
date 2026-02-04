#!/usr/bin/env python3
"""
Test script: display camera feed only (no face swap).
Use this to verify the camera and OpenCV window work on your system.
On macOS, the app uses the AVFoundation backend to avoid a black feed.

Interactive: type in the terminal (then Enter): n = next, p = previous, 0-9 = jump, q = quit.
On macOS the OpenCV window often doesn't get keyboard focus, so input is read from the terminal.

Usage:
    python scripts/test_camera.py
    python scripts/test_camera.py --list          # list available cameras and exit
    python scripts/test_camera.py --camera 0
    python scripts/test_camera.py --camera 1       # if 0 is black (e.g. IR camera)
"""

import queue
import sys
import threading
import time

import cv2

from dot.commons.video.videocaptureasync import VideoCaptureAsync, list_cameras


def print_devices(devices):
    """Print available camera devices (list of (index, res))."""
    if not devices:
        print("No camera devices found.")
        return
    print(f"Found {len(devices)} camera(s):")
    for idx, res in devices:
        res_str = f" {res}" if res else " (no frame)"
        print(f"  [{idx}] Camera {idx}{res_str}")
    print("Use --camera <index> to choose a device.")


def warmup(cap, n=5):
    """Discard first few frames so the camera delivers non-black frames."""
    for _ in range(n):
        cap.read()
        time.sleep(0.05)


def open_camera(device_index: int):
    """Open and start camera at given device index; return cap or None."""
    try:
        cap = VideoCaptureAsync(device_index)
        cap.start()
        warmup(cap)
        return cap
    except RuntimeError as e:
        print(f"Error opening camera {device_index}: {e}")
        return None


def start_terminal_input_queue():
    """Start a daemon thread that reads single-char commands from the terminal into a queue."""
    cmd_queue = queue.Queue()

    def read_stdin():
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip().lower()
                if line:
                    cmd_queue.put(line[0])
        except (EOFError, KeyboardInterrupt, OSError):
            pass

    t = threading.Thread(target=read_stdin, daemon=True)
    t.start()
    return cmd_queue


def draw_overlay(frame, device_index: int, device_count: int, resolution: str):
    """Draw controls and current device on the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 800)
    thickness = max(1, int(scale * 2))
    y0 = int(25 * scale)
    line_h = int(28 * scale)
    bg = frame.copy()
    cv2.rectangle(bg, (0, 0), (w, int(70 * scale)), (0, 0, 0), -1)
    cv2.addWeighted(bg, 0.5, frame, 0.5, 0, frame)
    text = f"Camera {device_index} ({resolution})  |  n: next  p: prev  0-9: jump  q: quit"
    cv2.putText(frame, text, (10, y0 + line_h), font, scale, (255, 255, 255), thickness)
    cv2.putText(frame, f"Device {device_index} of {device_count}", (10, y0 + 2 * line_h), font, scale, (200, 200, 200), thickness)


def main(camera_id: int = 0, list_only: bool = False):
    if list_only:
        print("Detecting cameras (this may take a few seconds)...")
        devices = list_cameras()
        print_devices(devices)
        print("Use --camera <index> to choose a device.")
        return

    print("Detecting cameras (this may take a few seconds)...")
    devices = list_cameras()
    if not devices:
        print("No camera devices found.")
        return
    print_devices(devices)
    print("Type in this terminal then Enter: n = next, p = previous, 0-9 = jump, q = quit.")
    print()

    # Resolve starting device: camera_id might be the OS index; find it in our list or use first
    device_list_indices = [i for i, (idx, _) in enumerate(devices) if idx == camera_id]
    if device_list_indices:
        current = device_list_indices[0]
    else:
        current = 0
    device_index = devices[current][0]
    resolution = devices[current][1] or "?"

    cap = open_camera(device_index)
    if cap is None:
        sys.exit(1)

    cv2.namedWindow("cam", cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow("cam", 500, 250)

    cmd_queue = start_terminal_input_queue()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame = cv2.flip(frame, 1)
            draw_overlay(frame, device_index, len(devices), resolution)
            cv2.imshow("cam", frame)

            # Prefer terminal input (works when OpenCV window has no focus on macOS)
            try:
                key_char = cmd_queue.get_nowait()
            except queue.Empty:
                key_char = None
            if key_char is None:
                k = cv2.waitKey(1) & 0xFF
                key_char = chr(k) if 0 < k < 256 else None

            if key_char == "q":
                break
            if key_char == "n":
                # Next device
                next_idx = (current + 1) % len(devices)
                if next_idx != current:
                    cap.stop()
                    device_index = devices[next_idx][0]
                    resolution = devices[next_idx][1] or "?"
                    cap = open_camera(device_index)
                    if cap is None:
                        cap = open_camera(devices[current][0])
                        if cap is None:
                            break
                    else:
                        current = next_idx
                        print(f"Switched to camera {device_index}")
            elif key_char == "p":
                # Previous device
                prev_idx = (current - 1) % len(devices)
                if prev_idx != current:
                    cap.stop()
                    device_index = devices[prev_idx][0]
                    resolution = devices[prev_idx][1] or "?"
                    cap = open_camera(device_index)
                    if cap is None:
                        cap = open_camera(devices[current][0])
                        if cap is None:
                            break
                    else:
                        current = prev_idx
                        print(f"Switched to camera {device_index}")
            elif key_char and key_char in "0123456789":
                # Jump to device 0-9 if present in list
                want = int(key_char)
                for i, (idx, _) in enumerate(devices):
                    if idx == want:
                        if i != current:
                            cap.stop()
                            device_index = devices[i][0]
                            resolution = devices[i][1] or "?"
                            cap = open_camera(device_index)
                            if cap is None:
                                cap = open_camera(devices[current][0])
                                if cap is None:
                                    break
                            else:
                                current = i
                                print(f"Switched to camera {device_index}")
                        break
    finally:
        if cap is not None:
            cap.stop()
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Display camera feed only (no face swap).")
    p.add_argument(
        "--list", "-l", action="store_true",
        help="List available camera devices and exit.",
    )
    p.add_argument(
        "--camera", "-c", type=int, default=0,
        help="Camera device ID (default: 0). Try 1 if feed is black (e.g. IR camera at 0).",
    )
    args = p.parse_args()
    main(camera_id=args.camera, list_only=args.list)
