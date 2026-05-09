#!/usr/bin/env python3

import os
import time

import torch

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV", ".AVI"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_project_root() -> str:
    current_file = os.path.abspath(__file__)
    commons_dir = os.path.dirname(current_file)
    src_dir = os.path.dirname(commons_dir)
    return os.path.dirname(src_dir)


def get_model_base_path() -> str:
    return os.path.join(get_project_root(), "saved_models", "simswap")


class TicToc:
    def __init__(self):
        self.t = None
        self.t_init = time.time()

    def tic(self):
        self.t = time.time()

    def toc(self, total=False):
        if total:
            return (time.time() - self.t_init) * 1000
        assert self.t, "You forgot to call tic()"
        return (time.time() - self.t) * 1000
