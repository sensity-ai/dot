#!/usr/bin/env python3

import glob
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

SEED = 42
np.random.seed(SEED)


def log(*args, file=sys.stderr, **kwargs):
    time_str = f"{time.time():.6f}"
    print(f"[{time_str}]", *args, file=file, **kwargs)


def info(*args, file=sys.stdout, **kwargs):
    print(*args, file=file, **kwargs)


def find_images_from_path(path):
    """
    @arguments:
        path              (str/int)    : Could be either path(str)
                                         or a CamID(int)
    """
    if os.path.isfile(path):
        return [path]

    try:
        return int(path)
    except ValueError:
        # supported extensions
        ext = ["png", "jpg", "jpeg"]
        files = []
        [files.extend(glob.glob(path + "**/*." + e, recursive=True)) for e in ext]

        return files


def find_files_from_path(path: str, ext: List, filter: str = None):
    """
    @arguments:
        path              (str)     Parent directory of files
        ext               (list)    List of desired file extensions
    """
    if os.path.isdir(path):
        files = []
        [
            files.extend(glob.glob(path + "**/*." + e, recursive=True)) for e in ext  # type: ignore
        ]
        np.random.shuffle(files)

        # filter
        if filter is not None:
            files = [file for file in files if filter in file]
            print("Filtered files: ", len(files))

        return files

    return [path]


def expand_bbox(
    bbox, image_width, image_height, scale=None
) -> Tuple[int, int, int, int]:
    if scale is None:
        raise ValueError("scale parameter is none")

    x1, y1, x2, y2 = bbox

    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    size_bb = round(max(x2 - x1, y2 - y1) * scale)

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    # Check for too big bb size for given x, y
    size_bb = min(image_width - x1, size_bb)
    size_bb = min(image_height - y1, size_bb)

    return (x1, y1, x1 + size_bb, y1 + size_bb)


def rand_idx_tuple(source_len, target_len):
    """
    pick a random tuple for source/target
    """
    return (random.randrange(source_len), random.randrange(target_len))


def generate_random_file_idx(length):
    return int("".join([str(random.randint(0, 10)) for _ in range(length)]))


class Tee(object):
    def __init__(self, filename, mode="w", terminal=sys.stderr):
        self.file = open(filename, mode, buffering=1)
        self.terminal = terminal

    def __del__(self):
        self.file.close()

    def write(self, *args, **kwargs):
        log(*args, file=self.file, **kwargs)
        log(*args, file=self.terminal, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.write(*args, **kwargs)

    def flush(self):
        self.file.flush()


class Logger:
    def __init__(self, filename, verbose=True):
        self.tee = Tee(filename)
        self.verbose = verbose

    def __call__(self, *args, important=False, **kwargs):
        if not self.verbose and not important:
            return

        self.tee(*args, **kwargs)


class Once:
    _id: Dict = {}

    def __init__(self, what, who=log, per=1e12):
        """Do who(what) once per seconds.
        what: args for who
        who: callable
        per: frequency in seconds.
        """
        assert callable(who)
        now = time.time()
        if what not in Once._id or now - Once._id[what] > per:
            who(what)
            Once._id[what] = now


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

    def tocp(self, str):
        t = self.toc()
        log(f"{str} took {t:.4f}ms")
        return t


class AccumDict:
    def __init__(self, num_f=3):
        self.d = defaultdict(list)
        self.num_f = num_f

    def add(self, k, v):
        self.d[k] += [v]

    def __dict__(self):
        return self.d

    def __getitem__(self, key):
        return self.d[key]

    def __str__(self):
        s = ""
        for k in self.d:
            if not self.d[k]:
                continue
            cur = self.d[k][-1]
            avg = np.mean(self.d[k])
            format_str = "{:.%df}" % self.num_f
            cur_str = format_str.format(cur)
            avg_str = format_str.format(avg)
            s += f"{k} {cur_str} ({avg_str})\t\t"
        return s

    def __repr__(self):
        return self.__str__()


def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def crop(img, p=0.7, offset_x=0, offset_y=0):
    h, w = img.shape[:2]
    x = int(min(w, h) * p)
    _l = (w - x) // 2
    r = w - _l
    u = (h - x) // 2
    d = h - u

    offset_x = clamp(offset_x, -_l, w - r)
    offset_y = clamp(offset_y, -u, h - d)

    _l += offset_x
    r += offset_x
    u += offset_y
    d += offset_y

    return img[u:d, _l:r], (offset_x, offset_y)


def pad_img(img, target_size, default_pad=0):
    sh, sw = img.shape[:2]
    w, h = target_size
    pad_w, pad_h = default_pad, default_pad
    if w / h > 1:
        pad_w += int(sw * (w / h) - sw) // 2
    else:
        pad_h += int(sh * (h / w) - sh) // 2
    out = np.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], "constant")
    return out


def resize(img, size, version="cv"):
    return cv2.resize(img, size)


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
