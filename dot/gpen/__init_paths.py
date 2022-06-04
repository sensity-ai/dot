#!/usr/bin/env python3

"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

path = osp.join(this_dir, "retinaface")
add_path(path)

path = osp.join(this_dir, "face_model")
add_path(path)
