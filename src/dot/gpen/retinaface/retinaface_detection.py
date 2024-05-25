#!/usr/bin/env python3

"""
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
"""
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from .data import cfg_re50
from .facemodels.retinaface import RetinaFace
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms


class RetinaFaceDetection(object):
    def __init__(self, base_dir, network="RetinaFace-R50", use_gpu=True):
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        self.pretrained_path = os.path.join(base_dir, "weights", network + ".pth")
        if use_gpu:
            self.device = "mps" if torch.backends.mps.is_available() else "cuda"
        else:
            self.device = "cpu"
        self.cfg = cfg_re50
        self.net = RetinaFace(cfg=self.cfg, phase="test")
        if use_gpu:
            self.load_model()
            self.net = self.net.to(self.device)
        else:
            self.load_model(load_to_cpu=True)
            self.net = self.net.cpu()

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.net.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
        return True

    def remove_prefix(self, state_dict, prefix):
        """Old style model is stored with all names of parameters sharing common prefix 'module.'"""
        return {
            (lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x)(
                key
            ): value
            for key, value in state_dict.items()
        }

    def load_model(self, load_to_cpu=False):
        if load_to_cpu:
            pretrained_dict = torch.load(
                self.pretrained_path, map_location=lambda storage, loc: storage
            )
        else:
            # pretrained_dict = torch.load(
            #     self.pretrained_path, map_location=lambda storage, loc: storage.to("mps")#.cuda()
            # )
            pretrained_dict = torch.load(self.pretrained_path, map_location=self.device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(
                pretrained_dict["state_dict"], "module."
            )
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, "module.")
        self.check_keys(pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def detect(
        self,
        img_raw,
        resize=1,
        confidence_threshold=0.9,
        nms_threshold=0.4,
        top_k=5000,
        keep_top_k=750,
        save_image=False,
        use_gpu=True,
    ):
        img = np.float32(img_raw)

        im_height, im_width = img.shape[:2]
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        if use_gpu:
            img = img.to(self.device)
            scale = scale.to(self.device)
        else:
            img = img.cpu()
            scale = scale.cpu()

        loc, conf, landms = self.net(img)  # forward pass

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        if use_gpu:
            priors = priors.to(self.device)
        else:
            priors = priors.cpu()

        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg["variance"])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg["variance"])
        scale1 = torch.Tensor(
            [
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
                img.shape[3],
                img.shape[2],
            ]
        )
        if use_gpu:
            scale1 = scale1.to(self.device)
        else:
            scale1 = scale1.cpu()

        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        # sort faces(delete)
        """
        fscores = [det[4] for det in dets]
        sorted_idx = sorted(range(len(fscores)), key=lambda k:fscores[k], reverse=False) # sort index
        tmp = [landms[idx] for idx in sorted_idx]
        landms = np.asarray(tmp)
        """

        landms = landms.reshape((-1, 5, 2))
        landms = landms.transpose((0, 2, 1))
        landms = landms.reshape(
            -1,
            10,
        )
        return dets, landms
