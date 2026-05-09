#!/usr/bin/env python3

import os
import sys
from contextlib import contextmanager

import torch

from dot.commons.utils import get_device
from .models.base_model import BaseModel


@contextmanager
def legacy_simswap_import_path():
    """Expose SimSwap's old top-level modules only while unpickling checkpoints."""
    path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


class fsModel(BaseModel):
    def name(self):
        return "fsModel"

    def initialize(
        self,
        opt_gpu_ids,
        opt_checkpoints_dir,
        opt_name,
        opt_verbose,
        opt_crop_size,
        opt_resize_or_crop,
        opt_load_pretrain,
        opt_which_epoch,
        opt_continue_train,
        arcface_model_path,
        use_gpu=True,
    ):

        BaseModel.initialize(
            self, opt_gpu_ids, opt_checkpoints_dir, opt_name, opt_verbose
        )
        torch.backends.cudnn.benchmark = True

        device_str = get_device() if use_gpu else "cpu"
        device = torch.device(device_str)

        if opt_crop_size == 224:
            from .models.fs_networks import Generator_Adain_Upsample
        elif opt_crop_size == 512:
            from .models.fs_networks_512 import Generator_Adain_Upsample

        # Generator network
        self.netG = Generator_Adain_Upsample(
            input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False
        )
        self.netG.to(device)

        # Id network
        with legacy_simswap_import_path():
            netArc_checkpoint = torch.load(
                arcface_model_path, weights_only=False, map_location=device
            )

        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.to(device)
        self.netArc.eval()

        pretrained_path = ""
        self.load_network(self.netG, "G", opt_which_epoch, pretrained_path)
        return

    def forward(self, img_id, img_att, latent_id, latent_att, for_G=False):
        img_fake = self.netG.forward(img_att, latent_id)

        return img_fake


def create_model(
    opt_verbose,
    opt_crop_size,
    opt_fp16,
    opt_gpu_ids,
    opt_checkpoints_dir,
    opt_name,
    opt_resize_or_crop,
    opt_load_pretrain,
    opt_which_epoch,
    opt_continue_train,
    arcface_model_path,
    use_gpu=True,
):

    model = fsModel()

    model.initialize(
        opt_gpu_ids,
        opt_checkpoints_dir,
        opt_name,
        opt_verbose,
        opt_crop_size,
        opt_resize_or_crop,
        opt_load_pretrain,
        opt_which_epoch,
        opt_continue_train,
        arcface_model_path,
        use_gpu=use_gpu,
    )

    if opt_verbose:
        print("model [%s] was created" % (model.name()))

    return model
