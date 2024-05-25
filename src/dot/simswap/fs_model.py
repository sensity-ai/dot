#!/usr/bin/env python3

import os
import sys

import torch

from .models.base_model import BaseModel


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


sys.path.insert(0, determine_path())

# TODO: Move this class inside models


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

        if use_gpu:
            device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cuda"
            )
        else:
            device = torch.device("cpu")

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
        if use_gpu:
            netArc_checkpoint = torch.load(arcface_model_path)
        else:
            netArc_checkpoint = torch.load(
                arcface_model_path, map_location=torch.device("cpu")
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
