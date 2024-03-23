#!/usr/bin/env python3

import cv2
import kornia as K
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import transform as ko_transform
from torch.nn import functional as F


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)


def encode_segmentation_rgb(segmentation, device, no_neck=True):
    parse = segmentation

    face_part_ids = (
        [1, 2, 3, 4, 5, 6, 10, 12, 13]
        if no_neck
        else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    )
    mouth_id = [11]

    face_map = (
        isin(
            parse,
            torch.tensor(face_part_ids).to(device),
        )
        * 255.0
    ).to(device)
    mouth_map = (
        isin(
            parse,
            torch.tensor(mouth_id).to(device),
        )
        * 255.0
    ).to(device)
    mask_stack = torch.stack((face_map, mouth_map), axis=2)

    mask_out = torch.zeros([2, parse.shape[0], parse.shape[1]]).to(device)
    mask_out[0, :, :] = mask_stack[:, :, 0]
    mask_out[1, :, :] = mask_stack[:, :, 1]

    return mask_out


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(
            torch.arange(0.0, kernel_size),
            torch.arange(0.0, kernel_size),
            indexing="xy",
        )
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer("weight", kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(
                x,
                F.conv2d(
                    x, weight=self.weight, groups=x.shape[1], padding=self.padding
                ),
            )
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask


def postprocess(swapped_face, target, target_mask, smooth_mask, device):

    target_mask /= 255.0

    face_mask_tensor = target_mask[0] + target_mask[1]

    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask_tensor = soft_face_mask_tensor[None, :, :]

    result = swapped_face * soft_face_mask_tensor + target * (1 - soft_face_mask_tensor)

    return result


def reverse2wholeimage(
    b_align_crop_tenor_list,
    swaped_imgs,
    mats,
    crop_size,
    oriimg,
    pasring_model=None,
    norm=None,
    use_mask=True,
    use_gpu=True,
    use_cam=True,
):

    device = torch.device(
        ("mps" if torch.backends.mps.is_available() else "cuda") if use_gpu else "cpu"
    )
    if use_mask:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).to(
            device
        )

    img = K.utils.image_to_tensor(oriimg).float().to(device)
    img /= 255.0
    kernel_use_cam = torch.ones(5, 5).to(device)
    kernel_use_image = np.ones((40, 40), np.uint8)
    orisize = (oriimg.shape[0], oriimg.shape[1])
    mat_rev_initial = np.ones([3, 3])
    mat_rev_initial[2, :] = np.array([0.0, 0.0, 1.0])
    for swaped_img, mat, source_img in zip(swaped_imgs, mats, b_align_crop_tenor_list):

        img_white = torch.full((1, 3, crop_size, crop_size), 1.0, dtype=torch.float).to(
            device
        )

        # invert the Affine transformation matrix
        mat_rev_initial[0:2, :] = mat
        mat_rev = np.linalg.inv(mat_rev_initial.astype(np.float32))
        mat_rev = mat_rev[:2, :]
        mat_rev = torch.tensor(mat_rev[None, ...]).to(device)

        if use_mask:
            source_img_norm = norm(source_img, use_gpu=use_gpu)
            source_img_512 = F.interpolate(source_img_norm, size=(512, 512))
            out = pasring_model(source_img_512)[0]
            parsing = out.squeeze(0).argmax(0)

            tgt_mask = encode_segmentation_rgb(parsing, device)

            # If the mask is large
            if tgt_mask.sum() >= 5000:

                target_mask = ko_transform.resize(tgt_mask, (crop_size, crop_size))

                target_image_parsing = postprocess(
                    swaped_img,
                    source_img[0],
                    target_mask,
                    smooth_mask,
                    device=device,
                )

                target_image_parsing = target_image_parsing[None, ...]
                swaped_img = swaped_img[None, ...]

                target_image = ko_transform.warp_affine(
                    target_image_parsing, mat_rev, orisize
                )
            else:
                swaped_img = swaped_img[None, ...]
                target_image = ko_transform.warp_affine(
                    swaped_img,
                    mat_rev,
                    orisize,
                )
        else:
            swaped_img = swaped_img[None, ...]
            target_image = ko_transform.warp_affine(
                swaped_img,
                mat_rev,
                orisize,
            )

        img_white = ko_transform.warp_affine(img_white, mat_rev, orisize)

        img_white[img_white > 0.0784] = 1.0

        if use_cam:
            img_white = K.morphology.erosion(img_white, kernel_use_cam)
        else:
            img_white = K.utils.tensor_to_image(img_white) * 255
            img_white = cv2.erode(img_white, kernel_use_image, iterations=1)
            img_white = cv2.GaussianBlur(img_white, (41, 41), 0)
            img_white = K.utils.image_to_tensor(img_white).to(device)
            img_white /= 255.0

        target_image = K.color.rgb_to_bgr(target_image)

        img = img_white * target_image + (1 - img_white) * img

    final_img = K.utils.tensor_to_image(img)
    final_img = (final_img * 255).astype(np.uint8)

    return final_img
