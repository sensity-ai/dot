#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import transform as ko_transform
from torch.nn import functional as F


def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation

    face_part_ids = (
        [1, 2, 3, 4, 5, 6, 10, 12, 13]
        if no_neck
        else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    )
    mouth_id = 11
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse == valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse == mouth_id)
    mouth_map[valid_index] = 255
    return np.stack([face_map, mouth_map], axis=2)


class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(
            torch.arange(0.0, kernel_size), torch.arange(0.0, kernel_size)
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


def postprocess(swapped_face, target, target_mask, smooth_mask, use_gpu=True):
    if use_gpu:
        mask_tensor = (
            torch.from_numpy(target_mask.copy().transpose((2, 0, 1)))
            .float()
            .mul_(1 / 255.0)
            .cuda()
        )
    else:
        mask_tensor = (
            torch.from_numpy(target_mask.copy().transpose((2, 0, 1)))
            .float()
            .mul_(1 / 255.0)
            .cpu()
        )

    face_mask_tensor = mask_tensor[0] + mask_tensor[1]

    soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
    soft_face_mask_tensor.squeeze_()

    soft_face_mask = soft_face_mask_tensor.cpu().numpy()
    soft_face_mask = soft_face_mask[:, :, np.newaxis]

    result = swapped_face * soft_face_mask + target * (1 - soft_face_mask)
    result = result[:, :, ::-1]  # .astype(np.uint8)
    return result


# TODO: use device instead of use_gpu
def reverse2wholeimage(
    b_align_crop_tenor_list,
    swaped_imgs,
    mats,
    crop_size,
    oriimg,
    pasring_model=None,
    norm=None,
    use_mask=False,
    use_gpu=True,
):

    target_image_list = []
    img_mask_list = []
    if use_mask:
        if use_gpu:
            smooth_mask = SoftErosion(
                kernel_size=17, threshold=0.9, iterations=7
            ).cuda()
        else:
            smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cpu()
    else:
        pass

    for swaped_img, mat, source_img in zip(swaped_imgs, mats, b_align_crop_tenor_list):
        swaped_img = swaped_img.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size, crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        mat_rev = np.zeros([2, 3])
        div1 = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        mat_rev[0][0] = mat[1][1] / div1
        mat_rev[0][1] = -mat[0][1] / div1
        mat_rev[0][2] = -(mat[0][2] * mat[1][1] - mat[0][1] * mat[1][2]) / div1
        div2 = mat[0][1] * mat[1][0] - mat[0][0] * mat[1][1]
        mat_rev[1][0] = mat[1][0] / div2
        mat_rev[1][1] = -mat[0][0] / div2
        mat_rev[1][2] = -(mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / div2

        orisize = (oriimg.shape[1], oriimg.shape[0])
        if use_mask:
            source_img_norm = norm(source_img, use_gpu=use_gpu)
            source_img_512 = F.interpolate(source_img_norm, size=(512, 512))
            out = pasring_model(source_img_512)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            tgt_mask = encode_segmentation_rgb(vis_parsing_anno)
            if tgt_mask.sum() >= 5000:
                target_mask = cv2.resize(tgt_mask, (crop_size, crop_size))
                target_image_parsing = postprocess(
                    swaped_img,
                    source_img[0].cpu().detach().numpy().transpose((1, 2, 0)),
                    target_mask,
                    smooth_mask,
                    use_gpu=use_gpu,
                )

                if use_gpu:
                    target_image = ko_transform.warp_affine(
                        torch.tensor(target_image_parsing).cuda(),
                        torch.tensor(mat_rev).cuda(),
                        orisize,
                    )
                else:
                    target_image = ko_transform.warp_affine(
                        torch.tensor(target_image_parsing).cpu(),
                        torch.tensor(mat_rev).cpu(),
                        orisize,
                    )
                target_image = target_image.cpu().detach().numpy()
            else:
                if use_gpu:
                    target_image = ko_transform.warp_affine(
                        torch.tensor(swaped_img).cuda(),
                        torch.tensor(mat_rev).cuda(),
                        orisize,
                    )[..., ::-1]
                else:
                    target_image = ko_transform.warp_affine(
                        torch.tensor(swaped_img).cpu(),
                        torch.tensor(mat_rev).cpu(),
                        orisize,
                    )[..., ::-1]
                target_image = target_image.cpu().detach().numpy()
        else:
            if use_gpu:
                target_image = ko_transform.warp_affine(
                    torch.tensor(swaped_img).cuda(),
                    torch.tensor(mat_rev).cuda(),
                    orisize,
                )
            else:
                target_image = ko_transform.warp_affine(
                    torch.tensor(swaped_img).cpu(), torch.tensor(mat_rev).cpu(), orisize
                )
            target_image = target_image.cpu().detach().numpy()

        if use_gpu:
            img_white = ko_transform.warp_affine(
                torch.tensor(img_white).cuda(), torch.tensor(mat_rev).cuda(), orisize
            )
        else:
            img_white = ko_transform.warp_affine(
                torch.tensor(img_white).cpu(), torch.tensor(mat_rev).cpu(), orisize
            )
        img_white = img_white.cpu().detach().numpy()

        img_white[img_white > 20] = 255

        img_mask = img_white

        kernel = np.ones((40, 40), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel_size = (20, 20)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)

        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if use_mask:
            target_image = np.array(target_image, dtype=np.float) * 255
        else:
            target_image = np.array(target_image, dtype=np.float)[..., ::-1] * 255

        img_mask_list.append(img_mask)
        target_image_list.append(target_image)

    img = np.array(oriimg, dtype=np.float)
    for img_mask, target_image in zip(img_mask_list, target_image_list):
        img = img_mask * target_image + (1 - img_mask) * img

    final_img = img.astype(np.uint8)

    return final_img
