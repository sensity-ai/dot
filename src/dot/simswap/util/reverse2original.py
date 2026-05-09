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


def preserve_dark_occluders(result, target, threshold=0.10, strength=0.90):
    """Keep dark foreground objects from the target crop, such as microphones."""
    threshold = float(np.clip(threshold, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return result

    luminance = target.mean(dim=0, keepdim=True)
    occlusion = (luminance < threshold).float().unsqueeze(0)
    occlusion = F.max_pool2d(occlusion, kernel_size=5, stride=1, padding=2)
    occlusion = F.avg_pool2d(occlusion, kernel_size=7, stride=1, padding=3)
    occlusion = (occlusion.squeeze(0) * strength).clamp(0.0, 1.0)
    return (result * (1.0 - occlusion) + target * occlusion).clamp(0.0, 1.0)


def match_color_statistics(swapped_face, target_face, strength=0.65):
    """Match swapped face color statistics to the target crop for natural lighting."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return swapped_face

    dims = (1, 2)
    src_mean = swapped_face.mean(dim=dims, keepdim=True)
    src_std = swapped_face.std(dim=dims, keepdim=True).clamp_min(1e-4)
    tgt_mean = target_face.mean(dim=dims, keepdim=True)
    tgt_std = target_face.std(dim=dims, keepdim=True).clamp_min(1e-4)

    matched = (swapped_face - src_mean) / src_std
    matched = (matched * tgt_std + tgt_mean).clamp(0.0, 1.0)
    return (swapped_face * (1.0 - strength) + matched * strength).clamp(0.0, 1.0)


def enhance_face_detail(swapped_face, strength=0.12):
    """Apply a mild unsharp mask to reduce the plastic look of generator output."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return swapped_face

    x = swapped_face.unsqueeze(0)
    blur = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return (x + (x - blur) * strength).clamp(0.0, 1.0).squeeze(0)


def poisson_blend(src_tensor, dst_tensor, mask_tensor):
    """Blend the warped face into the frame with OpenCV seamlessClone."""
    src = (K.utils.tensor_to_image(src_tensor) * 255).astype(np.uint8)
    dst = (K.utils.tensor_to_image(dst_tensor) * 255).astype(np.uint8)
    mask = K.utils.tensor_to_image(mask_tensor.max(dim=1, keepdim=True)[0])
    mask = ((mask > 0.05).astype(np.uint8) * 255)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    coords = cv2.findNonZero(mask)
    if coords is None:
        return dst_tensor

    x, y, w, h = cv2.boundingRect(coords)
    center = (x + w // 2, y + h // 2)

    try:
        blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return dst_tensor

    return K.utils.image_to_tensor(blended).float().to(dst_tensor.device) / 255.0


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
    color_match=False,
    color_match_strength=0.65,
    detail_enhance=False,
    detail_enhance_strength=0.12,
    preserve_occluders=False,
    occluder_threshold=0.10,
    occluder_strength=0.90,
    blend_mode="alpha",
    blend_strength=1.0,
    mask_blur=0,
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
        if color_match:
            swaped_img = match_color_statistics(
                swaped_img,
                source_img[0],
                strength=color_match_strength,
            )
        if detail_enhance:
            swaped_img = enhance_face_detail(
                swaped_img,
                strength=detail_enhance_strength,
            )

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
                if preserve_occluders:
                    target_image_parsing = preserve_dark_occluders(
                        target_image_parsing,
                        source_img[0],
                        threshold=occluder_threshold,
                        strength=occluder_strength,
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

        if mask_blur:
            blur_size = int(mask_blur)
            if blur_size > 1:
                if blur_size % 2 == 0:
                    blur_size += 1
                sigma = max(1.0, blur_size / 6.0)
                img_white = K.filters.gaussian_blur2d(
                    img_white,
                    (blur_size, blur_size),
                    (sigma, sigma),
                ).clamp(0.0, 1.0)

        blend_strength = float(np.clip(blend_strength, 0.0, 1.0))
        if blend_strength < 1.0:
            img_white = (img_white * blend_strength).clamp(0.0, 1.0)

        target_image = K.color.rgb_to_bgr(target_image)

        if blend_mode == "poisson":
            img = poisson_blend(target_image, img, img_white)
        else:
            img = img_white * target_image + (1 - img_white) * img

    final_img = K.utils.tensor_to_image(img)
    final_img = (final_img * 255).astype(np.uint8)

    return final_img
