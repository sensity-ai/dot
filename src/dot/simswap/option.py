#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dot.commons import ModelOption
from dot.simswap.fs_model import create_model
from dot.simswap.mediapipe.face_mesh import FaceMesh
from dot.simswap.parsing_model.model import BiSeNet
from dot.simswap.util.norm import SpecificNorm
from dot.simswap.util.reverse2original import reverse2wholeimage
from dot.simswap.util.util import _totensor


class SimswapOption(ModelOption):
    """Extends `ModelOption` and initializes models."""

    def __init__(
        self,
        use_gpu=True,
        use_mask=False,
        crop_size=224,
        gpen_type=None,
        gpen_path=None,
    ):
        super(SimswapOption, self).__init__(
            gpen_type=gpen_type,
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_path=gpen_path,
        )
        self.use_mask = use_mask

    def create_model(  # type: ignore
        self,
        detection_threshold=0.6,
        det_size=(640, 640),
        opt_verbose=False,
        opt_crop_size=224,
        opt_gpu_ids=[0],
        opt_fp16=False,
        checkpoints_dir="./checkpoints",
        opt_name="people",
        opt_resize_or_crop="scale_width",
        opt_load_pretrain="",
        opt_which_epoch="latest",
        opt_continue_train="store_true",
        parsing_model_path="./parsing_model/checkpoint/79999_iter.pth",
        arcface_model_path="./arcface_model/arcface_checkpoint.tar",
        **kwargs
    ) -> None:
        # preprocess_f
        self.transformer_Arcface = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if opt_crop_size == 512:
            opt_which_epoch = 550000
            opt_name = "512"
            self.mode = "ffhq"
        else:
            self.mode = "None"

        self.detect_model = FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            mode=self.mode,
        )

        # Tod check if we need this
        self.spNorm = SpecificNorm(use_gpu=self.use_gpu)
        if self.use_mask:
            n_classes = 19
            self.net = BiSeNet(n_classes=n_classes)
            if self.use_gpu:
                device = "mps" if torch.backends.mps.is_available() else "cuda"
                self.net.to(device)
                self.net.load_state_dict(
                    torch.load(parsing_model_path, map_location=device)
                )
            else:
                self.net.cpu()
                self.net.load_state_dict(
                    torch.load(parsing_model_path, map_location=torch.device("cpu"))
                )

            self.net.eval()
        else:
            self.net = None

        torch.nn.Module.dump_patches = False

        # Model
        self.model = create_model(
            opt_verbose,
            opt_crop_size,
            opt_fp16,
            opt_gpu_ids,
            checkpoints_dir,
            opt_name,
            opt_resize_or_crop,
            opt_load_pretrain,
            opt_which_epoch,
            opt_continue_train,
            arcface_model_path,
            use_gpu=self.use_gpu,
        )
        self.model.eval()

    def change_option(self, image: np.array, **kwargs) -> None:
        """Sets the source image in source/target pair face-swap.

        Args:
            image (np.array): Source image.
        """
        img_a_align_crop, _ = self.detect_model.get(image, self.crop_size)
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
        )
        img_a = self.transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        if self.use_gpu:
            img_id = (
                img_id.to("mps")
                if torch.backends.mps.is_available()
                else img_id.to("cuda")
            )
        else:
            img_id = img_id.cpu()

        # create latent id
        img_id_downsample = F.interpolate(img_id, size=(112, 112))
        source_image = self.model.netArc(img_id_downsample)
        source_image = source_image.detach().to("cpu")
        source_image = source_image / np.linalg.norm(
            source_image, axis=1, keepdims=True
        )

        source_image = (
            source_image.to("mps" if torch.backends.mps.is_available() else "cuda")
            if self.use_gpu
            else source_image.to("cpu")
        )
        self.source_image = source_image

    def process_image(self, image: np.array, **kwargs) -> np.array:
        """Main process of simswap method. There are 3 main steps:
        * face detection and alignment of target image.
        * swap with `self.source_image`.
        * face segmentation and reverse to whole image.

        Args:
            image (np.array): Target frame where face from `self.source_image` will be swapped with.

        Returns:
            np.array: Resulted face-swap image
        """

        detect_results = self.detect_model.get(image, self.crop_size)
        if detect_results is not None:
            frame_align_crop_list = detect_results[0]
            frame_mat_list = detect_results[1]
            swap_result_list = []
            frame_align_crop_tenor_list = []
            for frame_align_crop in frame_align_crop_list:
                if self.use_gpu:
                    frame_align_crop_tenor = _totensor(
                        cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB)
                    )[None, ...].to(
                        "mps" if torch.backends.mps.is_available() else "cuda"
                    )
                else:
                    frame_align_crop_tenor = _totensor(
                        cv2.cvtColor(frame_align_crop, cv2.COLOR_BGR2RGB)
                    )[None, ...].cpu()

                swap_result = self.model(
                    None, frame_align_crop_tenor, self.source_image, None, True
                )[0]
                swap_result_list.append(swap_result)
                frame_align_crop_tenor_list.append(frame_align_crop_tenor)

            result_frame = reverse2wholeimage(
                frame_align_crop_tenor_list,
                swap_result_list,
                frame_mat_list,
                self.crop_size,
                image,
                pasring_model=self.net,
                use_mask=self.use_mask,
                norm=self.spNorm,
                use_gpu=self.use_gpu,
                use_cam=kwargs.get("use_cam", True),
            )
            return result_frame
        else:
            return image
