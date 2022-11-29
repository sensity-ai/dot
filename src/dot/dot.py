#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.
"""

from pathlib import Path
from typing import List, Optional, Union

from .commons import ModelOption
from .faceswap_cv2 import FaceswapCVOption
from .fomm import FOMMOption
from .simswap import SimswapOption

AVAILABLE_SWAP_TYPES = ["simswap", "fomm", "faceswap_cv2"]


class DOT:
    """Main DOT Interface.

    Supported Engines:
        - `simswap`
        - `fomm`
        - `faceswap_cv2`

    Attributes:
        use_cam (bool): Use camera descriptor and pipeline.
        use_video (bool): Use video-swap pipeline.
        use_image (bool): Use image-swap pipeline.
        save_folder (str): Output folder to store face-swaps and metadata file when `use_cam` is False.
    """

    def __init__(
        self,
        use_video: bool = False,
        use_image: bool = False,
        save_folder: str = None,
        *args,
        **kwargs,
    ):
        """Constructor method.

        Args:
            use_video (bool, optional): if True, use video-swap pipeline. Defaults to False.
            use_image (bool, optional): if True, use image-swap pipeline. Defaults to False.
            save_folder (str, optional): Output folder to store face-swaps and metadata file when `use_cam` is False.
                Defaults to None.
        """
        # init
        self.use_video = use_video
        self.save_folder = save_folder
        self.use_image = use_image

        # additional attributes
        self.use_cam = (not use_video) and (not use_image)

        # create output folder
        if self.save_folder and not Path(self.save_folder).exists():
            Path(self.save_folder).mkdir(parents=True, exist_ok=True)

    def build_option(
        self,
        swap_type: str,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int,
        **kwargs,
    ) -> ModelOption:
        """Build DOT option based on swap type.

        Args:
            swap_type (str): Swap type engine.
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int): crop size.

        Returns:
            ModelOption: DOT option.
        """
        if swap_type not in AVAILABLE_SWAP_TYPES:
            raise ValueError(f"Invalid swap type: {swap_type}")

        option: ModelOption = None
        if swap_type == "simswap":
            option = self.simswap(
                use_gpu=use_gpu,
                gpen_type=gpen_type,
                gpen_path=gpen_path,
                crop_size=crop_size,
            )
        elif swap_type == "fomm":
            option = self.fomm(
                use_gpu=use_gpu, gpen_type=gpen_type, gpen_path=gpen_path, **kwargs
            )
        elif swap_type == "faceswap_cv2":
            option = self.faceswap_cv2(
                use_gpu=use_gpu, gpen_type=gpen_type, gpen_path=gpen_path
            )

        return option

    def generate(
        self,
        option: ModelOption,
        source: str,
        target: Union[int, str],
        show_fps: bool = False,
        duration: int = None,
        **kwargs,
    ) -> Optional[List]:
        """Differentiates among different swap options.

        Available swap options:
            - `camera`
            - `image`
            - `video`

        Args:
            option (ModelOption): Swap engine class.
            source (str): File path of source image.
            target (Union[int, str]): Either `int` which indicates camera descriptor or target image file.
            show_fps (bool, optional): Displays FPS during camera pipeline. Defaults to False.
            duration (int, optional): Used to trim source video in seconds. Defaults to None.

        Returns:
            Optional[List]: None when using camera, otherwise metadata of successful and rejected face-swaps.
        """
        if self.use_cam:
            option.generate_from_camera(
                source, int(target), show_fps=show_fps, **kwargs
            )
            return None
        if isinstance(target, str):
            if self.use_video:
                option.generate_from_video(
                    source, target, self.save_folder, duration, **kwargs
                )
                return None
            elif self.use_image:
                [swappedDict, rejectedDict] = option.generate_from_image(
                    source, target, self.save_folder, **kwargs
                )
                return [swappedDict, rejectedDict]
            else:
                return None
        else:
            return None

    def simswap(
        self,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int = 224,
        use_mask: bool = True,
    ) -> SimswapOption:
        """Build Simswap Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 224.
            use_mask (bool, optional): If True, use mask. Defaults to True.

        Returns:
            SimswapOption: Simswap Option.
        """
        return SimswapOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            use_mask=use_mask,
        )

    def faceswap_cv2(
        self, use_gpu: bool, gpen_type: str, gpen_path: str, crop_size: int = 256
    ) -> FaceswapCVOption:
        """Build FaceswapCV Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 256.

        Returns:
            FaceswapCVOption: FaceswapCV Option.
        """
        return FaceswapCVOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
        )

    def fomm(
        self,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int = 256,
        **kwargs,
    ) -> FOMMOption:
        """Build FOMM Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 256.

        Returns:
            FOMMOption: FOMM Option.
        """
        return FOMMOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            offline=self.use_video,
        )
