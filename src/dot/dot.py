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

# Base swap types always available
AVAILABLE_SWAP_TYPES = ["simswap", "fomm", "faceswap_cv2"]

# Try to import diffusion-based models (optional dependencies)
try:
    from .reactor import ReActorOption
    AVAILABLE_SWAP_TYPES.append("reactor")
    REACTOR_AVAILABLE = True
except ImportError:
    REACTOR_AVAILABLE = False

try:
    from .ipadapter_face import IPAdapterFaceOption
    AVAILABLE_SWAP_TYPES.append("ipadapter_face")
    IPADAPTER_AVAILABLE = True
except ImportError:
    IPADAPTER_AVAILABLE = False

try:
    from .instantid import InstantIDOption
    AVAILABLE_SWAP_TYPES.append("instantid")
    INSTANTID_AVAILABLE = True
except ImportError:
    INSTANTID_AVAILABLE = False


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
        elif swap_type == "reactor":
            option = self.reactor(
                use_gpu=use_gpu,
                gpen_type=gpen_type,
                gpen_path=gpen_path,
                crop_size=crop_size,
                **kwargs,
            )
        elif swap_type == "ipadapter_face":
            option = self.ipadapter_face(
                use_gpu=use_gpu,
                gpen_type=gpen_type,
                gpen_path=gpen_path,
                crop_size=crop_size,
                **kwargs,
            )
        elif swap_type == "instantid":
            option = self.instantid(
                use_gpu=use_gpu,
                gpen_type=gpen_type,
                gpen_path=gpen_path,
                crop_size=crop_size,
                **kwargs,
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

    def reactor(
        self,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int = 512,
        use_face_restore: bool = True,
        face_restore_model: str = "codeformer",
        **kwargs,
    ) -> "ReActorOption":
        """Build ReActor Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 512.
            use_face_restore (bool, optional): Whether to use face restoration. Defaults to True.
            face_restore_model (str, optional): Face restore model. Defaults to "codeformer".

        Returns:
            ReActorOption: ReActor Option.
        """
        if not REACTOR_AVAILABLE:
            raise ImportError(
                "ReActor dependencies not installed. "
                "Install with: pip install dot[reactor]"
            )
        return ReActorOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            use_face_restore=use_face_restore,
            face_restore_model=face_restore_model,
        )

    def ipadapter_face(
        self,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_lcm: bool = False,
        use_sdxl: bool = False,
        ip_adapter_scale: float = 0.6,
        **kwargs,
    ) -> "IPAdapterFaceOption":
        """Build IP-Adapter Face Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 512.
            num_inference_steps (int, optional): Diffusion steps. Defaults to 20.
            guidance_scale (float, optional): CFG scale. Defaults to 7.5.
            use_lcm (bool, optional): Use LCM scheduler. Defaults to False.
            use_sdxl (bool, optional): Use SDXL model. Defaults to False.
            ip_adapter_scale (float, optional): IP-Adapter scale. Defaults to 0.6.

        Returns:
            IPAdapterFaceOption: IP-Adapter Face Option.
        """
        if not IPADAPTER_AVAILABLE:
            raise ImportError(
                "IP-Adapter dependencies not installed. "
                "Install with: pip install dot[ipadapter]"
            )
        return IPAdapterFaceOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_lcm=use_lcm,
            use_sdxl=use_sdxl,
            ip_adapter_scale=ip_adapter_scale,
        )

    def instantid(
        self,
        use_gpu: bool,
        gpen_type: str,
        gpen_path: str,
        crop_size: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_lcm: bool = False,
        controlnet_conditioning_scale: float = 0.8,
        ip_adapter_scale: float = 0.8,
        **kwargs,
    ) -> "InstantIDOption":
        """Build InstantID Option.

        Args:
            use_gpu (bool): If True, use GPU.
            gpen_type (str): GPEN type.
            gpen_path (str): path to GPEN model checkpoint.
            crop_size (int, optional): crop size. Defaults to 512.
            num_inference_steps (int, optional): Diffusion steps. Defaults to 20.
            guidance_scale (float, optional): CFG scale. Defaults to 7.5.
            use_lcm (bool, optional): Use LCM scheduler. Defaults to False.
            controlnet_conditioning_scale (float, optional): ControlNet scale. Defaults to 0.8.
            ip_adapter_scale (float, optional): IP-Adapter scale. Defaults to 0.8.

        Returns:
            InstantIDOption: InstantID Option.
        """
        if not INSTANTID_AVAILABLE:
            raise ImportError(
                "InstantID dependencies not installed. "
                "Install with: pip install dot[instantid]"
            )
        return InstantIDOption(
            use_gpu=use_gpu,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            crop_size=crop_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_lcm=use_lcm,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
        )
