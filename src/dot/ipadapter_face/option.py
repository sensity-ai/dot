#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

IP-Adapter Face face swap implementation.
"""

from typing import Optional

import cv2
import numpy as np

from ..diffusion import DiffusionOption

# Try to import required dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        LCMScheduler,
        DDIMScheduler,
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class IPAdapterFaceOption(DiffusionOption):
    """
    IP-Adapter Face face swap implementation.

    IP-Adapter Face uses a face embedding adapter with Stable Diffusion
    to generate images that preserve the identity of a source face.
    """

    # Default model paths
    DEFAULT_SD15_MODEL = "runwayml/stable-diffusion-v1-5"
    DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    IP_ADAPTER_REPO = "h94/IP-Adapter"
    IP_ADAPTER_FACE_MODEL = "ip-adapter-full-face_sd15.safetensors"
    IP_ADAPTER_FACEID_MODEL = "ip-adapter-faceid_sd15.safetensors"

    def __init__(
        self,
        use_gpu: bool = True,
        crop_size: int = 512,
        gpen_type: Optional[str] = None,
        gpen_path: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_lcm: bool = False,
        use_sdxl: bool = False,
        ip_adapter_scale: float = 0.6,
        **kwargs,
    ):
        """
        Initialize IPAdapterFaceOption.

        Args:
            use_gpu: Whether to use GPU acceleration
            crop_size: Size to crop/resize images to
            gpen_type: GPEN face enhancement type
            gpen_path: Path to GPEN models
            num_inference_steps: Number of diffusion inference steps
            guidance_scale: Classifier-free guidance scale
            use_lcm: Whether to use LCM scheduler for faster inference
            use_sdxl: Whether to use SDXL model (higher quality, slower)
            ip_adapter_scale: Scale for IP-Adapter influence (0.0-1.0)
        """
        super().__init__(
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_lcm=use_lcm,
        )

        self.use_sdxl = use_sdxl
        self.ip_adapter_scale = ip_adapter_scale

        # Pipeline (loaded lazily)
        self._pipeline = None
        self._ip_adapter_loaded = False

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for IP-Adapter Face. "
                "Install with: pip install torch"
            )
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for IP-Adapter Face. "
                "Install with: pip install diffusers>=0.25.0"
            )
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for IP-Adapter Face. "
                "Install with: pip install Pillow"
            )

    def _get_torch_dtype(self):
        """Get appropriate torch dtype for the device."""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float32  # MPS often has issues with float16
        return torch.float32

    def create_model(
        self,
        base_model_path: Optional[str] = None,
        ip_adapter_model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the IP-Adapter Face model.

        Args:
            base_model_path: Path to base Stable Diffusion model
            ip_adapter_model_path: Path to IP-Adapter model
        """
        self._check_dependencies()

        # Determine base model
        if base_model_path is None:
            base_model_path = self.DEFAULT_SDXL_MODEL if self.use_sdxl else self.DEFAULT_SD15_MODEL

        dtype = self._get_torch_dtype()

        # Load pipeline
        PipelineClass = StableDiffusionXLPipeline if self.use_sdxl else StableDiffusionPipeline

        self._pipeline = PipelineClass.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # Set up scheduler
        if self.use_lcm:
            self._pipeline.scheduler = LCMScheduler.from_config(
                self._pipeline.scheduler.config
            )
        else:
            self._pipeline.scheduler = DDIMScheduler.from_config(
                self._pipeline.scheduler.config
            )

        # Move to device
        self._pipeline.to(self.device)

        # Load IP-Adapter
        subfolder = "sdxl_models" if self.use_sdxl else "models"
        model_name = "ip-adapter-faceid_sdxl.bin" if self.use_sdxl else self.IP_ADAPTER_FACEID_MODEL

        self._pipeline.load_ip_adapter(
            self.IP_ADAPTER_REPO,
            subfolder=subfolder,
            weight_name=model_name,
        )
        self._pipeline.set_ip_adapter_scale(self.ip_adapter_scale)
        self._ip_adapter_loaded = True

        print("IP-Adapter Face model initialized successfully")

    def change_option(self, image: np.ndarray, **kwargs) -> None:
        """
        Set the source image for face swapping.

        Args:
            image: Source image (BGR format)
        """
        super().change_option(image, **kwargs)

        # Convert source image to PIL for IP-Adapter
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.source_pil = Image.fromarray(image_rgb)

    def process_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform face swap on target image using IP-Adapter Face.

        Args:
            image: Target image (BGR format)

        Returns:
            np.ndarray: Result image with swapped face
        """
        if self._pipeline is None or not self._ip_adapter_loaded:
            raise ValueError("Model not initialized. Call create_model() first.")

        if self.source_face_embedding is None:
            raise ValueError("Source face not set. Call change_option() first.")

        # Detect target face to get position
        target_face = self._detect_face(image)
        if target_face is None:
            return image

        # Convert target to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_pil = Image.fromarray(image_rgb)

        # Resize to model expected size
        height, width = image.shape[:2]
        target_size = (width, height)

        # Prepare face embedding as tensor
        face_embed = torch.tensor(self.source_face_embedding).unsqueeze(0)
        if self.device != "cpu":
            face_embed = face_embed.to(self.device, dtype=self._get_torch_dtype())

        # Generate with IP-Adapter
        with torch.no_grad():
            result = self._pipeline(
                prompt="a person",
                negative_prompt="blurry, low quality, distorted",
                image=target_pil,
                ip_adapter_image_embeds=[face_embed],
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=width - (width % 8),
                height=height - (height % 8),
            ).images[0]

        # Resize back if needed
        if result.size != target_size:
            result = result.resize(target_size, Image.LANCZOS)

        # Convert back to BGR numpy array
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr
