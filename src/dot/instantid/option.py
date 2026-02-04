#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

InstantID face swap implementation.
"""

from typing import Optional, Tuple

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
        StableDiffusionXLPipeline,
        ControlNetModel,
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


def draw_kps(image: np.ndarray, kps: np.ndarray, color: Tuple = (255, 0, 0)) -> np.ndarray:
    """Draw facial keypoints on an image."""
    out_img = image.copy()
    for idx, kp in enumerate(kps):
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(out_img, (x, y), 3, color, -1)
    return out_img


class InstantIDOption(DiffusionOption):
    """
    InstantID face swap implementation.

    InstantID uses ControlNet with facial keypoints and IP-Adapter for
    identity-preserving image generation. It provides the highest quality
    results but is slower than ReActor.
    """

    # Model repositories
    INSTANTID_REPO = "InstantX/InstantID"
    CONTROLNET_MODEL = "ControlNetModel"
    SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    ANTELOPEV2_REPO = "DIAMONIK7777/antelopev2"

    def __init__(
        self,
        use_gpu: bool = True,
        crop_size: int = 512,
        gpen_type: Optional[str] = None,
        gpen_path: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_lcm: bool = False,
        controlnet_conditioning_scale: float = 0.8,
        ip_adapter_scale: float = 0.8,
        **kwargs,
    ):
        """
        Initialize InstantIDOption.

        Args:
            use_gpu: Whether to use GPU acceleration
            crop_size: Size to crop/resize images to
            gpen_type: GPEN face enhancement type
            gpen_path: Path to GPEN models
            num_inference_steps: Number of diffusion inference steps
            guidance_scale: Classifier-free guidance scale
            use_lcm: Whether to use LCM scheduler for faster inference
            controlnet_conditioning_scale: Scale for ControlNet influence
            ip_adapter_scale: Scale for IP-Adapter influence
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

        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.ip_adapter_scale = ip_adapter_scale

        # Pipeline components (loaded lazily)
        self._pipeline = None
        self._controlnet = None

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "torch is required for InstantID. "
                "Install with: pip install torch"
            )
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is required for InstantID. "
                "Install with: pip install diffusers>=0.25.0"
            )
        if not PIL_AVAILABLE:
            raise ImportError(
                "Pillow is required for InstantID. "
                "Install with: pip install Pillow"
            )

    def _get_torch_dtype(self):
        """Get appropriate torch dtype for the device."""
        if self.device == "cuda":
            return torch.float16
        elif self.device == "mps":
            return torch.float32
        return torch.float32

    def _create_face_keypoint_image(self, kps: np.ndarray, size: Tuple[int, int]) -> Image.Image:
        """
        Create a keypoint conditioning image for ControlNet.

        Args:
            kps: Facial keypoints (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
            size: Output image size (width, height)

        Returns:
            PIL.Image: Keypoint visualization image
        """
        width, height = size
        kps_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw keypoints with different colors
        colors = [
            (255, 0, 0),    # Left eye - red
            (0, 255, 0),    # Right eye - green
            (0, 0, 255),    # Nose - blue
            (255, 255, 0),  # Left mouth - yellow
            (0, 255, 255),  # Right mouth - cyan
        ]

        for idx, (kp, color) in enumerate(zip(kps, colors)):
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(kps_image, (x, y), 5, color, -1)

        return Image.fromarray(kps_image)

    def create_model(
        self,
        instantid_model_path: Optional[str] = None,
        base_model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the InstantID model.

        Args:
            instantid_model_path: Path to InstantID model (downloads if None)
            base_model_path: Path to base SDXL model
        """
        self._check_dependencies()

        if base_model_path is None:
            base_model_path = self.SDXL_MODEL

        dtype = self._get_torch_dtype()

        # Load ControlNet for face keypoints
        self._controlnet = ControlNetModel.from_pretrained(
            self.INSTANTID_REPO,
            subfolder="ControlNetModel",
            torch_dtype=dtype,
        )

        # Load SDXL pipeline with ControlNet
        self._pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            controlnet=self._controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # Set up scheduler
        if self.use_lcm:
            self._pipeline.scheduler = LCMScheduler.from_config(
                self._pipeline.scheduler.config
            )
            # LCM typically uses fewer steps
            if self.num_inference_steps > 8:
                self.num_inference_steps = 4
        else:
            self._pipeline.scheduler = DDIMScheduler.from_config(
                self._pipeline.scheduler.config
            )

        # Load IP-Adapter for face identity
        self._pipeline.load_ip_adapter(
            self.INSTANTID_REPO,
            subfolder="",
            weight_name="ip-adapter.bin",
        )
        self._pipeline.set_ip_adapter_scale(self.ip_adapter_scale)

        # Move to device
        self._pipeline.to(self.device)

        # Enable memory optimizations if available
        if hasattr(self._pipeline, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        print("InstantID model initialized successfully")

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
        Perform face swap on target image using InstantID.

        Args:
            image: Target image (BGR format)

        Returns:
            np.ndarray: Result image with swapped face
        """
        if self._pipeline is None:
            raise ValueError("Model not initialized. Call create_model() first.")

        if self.source_face_embedding is None:
            raise ValueError("Source face not set. Call change_option() first.")

        # Detect target face to get keypoints
        target_face = self._detect_face(image)
        if target_face is None:
            return image

        # Get image dimensions
        height, width = image.shape[:2]

        # Create keypoint conditioning image
        target_kps = target_face.kps
        kps_image = self._create_face_keypoint_image(target_kps, (width, height))

        # Prepare face embedding
        face_embed = torch.tensor(self.source_face_embedding).unsqueeze(0)
        if self.device != "cpu":
            face_embed = face_embed.to(self.device, dtype=self._get_torch_dtype())

        # Generate with InstantID
        with torch.no_grad():
            result = self._pipeline(
                prompt="a person, high quality, detailed face",
                negative_prompt="blurry, low quality, distorted, deformed",
                image=kps_image,
                ip_adapter_image_embeds=[face_embed],
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=width - (width % 8),
                height=height - (height % 8),
            ).images[0]

        # Resize back if needed
        target_size = (width, height)
        if result.size != target_size:
            result = result.resize(target_size, Image.LANCZOS)

        # Convert back to BGR numpy array
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

        return result_bgr
