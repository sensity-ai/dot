#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

ReActor face swap implementation.
"""

import sys
from typing import Optional

import cv2
import numpy as np

from ..commons.model_option import _mask_from_bbox
from ..diffusion import DiffusionOption
from ..diffusion.model_downloader import download_model_file, get_cache_dir

# Try to import required dependencies
try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class ReActorOption(DiffusionOption):
    """
    ReActor face swap implementation.

    ReActor uses InsightFace's inswapper model for fast, high-quality face swapping.
    It's the fastest of the diffusion-based options as it doesn't use a full
    diffusion pipeline.
    """

    # Model repository and file info
    INSWAPPER_REPO = "ezioruan/inswapper_128.onnx"
    INSWAPPER_FILE = "inswapper_128.onnx"

    def __init__(
        self,
        use_gpu: bool = True,
        crop_size: int = 512,
        gpen_type: Optional[str] = None,
        gpen_path: Optional[str] = None,
        use_face_restore: bool = True,
        face_restore_model: str = "codeformer",
        filter_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize ReActorOption.

        Args:
            use_gpu: Whether to use GPU acceleration
            crop_size: Size to crop/resize images to
            gpen_type: GPEN face enhancement type
            gpen_path: Path to GPEN models
            use_face_restore: Whether to apply face restoration after swap
            face_restore_model: Face restoration model to use (codeformer or gfpgan)
            filter_type: Camera/film filter to apply
        """
        # ReActor doesn't use standard diffusion parameters
        super().__init__(
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_type=gpen_type,
            gpen_path=gpen_path,
            num_inference_steps=1,  # Not used, but required by parent
            guidance_scale=1.0,  # Not used, but required by parent
            use_lcm=False,
            filter_type=filter_type,
        )

        self.use_face_restore = use_face_restore
        self.face_restore_model = face_restore_model

        # Models (loaded lazily)
        self._swapper = None
        self._face_restorer = None

    def _check_dependencies(self):
        """Check that required dependencies are available."""
        if not ONNX_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for ReActor. "
                "Install with: pip install onnxruntime-gpu (or onnxruntime for CPU)"
            )
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface is required for ReActor. "
                "Install with: pip install insightface"
            )

    def _get_swapper_model_path(self) -> str:
        """Download and return path to inswapper model."""
        return download_model_file(
            repo_id=self.INSWAPPER_REPO,
            filename=self.INSWAPPER_FILE,
        )

    @property
    def swapper(self):
        """Lazy-load the face swapper model."""
        if self._swapper is None:
            self._check_dependencies()

            import insightface

            model_path = self._get_swapper_model_path()

            # Set up ONNX providers (CUDA on Linux/Windows; CoreML on macOS when use_gpu)
            if self.use_gpu:
                if sys.platform == "darwin":
                    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                    print("ReActor: using CoreML (GPU) for ONNX on macOS")
                else:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self._swapper = insightface.model_zoo.get_model(
                model_path,
                providers=providers,
            )

        return self._swapper

    def create_model(
        self,
        reactor_model_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ReActor model.

        Args:
            reactor_model_path: Optional path to custom inswapper model
        """
        self._check_dependencies()

        # Force initialization of face analyzer and swapper
        _ = self.face_analyzer
        _ = self.swapper

        print("ReActor model initialized successfully")

    def change_option(self, image: np.ndarray, **kwargs) -> None:
        """
        Set the source image for face swapping.

        Args:
            image: Source image (BGR format)
        """
        self.source_image = image.copy()

        # Get source face
        source_face = self._detect_face(image)
        if source_face is None:
            raise ValueError("No face detected in source image")

        self.source_face = source_face

    def process_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform face swap on target image.

        Args:
            image: Target image (BGR format)

        Returns:
            np.ndarray: Result image with swapped face
        """
        if self.source_face is None:
            raise ValueError("Source face not set. Call change_option() first.")

        # Detect target face
        target_face = self._detect_face(image)
        if target_face is None:
            # No face in target, return original
            return image

        # Share face region mask for face-only tint/effects (same region as swap)
        face_mask_dict = kwargs.get("_face_mask")
        if isinstance(face_mask_dict, dict) and hasattr(target_face, "bbox"):
            face_mask_dict["mask"] = _mask_from_bbox(
                image.shape, target_face.bbox, expand=1.15, blur=35
            )

        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform swap
        result = self.swapper.get(
            image_rgb,
            target_face,
            self.source_face,
            paste_back=True,
        )

        # Convert back to BGR
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        return result
