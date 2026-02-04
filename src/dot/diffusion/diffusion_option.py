#!/usr/bin/env python3
"""
Copyright (c) 2022, Sensity B.V. All rights reserved.
licensed under the BSD 3-Clause "New" or "Revised" License.

Base class for diffusion-based face swap models.
"""

import sys
from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from ..commons import ModelOption
from .model_downloader import download_model_file, download_model_repo, get_cache_dir

# Try to import insightface for face analysis
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class DiffusionOption(ModelOption):
    """
    Base class for diffusion-based face swap models.

    Provides common functionality for:
    - Face embedding extraction using InsightFace
    - Model downloading from HuggingFace Hub
    - Common diffusion parameters (inference steps, guidance scale, etc.)
    """

    def __init__(
        self,
        use_gpu: bool = True,
        crop_size: int = 512,
        gpen_type: Optional[str] = None,
        gpen_path: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        use_lcm: bool = False,
    ):
        """
        Initialize DiffusionOption.

        Args:
            use_gpu: Whether to use GPU acceleration
            crop_size: Size to crop/resize images to
            gpen_type: GPEN face enhancement type (gpen_256 or gpen_512)
            gpen_path: Path to GPEN models
            num_inference_steps: Number of diffusion inference steps
            guidance_scale: Classifier-free guidance scale
            use_lcm: Whether to use LCM scheduler for faster inference
        """
        super().__init__(
            gpen_type=gpen_type,
            use_gpu=use_gpu,
            crop_size=crop_size,
            gpen_path=gpen_path,
        )

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.use_lcm = use_lcm

        # Face analysis model (loaded lazily)
        self._face_analyzer = None

        # Source face data (set by change_option)
        self.source_face_embedding = None
        self.source_face_kps = None
        self.source_image = None

    @property
    def device(self) -> str:
        """Get the device to use for inference."""
        if self.use_gpu:
            try:
                import torch
                if torch.backends.mps.is_available():
                    return "mps"
                elif torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        return "cpu"

    @property
    def face_analyzer(self):
        """Lazy-load the InsightFace face analyzer."""
        if self._face_analyzer is None:
            if not INSIGHTFACE_AVAILABLE:
                raise ImportError(
                    "insightface is required for diffusion-based face swap. "
                    "Install with: pip install insightface"
                )

            if self.use_gpu and sys.platform == "darwin":
                providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            elif self.use_gpu:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            self._face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=providers,
            )
            self._face_analyzer.prepare(ctx_id=0 if self.use_gpu else -1)

        return self._face_analyzer

    def _download_model_if_needed(
        self,
        repo_id: str,
        filename: Optional[str] = None,
        subfolder: Optional[str] = None,
    ) -> str:
        """
        Download model from HuggingFace Hub if not already cached.

        Args:
            repo_id: HuggingFace repository ID
            filename: Optional specific file to download
            subfolder: Optional subfolder within the repository

        Returns:
            str: Path to the downloaded model
        """
        if filename:
            return download_model_file(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
            )
        else:
            return download_model_repo(repo_id=repo_id)

    def _get_face_embedding(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract face embedding and keypoints from an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Tuple of (embedding, keypoints) or (None, None) if no face found
        """
        # Convert BGR to RGB for InsightFace
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = self.face_analyzer.get(image_rgb)
        if not faces:
            return None, None

        # Use the largest face
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

        embedding = face.embedding
        kps = face.kps

        return embedding, kps

    def _detect_face(self, image: np.ndarray):
        """
        Detect face in image and return face object.

        Args:
            image: Input image (BGR format)

        Returns:
            Face object or None if no face found
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.face_analyzer.get(image_rgb)
        if not faces:
            return None

        # Return the largest face
        return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    def change_option(self, image: np.ndarray, **kwargs) -> None:
        """
        Set the source image for face swapping.

        Args:
            image: Source image (BGR format)
        """
        self.source_image = image.copy()
        embedding, kps = self._get_face_embedding(image)

        if embedding is None:
            raise ValueError("No face detected in source image")

        self.source_face_embedding = embedding
        self.source_face_kps = kps

    @abstractmethod
    def create_model(self, **kwargs) -> None:
        """
        Initialize the diffusion model.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process a target image and perform face swap.

        Args:
            image: Target image (BGR format)

        Returns:
            np.ndarray: Result image with swapped face
        """
        pass
