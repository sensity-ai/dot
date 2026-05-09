"""
Source video/image conditioning pipeline.

The 5 P's: PRIOR PREPARATION PREVENTS PISS POOR PERFORMANCE.

This module ensures input sources (reference videos/photos) are preprocessed to:
1. Consistent lighting and color balance (histogram matching).
2. Robust face detection and crop quality.
3. Temporal consistency for video sources (frame selection, deduplication).
4. Quality scoring to reject bad frames early.

Garbage in → garbage out. Get the source RIGHT, and downstream warp/blend works.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


class SourceConditioner:
    """Preprocess source images/videos for optimal warp-based face swapping."""

    def __init__(
        self,
        crop_size: int = 128,
        histogram_bins: int = 256,
        quality_threshold: float = 0.5,
        temporal_stride: int = 5,
    ):
        """
        Args:
            crop_size: Target face crop size (will be square).
            histogram_bins: Bins for histogram matching.
            quality_threshold: Min quality score (0-1) to accept a frame.
            temporal_stride: Extract every N frames from video source.
        """
        self.crop_size = crop_size
        self.histogram_bins = histogram_bins
        self.quality_threshold = quality_threshold
        self.temporal_stride = temporal_stride

    @staticmethod
    def adaptive_histogram_equalization(image: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.
        Aggressive: clip_limit=3.0 handles real-world variable lighting (overhead shadows, etc).
        Improves lighting consistency without washing out details.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to LAB, enhance L channel only
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_chan = lab[:, :, 0]
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(6, 6))  # Smaller tiles for fine detail
            l_chan_eq = clahe.apply(l_chan)
            lab[:, :, 0] = l_chan_eq
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(6, 6))
            return clahe.apply(image)

    @staticmethod
    def normalize_color_space(image: np.ndarray) -> np.ndarray:
        """
        Ensure image is in BGR, apply bilateral filtering to smooth noise without losing edges.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # Bilateral filter: smooth while preserving edges
        return cv2.bilateralFilter(image, 5, 50, 50)

    @staticmethod
    def correct_shadows(image: np.ndarray) -> np.ndarray:
        """
        Correct shadows from directional lighting (overhead, etc).
        Uses morphological operations to detect and correct dark regions.
        """
        # Convert to LAB, work on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)

        # Detect shadow regions (low L values)
        _, shadow_mask = cv2.threshold(l_chan, 70, 255, cv2.THRESH_BINARY_INV)
        shadow_mask = shadow_mask.astype(np.float32) / 255.0

        # Dilate shadow mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        shadow_mask = cv2.dilate(shadow_mask, kernel, iterations=1)

        # Brighten shadow regions using morphological reconstruction
        # Compute per-shadow intensity boost
        shadow_boost = 30  # Brighten shadows by ~30 levels
        l_corrected = l_chan + shadow_mask * shadow_boost

        # Clip and update
        lab[:, :, 0] = np.clip(l_corrected, 0, 255).astype(np.uint8)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def enhance_skin_texture(image: np.ndarray, strength: float = 0.15) -> np.ndarray:
        """
        Enhance skin texture (wrinkles, pores) using unsharp masking.
        Critical for real faces: preserve fine detail without oversharpening.
        """
        # Gaussian blur for high-pass filter
        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        # High-pass: original - blurred
        high_freq = cv2.subtract(image.astype(np.float32), blurred.astype(np.float32))
        # Add back with scaling
        enhanced = image.astype(np.float32) + high_freq * strength
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    @staticmethod
    def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Match histogram of source to reference using LAB color space.
        Ensures consistent lighting/color for temporal stability.
        """
        if source.shape != reference.shape:
            return source

        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            src_hist, _ = np.histogram(source_lab[:, :, ch].ravel(), bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(reference_lab[:, :, ch].ravel(), bins=256, range=(0, 256))

            src_cdf = src_hist.cumsum() / src_hist.sum()
            ref_cdf = ref_hist.cumsum() / ref_hist.sum()

            # Simple LUT-based histogram matching
            lut = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                idx = np.searchsorted(ref_cdf, src_cdf[i])
                lut[i] = min(idx, 255)

            source_lab[:, :, ch] = cv2.LUT(source_lab[:, :, ch].astype(np.uint8), lut)

        return cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def compute_quality_score(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        Score image quality: brightness, contrast, blur, face size.

        Args:
            image: BGR image.
            face_bbox: (x, y, w, h) face bounding box. If None, assess whole image.

        Returns:
            Quality score 0-1. Higher is better.
        """
        score = 0.0
        weights = {"brightness": 0.2, "contrast": 0.3, "blur": 0.3, "face_size": 0.2}

        # Crop to face region if provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            x, y = max(0, x), max(0, y)
            w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
            roi = image[y : y + h, x : x + w]
        else:
            roi = image

        if roi.size == 0:
            return 0.0

        # Brightness: mean pixel intensity in normalized range
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2.0  # Optimal ~0.5
        brightness_score = max(0.0, brightness_score)

        # Contrast: standard deviation of pixel intensities
        contrast = np.std(gray) / 127.0  # Optimal ~50-100 std
        contrast_score = min(1.0, contrast)

        # Blur detection: Laplacian variance
        laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 500.0)  # Optimal >500 variance

        # Face size: prefer larger crops (more detail)
        if face_bbox is not None:
            face_area = w * h
            image_area = image.shape[0] * image.shape[1]
            face_size_score = min(1.0, (face_area / image_area) * 10.0)
        else:
            face_size_score = 0.5

        # Weighted sum
        score = (
            weights["brightness"] * brightness_score
            + weights["contrast"] * contrast_score
            + weights["blur"] * blur_score
            + weights["face_size"] * face_size_score
        )
        return np.clip(score, 0.0, 1.0)

    def preprocess_source_image(
        self,
        image_path: Path,
        face_detection_fn=None,
    ) -> Optional[np.ndarray]:
        """
        Load, preprocess, and validate a source image.
        Aggressive pipeline for real-world source photos:
        1. Color normalization
        2. Shadow correction (handles overhead/directional lighting)
        3. Adaptive histogram equalization (handles variable lighting)
        4. Texture enhancement (preserves wrinkles, skin detail)
        5. Quality validation

        Args:
            image_path: Path to image file.
            face_detection_fn: Optional function (image) -> face_bbox or None.

        Returns:
            Preprocessed source image (BGR, normalized, equalized), or None if quality too low.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[dot.conditioning] Failed to load image: {image_path}")
            return None

        original_shape = image.shape
        print(f"[dot.conditioning] Loaded source: {original_shape}")

        # Step 1: Normalize color space
        image = self.normalize_color_space(image)

        # Step 2: Correct shadows (critical for variable lighting)
        image = self.correct_shadows(image)
        print("[dot.conditioning] Shadow correction applied")

        # Step 3: Adaptive histogram equalization (aggressive)
        image = self.adaptive_histogram_equalization(image, clip_limit=3.0)
        print("[dot.conditioning] Lighting equalization applied")

        # Step 4: Detect face and check quality
        face_bbox = None
        if face_detection_fn is not None:
            face_bbox = face_detection_fn(image)

        quality = self.compute_quality_score(image, face_bbox)
        if quality < self.quality_threshold:
            print(
                f"[dot.conditioning] Source image quality too low ({quality:.2f} < {self.quality_threshold}). "
                f"Ensure good lighting, clear face, and high contrast."
            )
            return None

        # Step 5: Texture enhancement (preserve wrinkles and skin detail)
        image = self.enhance_skin_texture(image, strength=0.15)
        print("[dot.conditioning] Skin texture enhancement applied")

        print(
            f"[dot.conditioning] Source image preprocessed "
            f"(quality={quality:.2f}, shape={image.shape})"
        )
        return image

    def extract_frames_from_video(
        self,
        video_path: Path,
        face_detection_fn=None,
        max_frames: int = 30,
    ) -> List[np.ndarray]:
        """
        Extract and preprocess frames from a video source.

        Args:
            video_path: Path to video file.
            face_detection_fn: Optional function (image) -> face_bbox or None.
            max_frames: Max frames to extract.

        Returns:
            List of preprocessed frames (BGR, normalized, equalized).
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[dot.conditioning] Failed to open video: {video_path}")
            return []

        frames = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Temporal stride: extract every N frames
            if frame_idx % self.temporal_stride != 0:
                frame_idx += 1
                continue

            # Preprocess frame
            frame = self.normalize_color_space(frame)

            # Quality check
            face_bbox = None
            if face_detection_fn is not None:
                face_bbox = face_detection_fn(frame)

            quality = self.compute_quality_score(frame, face_bbox)
            if quality < self.quality_threshold:
                print(
                    f"[dot.conditioning] Video frame {frame_idx} skipped (quality={quality:.2f})"
                )
                frame_idx += 1
                continue

            # Enhance
            frame = self.adaptive_histogram_equalization(frame, clip_limit=2.0)
            frames.append(frame)
            frame_idx += 1

        cap.release()
        print(
            f"[dot.conditioning] Extracted {len(frames)} frames from video "
            f"(quality threshold={self.quality_threshold})"
        )
        return frames

    def select_best_source_frame(
        self,
        frames: List[np.ndarray],
        face_detection_fn=None,
    ) -> Optional[np.ndarray]:
        """
        Select the highest-quality frame from a list.

        Args:
            frames: List of BGR frames.
            face_detection_fn: Optional function (image) -> face_bbox or None.

        Returns:
            Best frame, or None if all below quality threshold.
        """
        best_frame = None
        best_score = 0.0

        for frame in frames:
            face_bbox = None
            if face_detection_fn is not None:
                face_bbox = face_detection_fn(frame)

            score = self.compute_quality_score(frame, face_bbox)
            if score > best_score:
                best_score = score
                best_frame = frame

        if best_frame is None or best_score < self.quality_threshold:
            print(
                f"[dot.conditioning] No frames met quality threshold. "
                f"Best score: {best_score:.2f}. Improve lighting, face size, or contrast."
            )
            return None

        print(f"[dot.conditioning] Selected best frame (quality={best_score:.2f})")
        return best_frame

    def build_source_pyramid(
        self,
        source: np.ndarray,
        levels: int = 3,
    ) -> List[np.ndarray]:
        """
        Build a Gaussian pyramid for multi-scale blending and robustness.
        Lower-level detail can improve temporal stability.
        """
        pyramid = [source]
        for _ in range(levels - 1):
            source = cv2.pyrDown(source)
            pyramid.append(source)
        return pyramid

    def histogram_match_to_target(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Match source histogram to target for temporal consistency in live swaps.
        Call per-frame in the main loop if lighting changes detected.
        """
        return self.histogram_match(source, target)
