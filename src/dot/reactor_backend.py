"""
ReActor-style motion-field face-swapping backend.

Key principle: Transfer motion (expressions) via landmark-based warping,
not per-frame GAN generation. This preserves temporal stability critical
for liveness detection and runs 3-5x faster on M2.

Pipeline:
1. Detect source face landmarks (468 MediaPipe points) → subsample → Delaunay triangulation.
2. Per-frame: detect target landmarks → EMA smooth → per-triangle affine warp.
3. Warp source texture to match target expression/pose.
4. Blend with target using mask + color matching.
5. Return swapped frame.
"""

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from scipy.spatial import Delaunay

from dot.conditioning import SourceConditioner
from dot.commons.shared_detector import get_static_detector, get_live_detector
from dot.temporal import TemporalSmoothing

# Key MediaPipe landmark indices for face shape coverage
# Subset of 478/468 that defines face contour, eyes, brows, nose, mouth
FACE_LANDMARK_SUBSET = np.array(
    [
        # Face oval (jaw to forehead)
        10,
        67,
        69,
        108,
        151,
        152,
        175,
        188,
        205,
        212,
        216,
        234,
        299,
        333,
        337,
        357,
        382,
        395,
        398,
        434,
        # Left eyebrow
        46,
        53,
        70,
        105,
        # Right eyebrow
        276,
        283,
        300,
        334,
        # Left eye
        33,
        133,
        159,
        145,
        154,
        155,
        # Right eye
        362,
        263,
        387,
        373,
        380,
        381,
        # Nose bridge
        1,
        2,
        98,
        195,
        # Nose bottom
        5,
        4,
        278,
        219,
        # Mouth outer
        61,
        291,
        0,
        17,
        37,
        267,
        13,
        14,
        # Mouth inner
        78,
        308,
        81,
        311,
        # Irises (for improved eye tracking)
        468,
        473,
    ],
    dtype=np.int32,
)


class ReactorBackend:
    """Motion-field warping backend for live face swapping."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crop_size = config.get("crop_size", 128)
        self.blend_mode = config.get("blend_mode", "alpha")
        self.blend_strength = config.get("blend_strength", 0.86)
        self.detection_threshold = config.get("detection_threshold", 0.55)
        self.smoothing_alpha = config.get("smoothing_alpha", 0.4)

        # Source data
        self.source_image = None
        self.source_landmarks = None
        self.source_triangulation = None
        self.source_tri_vertices = None

        # Face mesh (live tracking)
        self.detector = get_live_detector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_threshold,
            min_tracking_confidence=0.5,
            mode="None",
        )

        # Additional static detector for source processing
        self.static_detector = get_static_detector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_threshold,
            mode="None",
        )

        # Temporal smoothing
        self.temporal = TemporalSmoothing(alpha=self.smoothing_alpha)

        self.conditioner = SourceConditioner(crop_size=self.crop_size)
        print("[reactor] Backend initialized.")

    def load(self, source_path: Path):
        """Load and preprocess source image, build triangulation."""
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"[reactor] Source not found: {source_path}")

        # Face-aware quality scorer: scores frames using actual face bbox,
        # not the half-blind face_size_score=0.5 fallback.
        def _face_bbox(image):
            lms_list = self.static_detector.get_all_landmarks(image)
            if not lms_list:
                return None
            lms = lms_list[0][:, :2]
            x0, y0 = lms.min(axis=0)
            x1, y1 = lms.max(axis=0)
            return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        if source_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            frames = self.conditioner.extract_frames_from_video(
                source_path,
                face_detection_fn=_face_bbox,
                max_frames=30,
            )
            if not frames:
                raise RuntimeError(
                    f"[reactor] Failed to extract frames from {source_path}"
                )
            self.source_image = self.conditioner.select_best_source_frame(
                frames,
                face_detection_fn=_face_bbox,
            )
        else:
            self.source_image = self.conditioner.preprocess_source_image(
                source_path,
                face_detection_fn=_face_bbox,
            )

        if self.source_image is None:
            raise RuntimeError(
                f"[reactor] Failed to load source: {source_path}. "
                "Check image quality, lighting, and face visibility."
            )

        print(f"[reactor] Source loaded: {source_path} -> {self.source_image.shape}")

        # Detect source landmarks and build triangulation
        landmarks_list = self.static_detector.get_all_landmarks(self.source_image)
        if landmarks_list is None or len(landmarks_list) == 0:
            raise RuntimeError("[reactor] No face detected in source image.")

        all_landmarks = landmarks_list[0]  # (N, 3) where N=468 (or 478)

        # Subsample to key landmarks for speed
        valid_indices = FACE_LANDMARK_SUBSET[FACE_LANDMARK_SUBSET < len(all_landmarks)]
        sampled = all_landmarks[valid_indices, :2]  # Keep only (x, y)

        # Add image corners for triangulation stability
        h, w = self.source_image.shape[:2]
        corner_pts = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        self.source_landmarks = np.vstack([sampled, corner_pts])

        # Build Delaunay triangulation
        try:
            tri = Delaunay(self.source_landmarks)
            self.source_triangulation = tri
            self.source_tri_vertices = self.source_landmarks[tri.simplices]
            print(
                f"[reactor] Triangulation: {len(tri.simplices)} triangles "
                f"from {len(self.source_landmarks)} vertices."
            )
        except Exception as e:
            raise RuntimeError(f"[reactor] Triangulation failed: {e}")

        print("[reactor] Ready.")

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Warp source face onto target frame via landmark triangulation."""
        if (
            self.source_image is None
            or self.source_triangulation is None
            or self.source_tri_vertices is None
        ):
            raise RuntimeError("[reactor] Source not loaded. Call load() first.")

        # Detect target landmarks
        landmarks_list = self.detector.get_all_landmarks(frame)
        if landmarks_list is None or len(landmarks_list) == 0:
            self.temporal.reset()
            return frame  # No face: pass through

        all_landmarks = landmarks_list[0]

        # Subsample
        valid_indices = FACE_LANDMARK_SUBSET[FACE_LANDMARK_SUBSET < len(all_landmarks)]
        target_lms = all_landmarks[valid_indices, :2]

        # Add corner points to match source triangulation
        h, w = frame.shape[:2]
        corner_pts = np.array(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
        )
        target_landmarks = np.vstack([target_lms, corner_pts])

        # EMA smooth
        target_landmarks[:, :2] = self.temporal.smooth(target_landmarks[:, :2])

        # Warp via triangulation
        warped = self._warp_triangles(
            self.source_image,
            self.source_tri_vertices,
            target_landmarks[self.source_triangulation.simplices, :2],
            output_shape=frame.shape,
        )

        # Blend using face landmark polygon mask
        return self._blend(
            warped, frame, target_landmarks[:-4, :2]
        )  # exclude corner points

    @staticmethod
    def _warp_triangles(
        source: np.ndarray,
        src_tris: np.ndarray,
        dst_tris: np.ndarray,
        output_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Warp source via per-triangle affine transforms.
        src_tris, dst_tris: (N, 3, 2) arrays of triangle vertices.

        Uses relative coordinates: affine transform is computed from
        triangle vertices relative to each triangle's bounding box,
        not absolute image coordinates.
        """
        src_h, src_w = source.shape[:2]
        out_h, out_w = (output_shape or source.shape)[:2]
        warped = np.zeros((out_h, out_w, 3), dtype=np.float32)
        weight_map = np.zeros((out_h, out_w, 3), dtype=np.float32)

        for src_tri, dst_tri in zip(src_tris, dst_tris):
            src_tri = src_tri.astype(np.float32)
            dst_tri = dst_tri.astype(np.float32)

            # Bounding boxes
            sx, sy, sw, sh = cv2.boundingRect(src_tri)
            dx, dy, dw, dh = cv2.boundingRect(dst_tri)

            if sw < 1 or sh < 1 or dw < 1 or dh < 1:
                continue

            # Clamp to image bounds
            sx, sy = max(0, sx), max(0, sy)
            dx, dy = max(0, dx), max(0, dy)
            sw = min(sw, src_w - sx)
            sh = min(sh, src_h - sy)
            dw = min(dw, out_w - dx)
            dh = min(dh, out_h - dy)

            if sw < 1 or sh < 1 or dw < 1 or dh < 1:
                continue

            # Use coordinates relative to bounding box for affine transform
            src_local = src_tri - np.array([sx, sy], dtype=np.float32)
            dst_local = dst_tri - np.array([dx, dy], dtype=np.float32)

            try:
                M = cv2.getAffineTransform(src_local, dst_local)
            except cv2.error:
                continue

            src_patch = source[sy : sy + sh, sx : sx + sw]
            if src_patch.size == 0:
                continue

            warped_patch = cv2.warpAffine(
                src_patch,
                M,
                (dw, dh),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            if warped_patch.size == 0:
                continue

            # Triangle mask in destination space
            mask = np.zeros((dh, dw), dtype=np.float32)
            pts_local = dst_tri - np.array([dx, dy])
            cv2.fillPoly(mask, [np.maximum(pts_local, 0).astype(np.int32)], 1.0)

            for c in range(3):
                warped[dy : dy + dh, dx : dx + dw, c] += (
                    warped_patch[:dh, :dw, c] * mask
                )
                weight_map[dy : dy + dh, dx : dx + dw, c] += mask

        weight_map = np.maximum(weight_map, 1e-6)
        result = (warped / weight_map).astype(np.uint8)
        return result

    def _blend(
        self, warped: np.ndarray, target: np.ndarray, face_landmarks: np.ndarray
    ) -> np.ndarray:
        """Blend warped source onto target with mask from face landmark hull."""
        h, w = target.shape[:2]

        # Face mask from landmark convex hull
        mask = np.zeros((h, w), dtype=np.float32)
        hull = cv2.convexHull(face_landmarks.astype(np.int32))
        cv2.fillPoly(mask, [hull], 1.0)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)

        # Blend
        blended = warped.astype(np.float32) * mask[:, :, None] + target.astype(
            np.float32
        ) * (1.0 - mask[:, :, None])

        # Color match
        return self._match_color(blended, target)

    @staticmethod
    def _match_color(swapped: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match L channel of swapped region to target for lighting consistency."""
        swapped_lab = cv2.cvtColor(swapped.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        target_lab = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(
            np.float32
        )

        # Match brightness
        s_mean = np.mean(swapped_lab[:, :, 0])
        t_mean = np.mean(target_lab[:, :, 0])
        swapped_lab[:, :, 0] += t_mean - s_mean

        result = cv2.cvtColor(swapped_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return np.clip(result.astype(np.float32), 0, 255).astype(np.uint8)
