"""
Temporal smoothing for DOT face swap.

Applies EMA smoothing to affine matrices and landmarks to reduce jitter in live video.
Optional - disabled by default to preserve golden path performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TemporalSmoothing:
    """EMA-based temporal smoothing for face landmarks (used by ReactorBackend)."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.prev_landmarks: np.ndarray | None = None
        self.landmark_history: list[np.ndarray] = []
        self.velocity_history: list[float] = []
        self._max_history = 120

    def smooth(self, points: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to Nx2 points array."""
        points = points.astype(np.float32, copy=False)
        if self.prev_landmarks is None or points.shape != self.prev_landmarks.shape:
            self.prev_landmarks = points.copy()
            self.landmark_history = [self.prev_landmarks.copy()]
            self.velocity_history = []
            return points

        # Track motion before updating the previous frame state.
        displacement = np.linalg.norm(points - self.prev_landmarks, axis=1)
        self.velocity_history.append(float(displacement.mean()))
        if len(self.velocity_history) > self._max_history:
            self.velocity_history = self.velocity_history[-self._max_history :]

        smoothed = self.alpha * points + (1 - self.alpha) * self.prev_landmarks
        self.prev_landmarks = smoothed.copy()
        self.landmark_history.append(self.prev_landmarks.copy())
        if len(self.landmark_history) > self._max_history:
            self.landmark_history = self.landmark_history[-self._max_history :]
        return smoothed

    def compute_jitter(self) -> tuple[float, float]:
        """Return mean/std pixel displacement from velocity history."""
        if not self.velocity_history:
            return 0.0, 0.0
        values = np.asarray(self.velocity_history, dtype=np.float32)
        return float(values.mean()), float(values.std())

    def reset(self):
        """Reset state - call when no face detected."""
        self.prev_landmarks = None
        self.landmark_history.clear()
        self.velocity_history.clear()


@dataclass
class TemporalConfig:
    enabled: bool = False
    alpha_matrix: float = 0.35


class TemporalSmoother:
    """EMA-based temporal smoothing for affine matrices."""

    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self._prev_matrix = None
        self._frame_count = 0

        if cfg.enabled:
            print(f"[temporal] Enabled (alpha={cfg.alpha_matrix})")

    def smooth_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to affine matrix (2x3 float32)."""
        if not self.cfg.enabled:
            return matrix

        self._frame_count += 1

        if self._prev_matrix is None:
            self._prev_matrix = matrix.copy().astype(np.float32)
            return matrix

        alpha = self.cfg.alpha_matrix
        smoothed = alpha * matrix.astype(np.float32) + (1 - alpha) * self._prev_matrix
        self._prev_matrix = smoothed.copy()
        return smoothed

    def reset(self):
        """Reset state - call when no face detected in frame."""
        self._prev_matrix = None
