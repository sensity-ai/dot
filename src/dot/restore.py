"""
Face restoration stage for DOT face swap.

Adds GFPGAN restoration after swap but before paste, with frame-skipping
for performance. Optional - disabled by default to preserve "golden path".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

RestoreMode = Literal["none", "gfpgan", "codeformer"]


@dataclass
class RestoreConfig:
    mode: RestoreMode = "none"
    strength: float = 0.4
    every: int = 3  # restore every N frames (1 = every frame, 3 = ~3x faster)


class FaceRestorer:
    """GFPGAN/CodeFormer restorer with frame-skipping for live performance."""

    MAX_ERRORS = 3

    def __init__(self, cfg: RestoreConfig):
        self.cfg = cfg
        self._impl = None
        self._frame_count = 0
        self._error_count = 0
        self._disabled = False
        self._last = None  # cache last restored crop

        if cfg.mode == "none":
            print("[restore] Restoration disabled")
            return

        if cfg.mode == "gfpgan":
            self._init_gfpgan()
        elif cfg.mode == "codeformer":
            self._init_codeformer()

    def _init_gfpgan(self):
        try:
            from gfpgan import GFPGANer

            model_path = "saved_models/gfpgan/GFPGANv1.4.pth"
            self._impl = GFPGANer(
                model_path=model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device="cpu",
            )
            print(f"[restore] GFPGAN enabled (every {self.cfg.every} frames, strength={self.cfg.strength})")
        except ImportError:
            print("[restore] GFPGAN not installed. Install: pip install gfpgan")
            self.cfg.mode = "none"
        except FileNotFoundError:
            print(f"[restore] GFPGAN model not found at {model_path}")
            print("[restore] Falling back to no restoration")
            self.cfg.mode = "none"

    def _init_codeformer(self):
        raise RuntimeError("CodeFormer not wired yet. Use gfpgan first.")

    def restore_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
        """Restore a BGR face crop. Applies frame-skipping for performance."""
        if self.cfg.mode == "none" or self._impl is None or self._disabled:
            return face_bgr

        self._frame_count += 1

        # Frame skipping: reuse last restored result
        if self.cfg.every > 1 and (self._frame_count % self.cfg.every != 0) and self._last is not None:
            return self._last

        if self.cfg.mode == "gfpgan":
            try:
                cropped_faces, restored_faces, _ = self._impl.enhance(
                    face_bgr,
                    has_aligned=True,
                    only_center_face=True,
                    paste_back=False,
                    weight=self.cfg.strength,
                )
                out = restored_faces[0] if restored_faces else face_bgr
                out = np.ascontiguousarray(out)
                self._last = out
                return out
            except Exception as e:
                self._error_count += 1
                if self._error_count <= self.MAX_ERRORS:
                    print(f"[restore] GFPGAN error ({self._error_count}/{self.MAX_ERRORS}): {e}")
                elif self._error_count == self.MAX_ERRORS + 1:
                    print("[restore] Too many errors, disabling restoration for remainder of session")
                    self._disabled = True
                return face_bgr

        return face_bgr