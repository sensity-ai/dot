#!/usr/bin/env python3
"""Standalone live face swap.

Input: source identity image/video + camera or liveness driver video.
Output: OpenCV window by default, optional Python virtual camera or recording.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")

ROOT = Path(__file__).resolve().parent
WINDOW_NAME = "DOT - Live Deepfake"
MIN_VALID_FPS = 1.0
MAX_VALID_FPS = 240.0
DEFAULT_FPS = 20.0

sys.path.insert(0, str(ROOT / "src"))

from dot.commons.utils import get_device  # noqa: E402
from dot.simswap import SimswapOption  # noqa: E402
from dot.simswap.fs_model import legacy_simswap_import_path  # noqa: E402
from dot.commons.shared_detector import get_static_detector, get_live_detector  # noqa: E402
from dot.reactor_backend import ReactorBackend  # noqa: E402
from dot.restore import RestoreConfig, FaceRestorer  # noqa: E402
from dot.temporal import TemporalConfig, TemporalSmoother  # noqa: E402

PresetName = Literal["fast", "balanced", "natural", "natural-max", "reactor"]
BackendName = Literal["simswap", "onnx", "reactor"]
OutputName = Literal["window", "virtualcam", "both", "none"]
StyleName = Literal["swap", "avatar"]

PRESETS: dict[PresetName, Path] = {
    "fast": ROOT / "configs" / "simswap.yaml",
    "balanced": ROOT / "configs" / "m2_8gb_balanced.yaml",
    "natural": ROOT / "configs" / "m2_8gb_natural.yaml",
    "natural-max": ROOT / "configs" / "m2_8gb_natural_max.yaml",
    "reactor": ROOT / "configs" / "m2_8gb_reactor.yaml",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_supported_model_config(config: dict) -> dict:
    if config.get("swap_type", "simswap") != "simswap":
        return config
    if int(config.get("crop_size", 224)) != 512:
        return config

    checkpoints_dir = Path(config.get("checkpoints_dir", "saved_models/simswap/checkpoints"))
    checkpoint_512 = checkpoints_dir / "512" / "550000_net_G.pth"
    if checkpoint_512.exists():
        return config

    print("[dot] 512px SimSwap checkpoint missing; falling back to 224px.")
    print(f"[dot] Missing: {checkpoint_512}")
    config = dict(config)
    config["crop_size"] = 224
    return config


def normalize_fps(
    value: float | int | None,
    default: float = DEFAULT_FPS,
    min_fps: float = MIN_VALID_FPS,
    max_fps: float = MAX_VALID_FPS,
) -> float:
    try:
        fps = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(fps) or fps < min_fps or fps > max_fps:
        return float(default)
    return fps


def estimate_capture_fps(capture: cv2.VideoCapture, sample_frames: int = 48) -> float:
    timestamps: list[float] = []
    for _ in range(sample_frames):
        ok, _ = capture.read()
        if not ok:
            break
        ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        if ms > 0:
            timestamps.append(ms)

    if len(timestamps) < 2:
        return 0.0

    deltas = [next_ms - prev_ms for prev_ms, next_ms in zip(timestamps, timestamps[1:]) if next_ms > prev_ms]
    if not deltas:
        return 0.0

    median_delta = float(np.median(np.asarray(deltas, dtype=np.float32)))
    if median_delta <= 0:
        return 0.0
    return 1000.0 / median_delta


def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera {camera_id} is not available. On macOS, grant Camera "
            "permission to the terminal app, then restart the terminal."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        raise RuntimeError(
            f"Camera {camera_id} opened but returned no frames. Close other apps "
            "using the camera and rerun."
        )
    return cap


@dataclass(frozen=True)
class DriverSpec:
    kind: Literal["camera", "video"]
    camera_id: int | None
    video_path: Path | None
    width: int | None
    height: int | None
    fps: float
    loop: bool
    flip: bool


class FrameSource:
    """Frame driver for either a live camera or prerecorded liveness video."""

    eof_is_normal = False

    def __init__(self, width: int, height: int, fps: float, flip: bool = False):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.flip = flip

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class CameraFrameSource(FrameSource):
    def __init__(self, camera_id: int, width: int, height: int, fps: float):
        self.camera_id = camera_id
        self._cap = open_camera(camera_id, width, height)
        super().__init__(width, height, normalize_fps(fps, default=DEFAULT_FPS), flip=True)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if ok and frame is not None and self.flip:
            frame = cv2.flip(frame, 1)
        return ok, frame

    def close(self) -> None:
        self._cap.release()


class VideoFrameSource(FrameSource):
    eof_is_normal = True

    def __init__(
        self,
        path: Path,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        loop: bool = False,
        flip: bool = False,
    ):
        self.path = path
        self.loop = loop
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open driver video: {path}")

        native_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        native_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        native_fps = normalize_fps(self._cap.get(cv2.CAP_PROP_FPS), default=0.0)
        if native_fps <= 0:
            estimated_fps = normalize_fps(estimate_capture_fps(self._cap), default=0.0)
            native_fps = estimated_fps if estimated_fps > 0 else DEFAULT_FPS
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
            self._reopen()

        super().__init__(
            width or native_width,
            height or native_height,
            normalize_fps(fps, default=native_fps),
            flip=flip,
        )

    def _rewind(self) -> None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _reopen(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot reopen driver video: {self.path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if (not ok or frame is None) and self.loop:
            self._rewind()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._reopen()
                ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.flip:
            frame = cv2.flip(frame, 1)
        return True, frame

    def close(self) -> None:
        self._cap.release()


class VideoRecorder:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = normalize_fps(fps, default=DEFAULT_FPS)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        suffix = self.path.suffix.lower()
        codec = "MJPG" if suffix == ".avi" else "mp4v"
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*codec),
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open output recorder: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self._writer.write(np.ascontiguousarray(frame))

    def close(self) -> None:
        self._writer.release()


def resolve_driver_spec(args: argparse.Namespace) -> DriverSpec:
    camera_arg = args.driver_camera if args.driver_camera is not None else args.camera
    explicit_camera = camera_arg is not None
    has_video = args.driver_video is not None
    if has_video and explicit_camera:
        raise ValueError("Use either --driver-video or --driver-camera/--camera, not both.")

    if has_video:
        driver_video = Path(args.driver_video)
        if not driver_video.exists():
            raise ValueError(f"Driver video not found: {driver_video}")
        return DriverSpec(
            kind="video",
            camera_id=None,
            video_path=driver_video,
            width=args.width,
            height=args.height,
            fps=args.fps or 0.0,
            loop=args.loop_driver,
            flip=args.flip_driver,
        )

    return DriverSpec(
        kind="camera",
        camera_id=int(camera_arg if camera_arg is not None else 1),
        video_path=None,
        width=args.width or 640,
        height=args.height or 480,
        fps=args.fps or 20.0,
        loop=False,
        flip=True,
    )


def open_frame_source(spec: DriverSpec) -> FrameSource:
    if spec.kind == "video":
        assert spec.video_path is not None
        return VideoFrameSource(
            spec.video_path,
            width=spec.width,
            height=spec.height,
            fps=spec.fps if spec.fps > 0 else None,
            loop=spec.loop,
            flip=spec.flip,
        )
    assert spec.camera_id is not None and spec.width is not None and spec.height is not None
    return CameraFrameSource(spec.camera_id, spec.width, spec.height, spec.fps)


def draw_fps(frame, fps: float):
    frame = np.ascontiguousarray(frame)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    return frame


def read_source_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(path))
        try:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        finally:
            cap.release()
        raise RuntimeError(f"Cannot read a frame from source video: {path}")

    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Cannot read source image: {path}")
    return frame


def prepare_source_crop(crop: np.ndarray) -> np.ndarray:
    """Normalize source crop lighting before extracting identity."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    light, a_chan, b_chan = cv2.split(lab)
    light = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(light)
    normalized = cv2.cvtColor(cv2.merge((light, a_chan, b_chan)), cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(normalized, 5, 35, 35)


def arcface_embedding(net_arc, crop_bgr: np.ndarray, device: str) -> np.ndarray:
    crop_bgr = prepare_source_crop(crop_bgr)
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).to(device)
    tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
    with torch.no_grad():
        embedding = net_arc(tensor).detach().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-6)
    return embedding / norm


def prepare_source_file(source: Path, output: Path, size: int = 256) -> Path:
    frame = read_source_frame(source)
    detector = get_static_detector(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        mode="None",
    )
    result = detector.get(frame, size)
    if result is None:
        raise RuntimeError(f"No face detected in source: {source}")
    crop = prepare_source_crop(result[0][0])
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), crop):
        raise RuntimeError(f"Could not write prepared source: {output}")
    return output


def soft_paste(frame: np.ndarray, patch: np.ndarray, matrix: np.ndarray, blur: int = 21) -> np.ndarray:
    height, width = frame.shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    warped = cv2.warpAffine(patch, inverse, (width, height), borderMode=cv2.BORDER_REFLECT)
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        (patch.shape[1] // 2, patch.shape[0] // 2),
        (int(patch.shape[1] * 0.43), int(patch.shape[0] * 0.50)),
        0,
        0,
        360,
        255,
        -1,
    )
    mask = cv2.warpAffine(mask, inverse, (width, height), borderMode=cv2.BORDER_CONSTANT)
    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = warped.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


class VirtualCameraSink:
    def __init__(self, width: int, height: int, fps: int = 20):
        try:
            import pyvirtualcam
        except ImportError as exc:
            raise RuntimeError(
                "Virtual camera output requires pyvirtualcam. Install it in the "
                "dot environment and initialize a supported virtual camera provider "
                "first, for example OBS 30+ on macOS 13+."
            ) from exc

        self._pyvirtualcam = pyvirtualcam
        self._cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        print(f"[dot] Virtual camera: {self._cam.device}")

    def send_bgr(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._cam.send(rgb)
        if hasattr(self._cam, "sleep_until_next_frame"):
            self._cam.sleep_until_next_frame()

    def close(self):
        self._cam.close()


class SimSwapBackend:
    def __init__(self, config: dict):
        self.config = config
        self.option = SimswapOption(
            use_gpu=True,
            crop_size=config.get("crop_size", 224),
            use_mask=config.get("use_mask", False),
        )

    def load(self, source: Path):
        config = self.config
        crop_size = config.get("crop_size", 224)
        print("[dot] Loading SimSwap models ...")
        self.option.create_model(
            opt_crop_size=crop_size,
            opt_fp16=config.get("use_fp16", False),
            max_num_faces=config.get("max_num_faces", 1),
            detection_threshold=config.get("detection_threshold", 0.6),
            min_detection_confidence=config.get("detection_threshold", 0.6),
            parsing_model_path=config.get("parsing_model_path"),
            arcface_model_path=config.get("arcface_model_path"),
            checkpoints_dir=config.get("checkpoints_dir"),
        )
        print("[dot] Loading source identity ...")
        self.option.change_option(str(source))
        print("[dot] Ready.")

    def process(self, frame):
        config = self.config
        return self.option.process_image(
            frame,
            use_cam=True,
            natural_color_match=config.get("natural_color_match", False),
            natural_color_match_strength=config.get("natural_color_match_strength", 0.65),
            natural_detail_enhance=config.get("natural_detail_enhance", False),
            natural_detail_enhance_strength=config.get("natural_detail_enhance_strength", 0.12),
            natural_preserve_occluders=config.get("natural_preserve_occluders", False),
            natural_occluder_threshold=config.get("natural_occluder_threshold", 0.10),
            natural_occluder_strength=config.get("natural_occluder_strength", 0.90),
            natural_blend_mode=config.get("natural_blend_mode", "alpha"),
            natural_blend_strength=config.get("natural_blend_strength", 1.0),
            natural_mask_blur=config.get("natural_mask_blur", 0),
        )


class OnnxBackend:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.model_path = ROOT / "saved_models" / "onnx" / "inswapper_128_fp16.onnx"
        self.arcface_path = ROOT / "saved_models" / "simswap" / "arcface_model" / "arcface_checkpoint.tar"
        if not self.model_path.exists():
            raise RuntimeError(
                "ONNX backend requires saved_models/onnx/inswapper_128_fp16.onnx. "
                "The SimSwap backend is the installed default: use --backend simswap."
            )
        if not self.arcface_path.exists():
            raise RuntimeError(
                "ONNX backend requires saved_models/simswap/arcface_model/arcface_checkpoint.tar "
                "for source identity extraction."
            )
        self.detector = get_live_detector(
            max_num_faces=int(self.config.get("max_num_faces", 1)),
            refine_landmarks=True,
            min_detection_confidence=float(self.config.get("detection_threshold", 0.55)),
            min_tracking_confidence=0.5,
            mode="None",
        )
        self.session = None
        self.source_embedding: np.ndarray | None = None
        self._color_match = bool(self.config.get("natural_color_match", True))
        self._color_match_strength = float(self.config.get("natural_color_match_strength", 0.95))
        self._detail_enhance = bool(self.config.get("natural_detail_enhance", True))
        self._detail_enhance_strength = float(self.config.get("natural_detail_enhance_strength", 0.08))
        self._preserve_occluders = bool(self.config.get("natural_preserve_occluders", True))
        self._occluder_threshold = float(self.config.get("natural_occluder_threshold", 0.10))
        self._occluder_strength = float(self.config.get("natural_occluder_strength", 0.90))
        self._mask_blur = int(self.config.get("natural_mask_blur", 31))
        self._blend_strength = float(self.config.get("natural_blend_strength", 0.86))
        self._restore_mode = self.config.get("restore_mode", "none")
        self._restore_strength = float(self.config.get("restore_strength", 0.4))
        self._restore_every = int(self.config.get("restore_every", 3))
        self.restorer = FaceRestorer(RestoreConfig(
            mode=self._restore_mode,
            strength=self._restore_strength,
            every=self._restore_every,
        ))

        self._temporal_enabled = bool(self.config.get("temporal_enabled", False))
        self._temporal_alpha = float(self.config.get("temporal_alpha", 0.35))
        self.smoother = TemporalSmoother(TemporalConfig(
            enabled=self._temporal_enabled,
            alpha_matrix=self._temporal_alpha,
        ))

    def _load_arcface(self):
        device = get_device()
        with legacy_simswap_import_path():
            net_arc = torch.load(self.arcface_path, weights_only=False, map_location=device)
        net_arc = net_arc.to(device)
        net_arc.eval()
        return net_arc

    def _load_embedding_map(self) -> np.ndarray:
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise RuntimeError(
                "ONNX backend needs the onnx package to extract inswapper's buff2fs "
                "embedding map. Install it with: conda run -n dot python -m pip install onnx"
            ) from exc

        model = onnx.load(str(self.model_path))
        for initializer in model.graph.initializer:
            if initializer.name == "buff2fs":
                return numpy_helper.to_array(initializer).astype(np.float32)
        raise RuntimeError("ONNX model is missing the buff2fs source embedding map.")

    def load(self, source: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX backend requires onnxruntime in the dot environment.") from exc

        available = ort.get_available_providers()
        providers = [provider for provider in ("CoreMLExecutionProvider", "CPUExecutionProvider") if provider in available]
        providers = providers or available
        print("[dot] Loading ONNX inswapper ...")
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        print(f"[dot] ONNX providers: {', '.join(self.session.get_providers())}")

        source_frame = read_source_frame(source)
        detector = get_static_detector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(self.config.get("detection_threshold", 0.55)),
            mode="None",
        )
        result = detector.get(source_frame, 128)
        if result is None:
            raise RuntimeError(f"No face detected in source for ONNX backend: {source}")

        print("[dot] Preparing source crop and extracting ONNX identity ...")
        source_crop = result[0][0]
        net_arc = self._load_arcface()
        embedding = arcface_embedding(net_arc, source_crop, get_device())
        del net_arc
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()

        embedding_map = self._load_embedding_map()
        embedding = np.dot(embedding, embedding_map)
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True).clip(min=1e-6)
        self.source_embedding = embedding.astype(np.float32)
        print("[dot] Ready.")

    def process(self, frame):
        if self.session is None or self.source_embedding is None:
            raise RuntimeError("ONNX backend was not loaded.")

        result = self.detector.get(frame, 128)
        if result is None:
            self.smoother.reset()
            return frame

        output = frame
        crops, matrices = result
        for crop, matrix in zip(crops, matrices):
            matrix = self.smoother.smooth_matrix(matrix)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
            target = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None, ...].astype(np.float32)
            prediction = self.session.run(None, {"target": target, "source": self.source_embedding})[0]
            fake_rgb = prediction[0].transpose(1, 2, 0)
            fake_rgb = np.clip(fake_rgb * 127.5 + 127.5, 0, 255).astype(np.uint8)
            fake_bgr = cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2BGR)

            fake_bgr = self.restorer.restore_bgr(fake_bgr)
            fake_bgr = self._apply_quality(fake_bgr, crop)
            output = soft_paste(output, fake_bgr, matrix, blur=self._mask_blur)
        return output

    def _apply_quality(self, swapped: np.ndarray, target_crop: np.ndarray) -> np.ndarray:
        if self._color_match:
            swapped = self._match_color(swapped, target_crop, self._color_match_strength)
        if self._detail_enhance:
            swapped = self._enhance_detail(swapped, self._detail_enhance_strength)
        if self._preserve_occluders:
            swapped = self._preserve_occluders_impl(swapped, target_crop)
        if self._blend_strength < 1.0:
            swapped = cv2.addWeighted(swapped, self._blend_strength, target_crop, 1.0 - self._blend_strength, 0)
        return swapped

    @staticmethod
    def _match_color(swapped: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0:
            return swapped
        lab_swap = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB)
        lab_target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
        src_mean, src_std = lab_swap.mean(axis=(0, 1)), lab_swap.std(axis=(0, 1))
        tgt_mean, tgt_std = lab_target.mean(axis=(0, 1)), lab_target.std(axis=(0, 1))
        matched = ((lab_swap.astype(np.float32) - src_mean) / src_std.clip(min=1e-4)) * tgt_std + tgt_mean
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        return cv2.addWeighted(swapped, 1.0 - strength, matched, strength, 0)

    @staticmethod
    def _enhance_detail(swapped: np.ndarray, strength: float) -> np.ndarray:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength <= 0:
            return swapped
        blurred = cv2.GaussianBlur(swapped, (0, 0), 1.5)
        high_freq = cv2.subtract(swapped.astype(np.float32), blurred.astype(np.float32))
        enhanced = swapped.astype(np.float32) + high_freq * strength
        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _preserve_occluders_impl(self, swapped: np.ndarray, target: np.ndarray) -> np.ndarray:
        luminance = target.mean(axis=2).astype(np.float32) / 255.0
        occlusion = (luminance < self._occluder_threshold).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion = cv2.dilate(occlusion, kernel, iterations=1)
        occlusion = cv2.GaussianBlur(occlusion, (7, 7), 0)
        occlusion = np.clip(occlusion * self._occluder_strength, 0.0, 1.0)
        occlusion = occlusion[:, :, None]
        return np.clip(swapped.astype(np.float32) * (1.0 - occlusion) + target.astype(np.float32) * occlusion, 0, 255).astype(np.uint8)


class AvatarBackend:
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    IRIS_LEFT = [474, 475, 476, 477]
    IRIS_RIGHT = [469, 470, 471, 472]

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            self._mesh = None
        self.detector = get_live_detector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(self.config.get("detection_threshold", 0.55)),
            min_tracking_confidence=0.5,
            mode="None",
        )
        self.static_detector = get_static_detector(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=float(self.config.get("detection_threshold", 0.55)),
            mode="None",
        )
        self.source_palette: dict | None = None

    @staticmethod
    def _landmarks_to_array(landmarks, width: int, height: int) -> np.ndarray:
        return np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float32)

    def _extract_palette(self, frame: np.ndarray, landmarks) -> dict:
        h, w = frame.shape[:2]
        points = self._landmarks_to_array(landmarks, w, h).astype(np.int32)
        skin_mask = np.zeros((h, w), dtype=np.uint8)
        oval = points[np.array(self.FACE_OVAL)]
        cv2.fillPoly(skin_mask, [oval], 255)
        for eye_idx in self.LEFT_EYE + self.RIGHT_EYE:
            pt = tuple(points[eye_idx])
            cv2.circle(skin_mask, pt, max(1, int(np.linalg.norm(points[eye_idx] - points[eye_idx + 1] if eye_idx + 1 < len(points) else points[eye_idx]))), 0, -1)
        skin_pixels = frame[skin_mask > 0]
        if len(skin_pixels) == 0:
            skin_pixels = frame[h // 3: 2 * h // 3, w // 3: 2 * w // 3].reshape(-1, 3)
        return {
            "skin": skin_pixels.mean(axis=0).astype(np.float32) if len(skin_pixels) > 0 else np.array([200.0, 180.0, 170.0]),
            "lips": frame[int(points[0][1]), int(points[0][0])].astype(np.float32) if 0 <= int(points[0][1]) < h and 0 <= int(points[0][0]) < w else np.array([100.0, 80.0, 120.0]),
            "left_eye": frame[int(points[33][1]), int(points[33][0])].astype(np.float32) if 0 <= int(points[33][1]) < h and 0 <= int(points[33][0]) < w else np.array([30.0, 30.0, 30.0]),
            "right_eye": frame[int(points[362][1]), int(points[362][0])].astype(np.float32) if 0 <= int(points[362][1]) < h and 0 <= int(points[362][0]) < w else np.array([30.0, 30.0, 30.0]),
        }

    def load(self, source: Path):
        frame = read_source_frame(source)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb) if self._mesh else None
        if results and results.multi_face_landmarks:
            self.source_palette = self._extract_palette(frame, results.multi_face_landmarks[0].landmark)
            print(f"[dot] Avatar palette: skin={self.source_palette['skin'].astype(int).tolist()}")
        else:
            result = self.static_detector.get(frame, 256)
            if result is not None:
                crop = result[0][0]
                self.source_palette = {
                    "skin": crop.reshape(-1, 3).mean(axis=0).astype(np.float32),
                    "lips": crop[64, 64].astype(np.float32),
                    "left_eye": np.array([30.0, 30.0, 30.0]),
                    "right_eye": np.array([30.0, 30.0, 30.0]),
                }
        if self.source_palette is None:
            self.source_palette = {
                "skin": np.array([200.0, 180.0, 170.0]),
                "lips": np.array([120.0, 90.0, 140.0]),
                "left_eye": np.array([30.0, 30.0, 30.0]),
                "right_eye": np.array([30.0, 30.0, 30.0]),
            }
        print("[dot] Avatar style ready.")

    def _stylize(self, crop: np.ndarray) -> np.ndarray:
        s = 256
        crop = cv2.resize(crop, (s, s), interpolation=cv2.INTER_CUBIC)
        smooth = cv2.bilateralFilter(crop, 7, 60, 60)
        levels = 10
        posterized = np.round(smooth.astype(np.float32) / (256 / levels)) * (256 / levels)
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        skin = self.source_palette["skin"] if self.source_palette else np.array([200.0, 180.0, 170.0])
        posterized = cv2.addWeighted(posterized.astype(np.float32), 0.75, np.full_like(posterized, skin), 0.25, 0)
        posterized = np.clip(posterized, 0, 255).astype(np.uint8)
        xc, yc = s // 2, s // 2 + 4
        for (ex, ey), factor in [((s // 2 - 38, s // 2 - 6), 0.24), ((s // 2 + 38, s // 2 - 6), 0.24)]:
            xx, yy = np.mgrid[0:s, 0:s].astype(np.float32)
            dx, dy = xx - ex, yy - ey
            dist = np.sqrt(dx * dx + dy * dy)
            radius = 26.0
            mask = np.clip(1.0 - dist / radius, 0.0, 1.0) ** 3
            scale = 1.0 - factor * mask
            map_x = np.clip(ex + (xx - ex) * scale, 0, s - 1).astype(np.float32)
            map_y = np.clip(ey + (yy - ey) * scale, 0, s - 1).astype(np.float32)
            posterized = cv2.remap(posterized, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        specular = np.zeros_like(posterized)
        for (sx, sy) in [(s // 2 - 42, s // 2 - 16), (s // 2 + 34, s // 2 - 16)]:
            cv2.ellipse(specular, (sx, sy), (6, 8), 0, 0, 360, (255, 240, 220), -1)
            cv2.ellipse(specular, (sx + 2, sy - 1), (3, 4), 0, 0, 360, (255, 255, 255), -1)
        specular = cv2.GaussianBlur(specular, (5, 5), 0)
        posterized = np.clip(posterized.astype(np.float32) + specular.astype(np.float32) * 0.3, 0, 255).astype(np.uint8)
        lips = self.source_palette["lips"] if self.source_palette else np.array([140.0, 100.0, 130.0])
        lip_pts = np.array([[64, 112], [80, 118], [96, 112], [128, 112], [144, 118], [160, 112],
                            [160, 122], [144, 126], [128, 126], [96, 126], [80, 126], [64, 122]], dtype=np.int32)
        lip_mask = np.zeros((s, s), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [lip_pts], 255)
        lip_mask = cv2.GaussianBlur(lip_mask, (5, 5), 0)
        lip_region = lip_mask.astype(np.float32)[:, :, None] / 255.0
        posterized = np.clip(posterized.astype(np.float32) * (1.0 - lip_region * 0.5) + lips * lip_region * 0.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(posterized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 85)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        posterized[edges > 0] = (18, 18, 18)
        cv2.ellipse(posterized, (xc, yc), (int(s * 0.41), int(s * 0.46)), 0, 0, 360, (28, 28, 28), 2)
        return posterized

    def process(self, frame):
        landmarks = None
        if self._mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mesh.process(rgb)
            if results and results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
        if landmarks:
            h, w = frame.shape[:2]
            points = self._landmarks_to_array(landmarks, w, h)
            left_eye = points[[33, 133]].mean(axis=0)
            right_eye = points[[362, 263]].mean(axis=0)
            dst_pts = np.array([left_eye, right_eye, points[1]], dtype=np.float32)
            src_pts = np.array([[60, 96], [196, 96], [128, 60]], dtype=np.float32)
            matrix = cv2.getAffineTransform(src_pts, dst_pts)
            res = self.detector.get(frame, 256)
            if res is None:
                return frame
            stylized = self._stylize(res[0][0])
            return soft_paste(frame, stylized, res[1][0], blur=27)
        res = self.detector.get(frame, 256)
        if res is None:
            return frame
        stylized = self._stylize(res[0][0])
        return soft_paste(frame, stylized, res[1][0], blur=27)


def make_backend(name: str, style: str, config: dict):
    if style == "avatar":
        return AvatarBackend(config)
    if name == "reactor":
        return ReactorBackend(config)
    if name == "onnx":
        return OnnxBackend(config)
    return SimSwapBackend(config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local live face swap: source identity + camera/liveness video -> output."
    )
    parser.add_argument("--source", default="data/source_face.webm")
    parser.add_argument("--camera", type=int, default=None, help="legacy alias for --driver-camera")
    parser.add_argument("--driver-camera", type=int, default=None, help="physical camera id used as motion driver")
    parser.add_argument("--driver-video", help="prerecorded liveness video used as motion driver")
    parser.add_argument("--backend", choices=["simswap", "onnx", "reactor"], default="simswap")
    parser.add_argument("--style", choices=["swap", "avatar"], default="swap")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="natural")
    parser.add_argument("--output", choices=["window", "virtualcam", "both", "none"], default="window")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="recording/virtual camera FPS; video drivers default to native FPS")
    parser.add_argument("--loop-driver", action="store_true", help="loop --driver-video instead of stopping at EOF")
    parser.add_argument("--flip-driver", action="store_true", help="mirror prerecorded driver video frames")
    parser.add_argument("--record-output", help="write processed output video to this path")
    parser.add_argument("--prepare-source", help="write an aligned, lighting-normalized source image and exit")
    parser.add_argument("--restore", choices=["none", "gfpgan"], default="none", help="face restoration after swap (default: none)")
    parser.add_argument("--restore-strength", type=float, default=0.4, help="restoration strength 0.0-1.0 (default: 0.4)")
    parser.add_argument("--restore-every", type=int, default=3, help="restore every N frames for speed (default: 3)")
    parser.add_argument("--temporal", action="store_true", help="enable temporal smoothing to reduce jitter")
    parser.add_argument("--temporal-alpha", type=float, default=0.35, help="temporal smoothing alpha (default: 0.35)")
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"Source not found: {source}", file=sys.stderr)
        return 1

    if args.prepare_source:
        output = prepare_source_file(source, Path(args.prepare_source))
        print(output)
        return 0

    try:
        driver_spec = resolve_driver_spec(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config_path = PRESETS[args.preset]
    config = ensure_supported_model_config(load_config(config_path))
    config["restore_mode"] = args.restore
    config["restore_strength"] = args.restore_strength
    config["restore_every"] = args.restore_every
    config["temporal_enabled"] = args.temporal
    config["temporal_alpha"] = args.temporal_alpha

    print("DOT live")
    print(f"  Source : {source}")
    if driver_spec.kind == "video":
        print(f"  Driver : video {driver_spec.video_path}")
    else:
        print(f"  Driver : camera {driver_spec.camera_id}")
    print(f"  Backend: {args.backend}")
    print(f"  Style  : {args.style}")
    print(f"  Preset : {args.preset}")
    print(f"  Output : {args.output}")
    if args.record_output:
        print(f"  Record : {args.record_output}")
    print(f"  Device : {get_device()}")
    print(f"  Memory : {psutil.virtual_memory().available / (1024 ** 3):.1f}GB available")
    print()

    virtualcam = None
    driver = None
    recorder = None
    show_window = args.output in {"window", "both"}
    try:
        backend = make_backend(args.backend, args.style, config)
        backend.load(source)
        driver = open_frame_source(driver_spec)
        if args.output in {"virtualcam", "both"}:
            virtualcam = VirtualCameraSink(driver.width, driver.height, int(round(driver.fps)))
        if args.record_output:
            recorder = VideoRecorder(Path(args.record_output), driver.width, driver.height, driver.fps)
            print(f"[dot] Recording: {args.record_output} ({driver.width}x{driver.height} @ {driver.fps:.2f} FPS)")

        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
            cv2.moveWindow(WINDOW_NAME, 500, 250)

        last = time.monotonic()
        fps = 0.0
        failed_reads = 0
        while True:
            ok, frame = driver.read()
            if not ok or frame is None:
                if driver.eof_is_normal:
                    break
                failed_reads += 1
                if failed_reads > 30:
                    raise RuntimeError("Frame driver stopped returning frames.")
                time.sleep(0.03)
                continue
            failed_reads = 0
            swapped = np.ascontiguousarray(backend.process(frame))

            now = time.monotonic()
            elapsed = max(now - last, 1e-6)
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            if not args.no_fps and show_window:
                swapped = draw_fps(swapped, fps)
            if show_window:
                cv2.imshow(WINDOW_NAME, swapped)
            if virtualcam:
                virtualcam.send_bgr(swapped)
            if recorder:
                recorder.write(swapped)

            key = cv2.waitKey(1) if show_window else -1
            if key in {ord("q"), 27}:
                break
    finally:
        if driver:
            driver.close()
        if virtualcam:
            virtualcam.close()
        if recorder:
            recorder.close()
        if show_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "--debug" in sys.argv:
            import traceback

            traceback.print_exc()
        raise SystemExit(1)
