from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import live


def test_presets_load():
    for path in live.PRESETS.values():
        config = live.load_config(path)
        assert config.get("swap_type", "simswap") == "simswap"


def test_natural_blend_modes():
    natural = live.load_config(live.PRESETS["natural"])
    natural_max = live.load_config(live.PRESETS["natural-max"])
    assert natural["natural_blend_mode"] == "alpha"
    assert natural_max["natural_blend_mode"] == "poisson"
    assert natural["natural_blend_strength"] < 1.0
    assert natural_max["natural_mask_blur"] >= natural["natural_mask_blur"]


def test_512_fallback(tmp_path):
    config = {
        "swap_type": "simswap",
        "crop_size": 512,
        "checkpoints_dir": str(tmp_path / "missing"),
    }
    assert live.ensure_supported_model_config(config)["crop_size"] == 224


def test_virtualcam_missing_is_clear(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pyvirtualcam":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(RuntimeError, match="Virtual camera output requires pyvirtualcam"):
        live.VirtualCameraSink(640, 480)


def test_onnx_missing_model_is_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", Path(tmp_path))
    with pytest.raises(RuntimeError, match="inswapper_128_fp16.onnx"):
        live.OnnxBackend()


def test_source_crop_preparation_preserves_shape():
    crop = np.full((128, 128, 3), 80, dtype=np.uint8)
    crop[:32, :, :] = 230
    crop[:, 40:88, 1] = 120
    prepared = live.prepare_source_crop(crop)
    assert prepared.shape == crop.shape
    assert prepared.dtype == np.uint8
    assert prepared.std() > 0


def test_avatar_style_selects_avatar_backend():
    backend = live.make_backend("simswap", "avatar", {"detection_threshold": 0.5})
    assert isinstance(backend, live.AvatarBackend)


def test_onnx_backend_reads_quality_config(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "ROOT", Path(tmp_path))
    onnx_dir = tmp_path / "saved_models" / "onnx"
    onnx_dir.mkdir(parents=True)
    (onnx_dir / "inswapper_128_fp16.onnx").touch()
    arcface_dir = tmp_path / "saved_models" / "simswap" / "arcface_model"
    arcface_dir.mkdir(parents=True)
    (arcface_dir / "arcface_checkpoint.tar").touch()

    config = {
        "natural_color_match": True,
        "natural_color_match_strength": 0.92,
        "natural_detail_enhance": True,
        "natural_detail_enhance_strength": 0.12,
        "natural_preserve_occluders": True,
        "natural_occluder_threshold": 0.08,
        "natural_occluder_strength": 0.85,
        "natural_mask_blur": 27,
        "natural_blend_strength": 0.80,
    }
    backend = live.OnnxBackend(config)
    assert backend._color_match is True
    assert backend._color_match_strength == 0.92
    assert backend._detail_enhance is True
    assert backend._detail_enhance_strength == 0.12
    assert backend._preserve_occluders is True
    assert backend._occluder_threshold == 0.08
    assert backend._occluder_strength == 0.85
    assert backend._mask_blur == 27
    assert backend._blend_strength == 0.80


def test_onnx_quality_methods_are_callable():
    assert callable(live.OnnxBackend._match_color)
    assert callable(live.OnnxBackend._enhance_detail)


def _write_test_video(path: Path, size: tuple[int, int] = (64, 48), fps: float = 8.0, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        size,
    )
    assert writer.isOpened(), "test video writer did not open"
    for idx in range(frames):
        frame = np.full((size[1], size[0], 3), idx * 40, dtype=np.uint8)
        cv2.putText(frame, str(idx), (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        writer.write(frame)
    writer.release()
    assert path.stat().st_size > 0


def _driver_args(**overrides):
    values = {
        "driver_camera": None,
        "camera": None,
        "driver_video": None,
        "width": None,
        "height": None,
        "fps": None,
        "loop_driver": False,
        "flip_driver": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_driver_is_legacy_camera():
    spec = live.resolve_driver_spec(_driver_args())
    assert spec.kind == "camera"
    assert spec.camera_id == 1
    assert spec.width == 640
    assert spec.height == 480
    assert spec.fps == 20.0


def test_driver_video_and_camera_conflict(tmp_path):
    video = tmp_path / "driver.avi"
    _write_test_video(video)
    with pytest.raises(ValueError, match="either --driver-video or --driver-camera"):
        live.resolve_driver_spec(_driver_args(driver_video=str(video), driver_camera=1))


def test_video_driver_reads_size_fps_and_eof(tmp_path):
    video = tmp_path / "driver.avi"
    _write_test_video(video, size=(80, 60), fps=11.0, frames=2)
    spec = live.resolve_driver_spec(_driver_args(driver_video=str(video)))
    driver = live.open_frame_source(spec)
    try:
        assert driver.width == 80
        assert driver.height == 60
        assert driver.fps > 0
        ok, frame = driver.read()
        assert ok
        assert frame.shape == (60, 80, 3)
        assert driver.read()[0]
        assert not driver.read()[0]
        assert driver.eof_is_normal
    finally:
        driver.close()


def test_video_driver_can_resize_and_loop(tmp_path):
    video = tmp_path / "driver.avi"
    _write_test_video(video, size=(80, 60), frames=1)
    spec = live.resolve_driver_spec(
        _driver_args(driver_video=str(video), width=40, height=30, loop_driver=True)
    )
    driver = live.open_frame_source(spec)
    try:
        for _ in range(3):
            ok, frame = driver.read()
            assert ok
            assert frame.shape == (30, 40, 3)
    finally:
        driver.close()


def test_video_recorder_writes_playable_file(tmp_path):
    output = tmp_path / "out.avi"
    recorder = live.VideoRecorder(output, 64, 48, 8.0)
    try:
        for idx in range(3):
            recorder.write(np.full((48, 64, 3), idx * 50, dtype=np.uint8))
    finally:
        recorder.close()

    assert output.stat().st_size > 0
    cap = cv2.VideoCapture(str(output))
    try:
        ok, frame = cap.read()
        assert ok
        assert frame.shape[:2] == (48, 64)
    finally:
        cap.release()


def test_normalize_fps_filters_invalid_values():
    assert live.normalize_fps(29.97, default=20.0) == pytest.approx(29.97)
    assert live.normalize_fps(0.0, default=20.0) == 20.0
    assert live.normalize_fps(1000.0, default=20.0) == 20.0
    assert live.normalize_fps(float("nan"), default=20.0) == 20.0


class _FakeTimestampCapture:
    def __init__(self, timestamps_ms):
        self.timestamps = list(timestamps_ms)
        self.index = 0

    def read(self):
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC and self.index:
            return float(self.timestamps[self.index - 1])
        return 0.0


def test_estimate_capture_fps_uses_frame_timestamps():
    fake = _FakeTimestampCapture([0.0, 33.0, 66.0, 100.0, 133.0])
    fps = live.estimate_capture_fps(fake, sample_frames=8)
    assert fps == pytest.approx(30.3, rel=0.05)
