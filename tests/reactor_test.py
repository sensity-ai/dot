"""Tests for reactor backend and temporal smoothing modules."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from scipy.spatial import Delaunay

from dot.reactor_backend import ReactorBackend, FACE_LANDMARK_SUBSET
from dot.temporal import TemporalSmoothing
from dot.conditioning import SourceConditioner


# ── Identity warp test ──────────────────────────────────────────────────────

def test_warp_identity():
    """
    Warping a triangle to itself should produce identical output.
    Uses _warp_triangles directly with a 2-triangle mesh on a synthetic grid.
    """
    source = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(source, (50, 50), (150, 150), (0, 255, 0), -1)  # green square

    source_lms = np.array([[50, 50], [150, 50], [150, 150], [50, 150],
                            [100, 100], [50, 100], [150, 100], [100, 50]], dtype=np.float32)
    tri = Delaunay(source_lms)
    src_tris = source_lms[tri.simplices]
    dst_tris = src_tris.copy()  # identity

    result = ReactorBackend._warp_triangles(source, src_tris, dst_tris)

    diff = np.mean(np.abs(result.astype(np.float32) - source.astype(np.float32)))
    print(f"[test] identity warp diff: {diff:.2f} (target: <1.0)")
    assert diff < 1.0, f"Identity warp changed pixels by {diff:.2f}"


def test_warp_translation():
    """
    Warp a triangle 30px right should move a blue dot 30px right.
    Uses a single triangle with a blue circle inside.
    """
    source = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(source, (100, 100), 10, (255, 0, 0), -1)  # blue dot

    src_tri = np.array([[80, 80], [120, 80], [100, 120]], dtype=np.float32)
    dst_tri = src_tri + np.array([30, 0], dtype=np.float32)  # shift 30px right

    src_tris = src_tri[np.newaxis, :, :]
    dst_tris = dst_tri[np.newaxis, :, :]

    result = ReactorBackend._warp_triangles(source, src_tris, dst_tris)

    # Blue should now be at (130, 100)
    blue_at_old = result[100, 100, 0].astype(np.float32)
    blue_at_new = result[100, 130, 0].astype(np.float32)
    print(f"[test] blue at old pos: {blue_at_old:.0f} (target: ~0)")
    print(f"[test] blue at new pos: {blue_at_new:.0f} (target: ~255)")
    assert blue_at_new > 200, f"Blue dot not translated to (130,100), value={blue_at_new}"
    assert blue_at_old < 50, f"Blue dot still at (100,100), value={blue_at_old}"


def test_warp_target_shape_can_differ_from_source():
    """
    Camera frames are commonly landscape while the conditioned source may be
    portrait. The warped output must match the target frame shape for blending.
    """
    source = np.zeros((640, 480, 3), dtype=np.uint8)
    cv2.circle(source, (240, 320), 30, (0, 0, 255), -1)

    src_tri = np.array([[200, 260], [280, 260], [240, 380]], dtype=np.float32)
    dst_tri = np.array([[280, 180], [360, 180], [320, 300]], dtype=np.float32)

    result = ReactorBackend._warp_triangles(
        source,
        src_tri[np.newaxis, :, :],
        dst_tri[np.newaxis, :, :],
        output_shape=(480, 640, 3),
    )

    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8
    assert result[240, 320, 2] > 100


# ── Temporal smoothing tests ────────────────────────────────────────────────

def test_temporal_velocity_tracking():
    """
    smooth() should compute velocity before updating prev_landmarks.
    A constant motion should produce nonzero velocity in history.
    """
    smoother = TemporalSmoothing(alpha=0.4)

    # Simulate a face moving right by 2px per frame
    base = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float32)
    for i in range(10):
        lms = base + np.array([i * 2, 0], dtype=np.float32)
        smoother.smooth(lms)

    mean_disp, std_disp = smoother.compute_jitter()
    print(f"[test] jitter mean: {mean_disp:.2f}px (target: >0)")
    assert mean_disp > 0, f"Velocity tracking broken: jitter={mean_disp:.2f} (should be >0)"


def test_temporal_no_motion():
    """
    Static face on a loop should produce very low jitter.
    """
    smoother = TemporalSmoothing(alpha=0.4)

    face = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], dtype=np.float32)
    for _ in range(10):
        smoother.smooth(face)

    mean_disp, _ = smoother.compute_jitter()
    print(f"[test] static jitter: {mean_disp:.4f}px (target: <0.1)")
    assert mean_disp < 0.1, f"Jitter too high for static face: {mean_disp:.4f}"


def test_temporal_reset():
    """
    After reset(), smooth() should reinitialize state.
    Velocity history should be clear after reset.
    """
    smoother = TemporalSmoothing(alpha=0.4)

    face = np.array([[100, 100], [150, 100]], dtype=np.float32)
    smoother.smooth(face)                          # call 1: init (no velocity)
    smoother.smooth(face + 10)                     # call 2: computes velocity → 1 entry
    assert len(smoother.velocity_history) == 1, \
        f"Expected 1 velocity after 2 smooth calls, got {len(smoother.velocity_history)}"

    smoother.reset()
    assert smoother.prev_landmarks is None
    assert len(smoother.velocity_history) == 0
    assert len(smoother.landmark_history) == 0, "Landmark history not cleared"

    # After reset, smooth should reinitialize with current landmarks
    new_face = np.array([[200, 200], [250, 200]], dtype=np.float32)
    result = smoother.smooth(new_face)
    assert np.allclose(result, new_face), "Post-reset smooth should return landmarks as-is"
    print("[test] temporal reset: OK")


# ── Conditioner tests ──────────────────────────────────────────────────────

def test_conditioner_loads_source():
    """SourceConditioner should load and preprocess the test image."""
    cond = SourceConditioner(crop_size=128)
    src = cond.preprocess_source_image(str(SRC.parent / "data" / "source_face.jpg"))
    assert src is not None, "SourceConditioner returned None for valid image"
    assert src.shape[2] == 3, f"Expected 3 channels, got {src.shape}"
    print(f"[test] conditioner: {src.shape} OK")


# ── Full pipeline smoke test ───────────────────────────────────────────────

def test_full_pipeline():
    """
    Load source_face.jpg, process it as both source and target.
    Should return a swapped frame without crashing.
    """
    frame_path = str(SRC.parent / "data" / "source_face.jpg")
    frame = cv2.imread(frame_path)
    assert frame is not None, f"Could not load {frame_path}"

    backend = ReactorBackend({
        "detection_threshold": 0.55,
        "smoothing_alpha": 0.4,
        "crop_size": 128,
    })
    backend.load(frame_path)
    output = backend.process(frame)
    assert output is not None, "process() returned None"
    assert output.shape == frame.shape, f"Shape mismatch: {output.shape} vs {frame.shape}"
    assert output.dtype == np.uint8, f"Expected uint8, got {output.dtype}"

    diff = np.mean(np.abs(output.astype(np.float32) - frame.astype(np.float32)))
    print(f"[test] pipeline diff: {diff:.1f}")
    assert diff > 0, "Output identical to input (swap not running)"


if __name__ == "__main__":
    test_warp_identity()
    test_warp_translation()
    test_temporal_velocity_tracking()
    test_temporal_no_motion()
    test_temporal_reset()
    test_conditioner_loads_source()
    test_full_pipeline()
    print("\n=== ALL TESTS PASSED ===")
