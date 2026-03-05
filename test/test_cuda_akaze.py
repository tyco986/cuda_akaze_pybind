"""Tests for cuda_akaze module."""
import numpy as np
import pytest

try:
    import cuda_akaze
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip(f"cuda_akaze not installed: {e}", allow_module_level=True)


def test_set_device():
    """set_device should not raise."""
    cuda_akaze.set_device(0)


def test_detect_and_compute_fast():
    """detect_and_compute (fast mode) returns valid keypoints and descriptors."""
    np.random.seed(42)
    img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    img = np.ascontiguousarray(img)

    result = cuda_akaze.detect_and_compute(img, use_fast=True, max_pts=1000)

    assert "keypoints" in result
    assert "descriptors" in result

    kp = result["keypoints"]
    desc = result["descriptors"]

    assert kp.ndim == 2
    assert kp.shape[1] == 6  # x, y, octave, response, size, angle
    assert desc.ndim == 2
    assert desc.shape[1] == 61  # AKAZE MLDB descriptor size
    assert kp.shape[0] == desc.shape[0]


def test_detect_and_compute_float():
    """detect_and_compute (standard mode) with float32 input."""
    np.random.seed(42)
    img = np.random.rand(128, 128).astype(np.float32)  # [0,1] normalized
    img = np.ascontiguousarray(img)

    result = cuda_akaze.detect_and_compute(img, use_fast=False, max_pts=1000)

    assert "keypoints" in result
    assert "descriptors" in result
    assert result["keypoints"].shape[0] == result["descriptors"].shape[0]


def test_match():
    """match returns valid structure."""
    np.random.seed(123)
    img1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)

    result = cuda_akaze.match(img1, img2, use_fast=True, max_pts=1000)

    assert "keypoints1" in result
    assert "keypoints2" in result
    assert "descriptors1" in result
    assert "descriptors2" in result
    assert "matches" in result

    assert result["keypoints1"].shape[1] == 6
    assert result["descriptors1"].shape[1] == 61
    for m in result["matches"]:
        assert len(m) == 3
