"""
CUDA-accelerated AKAZE feature detection and matching.
"""

from ._cuda_akaze import Akazer, init_device, match
from .akaze_aligner import AkazeAligner

__all__ = ["Akazer", "init_device", "match", "detect_and_compute", "AkazeAligner"]


def detect_and_compute(image, use_fast=True, compute_descriptor=True):
    """
    Detect AKAZE keypoints and compute descriptors.

    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image. float32 (0-1) for standard path, uint8 for fast path.
    use_fast : bool
        If True, use uint8 input (fast path). If False, use float32 input.
    compute_descriptor : bool
        Whether to compute descriptors.

    Returns
    -------
    keypoints : numpy.ndarray
        Shape (N, 6): [x, y, size, angle, response, octave]
    descriptors : numpy.ndarray or None
        Shape (N, 61) if compute_descriptor else None
    """
    import numpy as np

    init_device(0)
    detector = Akazer()

    if use_fast:
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return detector.fast_detect_and_compute(image, compute_descriptor)
    else:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        return detector.detect_and_compute(image, compute_descriptor)
