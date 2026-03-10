"""
AKAZE CUDA aligner: A-KAZE features + match + RANSAC homography.
Feature-based homography alignment using GPU-accelerated A-KAZE.
"""

import cv2
import numpy as np
from typing import Dict, Tuple

from ._cuda_akaze import AkazerPipeline, init_device

DRATIO = 0.8

MODE_FAST = "fast"
MODE_ACCURATE = "accurate"


def _to_gray_u8(x) -> np.ndarray:
    """(B,H,W) or (B,C,H,W) -> (B,H,W) uint8 [0,255]. Accepts numpy or array-like."""
    arr = np.asarray(x)
    if arr.ndim == 4:
        if arr.shape[1] == 3:
            arr = (arr[:, 0] * 0.299 + arr[:, 1] * 0.587 + arr[:, 2] * 0.114)
        else:
            arr = arr[:, 0]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        if arr.shape[0] == 3:
            arr = np.dot(arr.transpose(1, 2, 0), [0.299, 0.587, 0.114])
        else:
            arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            return (arr * 255.0).clip(0, 255).astype(np.uint8)
        return arr.clip(0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


class AkazeAligner:
    """
    AKAZE CUDA: GPU-accelerated A-KAZE features + match + RANSAC homography.
    Feature-based homography alignment using AkazerPipeline (detect + match)
    and cv2.findHomography (RANSAC).
    """

    def __init__(
        self,
        ransac_thresh: float = 2.5,
        ransac_max_iters: int = 2000,
        ransac_confidence: float = 0.995,
        ransac_seed: int | None = None,
        nndr: float = DRATIO,
        max_pts: int = 10000,
        device: int = 0,
        use_nndr: bool = True,
        mode: str = MODE_FAST,
    ):
        self.ransac_thresh = ransac_thresh
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.ransac_seed = ransac_seed
        self.nndr = nndr if use_nndr else 1.0
        self.device = device
        self.mode = mode
        init_device(device)
        self._pipeline = AkazerPipeline(max_pts)

    def _find_transform_one(
        self, template: np.ndarray, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single pair. Returns (H 3x3, motion 2,)."""
        pts0, pts1 = self._pipeline.detect_and_match(
            template, image, self.nndr, self.mode,
        )

        if pts0.shape[0] < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        if self.ransac_seed is not None:
            cv2.setRNGSeed(self.ransac_seed)

        H, mask = cv2.findHomography(
            pts0, pts1, cv2.RANSAC,
            self.ransac_thresh,
            maxIters=self.ransac_max_iters,
            confidence=self.ransac_confidence,
        )
        if H is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        return H.astype(np.float32), H[:2, 2].astype(np.float32)

    def find_transform(self, template, input_image) -> Dict:
        """
        template, input_image: (B,H,W) or (B,C,H,W), numpy or array-like.
        Returns: {"warp_matrix": (B,3,3), "motion": (B,2)}
        """
        t = _to_gray_u8(template)
        i = _to_gray_u8(input_image)
        if t.ndim == 2:
            t = t[np.newaxis]
            i = i[np.newaxis]

        B = t.shape[0]
        warps = []
        motions = []
        for b in range(B):
            H, mot = self._find_transform_one(t[b], i[b])
            warps.append(H)
            motions.append(mot)

        warp = np.stack(warps, axis=0)
        motion = np.stack(motions, axis=0)
        return {"warp_matrix": warp, "motion": motion}


__all__ = ["AkazeAligner", "DRATIO", "MODE_FAST", "MODE_ACCURATE"]
