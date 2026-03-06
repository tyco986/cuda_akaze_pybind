"""
AKAZE CUDA aligner: A-KAZE features + match + RANSAC homography (skimage).
Feature-based homography alignment using GPU-accelerated A-KAZE.
"""

import numpy as np
from typing import Dict, Tuple

from skimage.transform import ProjectiveTransform
from skimage.measure import ransac

from ._cuda_akaze import Akazer, init_device, match

# Fixed seed for reproducible RANSAC
RANSAC_SEED = 42

# NNDR threshold for match filtering (ratio of best to second-best distance)
DRATIO = 0.80

# DiffusivityType: PM_G1=0, PM_G2=1, WEICKERT=2, CHARBONNIER=3
DIFFUSIVITY_PM_G2 = 1


def _to_gray_float32(x) -> np.ndarray:
    """(B,H,W) or (B,C,H,W) -> (B,H,W) float32 [0,1]. Accepts numpy or array-like."""
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
    if arr.dtype in (np.float32, np.float64):
        if arr.max() > 1.0:
            arr = (arr / 255.0).clip(0, 1)
    else:
        arr = (arr.astype(np.float32) / 255.0).clip(0, 1)
    return arr.astype(np.float32)


def _nndr_filter(
    kpts0: np.ndarray,
    desc0: np.ndarray,
    kpts1: np.ndarray,
    desc1: np.ndarray,
    matches: np.ndarray,
    nndr: float,
) -> np.ndarray:
    """
    Filter matches by NNDR (Nearest Neighbor Distance Ratio).
    matches: (N, 2) with [query_idx, train_idx].
    Returns indices of matches passing NNDR.
    Uses scipy.spatial.distance.cdist when available for speed (O(N*M) naive is slow).
    """
    if matches.shape[0] < 2:
        return np.arange(matches.shape[0])
    qidx = matches[:, 0].astype(np.int32)
    desc_q = desc0[qidx].astype(np.float64)
    desc_t = desc1.astype(np.float64)
    try:
        from scipy.spatial.distance import cdist
        dists_sq = cdist(desc_q, desc_t, "sqeuclidean")
    except ImportError:
        dists_sq = ((desc_q[:, np.newaxis] - desc_t[np.newaxis, :]) ** 2).sum(axis=2)
    part = np.partition(dists_sq, 1, axis=1)[:, :2]
    dist0_sq = np.minimum(part[:, 0], part[:, 1])
    dist1_sq = np.maximum(part[:, 0], part[:, 1])
    mask = (dist1_sq > 1e-20) & (dist0_sq < (nndr * nndr) * dist1_sq)
    return np.where(mask)[0]


class AkazeAligner:
    """
    AKAZE CUDA: GPU-accelerated A-KAZE features + match + RANSAC homography (skimage).
    Feature-based homography alignment using _cuda_akaze (Akazer, match).
    All Akazer parameters are exposed and passed to the detector.
    """

    def __init__(
        self,
        # Akazer / AKAZE detector parameters
        noctaves: int = 4,
        max_scale: int = 4,
        per: float = 0.7,
        kcontrast: float = 0.03,
        soffset: float = 1.6,
        reordering: bool = True,
        derivative_factor: float = 1.5,
        dthreshold: float = 0.001,
        diffusivity: int = DIFFUSIVITY_PM_G2,
        descriptor_pattern_size: int = 10,
        # RANSAC / matching parameters
        ransac_thresh: float = 2.5,
        ransac_max_iters: int = 2000,
        ransac_confidence: float = 0.995,
        nndr: float = DRATIO,
        device: int = 0,
        use_nndr: bool = True,
    ):
        self.noctaves = noctaves
        self.max_scale = max_scale
        self.per = per
        self.kcontrast = kcontrast
        self.soffset = soffset
        self.reordering = reordering
        self.derivative_factor = derivative_factor
        self.dthreshold = dthreshold
        self.diffusivity = diffusivity
        self.descriptor_pattern_size = descriptor_pattern_size
        self.ransac_thresh = ransac_thresh
        self.ransac_max_iters = ransac_max_iters
        self.ransac_confidence = ransac_confidence
        self.nndr = nndr
        self.device = device
        self.use_nndr = use_nndr
        init_device(device)
        self._detector = Akazer()

    def _find_transform_one(
        self, template: np.ndarray, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single pair. Returns (H 3x3, motion 2,)."""
        if template.dtype != np.uint8:
            template = (np.clip(template, 0, 1) * 255).astype(np.uint8)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        kpts_t, desc_t = self._detector.fast_detect_and_compute(
            template, True,
            noctaves=self.noctaves,
            max_scale=self.max_scale,
            per=self.per,
            kcontrast=self.kcontrast,
            soffset=self.soffset,
            reordering=self.reordering,
            derivative_factor=self.derivative_factor,
            dthreshold=self.dthreshold,
            diffusivity=self.diffusivity,
            descriptor_pattern_size=self.descriptor_pattern_size,
        )
        kpts_i, desc_i = self._detector.fast_detect_and_compute(
            image, True,
            noctaves=self.noctaves,
            max_scale=self.max_scale,
            per=self.per,
            kcontrast=self.kcontrast,
            soffset=self.soffset,
            reordering=self.reordering,
            derivative_factor=self.derivative_factor,
            dthreshold=self.dthreshold,
            diffusivity=self.diffusivity,
            descriptor_pattern_size=self.descriptor_pattern_size,
        )

        if kpts_t is None or desc_t is None or kpts_i is None or desc_i is None:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        n_t, n_i = kpts_t.shape[0], kpts_i.shape[0]
        if n_t < 4 or n_i < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        matches = match(kpts_t, desc_t, kpts_i, desc_i)
        if matches is None or matches.shape[0] == 0:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        if self.use_nndr:
            valid = _nndr_filter(kpts_t, desc_t, kpts_i, desc_i, matches, self.nndr)
        else:
            valid = np.arange(matches.shape[0])
        if len(valid) < 4:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

        m = matches[valid]
        pts0 = kpts_t[m[:, 0], :2].astype(np.float64)
        pts1 = kpts_i[m[:, 1], :2].astype(np.float64)

        rng = np.random.default_rng(RANSAC_SEED)
        model, inliers = ransac(
            (pts0, pts1),
            ProjectiveTransform,
            min_samples=4,
            residual_threshold=self.ransac_thresh,
            max_trials=self.ransac_max_iters,
            stop_probability=self.ransac_confidence,
            rng=rng,
        )
        if model is None or not model:
            return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
        H = model.params
        return H.astype(np.float32), H[:2, 2]

    def find_transform(self, template, input_image) -> Dict:
        """
        template, input_image: (B,H,W) or (B,C,H,W), numpy or array-like.
        Returns: {"warp_matrix": (B,3,3), "motion": (B,2)}
        """
        t = _to_gray_float32(template)
        i = _to_gray_float32(input_image)
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


__all__ = ["AkazeAligner", "DRATIO", "DIFFUSIVITY_PM_G2"]
