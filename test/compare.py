#!/usr/bin/env python3
"""
Compare fast, accurate, opencv: 3-way AKAZE implementation comparison.
- Output comparison (warp_matrix, motion)
- Timing comparison
Args: image, template, batch
"""

import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from skimage import io

import cuda_akaze


def _to_gray_u8(img) -> np.ndarray:
    """Convert to grayscale uint8 [0,255]."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def opencv_akaze_find_transform(
    template: np.ndarray,
    image: np.ndarray,
    nndr: float = 0.80,
    ransac_thresh: float = 2.5,
    ransac_max_iters: int = 2000,
    ransac_confidence: float = 0.995,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OpenCV AKAZE: detect + match + RANSAC homography.
    Returns (H 3x3, motion 2).
    """
    detector = cv2.AKAZE_create()
    kp1, desc1 = detector.detectAndCompute(template, None)
    kp2, desc2 = detector.detectAndCompute(image, None)

    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    pts0, pts1 = [], []
    for m_n in matches:
        if len(m_n) < 2:
            continue
        m, n = m_n[0], m_n[1]
        if m.distance < nndr * n.distance:
            pts0.append(kp1[m.queryIdx].pt)
            pts1.append(kp2[m.trainIdx].pt)

    pts0 = np.array(pts0, dtype=np.float32)
    pts1 = np.array(pts1, dtype=np.float32)

    if len(pts0) < 4:
        return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)

    H, mask = cv2.findHomography(
        pts0, pts1, cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=ransac_max_iters,
        confidence=ransac_confidence,
    )
    if H is None:
        return np.eye(3, dtype=np.float32), np.zeros(2, dtype=np.float32)
    motion = H[:2, 2].astype(np.float32)
    return H.astype(np.float32), motion


def run_mode(mode: str, template_batch, image_batch, nndr: float, device: int, warmup: int, runs: int):
    """Run one mode and return (result dict, mean_ms, std_ms)."""
    if mode == "opencv":
        for _ in range(warmup):
            for b in range(template_batch.shape[0]):
                opencv_akaze_find_transform(template_batch[b], image_batch[b], nndr=nndr)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            warps, motions = [], []
            for b in range(template_batch.shape[0]):
                H, mot = opencv_akaze_find_transform(
                    template_batch[b], image_batch[b], nndr=nndr
                )
                warps.append(H)
                motions.append(mot)
            result = {
                "warp_matrix": np.stack(warps, axis=0),
                "motion": np.stack(motions, axis=0),
            }
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        return result, np.mean(times), np.std(times)
    else:
        aligner = cuda_akaze.AkazeAligner(
            device=device, use_nndr=(nndr < 1.0), mode=mode
        )
        for _ in range(warmup):
            _ = aligner.find_transform(template_batch, image_batch)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            result = aligner.find_transform(template_batch, image_batch)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        return result, np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser(
        description="Compare fast, accurate, opencv: 3-way AKAZE comparison"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test/image.png",
        help="Input image path (default: test/image.png)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="test/template.png",
        help="Template image path (default: test/template.png)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID (default: 0)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs before timing (default: 2)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timing runs (default: 5)",
    )
    parser.add_argument(
        "--no-nndr",
        action="store_true",
        help="Disable NNDR filter (use 1.0 ratio)",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    template_path = Path(args.template).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    img = io.imread(str(image_path), as_gray=True)
    tpl = io.imread(str(template_path), as_gray=True)

    img_u8 = _to_gray_u8(img)
    tpl_u8 = _to_gray_u8(tpl)

    batch = args.batch
    nndr = 1.0 if args.no_nndr else 0.80

    template_batch = np.tile(tpl_u8[np.newaxis, :, :], (batch, 1, 1))
    image_batch = np.tile(img_u8[np.newaxis, :, :], (batch, 1, 1))

    modes = ["fast", "accurate", "opencv"]
    results = {}
    timings = {}

    for mode in modes:
        result, mean_ms, std_ms = run_mode(
            mode, template_batch, image_batch,
            nndr, args.device, args.warmup, args.runs,
        )
        results[mode] = result
        timings[mode] = (mean_ms, std_ms)

    print("=" * 60)
    print("fast vs accurate vs opencv: 3-way AKAZE comparison")
    print("=" * 60)
    print(f"  Image:      {image_path.name}")
    print(f"  Template:   {template_path.name}")
    print(f"  Batch:      {batch}")
    print(f"  Shape:      img {img.shape}, tpl {tpl.shape}")
    print()

    print("--- Timing ---")
    for mode in modes:
        mean_ms, std_ms = timings[mode]
        per = mean_ms / batch
        print(f"  {mode:10s}: {mean_ms:.2f} +/- {std_ms:.2f} ms  (batch={batch}, per-sample {per:.2f} ms)")
    if timings["opencv"][0] > 0:
        for m in ["fast", "accurate"]:
            speedup = timings["opencv"][0] / timings[m][0]
            print(f"  {m} vs opencv: ~{speedup:.2f}x faster")
    print()

    print("--- Output comparison (reference: opencv) ---")
    ref = results["opencv"]
    for mode in ["fast", "accurate"]:
        res = results[mode]
        diff_H = np.abs(res["warp_matrix"] - ref["warp_matrix"])
        diff_mot = np.abs(res["motion"] - ref["motion"])
        max_h = np.max(diff_H)
        max_m = np.max(diff_mot)
        match = np.allclose(res["warp_matrix"], ref["warp_matrix"], rtol=1e-3, atol=1e-2)
        status = "✓" if match else "✗"
        print(f"  {mode:10s} vs opencv: {status} max_diff_H={max_h:.6f}, max_diff_motion={max_m:.6f}")
    print()

    print("--- Warp matrix (sample 0) ---")
    np.set_printoptions(precision=4, suppress=True)
    for mode in modes:
        print(f"  {mode}:\n{results[mode]['warp_matrix'][0]}")
        print()


if __name__ == "__main__":
    main()
