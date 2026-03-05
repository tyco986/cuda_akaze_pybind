#!/usr/bin/env python3
"""
Benchmark test for AkazeAligner: GPU/CPU/memory usage and computation time.
"""

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


def _monitor_loop(stop_event, gpu_peak, cpu_peak, mem_peak, gpu_handle, process):
    """Background thread to track peak GPU, CPU, and memory usage."""
    while not stop_event.is_set():
        if process is not None:
            try:
                cpu_peak[0] = max(cpu_peak[0], process.cpu_percent())
                mem_info = process.memory_info()
                mem_peak[0] = max(mem_peak[0], mem_info.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if HAS_NVML and gpu_handle is not None:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                gpu_peak[0] = max(gpu_peak[0], mem.used)
            except Exception:
                pass
        stop_event.wait(0.05)


def main():
    parser = argparse.ArgumentParser(description="Benchmark AkazeAligner")
    parser.add_argument(
        "--image",
        type=str,
        default="image.png",
        help="Input image path (default: image.png)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="template.png",
        help="Template image path (default: template.png)",
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
        "--no-nndr",
        action="store_true",
        help="Disable NNDR filter for speed (RANSAC still filters outliers)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    image_path = script_dir / args.image
    template_path = script_dir / args.template

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    if tpl is None:
        raise RuntimeError(f"Failed to load template: {template_path}")

    batch = args.batch
    template_batch = np.tile(tpl[np.newaxis, :, :], (batch, 1, 1))
    image_batch = np.tile(img[np.newaxis, :, :], (batch, 1, 1))

    import cuda_akaze

    aligner = cuda_akaze.AkazeAligner(device=args.device, use_nndr=not args.no_nndr)

    # Warmup
    _ = aligner.find_transform(template_batch[:1], image_batch[:1])
    if HAS_NVML:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(args.device)
    else:
        gpu_handle = None

    process = psutil.Process()
    gpu_peak = [0]
    cpu_peak = [0.0]
    mem_peak = [0]

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_loop,
        args=(stop_event, gpu_peak, cpu_peak, mem_peak, gpu_handle, process),
    )
    monitor.start()

    t0 = time.perf_counter()
    result = aligner.find_transform(template_batch, image_batch)
    t1 = time.perf_counter()
    stop_event.set()
    monitor.join()

    if HAS_NVML:
        pynvml.nvmlShutdown()

    elapsed_ms = (t1 - t0) * 1000
    gpu_mb = gpu_peak[0] / (1024 * 1024)
    mem_mb = mem_peak[0] / (1024 * 1024)

    warp_matrix = result["warp_matrix"]

    print("--- AkazeAligner benchmark ---")
    print(f"  Batch size:        {batch}")
    print(f"  Image shape:       {img.shape}")
    print(f"  Template shape:    {tpl.shape}")
    print(f"  GPU peak (MB):     {gpu_mb:.2f}")
    print(f"  CPU peak (%):      {cpu_peak[0]:.1f}")
    print(f"  Memory peak (MB):  {mem_mb:.2f}")
    print(f"  Time (ms):         {elapsed_ms:.2f}")
    print(f"  Time per sample:   {elapsed_ms / batch:.2f} ms")
    print("  Warp matrix (3x3 per sample):")
    np.set_printoptions(precision=4, suppress=True)
    for i in range(batch):
        print(f"    [{i}]:\n{warp_matrix[i]}")


if __name__ == "__main__":
    main()
