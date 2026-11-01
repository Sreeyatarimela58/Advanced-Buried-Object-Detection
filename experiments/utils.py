"""
experiments/utils.py: Common utilities for GPR buried-object detection experiments.
Includes metric calculations, latency benchmarking, and label transformations.
"""
import os
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path


def calculate_metrics(precision, recall):
    """
    Compute F1-score from precision and recall.
    Formula: F1 = 2 * (P * R) / (P + R + 1e-16)
    """
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def save_metrics_summary(results_dict, output_dir, prefix="summary"):
    """
    Save mean and std metrics to both JSON and CSV.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_file = out_path / f"{prefix}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)

    # If flat dictionary or structured, save as CSV
    try:
        df = pd.DataFrame(results_dict)
        csv_file = out_path / f"{prefix}.csv"
        df.to_csv(csv_file, index=False)
    except Exception:
        pass


def benchmark_cpu_latency(model, input_size=(1, 3, 224, 224), warmup=20, runs=100, threads=None):
    """
    Benchmark CPU inference latency cleanly.
    Returns:
        dict: mean, std, median, p95, fps, and threads used.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    actual_threads = torch.get_num_threads()

    device = torch.device("cpu")
    model.eval()
    model.to(device)

    dummy_input = torch.randn(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    timings = np.array(timings)
    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    median_ms = float(np.median(timings))
    p95_ms = float(np.percentile(timings, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "fps": fps,
        "threads": actual_threads,
        "runs": runs,
    }


def yolo_to_voc_bbox(box_yolo, img_width=224, img_height=224):
    """
    Convert YOLO normalized [x_center, y_center, width, height] to VOC [x1, y1, x2, y2].
    """
    xc, yc, w, h = box_yolo
    x1 = (xc - w / 2.0) * img_width
    y1 = (yc - h / 2.0) * img_height
    x2 = (xc + w / 2.0) * img_width
    y2 = (yc + h / 2.0) * img_height
    # Clip to boundary
    x1 = max(0.0, min(float(img_width), x1))
    y1 = max(0.0, min(float(img_height), y1))
    x2 = max(0.0, min(float(img_width), x2))
    y2 = max(0.0, min(float(img_height), y2))
    return [x1, y1, x2, y2]
