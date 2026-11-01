# -*- coding: utf-8 -*-
"""
Empirical Latency & Throughput Benchmark for Phase 2 Models on Laptop CPU.
Benchmarks:
  - YOLOv8 Nano
  - YOLOv5 Nano
  - YOLO11 Nano
  - YOLOv8 Small

Protocol:
  - Input resolution: 224x224 px, batch size 1
  - Device: CPU (fixed thread count logged)
  - Iterations: 20 warm-up + 100 timed runs
  - Measures: Mean, Std, Median (P50), P95, and Real-Time FPS (1000 / median_ms)
  - Saves: results/model_comparison/latency_benchmark.json
"""

import os
import json
import time
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

MODELS_TO_BENCHMARK = [
    {"name": "yolov8n", "weights": "yolov8n.pt", "display": "YOLOv8 Nano"},
    {"name": "yolov5nu", "weights": "yolov5nu.pt", "display": "YOLOv5 Nano"},
    {"name": "yolo11n", "weights": "yolo11n.pt", "display": "YOLO11 Nano"},
    {"name": "yolov8s", "weights": "yolov8s.pt", "display": "YOLOv8 Small"},
]

IMG_SIZE = 224
NUM_WARMUP = 20
NUM_TEST = 100


def benchmark_single_model(model_info, imgsz=IMG_SIZE, warmup=NUM_WARMUP, test_runs=NUM_TEST):
    weights_path = model_info["weights"]
    display_name = model_info["display"]
    
    print("\n" + "=" * 70)
    print(f"BENCHMARKING: {display_name} ({weights_path})")
    print("=" * 70)

    model = YOLO(weights_path)
    threads = torch.get_num_threads()
    
    # Model parameters & GFLOPs
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    try:
        info = model.info(detailed=False)
        gflops = round(float(info[3]), 2)
    except Exception:
        gflops = 0.0

    print(f"  Architecture:     {model.model.__class__.__name__}")
    print(f"  Parameters:       {params:.2f} M")
    print(f"  GFLOPs:           {gflops}")
    print(f"  Resolution:       {imgsz}x{imgsz} px (batch 1)")
    print(f"  CPU Threads:      {threads}")
    print("-" * 70)

    dummy_input = np.random.randint(0, 256, (imgsz, imgsz, 3), dtype=np.uint8)

    # 1. Warm-up
    print(f"  Warming up cache ({warmup} runs)...")
    for _ in range(warmup):
        _ = model(dummy_input, imgsz=imgsz, verbose=False)

    # 2. Timed profiling
    print(f"  Profiling {test_runs} timed runs...")
    pre_times = []
    inf_times = []
    post_times = []
    total_times = []

    for _ in range(test_runs):
        t0 = time.perf_counter()
        res = model(dummy_input, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()

        speed = res[0].speed
        pre_times.append(speed.get("preprocess", 0.0))
        inf_times.append(speed.get("inference", 0.0))
        post_times.append(speed.get("postprocess", 0.0))
        total_times.append((t1 - t0) * 1000.0)

    mean_total = float(np.mean(total_times))
    std_total = float(np.std(total_times))
    median_total = float(np.percentile(total_times, 50))
    p95_total = float(np.percentile(total_times, 95))
    fps = round(1000.0 / median_total, 1)

    print(f"  Results:")
    print(f"    Mean ± Std:     {mean_total:.2f} ± {std_total:.2f} ms")
    print(f"    Median (P50):   {median_total:.2f} ms")
    print(f"    95th Pct (P95): {p95_total:.2f} ms")
    print(f"    Throughput:     {fps:.1f} FPS")

    return {
        "display_name": display_name,
        "weights": weights_path,
        "params_M": round(params, 2),
        "gflops": gflops,
        "threads": threads,
        "mean_ms": round(mean_total, 2),
        "std_ms": round(std_total, 2),
        "median_ms": round(median_total, 2),
        "p95_ms": round(p95_total, 2),
        "fps": fps,
    }


def main():
    out_dir = Path("results/model_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_latencies = {}

    for m_info in MODELS_TO_BENCHMARK:
        res = benchmark_single_model(m_info)
        all_latencies[m_info["name"]] = res

    out_file = out_dir / "latency_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(all_latencies, fp, indent=4)

    print("\n" + "=" * 70)
    print(f"[BENCHMARK COMPLETE] Saved all model latencies to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
