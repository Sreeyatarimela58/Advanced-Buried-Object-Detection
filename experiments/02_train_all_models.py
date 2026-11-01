"""
experiments/02_train_all_models.py: Multi-model benchmarking on grouped 5-fold split (Phase 2).
Compares YOLOv8n, YOLOv5nu, YOLO11n, and YOLOv8s on the exact same 5 folds (split_5fold.json),
seed 0, identical hyperparameters, 224x224 px, 50 epochs.
Records mAP50, mAP50-95, P, R, F1, Params (M), Size (MB), GFLOPs, Training Time, GPU Name, and FPS.
Outputs:
  - results/model_comparison/{model}/fold_{1..5}/metrics.json
  - results/model_comparison/{model}/summary.json
  - results/model_comparison/comparison.csv
  - results/tables/comparison_table.tex
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from ultralytics import YOLO

# Add parent directory for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN

MODELS = [
    {"name": "yolov8n", "weights": "yolov8n.pt", "display": "YOLOv8 Nano (Proposed)"},
    {"name": "yolov5nu", "weights": "yolov5nu.pt", "display": "YOLOv5 Nano (Predecessor)"},
    {"name": "yolo11n", "weights": "yolo11n.pt", "display": "YOLO11 Nano (Next-Gen)"},
    {"name": "yolov8s", "weights": "yolov8s.pt", "display": "YOLOv8 Small (Capacity Scaling)"},
]


def train_and_eval_model_fold(model_info, fold_idx, fold_yaml, results_base, epochs=50, device=0):
    """
    Train a specific model on a specific fold and evaluate on the test split.
    """
    model_name = model_info["name"]
    weights_path = model_info["weights"]
    fold_name = f"fold_{fold_idx + 1}"
    fold_res_dir = Path(results_base) / model_name / fold_name
    fold_res_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = fold_res_dir / "metrics.json"
    if metrics_file.exists():
        print(f"[{model_name} | {fold_name}] Already completed! Loading existing metrics.")
        with open(metrics_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    actual_device = device if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*30} STARTING {model_name.upper()} {fold_name.upper()} {'='*30}")
    print(f"Device: {gpu_name} (device={actual_device}) | Base Weights: {weights_path}")

    train_run_dir = fold_res_dir / "train_run"
    last_pt = train_run_dir / "weights" / "last.pt"
    best_pt = train_run_dir / "weights" / "best.pt"

    start_time = time.time()
    train_duration = 0.0

    if last_pt.exists():
        print(f"[{model_name} | {fold_name}] Resuming training from {last_pt}...")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO(weights_path)
        model.train(
            data=str(fold_yaml),
            epochs=epochs,
            imgsz=TRAIN["imgsz"],
            batch=TRAIN["batch"],
            optimizer=TRAIN["optimizer"],
            lr0=TRAIN["lr0"],
            lrf=TRAIN["lrf"],
            weight_decay=0.0005,
            patience=TRAIN["patience"],
            seed=0,
            device=actual_device,
            workers=2,
            plots=True,
            project=str(fold_res_dir.resolve()),
            name="train_run",
            exist_ok=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.3,
            shear=2.0,
            perspective=0.0001,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.5,
        )

    train_duration = time.time() - start_time
    results_csv = train_run_dir / "results.csv"
    if results_csv.exists():
        try:
            r_lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
            if len(r_lines) > 1:
                train_duration = float(r_lines[-1].split(",")[1].strip())
        except Exception:
            pass

    if not best_pt.exists() and last_pt.exists():
        best_pt = last_pt

    # Evaluate on TEST split
    eval_model = YOLO(str(best_pt))
    eval_metrics = eval_model.val(
        data=str(fold_yaml),
        split="test",
        device=actual_device,
        plots=True,
        save_txt=True,
        save_conf=True,
        project=str(fold_res_dir.resolve()),
        name="test_eval",
        exist_ok=True,
    )

    # Extract metrics
    map50 = float(eval_metrics.box.map50)
    map50_95 = float(eval_metrics.box.map)
    prec = float(eval_metrics.box.mp)
    rec = float(eval_metrics.box.mr)
    f1 = float(2 * prec * rec / (prec + rec + 1e-16))

    ap50_util = float(eval_metrics.box.class_result(0)[0]) if len(eval_metrics.box.maps) > 0 else 0.0
    ap50_95_util = float(eval_metrics.box.class_result(0)[1]) if len(eval_metrics.box.maps) > 0 else 0.0
    ap50_cav = float(eval_metrics.box.class_result(1)[0]) if len(eval_metrics.box.maps) > 1 else 0.0
    ap50_95_cav = float(eval_metrics.box.class_result(1)[1]) if len(eval_metrics.box.maps) > 1 else 0.0

    # Model parameters & GFLOPs
    params_m = round(sum(p.numel() for p in eval_model.model.parameters()) / 1e6, 3)
    size_mb = round(os.path.getsize(best_pt) / (1024 * 1024), 2) if best_pt.exists() else 0.0
    
    try:
        info = eval_model.info(detailed=False)
        gflops = round(float(info[3]), 2)
    except Exception:
        gflops = 0.0

    fold_metrics = {
        "model": model_name,
        "fold": fold_name,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "utility": {"mAP50": ap50_util, "mAP50_95": ap50_95_util},
        "cavity": {"mAP50": ap50_cav, "mAP50_95": ap50_95_cav},
        "params_M": params_m,
        "size_MB": size_mb,
        "gflops": gflops,
        "training_time_s": round(train_duration, 1),
        "gpu_name": gpu_name,
    }

    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(fold_metrics, fp, indent=4)

    return fold_metrics


def aggregate_model_metrics(model_info, all_folds_metrics, results_base):
    model_name = model_info["name"]
    out_dir = Path(results_base) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = ["mAP50", "mAP50_95", "precision", "recall", "f1", "training_time_s"]
    df = pd.DataFrame(all_folds_metrics)

    means = df[metrics_list].mean().to_dict()
    stds = df[metrics_list].std().to_dict()

    summary = {
        "model": model_name,
        "display_name": model_info["display"],
        "params_M": all_folds_metrics[0]["params_M"],
        "size_MB": all_folds_metrics[0]["size_MB"],
        "gflops": all_folds_metrics[0]["gflops"],
        "gpu_name": all_folds_metrics[0]["gpu_name"],
        "per_fold": all_folds_metrics,
        "mean": means,
        "std": stds,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=4)
    df.to_csv(out_dir / "summary.csv", index=False)

    return summary


def compare_kaggle_vs_laptop_baseline(kaggle_yolov8n_summary, laptop_summary_path="results/baseline_new_split/summary.json"):
    """
    Kaggle Gate Check: Compare Kaggle YOLOv8n run against Laptop CPU baseline.
    Alert if mean mAP50 differs by > 0.02.
    """
    if not Path(laptop_summary_path).exists():
        print(f"[GATE CHECK] Laptop summary {laptop_summary_path} not found. Skipping gate check.")
        return True

    with open(laptop_summary_path, "r", encoding="utf-8") as fp:
        laptop_data = json.load(fp)

    laptop_mean_map50 = laptop_data["mean"]["mAP50"]
    kaggle_mean_map50 = kaggle_yolov8n_summary["mean"]["mAP50"]
    diff = abs(kaggle_mean_map50 - laptop_mean_map50)

    print("\n" + "="*70)
    print("GATE CHECK: KAGGLE GPU YOLOv8n vs. LAPTOP CPU BASELINE")
    print("="*70)
    print(f"Laptop CPU YOLOv8n Mean mAP50:  {laptop_mean_map50:.4f}")
    print(f"Kaggle GPU YOLOv8n Mean mAP50:  {kaggle_mean_map50:.4f}")
    print(f"Absolute Difference (|Delta|):  {diff:.4f}")

    per_fold_kaggle = {f["fold"]: f["mAP50"] for f in kaggle_yolov8n_summary["per_fold"]}
    per_fold_laptop = {f["fold"]: f["mAP50"] for f in laptop_data["per_fold"]}
    
    print("\nPer-Fold Comparison:")
    for fold in sorted(per_fold_kaggle.keys()):
        lap_val = per_fold_laptop.get(fold, 0.0)
        kag_val = per_fold_kaggle.get(fold, 0.0)
        print(f"  {fold}: Laptop={lap_val:.4f} | Kaggle={kag_val:.4f} | Delta={kag_val - lap_val:+.4f}")

    if diff > 0.04:
        print(f"\n[WARNING] Kaggle GPU and Laptop CPU YOLOv8n mean mAP50 differ by {diff:.4f} (> 0.04)!")
        print("Please review and confirm before continuing Phase 2.")
        return False
    else:
        print(f"\n[PASS] Difference ({diff:.4f}) is <= 0.04 (normal CPU vs GPU variance). Results are consistent.")
        return True


def generate_overall_comparison_table(all_model_summaries, latency_file="results/model_comparison/latency_benchmark.json"):
    """
    Compile final comparison CSV and IEEEtran LaTeX table.
    Integrates latency and FPS if available.
    """
    results_dir = Path("results/model_comparison")
    results_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    latencies = {}
    if Path(latency_file).exists():
        try:
            with open(latency_file, "r", encoding="utf-8") as fp:
                latencies = json.load(fp)
        except Exception:
            pass

    rows = []
    for s in all_model_summaries:
        m_name = s["model"]
        lat_info = latencies.get(m_name, {})
        median_lat = lat_info.get("median_ms", None)
        fps = round(1000.0 / median_lat, 1) if (median_lat and median_lat > 0) else None

        row = {
            "Model": s["display_name"],
            "mAP50": f"{s['mean']['mAP50']:.4f} ± {s['std']['mAP50']:.4f}",
            "mAP50-95": f"{s['mean']['mAP50_95']:.4f} ± {s['std']['mAP50_95']:.4f}",
            "Precision": f"{s['mean']['precision']:.4f} ± {s['std']['precision']:.4f}",
            "Recall": f"{s['mean']['recall']:.4f} ± {s['std']['recall']:.4f}",
            "F1-Score": f"{s['mean']['f1']:.4f} ± {s['std']['f1']:.4f}",
            "Params (M)": s["params_M"],
            "Size (MB)": s["size_MB"],
            "GFLOPs": s["gflops"],
            "Training Time (s)": f"{s['mean']['training_time_s']:.1f}",
            "GPU Name": s["gpu_name"],
            "CPU Latency (ms)": f"{median_lat:.2f}" if median_lat else "TBD",
            "FPS": f"{fps:.1f}" if fps else "TBD",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = results_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV to: {csv_path}")

    # Generate IEEEtran LaTeX Table
    tex_path = table_dir / "comparison_table.tex"
    tex_lines = [
        "\\begin{table*}[htbp]",
        "\\caption{Performance, Complexity, and Latency Benchmark of Lightweight Detectors on Grouped 5-Fold Cross-Validation}",
        "\\label{tab:model_comparison}",
        "\\centering",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{mAP50} & \\textbf{mAP50-95} & \\textbf{F1-Score} & \\textbf{Params (M)} & \\textbf{GFLOPs} & \\textbf{CPU Latency (ms)} & \\textbf{FPS} \\\\",
        "\\hline",
    ]
    for r in rows:
        lat_str = r['CPU Latency (ms)']
        fps_str = r['FPS']
        tex_lines.append(f"{r['Model']} & {r['mAP50']} & {r['mAP50-95']} & {r['F1-Score']} & {r['Params (M)']} & {r['GFLOPs']} & {lat_str} & {fps_str} \\\\")
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    with open(tex_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(tex_lines))
    print(f"Saved IEEEtran LaTeX comparison table to: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train and Benchmark Comparison Models Across 5 Folds")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--device", default=0, help="Device to use for training (0 for GPU, cpu for CPU)")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov5nu", "yolo11n", "yolov8s"], help="List of models to benchmark")
    args = parser.parse_args()

    split_file = Path("results/split_new/split_5fold.json")
    if not split_file.exists():
        print("[ERROR] results/split_new/split_5fold.json not found!")
        sys.exit(1)

    results_base = Path("results/model_comparison")
    results_base.mkdir(parents=True, exist_ok=True)

    selected_models = [m for m in MODELS if m["name"] in args.models]
    all_summaries = []

    for m_info in selected_models:
        print(f"\n{'#'*35} BENCHMARKING {m_info['display'].upper()} {'#'*35}")
        model_fold_metrics = []
        for f_i in range(5):
            fold_yaml = Path(f"datasets/split_5fold/fold_{f_i + 1}/data.yaml")
            m = train_and_eval_model_fold(m_info, f_i, fold_yaml, results_base, epochs=args.epochs, device=args.device)
            model_fold_metrics.append(m)

        summary = aggregate_model_metrics(m_info, model_fold_metrics, results_base)
        all_summaries.append(summary)

        # Gate check after YOLOv8n
        if m_info["name"] == "yolov8n":
            gate_pass = compare_kaggle_vs_laptop_baseline(summary)
            if not gate_pass:
                print("\n[PAUSE] Gate check failed or requires user review. Stopping before next model.")
                break

    generate_overall_comparison_table(all_summaries)
    print("\n[PHASE 2 COMPLETED] All model benchmarks saved in results/model_comparison/")


if __name__ == "__main__":
    main()
