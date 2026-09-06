# -*- coding: utf-8 -*-
"""
Ablation Study Pipeline Runner for GPR Buried Object Detection.

Conducts a Leave-One-Out (LOO) study across 6 GPR data augmentation techniques:
    Run 0: All 6 augmentations ON (Baseline)
    Run 1: Exclude Gaussian Noise
    Run 2: Exclude Time Shift
    Run 3: Exclude Rotation
    Run 4: Exclude Horizontal Flip
    Run 5: Exclude Elastic Deformation
    Run 6: Exclude Spectral Shift

Usage:
    # Prepare all ablation datasets (instantaneous):
    python run_ablation.py --prepare-only

    # Train a single ablation run:
    python run_ablation.py --run 1

    # Check ablation progress and recorded metrics:
    python run_ablation.py --status

    # Evaluate an existing checkpoint and record metrics:
    python run_ablation.py --eval-only 1
"""

import os
import sys
import json
import shutil
import argparse
from collections import OrderedDict
import pandas as pd

# Define the 7 ablation runs
ABLATION_RUNS = OrderedDict([
    (0, {
        "key": "baseline",
        "name": "run0_baseline_all_on",
        "disabled": None,
        "desc": "All 6 Augmentations ON (Baseline)",
        "util_tags": [],
        "cav_tags": [],
    }),
    (1, {
        "key": "noise",
        "name": "run1_no_noise",
        "disabled": "noise",
        "desc": "All ON except Gaussian Noise",
        "util_tags": ["aug_1"],
        "cav_tags": ["aug_7"],
    }),
    (2, {
        "key": "time_shift",
        "name": "run2_no_time_shift",
        "disabled": "time_shift",
        "desc": "All ON except Time Shift",
        "util_tags": ["aug_2"],
        "cav_tags": ["aug_8"],
    }),
    (3, {
        "key": "rotation",
        "name": "run3_no_rotation",
        "disabled": "rotation",
        "desc": "All ON except Rotation",
        "util_tags": ["aug_3"],
        "cav_tags": ["aug_9"],
    }),
    (4, {
        "key": "flip",
        "name": "run4_no_flip",
        "disabled": "flip",
        "desc": "All ON except Horizontal Flip",
        "util_tags": ["aug_4"],
        "cav_tags": ["aug_11"],
    }),
    (5, {
        "key": "elastic",
        "name": "run5_no_elastic",
        "disabled": "elastic",
        "desc": "All ON except Elastic Deformation",
        "util_tags": ["aug_5"],
        "cav_tags": ["aug_12"],
    }),
    (6, {
        "key": "spectral",
        "name": "run6_no_spectral",
        "disabled": "spectral",
        "desc": "All ON except Spectral Shift",
        "util_tags": ["aug_6"],
        "cav_tags": ["aug_13"],
    }),
])

RESULTS_FILE = "ablation_results.json"
CSV_FILE = "ablation_summary.csv"
EXPERIMENTS_DIR = "ablation_experiments"


def load_results():
    """Load existing ablation results from JSON, saving baseline if not yet written."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Initialize with baseline Run 0 pre-populated and save to disk
    initial = {
        "0": {
            "run_id": 0,
            "name": "run0_baseline_all_on",
            "excluded_augmentation": "None (Baseline)",
            "desc": "All 6 Augmentations ON (Baseline)",
            "status": "completed",
            "weights": "models/best_model.pt",
            "train_images": 1070,
            "val_images": 269,
            "total_images": 1339,
            "metrics": {
                "mAP50": 0.9595,
                "mAP50-95": 0.6068,
                "precision": 0.9518,
                "recall": 0.9395,
                "utility_mAP50": 0.9463,
                "utility_precision": 0.9370,
                "utility_recall": 0.9118,
                "cavity_mAP50": 0.9726,
                "cavity_precision": 0.9668,
                "cavity_recall": 0.9672,
            }
        }
    }
    save_results(initial)
    return initial



def save_results(results):
    """Save results dictionary to JSON and export summary CSV."""
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Export to clean CSV table
    rows = []
    for run_id_str, d in results.items():
        m = d.get("metrics", {})
        rows.append({
            "Run ID": d.get("run_id"),
            "Configuration": d.get("desc"),
            "Excluded Augmentation": d.get("excluded_augmentation"),
            "Status": d.get("status"),
            "Train Images": d.get("train_images"),
            "Val Images": d.get("val_images"),
            "mAP50": m.get("mAP50"),
            "mAP50-95": m.get("mAP50-95"),
            "Precision": m.get("precision"),
            "Recall": m.get("recall"),
            "Utility mAP50": m.get("utility_mAP50"),
            "Cavity mAP50": m.get("cavity_mAP50"),
        })
    df = pd.DataFrame(rows)
    df.sort_values(by="Run ID", inplace=True)
    df.to_csv(CSV_FILE, index=False)


def prepare_ablation_dataset(run_id, base_dataset="yolo_dataset", output_root=EXPERIMENTS_DIR):
    """
    Constructs the filtered dataset for a specific ablation run.
    Excludes the target augmentation images and labels while maintaining the exact
    train/val split established in Run 0.
    """
    run_info = ABLATION_RUNS[run_id]
    run_name = run_info["name"]
    dataset_dir = os.path.abspath(os.path.join(output_root, "datasets", run_name))

    util_tags = set(run_info["util_tags"])
    cav_tags = set(run_info["cav_tags"])

    print(f"\n[{run_name}] Preparing dataset (Excluding: {run_info['disabled'] or 'None'})...")

    stats = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        img_out = os.path.join(dataset_dir, "images", split)
        lbl_out = os.path.join(dataset_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        src_img_dir = os.path.join(base_dataset, "images", split)
        src_lbl_dir = os.path.join(base_dataset, "labels", split)

        files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        for img_file in files:
            # Check if this image belongs to the excluded augmentation
            is_excluded = False
            base_no_ext, ext = os.path.splitext(img_file)

            if img_file.startswith("util_"):
                for tag in util_tags:
                    if f"_{tag}" in base_no_ext:
                        is_excluded = True
                        break
            elif img_file.startswith("cav_"):
                for tag in cav_tags:
                    if f"_{tag}" in base_no_ext:
                        is_excluded = True
                        break

            if is_excluded:
                continue

            # Link or copy image
            dst_img = os.path.join(img_out, img_file)
            src_img = os.path.abspath(os.path.join(src_img_dir, img_file))
            if not os.path.exists(dst_img):
                try:
                    os.link(src_img, dst_img)  # Hard link (fast, 0 disk space)
                except OSError:
                    shutil.copy2(src_img, dst_img)

            # Link or copy matching label
            lbl_file = base_no_ext + ".txt"
            src_lbl = os.path.abspath(os.path.join(src_lbl_dir, lbl_file))
            dst_lbl = os.path.join(lbl_out, lbl_file)
            if os.path.exists(src_lbl) and not os.path.exists(dst_lbl):
                try:
                    os.link(src_lbl, dst_lbl)
                except OSError:
                    shutil.copy2(src_lbl, dst_lbl)

            stats[split] += 1

    # Write data.yaml
    data_yaml_path = os.path.join(dataset_dir, "data.yaml")
    yaml_content = f"""# Ablation Run {run_id}: {run_name}
# Excluded Augmentation: {run_info['disabled'] or 'None'}

path: {dataset_dir}
train: images/train
val: images/val

nc: 2
names: ['utility', 'cavity']
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    total = stats["train"] + stats["val"]
    print(f"  ✓ Prepared: {total} images (Train: {stats['train']}, Val: {stats['val']})")
    print(f"  ✓ Config:   {data_yaml_path}")
    return dataset_dir, data_yaml_path, stats


def evaluate_model(model_path, data_yaml):
    """Run YOLO validation and extract comprehensive metrics."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device="cpu", verbose=False)

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)

    # Per-class metrics
    class_names = ["utility", "cavity"]
    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(metrics.box.p):
            per_class[f"{name}_precision"] = round(float(metrics.box.p[i]), 4)
            per_class[f"{name}_recall"] = round(float(metrics.box.r[i]), 4)
            per_class[f"{name}_mAP50"] = round(float(metrics.box.ap50[i]), 4)

    return {
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        **per_class,
    }


def train_ablation_run(run_id, epochs=50, batch=8, imgsz=224, workers=4, force=False):
    """Executes the full YOLOv8 training routine for a single ablation run."""
    from ultralytics import YOLO

    if run_id == 0:
        print("\n[Run 0] Baseline model is already trained and saved at models/best_model.pt.")
        return load_results().get("0")

    run_info = ABLATION_RUNS[run_id]
    run_name = run_info["name"]

    # Check if already completed in results
    all_results = load_results()
    if not force and str(run_id) in all_results and all_results[str(run_id)].get("status") == "completed":
        print(f"\n⏩ [Run {run_id}] ({run_info['desc']}) is already completed! Skipping. (Pass --force to retrain)")
        return all_results[str(run_id)]

    project_dir = os.path.abspath("runs/detect/ablation")
    best_weights = os.path.join(project_dir, run_name, "weights", "best.pt")

    # If best.pt already exists from a completed training run, evaluate and record instead of retraining
    if not force and os.path.exists(best_weights):
        print(f"\n⏩ [Run {run_id}] Found existing trained weights at {best_weights}.")
        print("Evaluating checkpoint and recording metrics without retraining (pass --force to retrain)...")
        _, data_yaml, stats = prepare_ablation_dataset(run_id)
        metrics = evaluate_model(best_weights, data_yaml)
        all_results[str(run_id)] = {
            "run_id": run_id,
            "name": run_name,
            "excluded_augmentation": run_info["disabled"],
            "desc": run_info["desc"],
            "status": "completed",
            "weights": best_weights,
            "train_images": stats["train"],
            "val_images": stats["val"],
            "total_images": stats["train"] + stats["val"],
            "metrics": metrics,
        }
        save_results(all_results)
        print(f"✓ Run {run_id} evaluated and recorded!")
        return all_results[str(run_id)]

    dataset_dir, data_yaml, stats = prepare_ablation_dataset(run_id)


    print("\n" + "=" * 60)
    print(f"STARTING ABLATION RUN {run_id}: {run_info['desc']}")
    print("=" * 60)
    print(f"  Excluded:   {run_info['disabled']}")
    print(f"  Dataset:    {data_yaml}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Device:     CPU")
    print("=" * 60 + "\n")

    project_dir = os.path.abspath("runs/detect/ablation")
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=run_name,
        project=project_dir,
        patience=20,
        device="cpu",
        workers=workers,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Locate best weights
    best_weights = os.path.join(project_dir, run_name, "weights", "best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.join(project_dir, run_name, "weights", "last.pt")

    print(f"\nEvaluating final checkpoint: {best_weights}...")
    metrics = evaluate_model(best_weights, data_yaml)

    # Update results
    all_results = load_results()
    all_results[str(run_id)] = {
        "run_id": run_id,
        "name": run_name,
        "excluded_augmentation": run_info["disabled"],
        "desc": run_info["desc"],
        "status": "completed",
        "weights": best_weights,
        "train_images": stats["train"],
        "val_images": stats["val"],
        "total_images": stats["train"] + stats["val"],
        "metrics": metrics,
    }
    save_results(all_results)
    print(f"\n✓ Run {run_id} complete! Metrics recorded to {RESULTS_FILE} and {CSV_FILE}.")
    return all_results[str(run_id)]


def print_status():
    """Prints a beautifully formatted status table of all 7 ablation runs."""
    results = load_results()
    print("\n" + "=" * 80)
    print(f"{'Run':<4} {'Configuration':<34} {'Status':<11} {'mAP50':>8} {'mAP50-95':>9} {'P':>7} {'R':>7}")
    print("=" * 80)

    for run_id, info in ABLATION_RUNS.items():
        r = results.get(str(run_id))
        if r and r.get("status") == "completed":
            m = r.get("metrics", {})
            print(
                f"{run_id:<4} {info['desc']:<34} {'✅ Done':<11} "
                f"{m.get('mAP50', 0):>8.4f} {m.get('mAP50-95', 0):>9.4f} "
                f"{m.get('precision', 0):>7.4f} {m.get('recall', 0):>7.4f}"
            )
        else:
            print(f"{run_id:<4} {info['desc']:<34} {'⏳ Pending':<11} {'-':>8} {'-':>9} {'-':>7} {'-':>7}")
    print("=" * 80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="GPR Buried Object Detection - LOO Ablation Study Runner")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare all ablation datasets without training")
    parser.add_argument("--status", action="store_true", help="Print current status and metric summary of all runs")
    parser.add_argument("--run", type=str, default=None, help="Run ID to execute (0-6, or 'all')")
    parser.add_argument("--eval-only", type=int, default=None, help="Evaluate an existing run checkpoint (e.g. --eval-only 1)")
    parser.add_argument("--force", action="store_true", help="Force retraining even if run is already marked completed")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4 for 16-core CPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.status:
        print_status()
        sys.exit(0)

    if args.prepare_only:
        print("Preparing all ablation datasets...")
        for rid in range(1, 7):
            prepare_ablation_dataset(rid)
        print("\nAll 6 ablation datasets prepared successfully!")
        sys.exit(0)

    if args.eval_only is not None:
        rid = args.eval_only
        run_info = ABLATION_RUNS[rid]
        _, data_yaml, stats = prepare_ablation_dataset(rid)
        weights_path = f"runs/detect/ablation/{run_info['name']}/weights/best.pt"
        if not os.path.exists(weights_path):
            weights_path = f"runs/detect/ablation/{run_info['name']}/weights/last.pt"
        if not os.path.exists(weights_path):
            print(f"Error: checkpoint not found at {weights_path}")
            sys.exit(1)
        print(f"Evaluating checkpoint for Run {rid}: {weights_path}...")
        metrics = evaluate_model(weights_path, data_yaml)
        all_res = load_results()
        all_res[str(rid)] = {
            "run_id": rid,
            "name": run_info["name"],
            "excluded_augmentation": run_info["disabled"],
            "desc": run_info["desc"],
            "status": "completed",
            "weights": weights_path,
            "train_images": stats["train"],
            "val_images": stats["val"],
            "total_images": stats["train"] + stats["val"],
            "metrics": metrics,
        }
        save_results(all_res)
        print_status()
        sys.exit(0)

    if args.run is not None:
        if args.run.lower() == "all":
            for rid in range(1, 7):
                train_ablation_run(rid, epochs=args.epochs, batch=args.batch, workers=args.workers, force=args.force)
            print_status()
        else:
            rid = int(args.run)
            train_ablation_run(rid, epochs=args.epochs, batch=args.batch, workers=args.workers, force=args.force)
            print_status()
    else:
        print_status()
        print("Tip: Use --prepare-only to create datasets, or --run <1-6> to start an ablation run.")

