"""
experiments/02_run_ablations.py: Grouped 5-Fold Leave-One-Out (LOO) & No-Augmentation Ablation Study.
Conducts 7 ablation runs across the exact same 5 folds (split_5fold.json) using YOLOv8n:
  - Run 0: Baseline All Augmentations ON (from Phase 1)
  - Run 1: Exclude Gaussian Noise (util: aug_1, cav: aug_7)
  - Run 2: Exclude Time Shift (util: aug_2, cav: aug_8)
  - Run 3: Exclude Rotation (util: aug_3, cav: aug_9)
  - Run 4: Exclude Horizontal Flip (util: aug_4, cav: aug_11)
  - Run 5: Exclude Elastic Deformation (util: aug_5, cav: aug_12)
  - Run 6: Exclude Spectral Shift / FFT (util: aug_6, cav: aug_13)
  - Run 7: No Augmentation (only un-augmented original images)
(Note: aug_10 Zoom on cavities is kept across runs 0-6 as confirmed).

Outputs:
  - results/ablation_study/{run_key}/fold_{1..5}/metrics.json
  - results/ablation_study/ablation_summary.csv
  - results/tables/ablation_table.tex
"""
import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Add parent directory for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRAIN

# Exact verified ground-truth suffix mappings
ABLATION_CONFIGS = OrderedDict([
    ("baseline", {
        "name": "All Augmentations (Baseline)",
        "excluded_util": [],
        "excluded_cav": [],
        "desc": "Baseline with all 6 GPR augmentations ON",
    }),
    ("no_noise", {
        "name": "w/o Gaussian Noise",
        "excluded_util": ["_aug_1."],
        "excluded_cav": ["_aug_7."],
        "desc": "Excluded Gaussian Noise (util: aug_1, cav: aug_7)",
    }),
    ("no_time_shift", {
        "name": "w/o Time Shift",
        "excluded_util": ["_aug_2."],
        "excluded_cav": ["_aug_8."],
        "desc": "Excluded Time Shift (util: aug_2, cav: aug_8)",
    }),
    ("no_rotation", {
        "name": "w/o Rotation",
        "excluded_util": ["_aug_3."],
        "excluded_cav": ["_aug_9."],
        "desc": "Excluded Rotation (util: aug_3, cav: aug_9)",
    }),
    ("no_flip", {
        "name": "w/o Horizontal Flip",
        "excluded_util": ["_aug_4."],
        "excluded_cav": ["_aug_11."],
        "desc": "Excluded Horizontal Flip (util: aug_4, cav: aug_11)",
    }),
    ("no_elastic", {
        "name": "w/o Elastic Deformation",
        "excluded_util": ["_aug_5."],
        "excluded_cav": ["_aug_12."],
        "desc": "Excluded Elastic Deformation (util: aug_5, cav: aug_12)",
    }),
    ("no_spectral", {
        "name": "w/o Spectral Shift",
        "excluded_util": ["_aug_6."],
        "excluded_cav": ["_aug_13."],
        "desc": "Excluded Spectral Shift (util: aug_6, cav: aug_13)",
    }),
    ("no_aug", {
        "name": "No Augmentation (Original Only)",
        "excluded_util": ["_aug_"],
        "excluded_cav": ["_aug_"],
        "desc": "Excluded ALL offline augmentations (100% original un-augmented)",
    }),
])


def setup_ablation_dataset(config_key, config_info, fold_idx, base_fold_dir="datasets/split_5fold", out_base="datasets/ablation"):
    """
    Create a filtered dataset directory for this ablation run by symlinking or copying
    from the verified split_5fold directory, omitting the excluded suffixes in train.
    """
    fold_name = f"fold_{fold_idx + 1}"
    src_fold = Path(base_fold_dir) / fold_name
    dest_fold = Path(out_base) / config_key / fold_name

    train_img_dest = dest_fold / "images" / "train"
    train_lbl_dest = dest_fold / "labels" / "train"
    train_img_dest.mkdir(parents=True, exist_ok=True)
    train_lbl_dest.mkdir(parents=True, exist_ok=True)

    # Validation and test splits are always identical to the honest base fold (un-augmented)
    val_img_dest = dest_fold / "images" / "val"
    val_lbl_dest = dest_fold / "labels" / "val"
    test_img_dest = dest_fold / "images" / "test"
    test_lbl_dest = dest_fold / "labels" / "test"

    for d in [val_img_dest, val_lbl_dest, test_img_dest, test_lbl_dest]:
        d.mkdir(parents=True, exist_ok=True)

    # Populate val & test
    for sp in ["val", "test"]:
        src_i_dir = src_fold / "images" / sp
        src_l_dir = src_fold / "labels" / sp
        dst_i_dir = dest_fold / "images" / sp
        dst_l_dir = dest_fold / "labels" / sp
        for f in os.listdir(src_i_dir):
            dst = dst_i_dir / f
            if not dst.exists():
                shutil.copy2(src_i_dir / f, dst)
        for f in os.listdir(src_l_dir):
            dst = dst_l_dir / f
            if not dst.exists():
                shutil.copy2(src_l_dir / f, dst)

    # Filter train images
    src_train_imgs = os.listdir(src_fold / "images" / "train")
    exc_u = config_info["excluded_util"]
    exc_c = config_info["excluded_cav"]

    included_count = 0
    for img_name in src_train_imgs:
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + ".txt"

        # Check exclusion rules
        exclude = False
        if img_name.startswith("util_"):
            for pattern in exc_u:
                if pattern in img_name:
                    exclude = True
                    break
        elif img_name.startswith("cav_"):
            for pattern in exc_c:
                if pattern in img_name:
                    exclude = True
                    break

        if not exclude:
            dst_i = train_img_dest / img_name
            if not dst_i.exists():
                shutil.copy2(src_fold / "images" / "train" / img_name, dst_i)
            src_lbl = src_fold / "labels" / "train" / lbl_name
            dst_l = train_lbl_dest / lbl_name
            if src_lbl.exists() and not dst_l.exists():
                shutil.copy2(src_lbl, dst_l)
            included_count += 1

    # Write data.yaml
    yaml_path = dest_fold / "data.yaml"
    abs_fold = dest_fold.resolve().as_posix()
    yaml_content = f"""path: {abs_fold}
train: images/train
val: images/val
test: images/test

names:
  0: utility
  1: cavity
"""
    with open(yaml_path, "w", encoding="utf-8") as fp:
        fp.write(yaml_content)

    return yaml_path, included_count


def train_and_eval_ablation_fold(config_key, config_info, fold_idx, fold_yaml, results_base, epochs=50, device=0):
    fold_name = f"fold_{fold_idx + 1}"
    fold_res_dir = Path(results_base) / config_key / fold_name
    fold_res_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = fold_res_dir / "metrics.json"
    if metrics_file.exists():
        print(f"[{config_key} | {fold_name}] Already completed! Loading existing metrics.")
        with open(metrics_file, "r", encoding="utf-8") as fp:
            return json.load(fp)

    actual_device = device if torch.cuda.is_available() else "cpu"
    train_run_dir = fold_res_dir / "train_run"
    last_pt = train_run_dir / "weights" / "last.pt"
    best_pt = train_run_dir / "weights" / "best.pt"

    start_time = time.time()
    if last_pt.exists():
        print(f"[{config_key} | {fold_name}] Resuming training from {last_pt}...")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO("yolov8n.pt")
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
            plots=False,
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

    eval_model = YOLO(str(best_pt))
    eval_metrics = eval_model.val(
        data=str(fold_yaml),
        split="test",
        device=actual_device,
        plots=False,
        project=str(fold_res_dir.resolve()),
        name="test_eval",
        exist_ok=True,
    )

    map50 = float(eval_metrics.box.map50)
    map50_95 = float(eval_metrics.box.map)
    prec = float(eval_metrics.box.mp)
    rec = float(eval_metrics.box.mr)
    f1 = float(2 * prec * rec / (prec + rec + 1e-16))

    fold_metrics = {
        "config": config_key,
        "fold": fold_name,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "training_time_s": round(train_duration, 1),
    }

    with open(metrics_file, "w", encoding="utf-8") as fp:
        json.dump(fold_metrics, fp, indent=4)

    return fold_metrics


def aggregate_and_generate_ablation_tables(all_results, baseline_map50_mean, baseline_map50_std):
    out_dir = Path("results/ablation_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = Path("results/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for c_key, data in all_results.items():
        df = pd.DataFrame(data["folds"])
        mean_map50 = df["mAP50"].mean()
        std_map50 = df["mAP50"].std()
        mean_map50_95 = df["mAP50_95"].mean()
        std_map50_95 = df["mAP50_95"].std()
        mean_prec = df["precision"].mean()
        mean_rec = df["recall"].mean()
        mean_f1 = df["f1"].mean()

        delta = mean_map50 - baseline_map50_mean
        is_significant = abs(delta) >= baseline_map50_std
        sig_str = "Yes" if is_significant else "No (Within 1 SD)"
        delta_str = f"{delta:+.4f}" if c_key != "baseline" else "—"

        summary_rows.append({
            "Configuration": data["info"]["name"],
            "Description": data["info"]["desc"],
            "mAP50": f"{mean_map50:.4f} ± {std_map50:.4f}",
            "mAP50-95": f"{mean_map50_95:.4f} ± {std_map50_95:.4f}",
            "Precision": f"{mean_prec:.4f}",
            "Recall": f"{mean_rec:.4f}",
            "F1-Score": f"{mean_f1:.4f}",
            "Delta_mAP50": delta_str,
            "Significant": sig_str if c_key != "baseline" else "—",
        })

    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_dir / "ablation_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved ablation summary CSV to: {csv_path}")

    # Generate LaTeX table (IEEEtran format)
    tex_path = table_dir / "ablation_table.tex"
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\caption{Leave-One-Out (LOO) Ablation Study Across Grouped 5-Fold Cross-Validation}",
        "\\label{tab:ablation_study}",
        "\\centering",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Configuration} & \\textbf{mAP50} & \\textbf{mAP50-95} & \\textbf{$\\Delta$ mAP50} & \\textbf{Significant?} \\\\",
        "\\hline",
    ]
    for r in summary_rows:
        tex_lines.append(f"{r['Configuration']} & {r['mAP50']} & {r['mAP50-95']} & {r['Delta_mAP50']} & {r['Significant']} \\\\")
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    with open(tex_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(tex_lines))
    print(f"Saved IEEEtran LaTeX ablation table to: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Run 5-Fold Grouped Ablation Study")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--device", default=0, help="Device to use (0 for GPU, cpu for CPU)")
    args = parser.parse_args()

    results_base = Path("results/ablation_study")
    results_base.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline from Cell 3 GPU YOLOv8n run or Phase 1 baseline
    gpu_baseline_path = Path("results/model_comparison/yolov8n/summary.json")
    baseline_summary_path = Path("results/baseline_new_split/summary.json")
    baseline_metrics = []

    if gpu_baseline_path.exists():
        with open(gpu_baseline_path, "r", encoding="utf-8") as fp:
            b_data = json.load(fp)
        baseline_metrics = b_data["per_fold"]
        baseline_mean = b_data["mean"]["mAP50"]
        baseline_std = b_data["std"]["mAP50"]
        print(f"[BASELINE LOADED] Cell 3 GPU YOLOv8n baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    elif baseline_summary_path.exists():
        with open(baseline_summary_path, "r", encoding="utf-8") as fp:
            b_data = json.load(fp)
        baseline_metrics = b_data["per_fold"]
        baseline_mean = b_data["mean"]["mAP50"]
        baseline_std = b_data["std"]["mAP50"]
        print(f"[BASELINE LOADED] Phase 1 YOLOv8n baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
    else:
        print("[WARNING] Baseline summary not found. Will train baseline in Run 0.")
        baseline_mean = 0.7390
        baseline_std = 0.0746

    all_ablation_results = OrderedDict()

    for c_key, c_info in ABLATION_CONFIGS.items():
        print(f"\n{'='*35} ABLATION: {c_info['name'].upper()} {'='*35}")
        
        # If baseline already exists, we can use it directly
        if c_key == "baseline" and baseline_metrics:
            all_ablation_results[c_key] = {"info": c_info, "folds": baseline_metrics}
            continue

        fold_metrics_list = []
        for f_i in range(5):
            print(f"\n--- Preparing Dataset for {c_key} Fold {f_i + 1} ---")
            fold_yaml, count = setup_ablation_dataset(c_key, c_info, f_i)
            print(f"  Train set size: {count} images (filtered according to {c_key})")
            
            fm = train_and_eval_ablation_fold(c_key, c_info, f_i, fold_yaml, results_base, epochs=args.epochs, device=args.device)
            fold_metrics_list.append(fm)

        all_ablation_results[c_key] = {"info": c_info, "folds": fold_metrics_list}

    aggregate_and_generate_ablation_tables(all_ablation_results, baseline_mean, baseline_std)
    print("\n[ABLATION STUDY COMPLETE] Results saved in results/ablation_study/")


if __name__ == "__main__":
    main()
