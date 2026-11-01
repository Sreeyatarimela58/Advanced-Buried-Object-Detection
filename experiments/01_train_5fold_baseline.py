"""
experiments/01_train_5fold_baseline.py: 5-Fold Stratified Grouped YOLOv8n Training & Evaluation.
- Trains YOLOv8n across all 5 folds using honest source-level grouped partitions.
- For each fold:
  - 1 fold is Test (un-augmented)
  - 1 fold is Validation (un-augmented, used for early stopping & checkpoint selection)
  - 3 folds are Train (augmented)
- Records per-fold metrics, logs online augmentation settings, and test predictions.
- Aggregates mean ± std, pooled confusion matrix, summary CSV/JSON, and IEEEtran LaTeX table.
"""
import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Single source of truth
from config import ROOT, RESULTS, CLASS_NAMES, TRAIN


def setup_fold_dataset(fold_idx, folds, base_dir="GPR_data", output_base="datasets/split_5fold"):
    """
    Prepare the image and label directories for a specific fold:
    Test: fold_idx
    Val: (fold_idx + 1) % 5
    Train: remaining 3 folds (augmented)
    """
    fold_name = f"fold_{fold_idx + 1}"
    val_idx = (fold_idx + 1) % 5
    train_indices = [i for i in range(5) if i != fold_idx and i != val_idx]

    fold_dir = Path(output_base) / fold_name
    for sp in ["train", "val", "test"]:
        (fold_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

    aug_util_dir = Path(base_dir) / "augmented_utilities"
    aug_util_lbl = aug_util_dir / "annotations" / "YOLO_format"
    aug_cav_dir = Path(base_dir) / "augmented_cavities"
    aug_cav_lbl = aug_cav_dir / "annotations" / "Yolo_format"

    orig_util_dir = Path(base_dir) / "Utilities"
    orig_cav_dir = Path(base_dir) / "cavities"
    orig_intact_dir = Path(base_dir) / "intact"

    def write_remapped_label(src_p, dst_p, target_cls):
        boxes = []
        if src_p and os.path.exists(src_p):
            with open(src_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for l in lines:
                parts = l.split()
                if len(parts) >= 5:
                    parts[0] = str(target_cls)
                    boxes.append(" ".join(parts))
        with open(dst_p, "w", encoding="utf-8") as f:
            for b in boxes:
                f.write(b + "\n")
        return len(boxes)

    # 1. Populate Test Fold (Un-augmented)
    test_data = folds[f"fold_{fold_idx + 1}"]
    for f in test_data["utility"]:
        base = os.path.splitext(f)[0]
        shutil.copy2(orig_util_dir / f, fold_dir / "images" / "test" / f"util_{f}")
        write_remapped_label(aug_util_lbl / f"{base}_aug_1.txt", fold_dir / "labels" / "test" / f"util_{base}.txt", 0)
    for f in test_data["cavity"]:
        base = os.path.splitext(f)[0]
        with Image.open(orig_cav_dir / f) as im:
            im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "test" / f"cav_{base}.jpg")
        write_remapped_label(aug_cav_lbl / f"{base}_aug_7.txt", fold_dir / "labels" / "test" / f"cav_{base}.txt", 1)
    for f in test_data["intact"]:
        safe = "intact_" + f.replace(" ", "_").replace("(", "").replace(")", "")
        with Image.open(orig_intact_dir / f) as im:
            im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "test" / safe)
        with open(fold_dir / "labels" / "test" / (os.path.splitext(safe)[0] + ".txt"), "w") as fp: pass

    # 2. Populate Val Fold (Un-augmented)
    val_data = folds[f"fold_{val_idx + 1}"]
    for f in val_data["utility"]:
        base = os.path.splitext(f)[0]
        shutil.copy2(orig_util_dir / f, fold_dir / "images" / "val" / f"util_{f}")
        write_remapped_label(aug_util_lbl / f"{base}_aug_1.txt", fold_dir / "labels" / "val" / f"util_{base}.txt", 0)
    for f in val_data["cavity"]:
        base = os.path.splitext(f)[0]
        with Image.open(orig_cav_dir / f) as im:
            im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "val" / f"cav_{base}.jpg")
        write_remapped_label(aug_cav_lbl / f"{base}_aug_7.txt", fold_dir / "labels" / "val" / f"cav_{base}.txt", 1)
    for f in val_data["intact"]:
        safe = "intact_" + f.replace(" ", "_").replace("(", "").replace(")", "")
        with Image.open(orig_intact_dir / f) as im:
            im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "val" / safe)
        with open(fold_dir / "labels" / "val" / (os.path.splitext(safe)[0] + ".txt"), "w") as fp: pass

    # 3. Populate Train Folds (Augmented)
    for t_idx in train_indices:
        t_data = folds[f"fold_{t_idx + 1}"]
        for f in t_data["utility"]:
            base = os.path.splitext(f)[0]
            # Copy original
            shutil.copy2(orig_util_dir / f, fold_dir / "images" / "train" / f"util_{f}")
            write_remapped_label(aug_util_lbl / f"{base}_aug_1.txt", fold_dir / "labels" / "train" / f"util_{base}.txt", 0)
            # Copy aug_1 to aug_6
            for a_i in range(1, 7):
                a_img = aug_util_dir / f"{base}_aug_{a_i}.jpg"
                if a_img.exists():
                    shutil.copy2(a_img, fold_dir / "images" / "train" / f"util_{base}_aug_{a_i}.jpg")
                    write_remapped_label(aug_util_lbl / f"{base}_aug_{a_i}.txt", fold_dir / "labels" / "train" / f"util_{base}_aug_{a_i}.txt", 0)

        for f in t_data["cavity"]:
            base = os.path.splitext(f)[0]
            # Resized original
            with Image.open(orig_cav_dir / f) as im:
                im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "train" / f"cav_{base}.jpg")
            write_remapped_label(aug_cav_lbl / f"{base}_aug_7.txt", fold_dir / "labels" / "train" / f"cav_{base}.txt", 1)
            # Copy aug_7 to aug_13
            for a_i in range(7, 14):
                a_img = aug_cav_dir / f"{base}_aug_{a_i}.jpg"
                if a_img.exists():
                    shutil.copy2(a_img, fold_dir / "images" / "train" / f"cav_{base}_aug_{a_i}.jpg")
                    write_remapped_label(aug_cav_lbl / f"{base}_aug_{a_i}.txt", fold_dir / "labels" / "train" / f"cav_{base}_aug_{a_i}.txt", 1)

        for f in t_data["intact"]:
            safe = "intact_" + f.replace(" ", "_").replace("(", "").replace(")", "")
            with Image.open(orig_intact_dir / f) as im:
                im.resize((224, 224), Image.Resampling.BILINEAR).save(fold_dir / "images" / "train" / safe)
            with open(fold_dir / "labels" / "train" / (os.path.splitext(safe)[0] + ".txt"), "w") as fp: pass

    # Write data.yaml for this fold
    yaml_path = fold_dir / "data.yaml"
    abs_fold = fold_dir.resolve().as_posix()
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

    # Perform strict leakage check on the created fold
    verify_fold_leakage(fold_dir)

    return yaml_path


def verify_fold_leakage(fold_dir):
    """
    Strict Leakage Verification:
    Extracts the root source identifier of every file in train, val, and test.
    Asserts that no source image appears in more than one partition.
    Prints [PASS] or [FAIL].
    """
    def get_source_id(filename):
        base = os.path.splitext(filename)[0]
        if "_aug_" in base:
            base = base.split("_aug_")[0]
        return base

    train_imgs = os.listdir(Path(fold_dir) / "images" / "train")
    val_imgs = os.listdir(Path(fold_dir) / "images" / "val")
    test_imgs = os.listdir(Path(fold_dir) / "images" / "test")

    train_sources = set(get_source_id(f) for f in train_imgs)
    val_sources = set(get_source_id(f) for f in val_imgs)
    test_sources = set(get_source_id(f) for f in test_imgs)

    leak_val = train_sources & val_sources
    leak_test = train_sources & test_sources
    leak_val_test = val_sources & test_sources

    if leak_val or leak_test or leak_val_test:
        print(f"  [FAIL] LEAKAGE DETECTED in {fold_dir}!")
        if leak_val: print(f"    Train/Val overlap ({len(leak_val)}): {sorted(list(leak_val))}")
        if leak_test: print(f"    Train/Test overlap ({len(leak_test)}): {sorted(list(leak_test))}")
        if leak_val_test: print(f"    Val/Test overlap ({len(leak_val_test)}): {sorted(list(leak_val_test))}")
        raise AssertionError(f"Leakage check failed for {fold_dir}!")
    else:
        print(f"  [PASS] Zero source-image leakage in {fold_dir}: "
              f"{len(train_sources)} train, {len(val_sources)} val, {len(test_sources)} test sources strictly disjoint.")
        return True



def is_training_complete(train_run_dir, target_epochs=50, patience=20):
    results_csv = train_run_dir / "results.csv"
    best_pt = train_run_dir / "weights" / "best.pt"
    if not (results_csv.exists() and best_pt.exists()):
        return False
    try:
        lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) <= 1:
            return False
        last_epoch = int(lines[-1].split(",")[0].strip())
        if last_epoch >= target_epochs:
            return True
        map50s = [float(l.split(",")[7].strip()) for l in lines[1:]]
        best_epoch_idx = int(np.argmax(map50s)) + 1
        if (last_epoch - best_epoch_idx) >= patience:
            return True
    except Exception:
        pass
    return False


def train_and_eval_fold(fold_idx, fold_yaml, results_base, epochs=50, smoke_test=False, device=0):
    """
    Train YOLOv8n on one fold and evaluate on the test split.
    """
    from ultralytics import YOLO
    import torch

    fold_name = f"fold_{fold_idx + 1}"
    fold_res_dir = Path(results_base) / fold_name
    fold_res_dir.mkdir(parents=True, exist_ok=True)

    # Use specified device if CUDA is available, else CPU
    actual_device = device if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"

    print(f"\n{'='*30} STARTING {fold_name.upper()} {'='*30}")
    print(f"Device: {gpu_name} (device={actual_device})")

    # Check if fold already completed
    metrics_file = fold_res_dir / "metrics.json"
    if metrics_file.exists() and not smoke_test:
        print(f"[{fold_name}] Already completed! Loading existing metrics.")
        with open(metrics_file, "r", encoding="utf-8") as fp:
            existing_metrics = json.load(fp)
        cm = np.array(existing_metrics["confusion_matrix"])
        return existing_metrics, cm

    abs_fold_res_dir = fold_res_dir.resolve()
    start_time = time.time()
    actual_epochs = 1 if smoke_test else epochs
    train_run_dir = fold_res_dir / "train_run"
    last_pt = train_run_dir / "weights" / "last.pt"
    best_pt = train_run_dir / "weights" / "best.pt"

    train_output = None
    if is_training_complete(train_run_dir, target_epochs=actual_epochs, patience=TRAIN["patience"]) and not smoke_test:
        print(f"[{fold_name}] Training already completed ({actual_epochs} epochs or early stopped). Proceeding to test evaluation.")
    elif last_pt.exists() and not smoke_test:
        print(f"[{fold_name}] Resuming training from {last_pt}...")
        model = YOLO(str(last_pt))
        train_output = model.train(resume=True)
    else:
        # Initialize model from scratch base
        model = YOLO("yolov8n.pt")
        train_output = model.train(
            data=str(fold_yaml),
            epochs=actual_epochs,
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
            project=str(abs_fold_res_dir),
            name="train_run",
            exist_ok=True,
            # Exact online augmentations matching original train.py
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
    if not best_pt.exists():
        if last_pt.exists():
            best_pt = last_pt

    # Evaluate best model on TEST split
    eval_model = YOLO(str(best_pt))
    eval_metrics = eval_model.val(
        data=str(fold_yaml),
        split="test",
        device=actual_device,
        plots=True,
        save_txt=True,
        save_conf=True,
        project=str(abs_fold_res_dir),
        name="test_eval",
        exist_ok=True,
    )

    # Extract metrics
    map50 = float(eval_metrics.box.map50)
    map50_95 = float(eval_metrics.box.map)
    prec = float(eval_metrics.box.mp)
    rec = float(eval_metrics.box.mr)
    f1 = float(2 * prec * rec / (prec + rec + 1e-16))

    # Per-class AP
    # class 0 = utility, class 1 = cavity
    ap50_util = float(eval_metrics.box.class_result(0)[0]) if len(eval_metrics.box.maps) > 0 else 0.0
    ap50_95_util = float(eval_metrics.box.class_result(0)[1]) if len(eval_metrics.box.maps) > 0 else 0.0
    ap50_cav = float(eval_metrics.box.class_result(1)[0]) if len(eval_metrics.box.maps) > 1 else 0.0
    ap50_95_cav = float(eval_metrics.box.class_result(1)[1]) if len(eval_metrics.box.maps) > 1 else 0.0

    # Confusion matrix
    conf_mat = eval_metrics.confusion_matrix.matrix.tolist()

    # Model parameters count
    params_m = round(sum(p.numel() for p in eval_model.model.parameters()) / 1e6, 3)

    # Cumulative training time from results.csv if available
    results_csv_file = train_run_dir / "results.csv"
    if results_csv_file.exists():
        try:
            r_lines = results_csv_file.read_text(encoding="utf-8").strip().splitlines()
            if len(r_lines) > 1:
                train_duration = float(r_lines[-1].split(",")[1].strip())
        except Exception:
            pass

    # Logged online augmentations
    online_aug_settings = {
        k: train_output.args.get(k)
        for k in ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste"]
        if train_output is not None and hasattr(train_output, "args") and k in train_output.args
    } if (train_output is not None and hasattr(train_output, "args")) else {}

    fold_metrics = {
        "fold": fold_name,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "utility": {
            "mAP50": ap50_util,
            "mAP50_95": ap50_95_util,
        },
        "cavity": {
            "mAP50": ap50_cav,
            "mAP50_95": ap50_95_cav,
        },
        "params_M": params_m,
        "size_MB": round(os.path.getsize(best_pt) / (1024 * 1024), 2) if best_pt.exists() else 0.0,
        "training_time_s": round(train_duration, 1),
        "gpu_name": gpu_name,
        "confusion_matrix": conf_mat,
        "online_augmentations": online_aug_settings,
    }

    with open(fold_res_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(fold_metrics, fp, indent=4)

    return fold_metrics, eval_metrics.confusion_matrix.matrix


def aggregate_5fold_results(all_fold_metrics, all_conf_matrices, results_base="results/baseline_new_split"):
    """
    Compute aggregate mean ± std across 5 folds, create pooled confusion matrix,
    and output summary CSV, JSON, and LaTeX tables.
    """
    out_dir = Path(results_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_list = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    summary_data = []

    for m in all_fold_metrics:
        row = {
            "fold": m["fold"],
            "mAP50": m["mAP50"],
            "mAP50_95": m["mAP50_95"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "utility_mAP50": m["utility"]["mAP50"],
            "cavity_mAP50": m["cavity"]["mAP50"],
            "training_time_s": m["training_time_s"],
        }
        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Compute mean and std
    means = df[metrics_list].mean()
    stds = df[metrics_list].std()

    summary_json = {
        "per_fold": summary_data,
        "mean": means.to_dict(),
        "std": stds.to_dict(),
        "leaky_old_baseline": {
            "mAP50": 0.9595,
            "mAP50_95": 0.6068,
            "precision": 0.952,
            "recall": 0.939,
            "f1": 0.9456
        }
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary_json, fp, indent=4)
    df.to_csv(out_dir / "summary.csv", index=False)

    # 1. Pooled Confusion Matrix
    pooled_raw = np.sum(all_conf_matrices, axis=0)
    # Normalize by row (true class)
    row_sums = pooled_raw.sum(axis=1, keepdims=True)
    pooled_norm = np.divide(pooled_raw, row_sums, out=np.zeros_like(pooled_raw, dtype=float), where=row_sums != 0)

    # Plot and save pooled confusion matrix
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["utility", "cavity", "background"]

    def plot_matrix(ax_sub, mat, title, fmt):
        im = ax_sub.imshow(mat, cmap="Blues")
        ax_sub.set_title(title)
        ax_sub.set_xticks(range(len(labels)))
        ax_sub.set_yticks(range(len(labels)))
        ax_sub.set_xticklabels(labels)
        ax_sub.set_yticklabels(labels)
        ax_sub.set_xlabel("Predicted")
        ax_sub.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mat[i, j]
                text = f"{val:{fmt}}"
                color = "white" if val > mat.max() / 2.0 else "black"
                ax_sub.text(j, i, text, ha="center", va="center", color=color)

    plot_matrix(ax[0], pooled_raw, "Pooled Confusion Matrix (Raw Counts)", ".0f")
    plot_matrix(ax[1], pooled_norm, "Pooled Confusion Matrix (Normalized)", ".3f")


    plt.tight_layout()
    pooled_cm_path = out_dir / "pooled_confusion_matrix.png"
    plt.savefig(pooled_cm_path, dpi=300)
    plt.close()
    print(f"Saved pooled confusion matrix: {pooled_cm_path}")

    # 2. LaTeX Table (IEEEtran style)
    latex_dir = Path("results/tables")
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = latex_dir / "baseline_table.tex"

    tex_content = f"""\\begin{{table}}[htbp]
\\caption{{Performance Comparison: Leaky Augmented Split vs.\\ Honest Grouped 5-Fold Cross-Validation}}
\\label{{tab:baseline_comparison}}
\\centering
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{Evaluation Setup}} & \\textbf{{mAP50}} & \\textbf{{mAP50-95}} & \\textbf{{Precision}} & \\textbf{{Recall}} & \\textbf{{F1-Score}} \\\\
\\hline
Augmented-split (old, leaky) & 0.9595 & 0.6068 & 0.9520 & 0.9390 & 0.9456 \\\\
Grouped 5-fold (new, honest) & {means['mAP50']:.4f} $\\pm$ {stds['mAP50']:.4f} & {means['mAP50_95']:.4f} $\\pm$ {stds['mAP50_95']:.4f} & {means['precision']:.4f} $\\pm$ {stds['precision']:.4f} & {means['recall']:.4f} $\\pm$ {stds['recall']:.4f} & {means['f1']:.4f} $\\pm$ {stds['f1']:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(tex_path, "w", encoding="utf-8") as fp:
        fp.write(tex_content)
    print(f"Saved LaTeX table to: {tex_path}")

    return summary_json


def main():
    parser = argparse.ArgumentParser(description="Run Grouped 5-Fold YOLOv8n Training & Evaluation")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test on fold 1 to verify code execution")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs per fold")
    parser.add_argument("--device", default=0, help="Device to use for training (e.g. 0 or cpu)")
    args = parser.parse_args()


    split_file = Path("results/split_new/split_5fold.json")
    if not split_file.exists():
        print("[ERROR] results/split_new/split_5fold.json not found! Run 01_split_source_level.py first.")
        sys.exit(1)

    with open(split_file, "r", encoding="utf-8") as fp:
        folds = json.load(fp)

    results_base = Path("results/baseline_new_split")
    results_base.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_matrices = []

    num_folds_to_run = 1 if args.smoke_test else 5

    for f_i in range(num_folds_to_run):
        print(f"\n--- Setting up dataset for Fold {f_i + 1} ---")
        fold_yaml = setup_fold_dataset(f_i, folds, base_dir="GPR_data", output_base="datasets/split_5fold")
        m, cm = train_and_eval_fold(f_i, fold_yaml, results_base, epochs=args.epochs, smoke_test=args.smoke_test)
        all_metrics.append(m)
        all_matrices.append(cm)

    if not args.smoke_test:
        aggregate_5fold_results(all_metrics, all_matrices, results_base=results_base)
        print("\n[ALL 5 FOLDS COMPLETE] Summary saved in results/baseline_new_split/")
    else:
        print("\n[SMOKE TEST COMPLETE] 1-epoch execution on Fold 1 passed successfully.")


if __name__ == "__main__":
    main()
