"""
experiments/03_eval_external.py: Evaluation on external GPR dataset (Phase 3).
Inspects dataset structure, checks for bounding boxes, and handles Test A (zero-shot) and Test B (fine-tuning).
DO NOT RUN until the external dataset path is confirmed and provided by user.
"""
import sys
import os
from pathlib import Path
import argparse

# Single source of truth
from config import ROOT, RESULTS, DATA_EXTERNAL, SEEDS


def inspect_external_dataset(dataset_path):
    """
    Inspect the external dataset structure.
    Checks:
    1. Image count
    2. Existence and format of bounding box labels
    3. Detectable classes
    4. Official train/val/test split presence
    """
    path = Path(dataset_path)
    if not path.exists():
        print(f"[ERROR] External dataset path does not exist: {dataset_path}")
        return False, {}

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = [p for p in path.rglob("*") if p.suffix.lower() in image_extensions]
    all_labels = [p for p in path.rglob("*") if p.suffix.lower() in (".txt", ".xml", ".json")]

    print("\n" + "=" * 50)
    print("EXTERNAL DATASET INSPECTION")
    print("=" * 50)
    print(f"Path: {dataset_path}")
    print(f"Total image files found: {len(all_images)}")
    print(f"Total label/annotation files found: {len(all_labels)}")

    if len(all_images) == 0:
        print("[ERROR] No images found in dataset directory.")
        return False, {}

    # Check for bounding box labels
    valid_box_labels = []
    classes_found = set()
    for lbl in all_labels:
        if lbl.suffix.lower() == ".txt":
            try:
                with open(lbl, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO bbox: class xc yc w h
                        valid_box_labels.append(lbl)
                        classes_found.add(parts[0])
            except Exception:
                continue

    print(f"Label files with valid bounding boxes: {len(valid_box_labels)}")
    print(f"Classes detected in labels: {sorted(list(classes_found))}")

    # Check for train / val / test structure
    subdirs = [p.name.lower() for p in path.iterdir() if p.is_dir()]
    has_official_split = any(k in subdirs for k in ["train", "val", "valid", "test"])
    print(f"Official split detected: {has_official_split} (Subdirectories: {subdirs})")

    info = {
        "num_images": len(all_images),
        "num_box_labels": len(valid_box_labels),
        "classes": sorted(list(classes_found)),
        "has_split": has_official_split,
        "path": str(path),
    }

    if len(valid_box_labels) == 0:
        print("\n" + "!" * 50)
        print("CRITICAL STOP: NO BOUNDING-BOX LABELS FOUND!")
        print("TASK.md requirement: If there are NO box labels, STOP and ask me.")
        print("!" * 50 + "\n")
        return False, info

    return True, info


def main():
    parser = argparse.ArgumentParser(description="External GPR Dataset Evaluation (Phase 3)")
    parser.add_argument("--data_dir", type=str, default=str(DATA_EXTERNAL) if DATA_EXTERNAL else "",
                        help="Path to external GPR dataset")
    args = parser.parse_args()

    if not args.data_dir or args.data_dir == "None":
        print("[STOP] External dataset path is not set in config.py or --data_dir.")
        print("Please provide the external dataset path before running Phase 3.")
        sys.exit(1)

    has_boxes, info = inspect_external_dataset(args.data_dir)
    if not has_boxes:
        print("[STOP] Cannot proceed without verified bounding box labels.")
        sys.exit(1)

    print("\n[OK] External dataset inspected successfully.")
    print("Ready for Test A (zero-shot) and Test B (fine-tuning) implementation once approved.")


if __name__ == "__main__":
    main()
