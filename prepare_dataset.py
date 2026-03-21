# -*- coding: utf-8 -*-
"""
Prepare YOLO Dataset for Multi-Class GPR Buried Object Detection.

Reorganizes the GPR_data/ folder into YOLO's required directory structure:
    yolo_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

Classes:
    0 = Utility (underground pipes/cables)
    1 = Cavity  (underground voids)

Intact images are excluded (no annotations).
"""

import os
import shutil
import random
from tqdm import tqdm


def prepare_dataset(
    base_dir="GPR_data",
    output_dir="yolo_dataset",
    split_ratio=0.8,
    seed=42,
):
    """
    Prepare the YOLO dataset from the existing GPR_data folder.

    Args:
        base_dir: Path to the GPR_data folder.
        output_dir: Path to create the YOLO dataset.
        split_ratio: Fraction of data for training (rest goes to val).
        seed: Random seed for reproducibility.
    """
    random.seed(seed)

    # Define sources: each has image dir, label dir, and optional class remap
    sources = {
        "utility": {
            "images": os.path.join(base_dir, "augmented_utilities"),
            "labels": os.path.join(base_dir, "augmented_utilities", "annotations", "YOLO_format"),
            "class_remap": {},  # Keep class 0 as-is
        },
        "cavity": {
            "images": os.path.join(base_dir, "augmented_cavities"),
            "labels": os.path.join(base_dir, "augmented_cavities", "annotations", "Yolo_format"),
            "class_remap": {0: 1},  # Remap class 0 → 1 for cavities
        },
    }

    # Create output directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    class_stats = {"train": {"utility": 0, "cavity": 0}, "val": {"utility": 0, "cavity": 0}}

    for category, paths in sources.items():
        img_dir = paths["images"]
        lbl_dir = paths["labels"]
        remap = paths["class_remap"]

        # Gather image files (exclude subdirectories like 'annotations')
        all_images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Shuffle for random split
        random.shuffle(all_images)

        split_idx = int(len(all_images) * split_ratio)
        splits = {
            "train": all_images[:split_idx],
            "val": all_images[split_idx:],
        }

        for split, img_list in splits.items():
            for img_file in tqdm(img_list, desc=f"{category}/{split}"):
                # Prefix to avoid filename collisions between categories
                prefix = "util_" if category == "utility" else "cav_"
                safe_name = prefix + img_file.replace(" ", "_")

                # --- Copy image ---
                src_img = os.path.join(img_dir, img_file)
                dst_img = os.path.join(output_dir, "images", split, safe_name)
                shutil.copy2(src_img, dst_img)
                stats[split]["images"] += 1

                # --- Copy / remap label ---
                label_name = os.path.splitext(img_file)[0] + ".txt"
                src_lbl = os.path.join(lbl_dir, label_name)

                safe_label = prefix + label_name.replace(" ", "_")
                dst_lbl = os.path.join(output_dir, "labels", split, safe_label)

                if os.path.exists(src_lbl):
                    if remap:
                        # Read, remap class IDs, and write
                        with open(src_lbl, "r", encoding='utf-8') as f:
                            lines = f.readlines()
                        with open(dst_lbl, "w", encoding='utf-8') as f:
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    parts[0] = str(remap.get(cls_id, cls_id))
                                    f.write(" ".join(parts) + "\n")
                    else:
                        shutil.copy2(src_lbl, dst_lbl)

                    stats[split]["labels"] += 1
                    class_stats[split][category] += 1
                else:
                    # Create empty label file (image has no objects)
                    with open(dst_lbl, "w") as f:
                        pass
                    stats[split]["labels"] += 1

    # Print summary
    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print(f"\n{'Split':<8} {'Images':>8} {'Labels':>8} {'Utilities':>10} {'Cavities':>10}")
    print("-" * 50)
    for split in ["train", "val"]:
        print(
            f"{split:<8} {stats[split]['images']:>8} {stats[split]['labels']:>8} "
            f"{class_stats[split]['utility']:>10} {class_stats[split]['cavity']:>10}"
        )
    total_imgs = stats["train"]["images"] + stats["val"]["images"]
    total_util = class_stats["train"]["utility"] + class_stats["val"]["utility"]
    total_cav = class_stats["train"]["cavity"] + class_stats["val"]["cavity"]
    print("-" * 50)
    print(f"{'TOTAL':<8} {total_imgs:>8} {total_imgs:>8} {total_util:>10} {total_cav:>10}")

    # Verify a sample cavity label was remapped
    print("\n--- Verification ---")
    sample_cav_labels = [
        f for f in os.listdir(os.path.join(output_dir, "labels", "train"))
        if f.startswith("cav_")
    ]
    if sample_cav_labels:
        sample_path = os.path.join(output_dir, "labels", "train", sample_cav_labels[0])
        with open(sample_path, "r") as f:
            content = f.read().strip()
        print(f"Sample cavity label ({sample_cav_labels[0]}):")
        print(f"  Content: {content}")
        if content.startswith("1 "):
            print("  ✓ Class correctly remapped to 1")
        else:
            print("  ✗ WARNING: Class may not be remapped correctly!")

    print(f"\nDataset ready for training! Use: python train.py")


if __name__ == "__main__":
    prepare_dataset()
