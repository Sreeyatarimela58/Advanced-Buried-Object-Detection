"""
experiments/01_split_source_level.py: 5-Fold Stratified Grouped Split and Baseline (Phase 1).
- Utility patches merged into groups with gap <= 3 (difference <= 4)
- 5-Fold cross-validation, stratified by class, keeping groups whole
- In each fold: 1 fold Test, 1 fold Validation, 3 folds Train (augmented)
- Validation and Test sets are strictly un-augmented original images
- Saves 5-fold split definition to results/split_new/split_5fold.json
- Generates side-by-side patch comparisons and visual bbox checks
"""
import os
import re
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw

from config import ROOT, RESULTS, DATA_SPLIT, SEEDS, CLASS_NAMES, TRAIN


def get_utility_groups(utility_dir, max_gap=3):
    """
    Merge utility patches separated by a gap of max_gap or fewer patches (diff <= max_gap + 1).
    Returns list of groups (lists of int patch IDs).
    """
    files = sorted([f for f in os.listdir(utility_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    nums = sorted([int(re.findall(r'\d+', f)[0]) for f in files if re.findall(r'\d+', f)])

    groups = []
    curr = [nums[0]]
    for n in nums[1:]:
        gap = n - curr[-1] - 1
        if gap <= max_gap:
            curr.append(n)
        else:
            groups.append(curr)
            curr = [n]
    groups.append(curr)
    return groups, files


def build_5fold_split(base_dir="GPR_data", seed=42):
    """
    Construct 5-fold stratified grouped split:
    - Utilities: 11 groups assigned across 5 folds keeping groups whole
    - Cavities: 79 independent images distributed across 5 folds
    - Intact: 75 independent images distributed across 5 folds
    """
    random.seed(seed)
    util_dir = os.path.join(base_dir, "Utilities")
    cav_dir = os.path.join(base_dir, "cavities")
    intact_dir = os.path.join(base_dir, "intact")

    # 1. Utilities (11 groups)
    groups, util_files = get_utility_groups(util_dir, max_gap=3)

    # Balanced assignment of the 11 groups:
    # G7 = [60], G2 = [17], G1+G11+G4 = [12+5+1=18], G3+G6 = [12+6=18], G5+G8+G9+G10 = [12+4+1+1=18]
    fold_util_groups = {
        0: [groups[1]],                                      # Group 2 (17)
        1: [groups[0], groups[10], groups[3]],               # Groups 1, 11, 4 (18)
        2: [groups[2], groups[5]],                           # Groups 3, 6 (18)
        3: [groups[4], groups[7], groups[8], groups[9]],     # Groups 5, 8, 9, 10 (18)
        4: [groups[6]],                                      # Group 7 (60)
    }

    fold_utilities = {}
    for f_idx in range(5):
        patches = []
        for g in fold_util_groups[f_idx]:
            patches.extend([f"{n:03d}.jpg" for n in g])
        fold_utilities[f_idx] = sorted(patches)

    # 2. Cavities (79 images)
    cav_files = sorted([f for f in os.listdir(cav_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(cav_files)
    fold_cavities = {
        0: sorted(cav_files[0:16]),
        1: sorted(cav_files[16:32]),
        2: sorted(cav_files[32:48]),
        3: sorted(cav_files[48:64]),
        4: sorted(cav_files[64:79]),
    }

    # 3. Intact (75 images)
    intact_files = sorted([f for f in os.listdir(intact_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(intact_files)
    fold_intact = {
        0: sorted(intact_files[0:15]),
        1: sorted(intact_files[15:30]),
        2: sorted(intact_files[30:45]),
        3: sorted(intact_files[45:60]),
        4: sorted(intact_files[60:75]),
    }

    folds = {}
    for f_idx in range(5):
        folds[f"fold_{f_idx + 1}"] = {
            "utility": fold_utilities[f_idx],
            "cavity": fold_cavities[f_idx],
            "intact": fold_intact[f_idx],
        }

    # Strict disjoint assertion across all 5 folds
    for i in range(5):
        for j in range(i + 1, 5):
            set_i = set([f"util_{x}" for x in fold_utilities[i]] + [f"cav_{x}" for x in fold_cavities[i]] + [f"intact_{x}" for x in fold_intact[i]])
            set_j = set([f"util_{x}" for x in fold_utilities[j]] + [f"cav_{x}" for x in fold_cavities[j]] + [f"intact_{x}" for x in fold_intact[j]])
            assert len(set_i & set_j) == 0, f"Folds {i+1} and {j+1} share items!"

    return folds


def count_instances_in_files(file_list, category, base_dir="GPR_data"):
    """
    Count bounding box instances in an image file list.
    """
    if category == "utility":
        lbl_dir = Path(base_dir) / "augmented_utilities" / "annotations" / "YOLO_format"
        suffix = "_aug_1.txt"
    elif category == "cavity":
        lbl_dir = Path(base_dir) / "augmented_cavities" / "annotations" / "Yolo_format"
        suffix = "_aug_7.txt"
    else:
        return 0

    count = 0
    for f in file_list:
        base = os.path.splitext(f)[0]
        lbl_p = lbl_dir / f"{base}{suffix}"
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as fp:
                count += len([line.strip() for line in fp if line.strip()])
    return count


def make_side_by_side(img_name1, img_name2, out_name, base_dir="GPR_data"):
    """
    Generate side-by-side comparison PNGs with ground truth bounding boxes.
    """
    p1 = Path(base_dir) / "Utilities" / img_name1
    p2 = Path(base_dir) / "Utilities" / img_name2

    im1 = Image.open(p1).convert("RGB")
    im2 = Image.open(p2).convert("RGB")

    lbl_dir = Path(base_dir) / "augmented_utilities" / "annotations" / "YOLO_format"
    base1 = os.path.splitext(img_name1)[0]
    base2 = os.path.splitext(img_name2)[0]

    def draw_boxes(im, base):
        draw = ImageDraw.Draw(im)
        w, h = im.size
        lbl_p = lbl_dir / f"{base}_aug_1.txt"
        if lbl_p.exists():
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2.0) * w
                        y1 = (yc - bh / 2.0) * h
                        x2 = (xc + bw / 2.0) * w
                        y2 = (yc + bh / 2.0) * h
                        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
                        draw.text((x1 + 3, y1 + 3), f"util ({xc:.2f}, {yc:.2f})", fill="cyan")

    draw_boxes(im1, base1)
    draw_boxes(im2, base2)

    banner_height = 30
    combined = Image.new("RGB", (im1.width + im2.width + 10, im1.height + banner_height), color=(30, 30, 30))
    combined.paste(im1, (0, banner_height))
    combined.paste(im2, (im1.width + 10, banner_height))

    draw = ImageDraw.Draw(combined)
    draw.text((10, 8), f"{img_name1} (Patch {base1})", fill="white")
    draw.text((im1.width + 20, 8), f"{img_name2} (Patch {base2})", fill="white")

    out_path = Path("results/split_new") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    return out_path


def main():
    print("=" * 65)
    print("PHASE 1: 5-FOLD STRATIFIED GROUPED SPLIT (GAP <= 3)")
    print("=" * 65)

    # 1. Generate side-by-side PNGs
    sb1 = make_side_by_side("011.jpg", "012.jpg", "side_by_side_011_012.png")
    sb2 = make_side_by_side("009.jpg", "011.jpg", "side_by_side_009_011.png")
    print(f"Generated side-by-side comparison: {sb1}")
    print(f"Generated side-by-side comparison: {sb2}")

    # 2. Build 5-Fold split
    folds = build_5fold_split(base_dir="GPR_data", seed=42)
    split_file = Path("results/split_new/split_5fold.json")
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=4)
    print(f"Saved 5-fold split definition to: {split_file}\n")

    # 3. Print counts per fold
    print("-" * 80)
    print(f"{'Fold':<8} {'Utility Img':<12} {'Cavity Img':<12} {'Intact Img':<12} {'Total Img':<12} {'Utility Boxes':<14} {'Cavity Boxes':<14}")
    print("-" * 80)

    summary_table = []
    for f_name, f_data in folds.items():
        u_imgs = len(f_data["utility"])
        c_imgs = len(f_data["cavity"])
        i_imgs = len(f_data["intact"])
        tot_imgs = u_imgs + c_imgs + i_imgs

        u_boxes = count_instances_in_files(f_data["utility"], "utility")
        c_boxes = count_instances_in_files(f_data["cavity"], "cavity")

        print(f"{f_name:<8} {u_imgs:<12} {c_imgs:<12} {i_imgs:<12} {tot_imgs:<12} {u_boxes:<14} {c_boxes:<14}")
        summary_table.append({
            "fold": f_name, "utility_images": u_imgs, "cavity_images": c_imgs, "intact_images": i_imgs,
            "total_images": tot_imgs, "utility_boxes": u_boxes, "cavity_boxes": c_boxes,
            "total_boxes": u_boxes + c_boxes
        })

    print("-" * 80)
    with open("results/split_new/counts_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=4)


if __name__ == "__main__":
    main()
